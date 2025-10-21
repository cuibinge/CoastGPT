"""
图像分类任务测评脚本
"""

import json
import logging
import os
import re
from difflib import SequenceMatcher  # 用于字符串相似度比较
from typing import Dict, List

import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn  # CUDA优化
import torch.distributed as dist  # 分布式训练支持
import wandb  # 实验跟踪工具
from Trainer import init_distributed
from Trainer.utils import ConfigArgumentParser, setup_logger, str2bool
# 从自定义数据加载器构建函数导入
from Dataset.build_loader import build_zero_shot_loader
# 从对话模板导入默认对话设置
from Dataset.conversation import default_conversation
# 导入CoastGPT模型
from Models.coastgpt import CoastGPT
from Models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    #build_model,
    tokenizer_image_token,
)
# 图像处理器
from transformers import CLIPImageProcessor
# 评估指标
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm  # 进度条

# 数据类型映射
type_dict = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
logger = logging.getLogger("train")

# 分类任务模板
CLS_TEMPLATE = [lambda c: f"[CLS] Choose the best categories describe the image from: {c}"]


def find_index_of_max_similar_substring(given_string, string_list):
    """
    在字符串列表中找到与给定字符串最相似的字符串索引
    
    参数:
        given_string: 待比较的字符串
        string_list: 字符串列表
        
    返回:
        max_index: 最相似字符串的索引
    """
    max_similarity = 0
    max_index = -1

    # 遍历字符串列表，计算每个字符串与给定字符串的最长公共子串长度
    for i, string in enumerate(string_list):
        similarity = (
            SequenceMatcher(None, given_string, string)
            .find_longest_match(0, len(given_string), 0, len(string))
            .size
        )
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = i

    return max_index


def classname_2_idx(preds: List[str], classes_to_idx: Dict[str, int]):
    """
    将预测的类别名称转换为对应的索引
    
    参数:
        preds: 预测的类别名称列表
        classes_to_idx: 类别名称到索引的映射字典
        
    返回:
        results: 对应的索引列表
    """
    results = []
    classes = list(classes_to_idx.keys())
    for pred in preds:
        pred = pred.strip()
        if pred in classes:
            # 如果预测结果在类别列表中，直接使用对应索引
            results.append(classes_to_idx[pred])
        else:
            # 否则找到最相似的类别
            index = find_index_of_max_similar_substring(pred, classes)
            results.append(classes_to_idx[classes[index]])
    return results


def parse_option():
    """
    解析命令行参数和配置文件
    
    返回:
        config: 配置对象
    """
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # 基本参数
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    parser.add_argument("--model-path", type=str, default=None, help="pretrained checkpoint path")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="mixed precision")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )

    # wandb参数
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    parser.add_argument("--project", type=str, default="MaskIndexNet", help="wandb project")

    # 硬件参数
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--local_rank", type=int, help="local rank")

    # 解析参数并转换为配置字典
    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config: ml_collections.ConfigDict):
    """主分类推理函数"""
    logger.info(f"Creating model")
    # 创建CoastGPT模型
    model = CoastGPT(config)
    # 设置数据类型
    dtype = type_dict[config.dtype]
    model.to(dtype)

    # 构建零样本分类数据加载器
    data_loader_train = build_zero_shot_loader(config, mode="zero_shot_cls")

    # 加载预训练模型权重
    if config.model_path is not None:
        logger.info(f"Loading pretrained checkpoint from {config.model_path}")
        if getattr(model, "custom_load_state_dict", False):
            # 使用自定义加载方法
            msg = model.custom_load_state_dict(config.model_path)
        else:
            # 标准加载方法
            ckpt = torch.load(config.model_path, map_location="cpu")
            msg = model.load_state_dict(ckpt["model"], strict=False)
        if msg is not None:
            logger.info(f"After loading, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
            logger.info(str(model))

    # 设置设备
    if config.accelerator == "gpu":
        if config.is_distribute:
            device = torch.device(getattr(config, "local_rank", 0))
        elif (
            "CUDA_VISABLE_DEVICE" in os.environ.keys() and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()  # 设置为评估模式

    # 获取类别信息
    if hasattr(data_loader_train.dataset, "classes"):
        all_classes = data_loader_train.dataset.classes
    else:
        all_classes = data_loader_train.dataset.CLASS_NAME
    # 规范化类别名称（小写并替换下划线）
    all_classes = [i.lower().replace("_", " ") for i in all_classes]
    # 创建类别名称到索引的映射
    classes_2_idx = {classname: idx for idx, classname in enumerate(all_classes)}
    # 构建分类提示
    inp = CLS_TEMPLATE[0](all_classes)

    # 设置对话模板
    conv = default_conversation.copy()
    roles = conv.roles

    # 根据配置添加图像标记
    if config.tune_im_start:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    # 构建对话提示
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 将提示转换为token ID
    input_ids = (
        tokenizer_image_token(prompt, model.language.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    # 重复输入以匹配批次大小
    input_ids = input_ids.repeat(config.batch_size, 1)
    model.eval()
    
    # 开始推理
    with torch.no_grad():
        preds = []  # 存储预测结果
        trues = []  # 存储真实标签
        
        # 使用进度条遍历数据加载器
        for image, target in tqdm(data_loader_train, unit_scale=config.batch_size, desc="Evaluating"):
            image = image.to(dtype).to(device)
            
            # 处理最后一个批次可能的大小不匹配问题
            if input_ids.shape[0] != image.shape[0]:
                input_ids = input_ids[: image.shape[0]]

            # 使用自动混合精度（如果启用）
            with torch.autocast(
                device_type="cuda" if config.accelerator == "gpu" else "cpu",
                enabled=config.enable_amp,
                dtype=dtype,
            ):
                # 使用模型生成预测
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image,
                    do_sample=False,  # 不使用采样
                    num_beams=1,  # 使用贪婪搜索
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=20 if config.eval.dataset != "METERML" else 30,  # 根据数据集调整生成长度
                )

            # 解码生成的token ID为文本
            outputs = model.language.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds += outputs  # 收集预测结果
            trues.append(target.cpu())  # 收集真实标签

    # 将预测的类别名称转换为索引
    preds = classname_2_idx(preds, classes_2_idx)
    # 合并所有真实标签
    trues = torch.cat(trues)
    # 计算平衡准确率
    mean_per_class_recall = balanced_accuracy_score(trues, preds)
    # 输出分类报告
    logger.info(classification_report(trues, preds, digits=3, target_names=all_classes))
    logger.info(mean_per_class_recall)  # 输出平衡准确率


if __name__ == "__main__":
    # 解析配置
    config = parse_option()

    # 初始化分布式设置
    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    config.adjust_norm = False
    print(config)

    # 设置日志记录器
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)

    # 设置随机种子以确保可重复性
    if config.is_distribute:
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True  # 启用CuDNN自动优化

    # 主进程保存完整配置
    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    # 初始化W&B（如果启用）
    if config.wandb and config.rank == 0:
        wandb.init(config=config.to_dict(), entity=config.entity, project=config.project)
        config = wandb.config

    # 运行主函数
    main(config)
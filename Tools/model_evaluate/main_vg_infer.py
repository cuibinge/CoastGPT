"""
视觉定位任务推理脚本，负责生成预测结果 json 文件
------------------------------------------------
功能：
1. 解析命令行配置
2. 初始化（包括分布式设置）
3. 加载模型（CoastGPT）和数据集
4. 对测试集进行推理（支持混合精度）
5. 生成预测结果 json 文件
"""
import json
import logging
import os
import re

import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn  # CUDA优化
import torch.distributed as dist  # 分布式训练支持
import wandb  # 实验跟踪工具
from Trainer import init_distributed
from Trainer.utils import ConfigArgumentParser, setup_logger, str2bool
from Trainer.utils.distribute import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
from Dataset import DataCollatorForVGSupervisedDataset, VGEvalDataset  # 视觉定位数据集和数据处理
from Dataset.conversation import default_conversation  # 默认对话模板
from Models.coastgpt import CoastGPT  # 主要模型

from tqdm import tqdm  # 进度条显示
from transformers import CLIPImageProcessor  # 图像预处理

# 数据类型映射
type_dict = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
logger = logging.getLogger("train")  # 创建日志记录器


def save_result(result, result_dir, filename, remove_duplicate=""):
    """
    保存推理结果到JSON文件，支持分布式环境下的结果合并
    
    参数:
        result: 要保存的结果数据
        result_dir: 结果目录路径
        filename: 输出文件名
        remove_duplicate: 去重字段名（可选）
    """
    # 为每个进程创建单独的结果文件
    result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
    # 最终合并后的结果文件
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    # 保存当前进程的结果
    json.dump(result, open(result_file, "w"))

    # 分布式环境下等待所有进程完成
    if is_distributed():
        dist.barrier()

    # 主进程负责合并所有结果
    if is_main_process():
        # 合并所有进程的结果
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
            res = json.load(open(result_file, "r"))
            result += res

        # 如果需要去重
        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        # 保存最终结果
        json.dump(result, open(final_result_file, "w"))
        logger.info("result file saved to %s" % final_result_file)

    return final_result_file


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
    parser.add_argument("--data-path", type=str, default="Data/Stage2Data/RSVG_Image", help="path to dataset")
    parser.add_argument("--data-target", type=str, default="Data/Stage2Data/RSVG.json", help="path to dataset annotation file ")
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    parser.add_argument("--model-path", type=str, default="Checkpoint/test/checkpoints/FINAL.pt", help="pretrained checkpoint path")
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
    """主视觉定位推理函数"""
    logger.info(f"Creating model")
    # 创建CoastGPT模型
    model = CoastGPT(config)
    # 设置数据类型
    dtype = type_dict[config.dtype]
    model.to(dtype)

    # 加载CLIP图像处理器用于图像预处理
    vis_transform = CLIPImageProcessor.from_pretrained("/root/autodl-tmp/clip-vit-large-patch14")
    
    # 创建视觉定位评估数据集
    dataset = VGEvalDataset(
        root=config.data_path,
        target=config.data_target,
        transform=vis_transform,
        tokenizer=model.language.tokenizer,
    )
    logger.info(f"Data Length: {len(dataset)}")

    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.workers,
        pin_memory=True,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForVGSupervisedDataset(model.language.tokenizer),
    )

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
            "CUDA_VISABLE_DEVICES" in os.environ.keys() and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()  # 设置为评估模式

    preds = []  # 存储预测结果
    with torch.no_grad():  # 禁用梯度计算
        # 使用进度条遍历数据加载器
        for image, input_ids, targets, file_name, attention_mask in tqdm(
            data_loader, unit_scale=config.batch_size, desc="Evaluating"
        ):
            # 将数据移动到相应设备
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 处理最后一个批次可能的大小不匹配问题
            if input_ids.shape[0] != image.shape[0]:
                input_ids = input_ids[: image.shape[0]]

            # 使用自动混合精度（如果启用）
            with torch.autocast(
                device_type="cuda" if config.accelerator == "gpu" else "cpu",
                enabled=config.enable_amp,
                dtype=dtype,
            ):
                # 使用模型生成边界框预测
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image,
                    num_beams=1,  # 使用贪婪搜索
                    attention_mask=attention_mask,
                    do_sample=False,  # 不使用采样
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=100,  # 最大生成长度
                )

            # 解码生成的token ID为文本
            outputs = model.language.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.strip() for output in outputs]
            
            # 收集预测结果
            for pred, target, name in zip(outputs, targets, file_name):
                preds.append(dict(pred=pred, target=target, filename=name))

    # 保存结果到JSON文件
    save_result(preds, config.output, "rsvg_result", "filename")
    
    # 注意：原代码中计算IoU指标的部分已被删除


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
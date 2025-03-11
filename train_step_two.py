import deepspeed
import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from Trainer import deepspeed_init_distributed
from Trainer.EpochBasedTrainer import EpochBasedTrainer
from Trainer.utils import (
    ConfigArgumentParser,
    auto_resume_helper,
    setup_logger,
    str2bool,
)
from Dataset.build_loader import build_loader
from models import build_model
from optimizer import build_optimizer
import logging
import os
import json

# 创建日志记录器，用于记录训练过程中的信息
logger = logging.getLogger("train")

# 构建DeepSpeed配置
def build_ds_config(config: ml_collections.ConfigDict):
    # 将优化器名称转换为小写
    opt_lower = config.optimizer.lower()
    if opt_lower == "adamw":
        # 定义AdamW优化器的参数
        optimizer = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,  # 学习率
                "eps": 1e-8,  # 防止除零的小常数
                "betas": (0.9, 0.95),  # 用于计算梯度一阶矩和二阶矩的指数衰减率
                "weight_decay": config.wd,  # 权重衰减系数
            },
        }

        # 构建DeepSpeed配置字典
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,  # 每个GPU的微批量大小
            "optimizer": optimizer,  # 优化器配置
            "fp16": {
                "enabled": True if config.fp16 else False,  # 是否启用FP16混合精度训练
                "auto_cast": False,  # 是否自动进行类型转换
                "initial_scale_power": 16,  # 初始缩放因子的幂
                "loss_scale_window": 500,  # 损失缩放窗口大小
            },
            "bf16": {
                "enabled": True if config.bf16 else False,  # 是否启用BF16混合精度训练
                "auto_cast": False,  # 是否自动进行类型转换
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO优化阶段
                "sub_group_size": 1e9,  # 子组大小
                "contiguous_gradients": True,  # 是否使用连续梯度
                "overlap_comm": True,  # 是否重叠通信和计算
                "stage3_gather_16bit_weights_on_model_save": True,  # 在保存模型时是否收集16位权重
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪的最大范数
        }

    else:
        # 其他优化器的DeepSpeed配置
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,  # 每个GPU的微批量大小
            "bf16": {
                "enabled": True,  # 是否启用BF16混合精度训练
                "auto_cast": True,  # 是否自动进行类型转换
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO优化阶段
                "offload_optimizer": {
                    "device": "cpu",  # 将优化器状态卸载到CPU
                },
                "offload_param": {"device": "cpu"},  # 将模型参数卸载到CPU
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪的最大范数
            "zero_force_ds_cpu_optimizer": False,  # 是否强制使用DeepSpeed的CPU优化器
            "zero_allow_untested_optimizer": True,  # 是否允许使用未测试的优化器
        }

    return ds_config

# 解析命令行参数
def parse_option():
    # 创建配置参数解析器
    parser = ConfigArgumentParser()
    # 添加可修改配置选项的参数
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # 基本配置参数
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")  # 单个GPU的批量大小
    parser.add_argument("--data-path", type=str, help="path to dataset")  # 数据集路径
    parser.add_argument("--eval-data-path", type=str, help="path to evaluate dataset")  # 评估数据集路径
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")  # 数据加载器的工作进程数
    parser.add_argument(
        "--auto-resume", action="store_true", help="resume from checkpoint"
    )  # 是否自动从检查点恢复训练
    parser.add_argument(
        "--resume-path", type=str, default=None, help="resume checkpoint path"
    )  # 恢复训练的检查点路径
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="pretrained checkpoint path for model (maybe stage 1)",
    )  # 模型的预训练检查点路径
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )  # 梯度累积步数
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )  # 是否使用梯度检查点来节省内存
    parser.add_argument(
        "--enable-amp", type=str2bool, default=False, help="mixed precision"
    )  # 是否启用混合精度训练
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )  # 输出文件夹的根路径
    parser.add_argument("--seed", type=int, default=322, help="random seed")  # 随机种子
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")  # GPU ID
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )  # 是否使用无限加载器
    parser.add_argument(
        "--torch-compile",
        type=str2bool,
        default=False,
        help="Use torch.compile to accelerate model or not",
    )  # 是否使用torch.compile加速模型

    # Wandb配置参数
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")  # 是否使用Wandb记录日志
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")  # Wandb实体名称
    parser.add_argument(
        "--project", type=str, default="MultiModal", help="wandb project"
    )  # Wandb项目名称
    parser.add_argument(
        "--job-type", type=str, default="vlm_test", help="wandb job_type"
    )  # Wandb作业类型
    parser.add_argument(
        "--tags", type=str, default="MultiModal", nargs="+", help="wandb tags"
    )  # Wandb标签
    parser.add_argument("--name", type=str, default="first_run", help="wandb run name")  # Wandb运行名称
    parser.add_argument("--notes", type=str, default=None, help="wandb run's notes")  # Wandb运行的备注

    # 硬件配置参数
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )  # 加速器类型
    parser.add_argument("--local_rank", type=int)  # 本地进程的排名

    # 解析命令行参数
    config = parser.parse_args(wandb=True)
    # 将解析结果转换为ConfigDict类型
    config = ml_collections.config_dict.ConfigDict(config)

    return config

# 主训练函数
def main(config):
    # 记录创建模型的信息
    logger.info(f"Creating model")
    # 构建模型
    model = build_model(
        config,
        activate_modal=("rgb", "text"),
    )
    # 记录模型信息
    logger.info(str(model) + "\n")

    # 记录构建数据集的信息
    logger.info(f"Building Dataset")
    # 构建训练数据加载器
    data_loader_train = build_loader(
        config,
        mode="pretrain",
        tokenizer=model.text.tokenizer,
        prompt_type=config.prompt_template,
    )

    # 根据配置确定计算数据类型
    compute_dtype = (
        torch.float16
        if config.fp16
        else (torch.bfloat16 if config.bf16 else torch.float32)
    )

    # 准备模型进行训练
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,  # 是否冻结视觉模块
        freeze_text=not config.lora.enable,  # 是否冻结文本模块
        tune_rgb_pooler=config.tune_rgb_pooler,  # 是否调整RGB池化层
        model_path=config.model_path,  # 模型的预训练路径
        tune_im_start=config.tune_im_start,  # 是否调整图像起始层
        compute_dtype=compute_dtype,  # 计算数据类型
    )

    # 根据优化器类型处理参数和优化器
    if config.optimizer.lower() == "adamw":
        parameter = None
        optimizer = None
    else:
        parameter = None
        optimizer = build_optimizer(model, config, is_pretrain=True)

    # 初始化DeepSpeed模型引擎、优化器等
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),  # DeepSpeed配置
        model=model,  # 模型
        optimizer=optimizer if optimizer is not None else None,  # 优化器
        model_parameters=parameter if parameter is not None else None,  # 模型参数
    )

    # 创建基于Epoch的训练器
    trainer = EpochBasedTrainer(
        model=model_engine,  # 模型引擎
        optimizer=optimizer,  # 优化器
        lr_scheduler=config.schedule,  # 学习率调度器
        data_loader=data_loader_train,  # 训练数据加载器
        max_epochs=config.epochs,  # 最大训练轮数
        work_dir=config.output,  # 工作目录
        log_period=1,  # 日志记录周期
        save_ckpt_by="iter",  # 保存检查点的方式
        ckpt_period=100,  # 检查点保存周期
        accelerator=config.accelerator,  # 加速器类型
        enable_amp=config.enable_amp,  # 是否启用混合精度训练
        wandb=config.wandb,  # 是否使用Wandb记录日志
        gpus=0,  # GPU ID
        max_num_checkpoints=1,  # 最大检查点数量
        clip_grad_norm=config.max_grad_norm,  # 梯度裁剪的最大范数
        is_distributed=config.is_distribute,  # 是否使用分布式训练
        torch_compile=config.torch_compile,  # 是否使用torch.compile加速模型
        dtype=compute_dtype,  # 计算数据类型
        deepspeed=True,  # 是否使用DeepSpeed
    )

    # 自动恢复训练
    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)  # 查找自动恢复的检查点文件
        if resume_file:
            if config.resume_path is not None:
                logger.warning(
                    f"auto-resume changing resume file from {config.resume_path} to {resume_file}"
                )
            config.resume_path = resume_file  # 更新恢复路径
            logger.info(f"auto resuming from {resume_file}")  # 记录自动恢复信息
        else:
            logger.info(
                f"no checkpoint found in {config.output}/checkpoint, ignoring auto resume"
            )  # 未找到检查点，忽略自动恢复

    # 开始训练
    trainer.train(load_checkpoint=config.resume_path)

    # 保存最终模型
    if config.local_rank == 0 or config.local_rank == -1:
        state_dict = model.custom_save_checkpoint(
            os.path.join(config.output, "checkpoints")
        )  # 自定义保存模型状态字典
        torch.save(
            state_dict,
            os.path.join(os.path.join(config.output, "checkpoints"), "FINAL.pt"),
        )  # 保存最终模型

if __name__ == "__main__":
    # 解析命令行参数
    config = parse_option()

    # 初始化分布式训练环境
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    # 判断是否使用分布式训练
    config.is_distribute = config.world_size > 1
    # 打印配置信息
    print(config)

    # 设置日志记录器
    setup_logger("train", output=config.output, rank=config.rank)
    # 创建输出文件夹
    os.makedirs(config.output, exist_ok=True)
    # 创建检查点文件夹
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

    # 根据是否使用分布式训练设置随机种子
    if config.is_distribute:
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 启用cuDNN基准测试，加速卷积运算
    cudnn.benchmark = True

    # 保存配置信息到JSON文件
    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    # 初始化Wandb
    if config.wandb and config.rank == 0:
        wandb.init(
            config=config.to_dict(),  # 配置信息
            entity=config.entity,  # Wandb实体名称
            project=config.project,  # Wandb项目名称
            job_type=config.job_type,  # Wandb作业类型
            tags=config.tags,  # Wandb标签
            name=config.name,  # Wandb运行名称
        )
        # 更新配置信息
        config = ml_collections.config_dict.ConfigDict(wandb.config)

    # 开始主训练
    main(config)

import json
import logging
import os

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

# 创建名为 'train' 的日志记录器
# 用于记录训练过程中的各种信息，方便调试和监控
logger = logging.getLogger("train")

# 构建 DeepSpeed 配置的函数
# 根据配置文件中的优化器类型，生成对应的 DeepSpeed 配置
def build_ds_config(config: ml_collections.ConfigDict):
    # 将优化器名称转换为小写，方便后续比较
    opt_lower = config.optimizer.lower()
    if opt_lower == "adamw":
        # 定义 AdamW 优化器的参数
        optimizer = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,  # 学习率
                "eps": 1e-8,  # 数值稳定性参数
                "betas": (0.9, 0.95),  # 一阶矩和二阶矩估计的指数衰减率
                "weight_decay": config.wd,  # 权重衰减系数
            },
        }
        # 定义使用 AdamW 优化器时的 DeepSpeed 配置
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,  # 每个 GPU 的微批量大小
            "optimizer": optimizer,  # 优化器配置
            "fp16": {
                "enabled": config.fp16,  # 是否启用 FP16 混合精度训练
                "auto_cast": False,  # 是否自动进行类型转换
                "initial_scale_power": 16,  # 初始缩放因子
                "loss_scale_window": 500,  # 损失缩放窗口
            },
            "bf16": {
                "enabled": config.bf16,  # 是否启用 BF16 混合精度训练
                "auto_cast": False,  # 是否自动进行类型转换
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO 优化阶段
                "sub_group_size": 1e9,  # 子组大小
                "contiguous_gradients": True,  # 是否将梯度连续存储
                "overlap_comm": True,  # 是否重叠通信和计算
                "stage3_gather_16bit_weights_on_model_save": True,  # 保存模型时是否收集 16 位权重
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪的最大范数
        }
    else:
        # 定义使用其他优化器时的 DeepSpeed 配置
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,  # 每个 GPU 的微批量大小
            "bf16": {
                "enabled": True,  # 是否启用 BF16 混合精度训练
                "auto_cast": True,  # 是否自动进行类型转换
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO 优化阶段
                "offload_optimizer": {
                    "device": "cpu",  # 优化器卸载到 CPU
                },
                "offload_param": {"device": "cpu"},  # 参数卸载到 CPU
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪的最大范数
            "zero_force_ds_cpu_optimizer": False,  # 是否强制使用 DeepSpeed CPU 优化器
            "zero_allow_untested_optimizer": True,  # 是否允许使用未测试的优化器
        }
    return ds_config

# 解析命令行参数的函数
# 用于解析用户输入的命令行参数，并将其转换为配置字典
def parse_option():
    # 创建自定义的配置参数解析器
    parser = ConfigArgumentParser()
    # 添加可修改配置选项的参数
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    # 添加基本配置参数
    add_basic_config(parser)
    # 添加 WandB 配置参数
    add_wandb_config(parser)
    # 添加硬件配置参数
    add_hardware_config(parser)

    # 解析命令行参数
    config = parser.parse_args(wandb=True)
    # 将解析结果转换为 ConfigDict 类型
    config = ml_collections.config_dict.ConfigDict(config)
    return config

# 添加基本配置参数的函数
def add_basic_config(parser):
    # 单 GPU 批量大小
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    # 数据集路径
    parser.add_argument("--data-path", type=str, help="path to dataset")
    # 评估数据集路径
    parser.add_argument("--eval-data-path", type=str, help="path to evaluate dataset")
    # 数据加载器的工作进程数
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    # 是否自动从检查点恢复训练
    parser.add_argument(
        "--auto-resume", action="store_true", help="resume from checkpoint"
    )
    # 恢复训练的检查点路径
    parser.add_argument(
        "--resume-path", type=str, default=None, help="resume checkpoint path"
    )
    # 模型预训练检查点路径
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="pretrained checkpoint path for model (maybe stage 1)",
    )
    # 梯度累积步数
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )
    # 是否使用梯度检查点以节省内存
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    # 是否启用混合精度训练
    parser.add_argument(
        "--enable-amp", type=str2bool, default=False, help="mixed precision"
    )
    # 输出文件夹根路径
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    # 随机种子
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    # GPU ID
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    # 是否使用无限数据加载器
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )
    # 是否使用 torch.compile 加速模型
    parser.add_argument(
        "--torch-compile",
        type=str2bool,
        default=False,
        help="Use torch.compile to accelerate model or not",
    )

# 添加 WandB 配置参数的函数
def add_wandb_config(parser):
    # 是否使用 WandB 日志记录
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    # WandB 实体名称
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    # WandB 项目名称
    parser.add_argument(
        "--project", type=str, default="MultiModal", help="wandb project"
    )
    # WandB 作业类型
    parser.add_argument(
        "--job-type", type=str, default="vlm_test", help="wandb job_type"
    )
    # WandB 标签
    parser.add_argument(
        "--tags", type=str, default="MultiModal", nargs="+", help="wandb tags"
    )
    # WandB 运行名称
    parser.add_argument("--name", type=str, default="first_run", help="wandb run name")
    # WandB 运行备注
    parser.add_argument("--notes", type=str, default=None, help="wandb run's notes")

# 添加硬件配置参数的函数
def add_hardware_config(parser):
    # 加速器类型
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    # 本地进程排名
    parser.add_argument("--local_rank", type=int)

# 初始化训练环境的函数
# 包括分布式训练环境的初始化、日志记录器的设置、随机种子的设置等
def init_training_environment(config):
    # 初始化分布式训练环境，获取当前进程的全局排名、本地排名和世界大小
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    # 判断是否为分布式训练
    config.is_distribute = config.world_size > 1
    # 打印配置信息
    print(config)

    # 设置日志记录器
    setup_logger("train", output=config.output, rank=config.rank)
    # 创建输出文件夹
    os.makedirs(config.output, exist_ok=True)
    # 创建检查点文件夹
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

    # 根据是否分布式训练设置随机种子
    if config.is_distribute:
        # 分布式训练时，每个进程使用不同的随机种子
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 启用 CuDNN 自动调优
    cudnn.benchmark = True

    # 如果是主进程
    if config.rank == 0:
        # 保存配置信息到 JSON 文件
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        # 记录配置信息保存的日志
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    # 如果启用 WandB 日志记录且是主进程
    if config.wandb and config.rank == 0:
        # 初始化 WandB
        wandb.init(
            config=config.to_dict(),  # 配置信息
            entity=config.entity,  # WandB 实体名称
            project=config.project,  # WandB 项目名称
            job_type=config.job_type,  # WandB 作业类型
            tags=config.tags,  # WandB 标签
            name=config.name,  # WandB 运行名称
        )
        # 更新配置信息
        config = ml_collections.config_dict.ConfigDict(wandb.config)

    return config

# 准备模型、优化器和数据加载器的函数
def prepare_model_optimizer_dataloader(config):
    # 记录创建模型的日志信息
    logger.info(f"Creating model")
    # 构建模型
    model = build_model(
        config,
        activate_modal=("rgb", "text"),
    )
    # 记录模型信息
    logger.info(str(model) + "\n")

    # 记录构建数据集的日志信息
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
        tune_rgb_pooler=config.tune_rgb_pooler,  # 是否调整 RGB 池化层
        model_path=config.model_path,  # 模型预训练路径
        tune_im_start=config.tune_im_start,  # 是否调整图像起始部分
        compute_dtype=compute_dtype,  # 计算数据类型
    )

    # 如果使用 AdamW 优化器
    if config.optimizer.lower() == "adamw":
        parameter = None
        optimizer = None
    else:
        # 构建其他优化器
        parameter = None
        optimizer = build_optimizer(model, config, is_pretrain=True)

    # 初始化 DeepSpeed 模型引擎、优化器等
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),  # DeepSpeed 配置
        model=model,  # 模型
        optimizer=optimizer if optimizer is not None else None,  # 优化器
        model_parameters=parameter if parameter is not None else None,  # 模型参数
    )

    return model_engine, optimizer, data_loader_train, compute_dtype

# 处理自动恢复训练的函数
def handle_auto_resume(config):
    # 如果启用自动恢复训练
    if config.auto_resume:
        # 自动查找恢复训练的检查点文件
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.resume_path is not None:
                # 记录自动恢复训练时检查点文件的变更信息
                logger.warning(
                    f"auto-resume changing resume file from {config.resume_path} to {resume_file}"
                )
            config.resume_path = resume_file
            # 记录自动恢复训练的信息
            logger.info(f"auto resuming from {resume_file}")
        else:
            # 记录未找到检查点文件的信息
            logger.info(
                f"no checkpoint found in {config.output}/checkpoint, ignoring auto resume"
            )
    return config

# 保存最终模型的函数
def save_final_model(model, config):
    # 如果是主进程
    if config.local_rank == 0 or config.local_rank == -1:
        # 自定义保存模型检查点
        state_dict = model.custom_save_checkpoint(
            os.path.join(config.output, "checkpoints")
        )
        # 保存最终模型
        torch.save(
            state_dict,
            os.path.join(os.path.join(config.output, "checkpoints"), "FINAL.pt"),
        )

# 主函数
def main():
    # 解析命令行参数
    config = parse_option()
    # 初始化训练环境
    config = init_training_environment(config)

    # 准备模型、优化器和数据加载器
    model_engine, optimizer, data_loader_train, compute_dtype = prepare_model_optimizer_dataloader(config)

    # 创建基于 epoch 的训练器
    trainer = EpochBasedTrainer(
        model=model_engine,  # 模型引擎
        optimizer=optimizer,  # 优化器
        lr_scheduler=config.schedule,  # 学习率调度器
        data_loader=data_loader_train,  # 训练数据加载器
        max_epochs=config.epochs,  # 最大训练轮数
        work_dir=config.output,  # 工作目录
        log_period=1,  # 日志记录周期
        save_ckpt_by="iter",  # 保存检查点的方式
        ckpt_period=1000,  # 检查点保存周期
        accelerator=config.accelerator,  # 加速器类型
        enable_amp=config.enable_amp,  # 是否启用混合精度训练
        wandb=config.wandb,  # 是否使用 WandB 日志记录
        gpus=0,  # GPU ID
        max_num_checkpoints=1,  # 最大检查点数量
        clip_grad_norm=config.max_grad_norm,  # 梯度裁剪的最大范数
        is_distributed=config.is_distribute,  # 是否分布式训练
        torch_compile=config.torch_compile,  # 是否使用 torch.compile 加速模型
        dtype=compute_dtype,  # 数据类型
        deepspeed=True,  # 是否使用 DeepSpeed
    )

    # 处理自动恢复训练
    config = handle_auto_resume(config)

    # 开始训练
    trainer.train(load_checkpoint=config.resume_path)

    # 保存最终模型
    save_final_model(model_engine.module, config)

if __name__ == "__main__":
    main()
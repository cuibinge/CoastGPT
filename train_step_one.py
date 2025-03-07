# 导入必要的库
import json
import logging
import os

import deepspeed
import ml_collections.config_dict  # 用于管理配置的库
import numpy as np
import torch
import torch.backends.cudnn as cudnn  # 用于优化CUDA性能
import torch.distributed as dist  # 分布式训练支持
import wandb  # 实验跟踪和可视化

# 项目自定义模块
from Trainer import deepspeed_init_distributed  # DeepSpeed分布式初始化
from Trainer.EpochBasedTrainer import EpochBasedTrainer  # 基于epoch的训练循环
from Trainer.utils import (
    ConfigArgumentParser,
    auto_resume_helper,
    setup_logger,
    str2bool,
)
from Dataset.build_loader import build_loader  # 数据加载器构建
from models import build_model  # 模型构建函数
from optimizer import build_optimizer  # 优化器构建

logger = logging.getLogger("train")  # 获取训练日志器


def build_ds_config(config: ml_collections.ConfigDict):
    """构建DeepSpeed配置字典"""
    opt_lower = config.optimizer.lower()
    if opt_lower == "adamw":
        # AdamW优化器配置
        optimizer = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "eps": 1e-8,
                "betas": (0.9, 0.95),
                "weight_decay": config.wd,
            },
        }

        # DeepSpeed配置参数
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "optimizer": optimizer,
            "fp16": {  # 混合精度训练配置
                "enabled": True if config.fp16 else False,
                "auto_cast": False,
                "initial_scale_power": 16,
                "loss_scale_window": 500,
            },
            "bf16": {
                "enabled": True if config.bf16 else False,
                "auto_cast": False,
            },
            "zero_optimization": {  # Zero优化策略（显存优化）
                "stage": 2,
                "sub_group_size": 1e9,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪阈值
        }
    else:
        # 其他优化器的默认配置
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "bf16": {"enabled": True, "auto_cast": True},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},  # 优化器卸载到CPU
                "offload_param": {"device": "cpu"},  # 参数卸载到CPU
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
            "zero_force_ds_cpu_optimizer": False,
            "zero_allow_untested_optimizer": True,
        }
    return ds_config


def parse_option():
    """解析命令行参数和配置文件"""
    parser = ConfigArgumentParser()
    # 基本训练参数
    parser.add_argument("--batch-size", type=int, help="单GPU的批大小")
    parser.add_argument("--data-path", type=str, help="训练数据集路径")
    # ...（其他参数类似，解释关键参数的作用）

    # 硬件相关参数
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "mps"], default="cpu")

    config = parser.parse_args(wandb=True)  # 整合wandb配置
    return ml_collections.config_dict.ConfigDict(config)


def main(config):
    """主训练流程"""
    # 1. 模型构建
    logger.info("创建模型中...")
    model = build_model(config, activate_modal=("rgb", "text"))

    # 2. 数据加载器准备
    logger.info("构建数据加载器...")
    data_loader_train = build_loader(config, mode="pretrain",
                                     tokenizer=model.text.tokenizer)

    # 3. 模型训练准备（冻结层/LoRA等）
    compute_dtype = torch.float16 if config.fp16 else torch.bfloat16 if config.bf16 else torch.float32
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,
        freeze_text=not config.lora.enable,
        tune_rgb_pooler=config.tune_rgb_pooler,
        model_path=config.model_path,  # 加载预训练权重
        compute_dtype=compute_dtype,
    )

    # 4. DeepSpeed引擎初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),
        model=model,
        # ... 初始化优化器和参数
    )

    # 5. 训练器配置
    trainer = EpochBasedTrainer(
        model=model_engine,
        optimizer=optimizer,
        data_loader=data_loader_train,
        max_epochs=config.epochs,
        # ... 其他训练参数
    )

    # 6. 自动恢复训练处理
    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        # ... 处理恢复逻辑

    # 7. 启动训练
    trainer.train(load_checkpoint=config.resume_path)

    # 8. 最终模型保存
    if config.local_rank in [0, -1]:
        torch.save(state_dict, os.path.join(output_dir, "FINAL.pt"))


if __name__ == "__main__":
    # 配置初始化
    config = parse_option()

    # 分布式环境初始化
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    config.is_distribute = config.world_size > 1  # 判断是否分布式

    # 日志系统初始化
    setup_logger("train", output=config.output, rank=config.rank)

    # 随机种子设置（保证可复现性）
    torch.manual_seed(config.seed + (dist.get_rank() if config.is_distribute else 0))

    # Wandb实验跟踪
    if config.wandb and config.rank == 0:
        wandb.init(config=config.to_dict(), ...)

    # 启动主训练流程
    main(config)

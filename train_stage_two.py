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
from Models import build_model
from Trainer.optimizer import build_optimizer

# 创建训练过程专用的日志记录器
logger = logging.getLogger("train")


def build_ds_config(config: ml_collections.ConfigDict):
    """构建DeepSpeed训练配置"""
    opt_lower = config.optimizer.lower()

    # 针对AdamW优化器的特殊配置
    if opt_lower == "adamw":
        optimizer = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,  # 基础学习率
                "eps": 1e-8,  # 数值稳定常数
                "betas": (0.9, 0.95),  # 动量参数
                "weight_decay": config.wd,  # 权重衰减系数
            },
        }

        # DeepSpeed核心配置项
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,  # 单GPU批大小
            "optimizer": optimizer,  # 优化器配置
            "fp16": {
                "enabled": bool(config.fp16),  # 启用FP16混合精度
                "auto_cast": False,  # 禁用自动类型转换
                "initial_scale_power": 16,  # 初始缩放指数
                "loss_scale_window": 500,  # 损失缩放窗口大小
            },
            "bf16": {
                "enabled": bool(config.bf16),  # 启用BF16混合精度
                "auto_cast": False,
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO优化阶段2
                "sub_group_size": 1e9,  # 子组参数大小
                "contiguous_gradients": True,  # 连续梯度存储
                "overlap_comm": True,  # 通信计算重叠
                "stage3_gather_16bit_weights_on_model_save": True,  # 保存时收集16位权重
            },
            "gradient_accumulation_steps": config.accumulation_steps,  # 梯度累积步数
            "gradient_clipping": config.max_grad_norm,  # 梯度裁剪阈值
        }
    else:
        # 其他优化器的通用配置
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "bf16": {"enabled": True, "auto_cast": True},  # 默认启用BF16
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},  # 优化器卸载至CPU
                "offload_param": {"device": "cpu"},  # 模型参数卸载
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
            "zero_force_ds_cpu_optimizer": False,  # 禁用强制CPU优化器
            "zero_allow_untested_optimizer": True,  # 允许实验性优化器
        }

    return ds_config


def parse_option():
    """解析命令行参数和配置文件"""
    # 初始化带WandB支持的配置解析器
    parser = ConfigArgumentParser(wandb=True)

    # 基本训练参数
    parser.add_argument("--batch-size", type=int, help="单GPU批大小")
    parser.add_argument("--data-path", type=str, help="训练数据集路径")
    parser.add_argument("--eval-data-path", type=str, help="验证数据集路径")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--auto-resume", action="store_true", help="自动恢复训练")
    parser.add_argument("--resume-path", type=str, default=None, help="手动指定恢复路径")
    parser.add_argument("--model-path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="梯度累积次数")
    parser.add_argument("--use-checkpoint", action="store_true", help="激活梯度检查点")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="启用混合精度")
    parser.add_argument("--output", default="output", type=str, help="输出目录根路径")
    parser.add_argument("--seed", type=int, default=322, help="全局随机种子")
    parser.add_argument("--gpus", type=int, default=0, help="使用的GPU索引")
    parser.add_argument("--inf_sampler", type=str2bool, default=False, help="使用无限采样器")
    parser.add_argument("--torch-compile", type=str2bool, default=False, help="启用模型编译")

    # WandB集成参数
    parser.add_argument("--wandb", type=str2bool, default=False, help="启用WandB日志")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="WandB团队名称")
    parser.add_argument("--project", type=str, default="MultiModal", help="WandB项目名称")
    parser.add_argument("--job-type", type=str, default="vlm_test", help="WandB任务类型")
    parser.add_argument("--tags", type=str, nargs="+", default="MultiModal", help="WandB标签")
    parser.add_argument("--name", type=str, default="first_run", help="WandB运行名称")
    parser.add_argument("--notes", type=str, default=None, help="WandB运行备注")

    # 硬件配置
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "mps"], default="cpu",
                        help="训练加速器类型")
    parser.add_argument("--local_rank", type=int, help="DeepSpeed自动分配的进程编号")

    # 解析并返回配置对象
    config = parser.parse_args()
    return ml_collections.config_dict.ConfigDict(config)


def main(config):
    """主训练流程"""
    # 模型初始化
    logger.info("构建多模态模型...")
    model = build_model(config, activate_modal=("rgb", "text"))
    logger.debug(f"模型结构:\n{model}")

    # 数据加载器配置
    logger.info("初始化训练数据加载器...")
    train_loader = build_loader(
        config,
        mode="pretrain",
        tokenizer=model.text.tokenizer,
        prompt_type=config.prompt_template
    )

    # 精度配置
    compute_dtype = torch.float16 if config.fp16 else (
        torch.bfloat16 if config.bf16 else torch.float32
    )

    # 模型训练准备（冻结层、LoRA等）
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,
        freeze_text=not config.lora.enable,
        tune_rgb_pooler=config.tune_rgb_pooler,
        model_path=config.model_path,
        tune_im_start=config.tune_im_start,
        compute_dtype=compute_dtype,
    )

    # 优化器初始化
    if config.optimizer.lower() == "adamw":
        params, optimizer = None, None  # DeepSpeed自动处理AdamW
    else:
        params = model.get_optimizer_params(config)
        optimizer = build_optimizer(params, config, is_pretrain=True)

    # DeepSpeed引擎初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),
        model=model,
        optimizer=optimizer,
        model_parameters=params,
    )

    # 训练器配置
    trainer = EpochBasedTrainer(
        model=model_engine,
        optimizer=optimizer,
        lr_scheduler=config.schedule,
        data_loader=train_loader,
        max_epochs=config.epochs,
        work_dir=config.output,
        log_period=1,
        save_ckpt_by="iter",
        ckpt_period=1000,
        accelerator=config.accelerator,
        enable_amp=config.enable_amp,
        wandb=config.wandb,
        max_num_checkpoints=1,
        clip_grad_norm=config.max_grad_norm,
        is_distributed=config.is_distribute,
        torch_compile=config.torch_compile,
        dtype=compute_dtype,
        deepspeed=True,
    )

    # 自动恢复机制
    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            config.resume_path = resume_file
            logger.info(f"自动恢复训练: {resume_file}")

    # 启动训练流程
    trainer.train(load_checkpoint=config.resume_path)

    # 最终模型保存（主进程）
    if config.local_rank in {0, -1}:
        ckpt_dir = os.path.join(config.output, "checkpoints")
        state_dict = model.custom_save_checkpoint(ckpt_dir)
        torch.save(state_dict, os.path.join(ckpt_dir, "FINAL.pt"))


if __name__ == "__main__":
    # 配置初始化
    config = parse_option()

    # 分布式环境设置
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    config.is_distribute = config.world_size > 1

    # 日志系统初始化
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

    # 随机种子设置
    seed = config.seed + (dist.get_rank() if config.is_distribute else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True  # 启用CuDNN自动优化

    # 配置持久化
    if config.rank == 0:
        config_path = os.path.join(config.output, "config.json")
        with open(config_path, "w") as f:
            json.dump(dict(config.to_dict()), f, indent=4)
        logger.info(f"训练配置已保存至 {config_path}")

    # WandB集成（仅主进程）
    if config.wandb and config.rank == 0:
        wandb.init(
            config=dict(config.to_dict()),
            entity=config.entity,
            project=config.project,
            job_type=config.job_type,
            tags=config.tags,
            name=config.name,
            notes=config.notes,
        )
        config = ml_collections.config_dict.ConfigDict(wandb.config)

    # 启动主训练流程
    main(config)

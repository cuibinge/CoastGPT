import json
import logging
import os
import sys
from datetime import datetime

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
from Models.coastgpt import CoastGPT
from Models.dynamic_tanh import convert_ln_to_dyt
from Trainer.optimizer import build_optimizer
import torch_npu

# Constants and global configuration
logger = logging.getLogger("train")
ASCEND_DEVICES = "0,1,2,3,4,5,6,7"


def _patch_torch_vector_norm():
    """Patch torch.linalg.vector_norm to be robust to non-floating inputs.

    Some DeepSpeed versions may stack integer zeros into a Long tensor when computing
    global grad norm. This wrapper ensures the norm is computed on a floating tensor.
    """

    """这段代码是PyTorch 函数补丁(Monkey Patch)，核心作用是修复`torch.linalg.vector_norm`对非浮点 / 复数输入的兼容性问题，专门解决 DeepSpeed 框架中整数张量计算梯度范数时的报错。

    1. 原生 PyTorch 限制：`torch.linalg.vector_norm`**仅支持浮点型 / 复数张量, 直接传入整数型张量(LongTensor/IntTensor) 会直接报错。

    2. DeepSpeed 的 bug:部分 DeepSpeed 版本计算全局梯度范数时，会把整数 0 堆叠成`LongTensor`，触发原生函数的报错。

    3. 解决方案：给原生函数套一层安全包装器，自动将非浮点输入转为浮点型，保证函数正常运行
    """

    orig_fn = torch.linalg.vector_norm

    float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    complex_dtypes = {torch.complex64, torch.complex128}

    def safe_vector_norm(x, *args, **kwargs):
        dtype = kwargs.get("dtype", None)

        def is_float_or_complex(dt):
            return (dt in float_dtypes) or (dt in complex_dtypes)

        # If input is not float/complex, cast to a safe floating dtype
        if isinstance(x, torch.Tensor) and not (x.is_floating_point() or x.is_complex()):
            target_dtype = dtype if dtype is not None else torch.float32
            if not is_float_or_complex(target_dtype):
                target_dtype = torch.float32
            x = x.to(target_dtype)
            kwargs["dtype"] = target_dtype
        else:
            # If a non-floating dtype was explicitly requested, override to float32
            if dtype is not None and not is_float_or_complex(dtype):
                kwargs["dtype"] = torch.float32

        return orig_fn(x, *args, **kwargs)

    torch.linalg.vector_norm = safe_vector_norm


def create_ds_config(config):
    """Create DeepSpeed configuration based on training settings"""
    opt_lower = config.optimizer.lower()
    
    if opt_lower == "adamw":
        optimizer_cfg = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "eps": 1e-8,
                "betas": (0.9, 0.95),
                "weight_decay": config.wd,
            },
        }

        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "optimizer": optimizer_cfg,
            "fp16": {"enabled": config.fp16, "auto_cast": False, "initial_scale_power": 16, "loss_scale_window": 500},
            "bf16": {"enabled": config.bf16, "auto_cast": False},
            "zero_optimization": {
                "stage": 2,
                "sub_group_size": 1e9,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
        }
    else:
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "bf16": {"enabled": True, "auto_cast": True},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
            "zero_force_ds_cpu_optimizer": False,
            "zero_allow_untested_optimizer": True,
        }
    
    return ds_config

def parse_config():
    """Parse command line arguments and configuration"""
    parser = ConfigArgumentParser() 
    
    # Basic training parameters
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size per NPU")
    parser.add_argument("--data-path", default="/home/share/Data/LHRS/PretrainData", type=str)
    parser.add_argument("--eval-data-path", type=str)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true", default=True)
    parser.add_argument("--enable-amp", type=str2bool, default=True)
    parser.add_argument("--output", default="Checkpoint/test", type=str)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--inf_sampler", type=str2bool, default=False)
    parser.add_argument("--torch-compile", type=str2bool, default=False)

    # WandB configuration
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--entity", type=str, default="hnguohao")
    parser.add_argument("--project", type=str, default="CoastGPT")
    parser.add_argument("--job-type", type=str, default="vlm_test")
    parser.add_argument("--tags", type=str, default="MultiModal", nargs="+")
    parser.add_argument("--name", type=str, default="first_run")
    parser.add_argument("--notes", type=str, default=None)

    # Hardware settings
    parser.add_argument("--accelerator", default="npu", choices=["cpu", "gpu", "mps", "npu"])
    parser.add_argument("--local_rank", type=int)

    return ml_collections.config_dict.ConfigDict(parser.parse_args(wandb=True))

def initialize_training_components(config):
    """Initialize model, data loader, and training engine"""
    # Model initialization
    logger.info("Initializing CoastGPT model")
    model = CoastGPT(config)
    
    # Datasets
    logger.info("Building dataset")
    compute_dtype = torch.float16 if config.fp16 else (torch.bfloat16 if config.bf16 else torch.float32)
    data_loader_train = build_loader(
        config,
        mode="pretrain",                      # 训练模式:预训练
        tokenizer=model.language.tokenizer,   # 模型的文本分词器
        prompt_type=config.prompt_template,   # 提示语模板类型
    )

    # Model preparation
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,                             # 启用tune_rgb_bk时冻结视觉模块
        freeze_text=not config.lora.enable,                               # 启用lora时冻结文本模块    
        # Some configs may not define tune_multimodal; default to False
        tune_multimodal=getattr(config, "tune_multimodal", False),        
        model_path=config.model_path,                                     # 预训练模型路径（如果提供）
        # Safely read tune_im_start with default False
        tune_im_start=getattr(config, "tune_im_start", False),            
        compute_dtype=compute_dtype,                                      # 计算精度(BF16/FP16/FP32) 
    )

    # Optimizer selection
    optimizer = build_optimizer(model, config, is_pretrain=True) if config.optimizer.lower() != "adamw" else None
    
    # DeepSpeed engine initialization
    ds_config = create_ds_config(config)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        optimizer=optimizer,
        model_parameters=None if optimizer else None
    )
    
    return model_engine, optimizer, data_loader_train, compute_dtype

def setup_logging_and_resume(config):
    """Set up logging and handle auto-resume functionality"""
    # Logging setup
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)
    
    # Random seed initialization
    seed = config.seed + (dist.get_rank() if config.is_distribute else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # Configuration saving
    if config.rank == 0:
        with open(os.path.join(config.output, "config.json"), "w") as f:
            json.dump(dict(config.to_dict()), f, indent=4)
    
    # Handle auto-resume
    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.resume_path:
                logger.warning(f"Auto-resume overwriting resume path: {resume_file}")
            config.resume_path = resume_file

def setup_wandb(config):
    """Initialize WandB if enabled"""
    if config.wandb and config.rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = getattr(config, "name", "run")
        run_name = f"{base_name}-{timestamp}"
        wandb.init(
            config=config.to_dict(),
            entity=config.entity,
            project=config.project,
            job_type=config.job_type,
            tags=config.tags,
            name=run_name,
        )
        return ml_collections.config_dict.ConfigDict(wandb.config)
    return config

def save_final_model(model, config):
    """Save the final model checkpoint"""
    if config.local_rank in [0, -1]:                                  # 仅在主进程保存模型检查点，避免多进程重复保存
        checkpoint_dir = os.path.join(config.output, "checkpoints")   # 模型检查点目录
        state_dict = model.custom_save_checkpoint(checkpoint_dir)     # 获取模型状态字典，可能包含额外信息如优化器状态、训练进度等
        torch.save(state_dict, os.path.join(checkpoint_dir, "FINAL.pt")) # 保存最终模型检查点，命名为 FINAL.pt

def main():
    """Main training pipeline"""
    # Config initialization
    _patch_torch_vector_norm()
    config = parse_config()
    
    local_rank = int(os.environ["LOCAL_RANK"])                  # 获取当前进程的NPU编号
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(local_rank)   # 设置当前进程可见的NPU设备
    torch_npu.npu.set_device(local_rank)                        # 将当前进程绑定到对应的NPU设备上
    
    # Device and distributed setup
    # local_rank = int(os.environ["LOCAL_RANK"])

    # Distributed initialization
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed() # 初始化分布式环境，获取全局rank、本地rank和世界大小
    config.is_distribute = config.world_size > 1                # 判断是否多卡分布式训练

    # Logging and resume setup
    setup_logging_and_resume(config)                            # 设置日志记录、断点续训
    config = setup_wandb(config)                                # 初始化WandB并更新配置
    
    # Model and data initialization
    model, optimizer, data_loader, compute_dtype = initialize_training_components(config) # 初始化模型、优化器和数据加载器
    
    # Trainer configuration
    trainer = EpochBasedTrainer(
        model=model,                       # 训练模型（已封装DeepSpeed引擎）
        optimizer=optimizer,               # 优化器（如果使用AdamW则为None，由DeepSpeed自动处理）
        lr_scheduler=config.schedule,      # 学习率调度器配置（如果提供）
        data_loader=data_loader,           # 训练数据加载器
        max_epochs=config.epochs,          # 最大训练轮数
        work_dir=config.output,            # 输出目录
        log_period=1,                      # 日志记录频率（每多少个迭代记录一次）
        save_ckpt_by="iter",               # 模型保存频率单位（按迭代次数）
        ckpt_period=250,                   # 模型保存频率（每多少个迭代保存一次）
        
        
        accelerator=config.accelerator,    # 训练加速器类型（如NPU）
        enable_amp=config.enable_amp,      # 是否启用自动混合精度（AMP），如果启用则根据配置选择FP16或BF16
        wandb=config.wandb,                # 是否启用WandB日志记录 
        gpus=0,                            # 使用的GPU数量（NPU训练时通常设置为0，因为DeepSpeed会自动处理设备分配） 
        max_num_checkpoints=5,             # 最多保留的模型检查点数量（超过后会删除最旧的检查点）
        clip_grad_norm=config.max_grad_norm,    # 梯度裁剪的最大范数值，防止梯度爆炸
        is_distributed=config.is_distribute,    # 是否启用分布式训练，DeepSpeed会根据这个设置自动处理分布式环境
        torch_compile=config.torch_compile,     # 是否启用PyTorch 2.0的torch.compile功能进行模型编译优化（如果支持）
        dtype=compute_dtype,                    # 训练使用的计算精度（BF16/FP16/FP32），根据配置自动选择
        deepspeed=True,                         # 是否启用DeepSpeed引擎进行训练加速和内存优化（NPU训练时通常设置为True）
    )

    # Start training
    trainer.train(load_checkpoint=config.resume_path)                             # 如果提供了resume_path，则从指定检查点恢复训练，否则从头开始训练
    
    # Final operations
    save_final_model(model.module if hasattr(model, 'module') else model, config) # 保存最终模型检查点（如果使用了DataParallel或DistributedDataParallel，则访问module属性）

if __name__ == "__main__":
    main()

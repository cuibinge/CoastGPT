
import json
import logging
import os
import sys

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

# Constants and global configuration
logger = logging.getLogger("train")
ASCEND_DEVICES = "0,1,2,3,4,5,6,7"


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
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size per GPU")
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
    parser.add_argument("--entity", type=str, default="pumpkinn")
    parser.add_argument("--project", type=str, default="MultiModal")
    parser.add_argument("--job-type", type=str, default="vlm_test")
    parser.add_argument("--tags", type=str, default="MultiModal", nargs="+")
    parser.add_argument("--name", type=str, default="first_run")
    parser.add_argument("--notes", type=str, default=None)

    # Hardware settings
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "mps", "npu"])
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
        mode="pretrain",
        tokenizer=model.language.tokenizer,
        prompt_type=config.prompt_template,
    )

    # Model preparation
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,
        freeze_text=not config.lora.enable,
        tune_multimodal=config.tune_multimodal,
        model_path=config.model_path,
        tune_im_start=config.tune_im_start,
        compute_dtype=compute_dtype,
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
        wandb.init(
            config=config.to_dict(),
            entity=config.entity,
            project=config.project,
            job_type=config.job_type,
            tags=config.tags,
            name=config.name,
        )
        return ml_collections.config_dict.ConfigDict(wandb.config)
    return config

def save_final_model(model, config):
    """Save the final model checkpoint"""
    if config.local_rank in [0, -1]:
        checkpoint_dir = os.path.join(config.output, "checkpoints")
        state_dict = model.custom_save_checkpoint(checkpoint_dir)
        torch.save(state_dict, os.path.join(checkpoint_dir, "FINAL.pt"))

def main():
    """Main training pipeline"""
    # Config initialization
    config = parse_config()
    
    # Device and distributed setup
    local_rank = int(os.environ["LOCAL_RANK"])

    # Distributed initialization
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    config.is_distribute = config.world_size > 1

    # Logging and resume setup
    setup_logging_and_resume(config)
    config = setup_wandb(config)
    
    # Model and data initialization
    model, optimizer, data_loader, compute_dtype = initialize_training_components(config)
    
    # Trainer configuration
    trainer = EpochBasedTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=config.schedule,
        data_loader=data_loader,
        max_epochs=config.epochs,
        work_dir=config.output,
        log_period=1,
        save_ckpt_by="iter",
        ckpt_period=250,
        accelerator=config.accelerator,
        enable_amp=config.enable_amp,
        wandb=config.wandb,
        gpus=0,
        max_num_checkpoints=5,
        clip_grad_norm=config.max_grad_norm,
        is_distributed=config.is_distribute,
        torch_compile=config.torch_compile,
        dtype=compute_dtype,
        deepspeed=True,
    )

    # Start training
    trainer.train(load_checkpoint=config.resume_path)
    
    # Final operations
    save_final_model(model.module if hasattr(model, 'module') else model, config)

if __name__ == "__main__":
    main()
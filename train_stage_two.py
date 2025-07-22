import json
import logging
import os
from PIL import Image
import deepspeed
import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from Trainer import deepspeed_init_distributed
from Trainer.EpochBasedTrainer import EpochBasedTrainer
from Trainer.utils import ConfigArgumentParser, auto_resume_helper, setup_logger, str2bool
from Dataset.build_loader import build_loader
from Models.coastgpt import CoastGPT
from Models.dynamic_tanh import convert_ln_to_dyt
from Trainer.optimizer import build_optimizer

# Configuration
logger = logging.getLogger("train")
Image.MAX_IMAGE_PIXELS = None


# DeepSpeed Configuration Builder
def build_ds_config(config: ml_collections.ConfigDict):
    opt_lower = config.optimizer.lower()
    
    if opt_lower == "adamw":
        optimizer = {
            "type": "AdamW",
            "params": {"lr": config.lr, "eps": 1e-8, "betas": (0.9, 0.95), "weight_decay": config.wd},
        }
        return {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "optimizer": optimizer,
            "fp16": {"enabled": bool(config.fp16), "auto_cast": False, "initial_scale_power": 16, "loss_scale_window": 500},
            "bf16": {"enabled": bool(config.bf16), "auto_cast": False},
            "zero_optimization": {"stage": 2, "sub_group_size": 1e9, "contiguous_gradients": True, "overlap_comm": True, 
                                  "stage3_gather_16bit_weights_on_model_save": True},
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
        }
    
    return {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "bf16": {"enabled": True, "auto_cast": True},
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}, "offload_param": {"device": "cpu"}},
        "gradient_accumulation_steps": config.accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
    }

# Configuration Parser
def parse_option():
    parser = ConfigArgumentParser()
    
    # Basic parameters
    parser.add_argument("--batch-size", type=int, help="Batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--eval-data-path", type=str, help="Path to evaluate dataset")
    parser.add_argument("--workers", type=int, default=8, help="Workers of dataloader")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--model-path", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="Enable mixed precision")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="Root of output folder")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--inf_sampler", type=str2bool, default=False, help="Use Infinite loader")
    parser.add_argument("--torch-compile", type=str2bool, default=False, help="Use torch.compile")
    
    # W&B parameters
    parser.add_argument("--wandb", type=str2bool, default=False, help="Enable wandb logging")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="Wandb entity")
    parser.add_argument("--project", type=str, default="MultiModal", help="Wandb project")
    parser.add_argument("--job-type", type=str, default="vlm_test", help="Wandb job_type")
    parser.add_argument("--tags", type=str, default="MultiModal", nargs="+", help="Wandb tags")
    parser.add_argument("--name", type=str, default="first_run", help="Wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="Wandb run notes")
    
    # Hardware parameters
    parser.add_argument("--accelerator", default="gpu", type=str, choices=["cpu", "gpu", "mps", "npu"])
    parser.add_argument("--local_rank", type=int)
    
    return parser.parse_args(wandb=True)

# Model Preparation
def prepare_model(config, model):
    compute_dtype = torch.float16 if config.fp16 else (torch.bfloat16 if config.bf16 else torch.float32)
    
    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,
        freeze_text=not config.lora.enable,
        tune_multimodal=config.tune_multimodal,
        model_path=config.model_path,
        tune_im_start=config.tune_im_start,
        compute_dtype=compute_dtype,
    )
    return model

# Setup Environment
def setup_environment(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    config.is_distribute = config.world_size > 1
    
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)
    
    seed = config.seed + dist.get_rank() if config.is_distribute else config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    return config

# Save Configuration
def save_config(config):
    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

# Initialize WandB
def init_wandb(config):
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

# Main Training Function
def train_model(config):
    logger.info("Creating model")
    model = CoastGPT(config)
    logger.info(f"Model created:\n{model}\n")
    
    logger.info("Building dataset")
    data_loader_train = build_loader(
        config,
        mode="pretrain",
        tokenizer=model.language.tokenizer,
        prompt_type=config.prompt_template,
    )
    
    # Prepare model and optimizer
    model = prepare_model(config, model)
    
    if config.optimizer.lower() != "adamw":
        optimizer = build_optimizer(model, config, is_pretrain=True)
    else:
        optimizer = None
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),
        model=model,
        optimizer=optimizer,
        model_parameters=None if config.optimizer.lower() == "adamw" else None,
    )
    
    # Create trainer
    trainer = EpochBasedTrainer(
        model=model_engine,
        optimizer=optimizer,
        lr_scheduler=config.schedule,
        data_loader=data_loader_train,
        max_epochs=config.epochs,
        work_dir=config.output,
        log_period=1,
        save_ckpt_by="iter",
        ckpt_period=100,
        accelerator=config.accelerator,
        enable_amp=config.enable_amp,
        wandb=config.wandb,
        max_num_checkpoints=10,
        clip_grad_norm=config.max_grad_norm,
        is_distributed=config.is_distribute,
        torch_compile=config.torch_compile,
        dtype=torch.float16 if config.fp16 else torch.bfloat16 if config.bf16 else torch.float32,
        deepspeed=True,
    )
    
    # Handle auto-resume
    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            config.resume_path = resume_file
            logger.info(f"Auto resuming from {resume_file}")
    
    # Start training
    trainer.train(load_checkpoint=config.resume_path)
    
    # Save final model
    if config.local_rank in [0, -1]:
        state_dict = model.custom_save_checkpoint(os.path.join(config.output, "checkpoints"))
        torch.save(state_dict, os.path.join(os.path.join(config.output, "checkpoints"), "FINAL.pt"))

# Main Execution
if __name__ == "__main__":
    config = ml_collections.config_dict.ConfigDict(parse_option())
    config = setup_environment(config)
    save_config(config)
    config = init_wandb(config)
    train_model(config)

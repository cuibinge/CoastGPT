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


def patch_vector_norm_for_npu():
    """Work around DeepSpeed ZeRO grad-norm dtype issue on some torch_npu stacks."""
    orig_vector_norm = torch.linalg.vector_norm
    if getattr(orig_vector_norm, "_coastgpt_patched", False):
        return

    def _safe_vector_norm(x, *args, **kwargs):
        if torch.is_tensor(x) and (not x.is_floating_point()) and (not torch.is_complex(x)):
            x = x.to(dtype=torch.float32)
        try:
            return orig_vector_norm(x, *args, **kwargs)
        except RuntimeError as err:
            msg = str(err)
            if "Expected a floating point or complex tensor as input" not in msg:
                raise
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if (not x.is_floating_point()) and (not torch.is_complex(x)):
                return orig_vector_norm(x.to(dtype=torch.float32), *args, **kwargs)
            raise

    _safe_vector_norm._coastgpt_patched = True
    torch.linalg.vector_norm = _safe_vector_norm
    logger.warning("Patched torch.linalg.vector_norm to cast non-floating tensors to float on NPU.")


def patch_deepspeed_zero_grad_norm_for_npu():
    """Patch DeepSpeed ZeRO grad-norm path for torch_npu stability."""
    try:
        from deepspeed.runtime.zero import stage_1_and_2 as ds_zero
    except Exception as exc:
        logger.warning("Skip DeepSpeed ZeRO grad-norm patch: %s", exc)
        return

    zero_cls = ds_zero.DeepSpeedZeroOptimizer
    if getattr(zero_cls, "_coastgpt_npu_grad_norm_patched", False):
        return

    orig_scaled_global_norm = zero_cls.scaled_global_norm

    def _safe_get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        all_norms = []

        if norm_type == ds_zero.inf:
            for g in gradients:
                if g is None:
                    continue
                all_norms.append(g.data.abs().max().float())
            total_norm = torch.stack(all_norms).max() if all_norms else torch.tensor(0.0, dtype=torch.float32, device=self.device)
            ds_zero.dist.all_reduce(total_norm, op=ds_zero.dist.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm, op=ds_zero.dist.ReduceOp.MAX)
            ds_zero.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
            return total_norm

        for g, p in zip(gradients, params):
            if g is None:
                continue
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, ds_zero.PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if ds_zero.is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                grad = g.data.detach()
                # torch_npu does not reliably support double in this path.
                grad = grad.float() if grad.dtype != torch.float32 else grad
                all_norms.append(
                    torch.linalg.vector_norm(grad, ord=norm_type).to(ds_zero.get_accelerator().current_device_name())
                )

        if all_norms:
            total_norm = torch.stack(all_norms).square().sum().float()
        else:
            total_norm = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        ds_zero.dist.all_reduce(total_norm, op=ds_zero.dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=ds_zero.dist.ReduceOp.SUM)
        total_norm = total_norm.pow(1.0 / norm_type)
        ds_zero.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
        return total_norm

    def _safe_scaled_global_norm(self, norm_type=2):
        if float(getattr(self, "clip_grad", 0.0)) <= 0.0:
            return torch.tensor(float(self.loss_scale), dtype=torch.float32, device=self.device)
        return orig_scaled_global_norm(self, norm_type=norm_type)

    zero_cls.get_grad_norm_direct = _safe_get_grad_norm_direct
    zero_cls.scaled_global_norm = _safe_scaled_global_norm
    zero_cls._coastgpt_npu_grad_norm_patched = True
    logger.warning(
        "Patched DeepSpeed ZeRO grad-norm for NPU: use float32 norms and skip norm all-reduce when clip_grad<=0."
    )


def apply_npu_stability_overrides(config):
    """Apply conservative defaults for NPU + DeepSpeed stability."""
    if str(getattr(config, "accelerator", "")).lower() != "npu":
        return
    opt_lower = str(getattr(config, "optimizer", "")).lower()
    if opt_lower and opt_lower != "adamw":
        logger.warning(
            "NPU + DeepSpeed ZeRO stage2 is unstable with optimizer=%s; override to adamw.",
            config.optimizer,
        )
        config.optimizer = "adamw"


# DeepSpeed Configuration Builder
def build_ds_config(config: ml_collections.ConfigDict):
    opt_lower = config.optimizer.lower()
    grad_clip = float(getattr(config, "max_grad_norm", 0.0) or 0.0)
    if grad_clip < 0:
        logger.warning("max_grad_norm=%s is invalid; clamp to 0.0.", grad_clip)
        grad_clip = 0.0
    overlap_comm = str(getattr(config, "accelerator", "")).lower() != "npu"
    
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
            "zero_optimization": {"stage": 2, "sub_group_size": 1e9, "contiguous_gradients": True, "overlap_comm": overlap_comm,
                                  "stage3_gather_16bit_weights_on_model_save": True},
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": grad_clip,
        }
    
    return {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "fp16": {"enabled": bool(config.fp16), "auto_cast": False, "initial_scale_power": 16, "loss_scale_window": 500},
        "bf16": {"enabled": bool(config.bf16), "auto_cast": False},
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}, "offload_param": {"device": "cpu"}},
        "gradient_accumulation_steps": config.accumulation_steps,
        "gradient_clipping": grad_clip,
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
    }

# Configuration Parser
def parse_option():
    parser = ConfigArgumentParser()
    
    # Basic parameters
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--eval-data-path", type=str, help="Path to evaluate dataset")
    parser.add_argument("--workers", type=int, default=8, help="Workers of dataloader")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--model-path", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm. Set 0 to disable clipping.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="Enable mixed precision")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="Root of output folder")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--inf_sampler", type=str2bool, default=False, help="Use Infinite loader")
    parser.add_argument("--torch-compile", type=str2bool, default=False, help="Use torch.compile")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs from config")
    parser.add_argument(
        "--ckpt-period",
        type=int,
        default=None,
        help="Checkpoint save period. Set <= 0 to disable intermediate checkpoints.",
    )
    parser.add_argument(
        "--max-debug-iters",
        type=int,
        default=0,
        help="If > 0, run only this many iterations for quick sanity check",
    )
    
    # W&B parameters
    parser.add_argument("--wandb", type=str2bool, default=False, help="Enable wandb logging")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="Wandb entity")
    parser.add_argument("--project", type=str, default="MultiModal", help="Wandb project")
    parser.add_argument("--job-type", type=str, default="vlm_test", help="Wandb job_type")
    parser.add_argument("--tags", type=str, default="MultiModal", nargs="+", help="Wandb tags")
    parser.add_argument("--name", type=str, default="first_run", help="Wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="Wandb run notes")
    
    # Hardware parameters
    parser.add_argument("--accelerator", default="npu", type=str, choices=["cpu", "gpu", "mps", "npu"])
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
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters after prepare_for_training. Please check tune_* switches.")
    num_trainable = sum(p.numel() for _, p in trainable_params)
    dtype_set = sorted({str(p.dtype) for _, p in trainable_params})
    logger.info(
        f"Trainable params: {num_trainable} ({len(trainable_params)} tensors), dtypes={dtype_set}"
    )
    branch_count = {}
    for name, param in trainable_params:
        branch = name.split(".", 1)[0]
        branch_count[branch] = branch_count.get(branch, 0) + param.numel()
    logger.info(
        "Trainable breakdown: "
        + ", ".join(f"{k}={v}" for k, v in sorted(branch_count.items(), key=lambda x: -x[1]))
    )
    return model

# Setup Environment
def setup_environment(config):
    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed(
        accelerator=config.accelerator
    )
    config.is_distribute = config.world_size > 1
    
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)
    
    seed = config.seed + dist.get_rank() if config.is_distribute else config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config.accelerator == "gpu":
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
        # Preserve optional runtime fields (e.g. resume_path=None) that wandb may drop.
        merged = ml_collections.config_dict.ConfigDict(config.to_dict())
        for key, value in dict(wandb.config).items():
            merged[key] = value
        return merged
    return config

# Main Training Function
def train_model(config):
    logger.info("Creating model")
    model = CoastGPT(config)
    logger.info(f"Model created:\n{model}\n")
    if int(getattr(config, "max_debug_iters", 0)) > 0:
        logger.info(f"Quick sanity mode enabled: max_debug_iters={int(config.max_debug_iters)}")
    
    logger.info("Building dataset")
    data_loader_train = build_loader(
        config,
        mode="pretrain",
        tokenizer=model.language.tokenizer,
        prompt_type=config.prompt_template,
    )
    if len(data_loader_train) == 0:
        raise ValueError(
            "Training dataloader is empty. Please verify data_path, cut300_path, and the cut300 train image/gt directories."
        )
    
    # Prepare model and optimizer
    model = prepare_model(config, model)
    
    if config.optimizer.lower() != "adamw":
        optimizer = build_optimizer(model, config, is_pretrain=True)
        deepspeed_model_parameters = None
    else:
        optimizer = None
        deepspeed_model_parameters = [p for p in model.parameters() if p.requires_grad]
        if len(deepspeed_model_parameters) == 0:
            raise RuntimeError("No trainable parameters available for DeepSpeed optimizer.")
        logger.info(
            "Passing %d explicitly trainable tensors to DeepSpeed optimizer.",
            len(deepspeed_model_parameters),
        )
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),
        model=model,
        optimizer=optimizer,
        model_parameters=deepspeed_model_parameters,
    )
    
    ckpt_period = getattr(config, "ckpt_period", None)
    ckpt_period = 100 if ckpt_period is None else int(ckpt_period)
    if ckpt_period <= 0:
        logger.info("Intermediate checkpoints disabled (ckpt_period=%s).", ckpt_period)
    else:
        logger.info("Intermediate checkpoints enabled every %s iterations.", ckpt_period)

    # Create trainer
    trainer = EpochBasedTrainer(
        model=model_engine,
        optimizer=optimizer,
        lr_scheduler=config.schedule,
        data_loader=data_loader_train,
        max_epochs=config.epochs,
        max_iters_override=int(getattr(config, "max_debug_iters", 0)),
        work_dir=config.output,
        log_period=1,
        save_ckpt_by="iter",
        ckpt_period=ckpt_period,
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
    if bool(getattr(config, "auto_resume", False)):
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            config.resume_path = resume_file
            logger.info(f"Auto resuming from {resume_file}")
    
    # Start training
    trainer.train(load_checkpoint=getattr(config, "resume_path", None))
    
    # Save final model. All ranks must participate in ZeRO parameter gathering;
    # only rank 0 writes the merged artifact to disk.
    final_ckpt_dir = os.path.join(config.output, "checkpoints")
    state_dict = model.custom_save_checkpoint(final_ckpt_dir)
    if config.rank == 0:
        final_path = os.path.join(final_ckpt_dir, "FINAL.pt")
        torch.save(state_dict, final_path)
        logger.info("Saved final checkpoint to %s", final_path)
    if config.is_distribute and dist.is_available() and dist.is_initialized():
        dist.barrier()

# Main Execution
if __name__ == "__main__":
    config = ml_collections.config_dict.ConfigDict(parse_option())
    if str(getattr(config, "accelerator", "")).lower() == "npu":
        patch_vector_norm_for_npu()
        patch_deepspeed_zero_grad_norm_for_npu()
    apply_npu_stability_overrides(config)
    config = setup_environment(config)
    save_config(config)
    config = init_wandb(config)
    train_model(config)

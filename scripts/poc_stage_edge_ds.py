#!/usr/bin/env python3
"""
PoC-3 Coastline Edge Head — DeepSpeed Distributed Training (A2+).

Adapts the single-NPU poc_stage_edge.py for multi-NPU DeepSpeed training.
Only FPN + EdgeHead are trainable (~4M params); vision encoder is frozen.
Uses ZeRO-0 (DDP with DeepSpeed AMP) since the trainable model is small.

Usage (via launch script):
    bash scripts/run_a2_8npu.sh

Or directly:
    deepspeed --num_nodes=1 --num_gpus=8 scripts/poc_stage_edge_ds.py \
        -c configs/poc3_edge_a2_soft_multiscale.yaml \
        --batch-size 2 --epochs 30 --accumulation-steps 8 \
        --device npu --output outputs/poc3_edge/a2_8npu
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from ml_collections import ConfigDict

import deepspeed

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainer.utils import ConfigArgumentParser, setup_logger, str2bool
from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.edge_head import SingleScaleEdgeHead, MultiScaleEdgeHead
from Dataset.coastline_dataset import (
    CoastlineEdgeDataset,
    coastline_collate_fn,
    build_coastline_manifest,
)
from utils.edge_losses import (
    edge_bce_dice_loss,
    edge_focal_dice_loss,
    DeepSupervisedEdgeLoss,
)
from utils.edge_postprocess import postprocess_edge
from utils.coastline_metrics import compute_all_edge_metrics

logger = logging.getLogger("train")


# =============================================================================
# NPU patches (same as train_stage_two.py)
# =============================================================================


def patch_vector_norm_for_npu():
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
    logger.warning("Patched torch.linalg.vector_norm for NPU.")


# =============================================================================
# Container module for DeepSpeed
# =============================================================================


class EdgeTrainingModel(nn.Module):
    """Container wrapping FPN + EdgeHead for DeepSpeed.

    DeepSpeed needs a single nn.Module. We keep vision encoder external
    (frozen, called with torch.no_grad() before this module).
    """

    def __init__(self, fpn: FPNNeck, edge_head: nn.Module, is_multi_scale: bool = False):
        super().__init__()
        self.fpn = fpn
        self.edge_head = edge_head
        self.is_multi_scale = is_multi_scale

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
        c32: torch.Tensor,
        vit_feat: Optional[torch.Tensor] = None,
    ):
        p1, p2, p3, p4 = self.fpn(c4, c8, c16, c32, vit_feat=vit_feat)
        return self.edge_head(p1, p2, p3, p4)


# =============================================================================
# Config parsing (compatible with train_stage_two.py CLI style)
# =============================================================================


def parse_option():
    parser = argparse.ArgumentParser(description="PoC-3 A2 DeepSpeed Training")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (npu/cuda/cpu)")
    parser.add_argument("--enable-amp", type=str2bool, default=None, help="Enable AMP")
    parser.add_argument("--accelerator", default="npu", type=str, help="Hardware accelerator")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm")
    return parser.parse_args()


# =============================================================================
# Vision encoder loading
# =============================================================================


def clean_vision_state_dict(state_dict: dict) -> dict:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("vision."):
            new_k = new_k[len("vision."):]
        cleaned[new_k] = v
    return cleaned


def build_vision_encoder(model_cfg: ConfigDict, ckpt_path: str) -> DualVisionEncoder:
    logger.info("Building DualVisionEncoder...")
    vision = DualVisionEncoder(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "vision_ckpt" in ckpt:
            state_dict = ckpt["vision_ckpt"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = clean_vision_state_dict(state_dict)
    vision.load_state_dict(state_dict, strict=False)
    return vision


# =============================================================================
# Distributed DataLoader helper
# =============================================================================


def build_distributed_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader with DistributedSampler when running multi-GPU."""
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # sampler handles shuffling

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=coastline_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )


# =============================================================================
# DeepSpeed config builder
# =============================================================================


def build_ds_config(args: dict) -> dict:
    """Build DeepSpeed config for small-model edge training (ZeRO-0)."""
    accelerator = str(args.get("accelerator", "npu")).lower()
    grad_clip = float(args.get("max_grad_norm", 1.0) or 0.0)
    if grad_clip < 0:
        grad_clip = 0.0
    use_bf16 = accelerator == "npu"

    return {
        "train_micro_batch_size_per_gpu": args["batch_size"],
        "gradient_accumulation_steps": args["accumulation_steps"],
        "gradient_clipping": grad_clip,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": float(args["lr"]),
                "eps": 1e-8,
                "betas": (0.9, 0.95),
                "weight_decay": float(args["wd"]),
            },
        },
        "fp16": {"enabled": not use_bf16 and bool(args.get("enable_amp", False)), "auto_cast": False},
        "bf16": {"enabled": use_bf16 and bool(args.get("enable_amp", False)), "auto_cast": False},
        "zero_optimization": {"stage": 0},
        "wall_clock_breakdown": False,
    }


# =============================================================================
# Environment setup
# =============================================================================


def setup_environment(config: ConfigDict) -> ConfigDict:
    # Initialize distributed (self-contained — avoids stale worktree Trainer code)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        deepspeed.init_distributed()

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.is_distribute = world_size > 1

    if config.is_distribute:
        import torch_npu
        torch_npu.npu.set_device(local_rank)

    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

    seed = config.seed + rank if config.is_distribute else config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    return config


# =============================================================================
# Training epoch
# =============================================================================


def train_epoch(
    model: EdgeTrainingModel,
    model_engine: deepspeed.DeepSpeedEngine,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    max_grad_norm: float = 1.0,
    loss_type: str = "deep_supervised",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    deep_sup_loss_fn: Optional[DeepSupervisedEdgeLoss] = None,
) -> float:
    vision.eval()
    model.train()

    total_loss_sum = 0.0
    total_steps = 0
    is_multi_scale = model.is_multi_scale

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if model.fpn.has_vit else None

        # Cast vision outputs to match FPN dtype (bf16 for NPU AMP)
        model_dtype = next(model.parameters()).dtype
        c4, c8, c16, c32 = c4.to(model_dtype), c8.to(model_dtype), c16.to(model_dtype), c32.to(model_dtype)
        if vit_feat is not None:
            vit_feat = vit_feat.to(model_dtype)

        if is_multi_scale:
            outputs = model(c4, c8, c16, c32, vit_feat=vit_feat)
            loss_dict = deep_sup_loss_fn(
                outputs["fused"],
                [outputs["side1"], outputs["side2"], outputs["side3"], outputs["side4"]],
                targets,
            )
            total_loss = loss_dict["total"]
            comp_name = "Focal"
            loss_dice_val = 0.0
            loss_comp1_val = loss_dict["loss_fused"].item()
        else:
            logits = model(c4, c8, c16, c32, vit_feat=vit_feat)
            if loss_type == "focal_dice":
                total_loss, loss_comp1, loss_dice = edge_focal_dice_loss(
                    logits, targets, alpha=focal_alpha, gamma=focal_gamma
                )
                comp_name = "Focal"
            else:
                total_loss, loss_comp1, loss_dice = edge_bce_dice_loss(logits, targets)
                comp_name = "BCE"
            loss_comp1_val = loss_comp1.item()
            loss_dice_val = loss_dice.item()

        # DeepSpeed backward
        model_engine.backward(total_loss)
        model_engine.step()

        total_loss_sum += float(total_loss.item())
        total_steps += 1

        if dist.get_rank() == 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            fg_ratio = float(targets.mean().item()) if targets.numel() > 0 else 0.0
            logger.info(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | {comp_name}: {loss_comp1_val:.3f} "
                f"Dice: {loss_dice_val:.3f} | GT_fg: {fg_ratio:.4f}"
            )

    avg_loss = total_loss_sum / max(total_steps, 1)
    if dist.get_rank() == 0:
        logger.info(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(
    model: EdgeTrainingModel,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    threshold: float = 0.5,
    max_batches: int = 0,
    threshold_values: Optional[List[float]] = None,
    sweep_metric: str = "buffered_f1_1px",
) -> dict:
    model.eval()
    vision.eval()

    is_multi_scale = model.is_multi_scale
    do_sweep = threshold_values is not None and len(threshold_values) > 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    val_output_dir = Path(output_dir) / "vis" / f"val_epoch_{epoch:03d}"
    if rank == 0:
        val_output_dir.mkdir(parents=True, exist_ok=True)
        geojson_dir = val_output_dir / "geojson"
        geojson_dir.mkdir(exist_ok=True)

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_metas: List[dict] = []
    all_images: List[torch.Tensor] = []

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets_np = targets[:, 0].numpy()

        image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if model.fpn.has_vit else None

        # Cast to model dtype for AMP compatibility
        model_dtype = next(model.parameters()).dtype
        c4, c8, c16, c32 = c4.to(model_dtype), c8.to(model_dtype), c16.to(model_dtype), c32.to(model_dtype)
        if vit_feat is not None:
            vit_feat = vit_feat.to(model_dtype)

        if is_multi_scale:
            outputs = model(c4, c8, c16, c32, vit_feat=vit_feat)
            logits = outputs["fused"]
        else:
            logits = model(c4, c8, c16, c32, vit_feat=vit_feat)

        probs = torch.sigmoid(logits)
        probs_np = probs[:, 0].cpu().numpy()

        for i in range(images.shape[0]):
            all_probs.append(probs_np[i])
            all_targets.append(targets_np[i])
            all_metas.append(metas[i])
            all_images.append(images[i].cpu())

    # ---- Gather metrics across ranks (naive: all ranks compute same) ----
    if rank != 0:
        # Only rank 0 does visualization and full metric reporting
        # Non-zero ranks return empty dict
        return {}

    # Determine best threshold via sweep
    if do_sweep:
        sweep_results = {}
        for t in threshold_values:
            all_m = [compute_all_edge_metrics(p, g, threshold=t)
                      for p, g in zip(all_probs, all_targets)]
            agg = _aggregate_metrics(all_m)
            sweep_results[t] = agg.get(sweep_metric, 0.0)

        best_t = max(sweep_results, key=sweep_results.get)
        best_f1 = sweep_results[best_t]
        logger.info(f"Threshold sweep (epoch {epoch}):")
        for t in threshold_values:
            marker = " <-- BEST" if t == best_t else ""
            logger.info(f"  t={t:.2f}: {sweep_metric}={sweep_results[t]:.4f}{marker}")
        threshold = best_t
    else:
        best_t = threshold
        best_f1 = 0.0

    all_metrics = [compute_all_edge_metrics(p, g, threshold=threshold)
                   for p, g in zip(all_probs, all_targets)]
    for m, meta in zip(all_metrics, all_metas):
        m["sample_id"] = meta.get("sample_id", "")

    # Save overlays (rank 0 only, first 12)
    overlay_count = 0
    max_overlay = 12
    for idx in range(min(len(all_images), len(all_metas))):
        if overlay_count >= max_overlay:
            break
        meta = all_metas[idx]
        _save_edge_overlay(
            all_images[idx], all_targets[idx], all_probs[idx],
            meta, epoch, val_output_dir, threshold=threshold,
        )
        overlay_count += 1

        if overlay_count <= 5:
            georef = {
                "source_crs": meta.get("source_crs", "EPSG:4326"),
                "model_transform": meta.get("model_transform", [1e-5, 0, 0, 0, -1e-5, 0]),
            }
            fc = postprocess_edge(
                all_probs[idx], georef,
                threshold=threshold, min_length=10,
                max_components=5, simplify_epsilon=1.0,
                sample_id=meta.get("sample_id", ""),
            )
            geo_path = val_output_dir / "geojson" / f"{meta.get('sample_id', idx)}.geojson"
            with open(geo_path, 'w', encoding='utf-8') as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)

    agg = _aggregate_metrics(all_metrics)
    agg["best_threshold"] = float(threshold)
    if do_sweep:
        agg["threshold_sweep"] = sweep_results

    logger.info(
        f"Validation epoch {epoch} (thresh={threshold:.2f}): "
        f"pixel_f1={agg.get('pixel_f1', 0):.4f}, "
        f"buffered_f1_1px={agg.get('buffered_f1_1px', 0):.4f}, "
        f"buffered_f1_3px={agg.get('buffered_f1_3px', 0):.4f}, "
        f"chamfer={agg.get('chamfer_distance_px', 0):.2f}px, "
        f"n_samples={len(all_metrics)}"
    )
    return agg


def _aggregate_metrics(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    agg = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if vals:
            agg[key] = float(np.mean(vals))
    return agg


def _save_edge_overlay(
    image: torch.Tensor,
    gt: np.ndarray,
    pred: np.ndarray,
    meta: dict,
    epoch: int,
    output_dir: Path,
    threshold: float = 0.5,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_id = meta.get("sample_id", "unknown")
    img = image.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("GT Edge")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(img)
    axes[0, 2].imshow(gt, cmap="Reds", alpha=0.5)
    axes[0, 2].set_title("GT Overlay")
    axes[0, 2].axis("off")
    axes[1, 0].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Pred Heatmap (thresh={threshold})")
    axes[1, 0].axis("off")
    pred_bin = (pred >= threshold).astype(np.uint8)
    axes[1, 1].imshow(pred_bin, cmap="gray")
    axes[1, 1].set_title("Pred Binary")
    axes[1, 1].axis("off")
    try:
        from utils.edge_postprocess import heatmap_to_binary, binary_to_skeleton
        binary = heatmap_to_binary(pred, threshold=threshold, min_area=8)
        skeleton = binary_to_skeleton(binary)
        axes[1, 2].imshow(img)
        axes[1, 2].imshow(skeleton, cmap="Greens", alpha=0.7)
        axes[1, 2].set_title("Pred Skeleton Overlay")
    except Exception:
        axes[1, 2].imshow(img)
        axes[1, 2].set_title("Skeleton (skimage not available)")
    axes[1, 2].axis("off")

    plt.suptitle(f"Epoch {epoch} — {sample_id}")
    plt.tight_layout()
    save_path = output_dir / f"{sample_id}_overlay.png"
    plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close()


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    model_engine: deepspeed.DeepSpeedEngine,
    epoch: int,
    output_dir: str,
    best_f1: float = 0.0,
):
    if dist.get_rank() != 0:
        return
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    # Save trainable state (FPN + EdgeHead) from the engine's fp32 master weights
    state_dict = model_engine.module.state_dict()  # module unwrap from DeepSpeed
    torch.save(
        {"epoch": epoch, "model": state_dict, "best_f1": best_f1},
        str(ckpt_path),
    )
    logger.info(f"Checkpoint saved to {ckpt_path}")


# =============================================================================
# Main
# =============================================================================


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    return ConfigDict(cfg_raw)


def main():
    args = parse_option()

    # Load YAML config (same pattern as single-NPU script)
    cfg = load_config(args.config)

    output_dir = args.output or cfg.get("experiment.output_dir", "outputs/poc3_edge/a2_8npu")
    device_str = args.device or cfg.get("train.device", "npu")

    if device_str == "npu":
        patch_vector_norm_for_npu()

    # Setup distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        deepspeed.init_distributed()
        import torch_npu
        torch_npu.npu.set_device(local_rank)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    setup_logger("train", output=output_dir, rank=rank)

    # Build manifest
    manifest_path = cfg.get("data.manifest_path", str(Path(output_dir) / "coastline_manifest.json"))
    if rank == 0:
        manifest = build_coastline_manifest(
            data_roots=list(cfg.get("data.roots", [])),
            output_path=manifest_path,
            val_ratio=float(cfg.get("data.val_ratio", 0.2)),
            seed=int(cfg.get("data.val_split_seed", 42)),
        )
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    if args.build_manifest_only:
        if rank == 0:
            logger.info("Manifest built. Exiting.")
        return

    # Device
    if device_str == "npu":
        device = torch.device("npu:0")
    elif device_str == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if rank == 0:
        logger.info(f"Device: {device}")

    # Build frozen vision encoder
    model_cfg = ConfigDict(cfg.get("model", {}))
    vision = build_vision_encoder(
        model_cfg,
        cfg.get("model.vision_checkpoint", "./output/stage2/checkpoints/iter_2879_consolidated.pt"),
    )
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False
    if rank == 0:
        logger.info("Vision encoder frozen")

    # Build FPN
    fpn_cfg = cfg.get("model.fpn", {})
    fpn = FPNNeck(
        in_channels=list(fpn_cfg.get("in_channels", [128, 256, 512, 1024])),
        out_channels=int(fpn_cfg.get("out_channels", 256)),
        vit_in_channels=int(fpn_cfg.get("vit_in_channels", 1024)),
    )

    # Build Edge Head
    edge_head_type = cfg.get("model.edge_head.type", "multi_scale")
    if edge_head_type == "multi_scale":
        edge_head = MultiScaleEdgeHead(
            in_channels=int(cfg.get("model.edge_head.in_channels", 256)),
            output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
        )
        is_multi_scale = True
    else:
        edge_head = SingleScaleEdgeHead(
            in_channels=int(cfg.get("model.edge_head.in_channels", 256)),
            decoder_channels=tuple(cfg.get("model.edge_head.decoder_channels", [256, 128])),
            output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
        )
        is_multi_scale = False

    if rank == 0:
        total_params = sum(p.numel() for p in fpn.parameters()) + \
                       sum(p.numel() for p in edge_head.parameters())
        logger.info(f"FPN + EdgeHead: {total_params:,} params (multi_scale={is_multi_scale})")

    # Warm-start FPN from A1 checkpoint (edge head architecture differs: SingleScale vs MultiScale)
    a1_ckpt = cfg.get("model.a0_checkpoint")
    if a1_ckpt and Path(a1_ckpt).exists():
        a1 = torch.load(a1_ckpt, map_location="cpu")
        if "fpn" in a1:
            fpn.load_state_dict(a1["fpn"])
            if rank == 0:
                logger.info(f"Warm-started FPN from {a1_ckpt} (epoch {a1.get('epoch', '?')})")
        # MultiScaleEdgeHead has different architecture — skip edge_head warm-start
        if "edge_head" in a1 and edge_head_type != "multi_scale":
            edge_head.load_state_dict(a1["edge_head"])
            if rank == 0:
                logger.info(f"Warm-started EdgeHead from {a1_ckpt}")

    # Wrap in container for DeepSpeed
    model = EdgeTrainingModel(fpn, edge_head, is_multi_scale=is_multi_scale)

    # Training params (CLI overrides > config)
    epochs = args.epochs or int(cfg.get("train.epochs", 30))
    batch_size = args.batch_size or int(cfg.get("train.batch_size", 2))
    accumulation_steps = args.accumulation_steps or int(cfg.get("train.accum_steps", 4))
    max_grad_norm = args.max_grad_norm or float(cfg.get("train.max_grad_norm", 1.0))
    enable_amp = args.enable_amp if args.enable_amp is not None else (cfg.get("train.precision") == "bf16")

    # Build DS config args
    ds_args = {
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "enable_amp": enable_amp,
        "lr": float(cfg.get("train.lr_edge_head", 5e-5)),
        "wd": float(cfg.get("train.weight_decay", 1e-4)),
        "accelerator": device_str,
    }

    # Initialize DeepSpeed
    ds_cfg = build_ds_config(ds_args)
    if rank == 0:
        logger.info(f"DeepSpeed ZeRO-0, bf16={ds_cfg['bf16']['enabled']}, "
                     f"batch={batch_size}, accum={accumulation_steps}")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_cfg,
        model=model,
        model_parameters=model.parameters(),
    )
    if rank == 0:
        logger.info("DeepSpeed engine initialized")

    # Datasets
    line_width_train = int(cfg.get("data.line_width_train", 3))
    soft_sigma = float(cfg.get("loss.soft_edge_sigma", 0.0))
    soft_radius = int(cfg.get("loss.soft_edge_radius", 3))

    train_ds = CoastlineEdgeDataset(
        manifest["train"], line_width=line_width_train,
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
    )
    val_ds = CoastlineEdgeDataset(
        manifest["val"], line_width=int(cfg.get("data.line_width_eval", 1)),
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
    )

    num_workers = args.workers or int(cfg.get("data.num_workers", 2))
    train_loader = build_distributed_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        pin_memory=(device_str != "cpu"),
    )
    val_loader = build_distributed_loader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=(device_str != "cpu"),
    )
    if rank == 0:
        logger.info(f"Data: {len(train_ds)} train, {len(val_ds)} val samples")

    # Loss config
    loss_type = str(cfg.get("loss.type", "deep_supervised"))
    focal_alpha = float(cfg.get("loss.focal_alpha", 0.75))
    focal_gamma = float(cfg.get("loss.focal_gamma", 2.0))
    deep_sup_loss_fn: Optional[DeepSupervisedEdgeLoss] = None

    if loss_type == "deep_supervised":
        ds_side_weights = tuple(cfg.get("loss.side_weights", [0.5, 0.3, 0.2, 0.1]))
        ds_fused_weight = float(cfg.get("loss.fused_weight", 1.0))
        deep_sup_loss_fn = DeepSupervisedEdgeLoss(
            alpha=focal_alpha, gamma=focal_gamma,
            fused_weight=ds_fused_weight, side_weights=ds_side_weights,
        )

    # Training params
    val_interval = int(cfg.get("train.val_interval", 5))
    save_interval = int(cfg.get("train.save_interval", 5))
    log_interval = int(cfg.get("train.log_interval", 20))
    threshold = float(cfg.get("postprocess.threshold", 0.5))
    max_val_batches = int(cfg.get("eval.max_val_batches", 0))

    threshold_values = None
    sweep_metric = "buffered_f1_1px"
    if cfg.get("postprocess.mode") == "threshold_sweep":
        threshold_values = cfg.get("postprocess.threshold_values")
        sweep_metric = str(cfg.get("postprocess.threshold_select_metric", "buffered_f1_1px"))
        if threshold_values is None:
            threshold = float(cfg.get("postprocess.threshold_default", 0.30))

    if rank == 0:
        eff_batch = batch_size * accumulation_steps * world_size
        logger.info(f"Training: {epochs} epochs, effective_batch={eff_batch}")
        logger.info(f"Loss: {loss_type} (alpha={focal_alpha}, gamma={focal_gamma})"
                     f"{' soft_sigma=' + str(soft_sigma) if soft_sigma > 0 else ''}")

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, 'set_epoch') and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        avg_loss = train_epoch(
            model, model_engine, vision, train_loader, device,
            epoch=epoch, log_interval=log_interval, max_grad_norm=max_grad_norm,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            deep_sup_loss_fn=deep_sup_loss_fn,
        )

        if epoch % val_interval == 0:
            metrics = validate(
                model, vision, val_loader, device,
                output_dir=output_dir, epoch=epoch,
                threshold=threshold, max_batches=max_val_batches,
                threshold_values=threshold_values, sweep_metric=sweep_metric,
            )
            if rank == 0 and metrics:
                f1_1px = metrics.get("buffered_f1_1px", 0)
                if f1_1px > best_f1:
                    best_f1 = f1_1px
                    save_checkpoint(model_engine, 0, output_dir, best_f1)
                    logger.info(f"  New best F1@1px: {best_f1:.4f}")

        if epoch % save_interval == 0:
            save_checkpoint(model_engine, epoch, output_dir, best_f1)

    # Final save
    save_checkpoint(model_engine, epochs, output_dir, best_f1)
    if rank == 0:
        logger.info(f"Training complete. Best F1@1px: {best_f1:.4f}")


if __name__ == "__main__":
    main()

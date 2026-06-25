#!/usr/bin/env python3
"""
PoC-3 Coastline Edge Head — DDP Distributed Training (A2).

Uses torchrun + DDP (no DeepSpeed). Simpler and more stable for small models
(FPN + EdgeHead ~4.9M params) on NPU.

Usage:
    torchrun --nproc_per_node=8 scripts/poc_stage_edge_ddp.py \
        -c configs/poc3_edge_a2_soft_multiscale.yaml \
        --batch-size 2 --epochs 30 --device npu
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
from torch.nn.parallel import DistributedDataParallel as DDP
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
# Config loading
# =============================================================================


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, "r", encoding="utf-8") as f:
        return ConfigDict(yaml.safe_load(f))


# =============================================================================
# Vision encoder
# =============================================================================


def clean_vision_state_dict(state_dict: dict) -> dict:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("vision."):
            nk = nk[len("vision."):]
        cleaned[nk] = v
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
# DDP container
# =============================================================================


class EdgeTrainingModel(nn.Module):
    def __init__(self, fpn: FPNNeck, edge_head: nn.Module, is_multi_scale: bool = False):
        super().__init__()
        self.fpn = fpn
        self.edge_head = edge_head
        self.is_multi_scale = is_multi_scale

    def forward(self, c4, c8, c16, c32, vit_feat=None):
        p1, p2, p3, p4 = self.fpn(c4, c8, c16, c32, vit_feat=vit_feat)
        return self.edge_head(p1, p2, p3, p4)


# =============================================================================
# Training
# =============================================================================


def train_epoch(
    model: nn.Module,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    loss_type: str = "deep_supervised",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    deep_sup_loss_fn: Optional[DeepSupervisedEdgeLoss] = None,
) -> float:
    vision.eval()
    model.train()

    total_loss_sum = 0.0
    total_steps = 0
    is_multi_scale = getattr(model, 'is_multi_scale', isinstance(model.module, EdgeTrainingModel) and model.module.is_multi_scale)
    # Handle DDP wrapper
    raw_model = model.module if hasattr(model, 'module') else model

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            _, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if raw_model.fpn.has_vit else None

        if is_multi_scale:
            outputs = model(c4, c8, c16, c32, vit_feat=vit_feat)
            loss_dict = deep_sup_loss_fn(
                outputs["fused"],
                [outputs["side1"], outputs["side2"], outputs["side3"], outputs["side4"]],
                targets,
            )
            total_loss = loss_dict["total"]
            comp_name = "Focal"
            loss_comp1_val = loss_dict["loss_fused"].item()
            loss_dice_val = 0.0
        else:
            logits = model(c4, c8, c16, c32, vit_feat=vit_feat)
            if loss_type == "focal_dice":
                total_loss, loss_comp1, loss_dice = edge_focal_dice_loss(
                    logits, targets, alpha=focal_alpha, gamma=focal_gamma)
                comp_name = "Focal"
            else:
                total_loss, loss_comp1, loss_dice = edge_bce_dice_loss(logits, targets)
                comp_name = "BCE"
            loss_comp1_val = loss_comp1.item()
            loss_dice_val = loss_dice.item()

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        total_steps += 1

        if dist.get_rank() == 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            fg_ratio = targets.mean().item() if targets.numel() > 0 else 0.0
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
    model: nn.Module,
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

    is_multi_scale = getattr(model, 'is_multi_scale', False)
    if hasattr(model, 'module'):
        is_multi_scale = model.module.is_multi_scale
        raw_model = model.module
    else:
        raw_model = model

    do_sweep = threshold_values is not None and len(threshold_values) > 1
    rank = dist.get_rank()

    val_output_dir = Path(output_dir) / "vis" / f"val_epoch_{epoch:03d}"
    if rank == 0:
        val_output_dir.mkdir(parents=True, exist_ok=True)

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_metas: List[dict] = []
    all_images: List[torch.Tensor] = []

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device)
        targets_np = targets[:, 0].numpy()
        _, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if raw_model.fpn.has_vit else None

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

    if rank != 0:
        return {}

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

    all_metrics = [compute_all_edge_metrics(p, g, threshold=threshold)
                   for p, g in zip(all_probs, all_targets)]
    agg = _aggregate_metrics(all_metrics)
    agg["best_threshold"] = float(threshold)
    if do_sweep:
        agg["threshold_sweep"] = sweep_results

    logger.info(
        f"Val epoch {epoch} (thresh={threshold:.2f}): "
        f"pixel_f1={agg.get('pixel_f1', 0):.4f}, "
        f"buff_f1_1px={agg.get('buffered_f1_1px', 0):.4f}, "
        f"chamfer={agg.get('chamfer_distance_px', 0):.2f}px, "
        f"n={len(all_metrics)}"
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


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, output_dir: str, best_f1: float = 0.0):
    if dist.get_rank() != 0:
        return
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save({
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1,
    }, str(ckpt_dir / f"epoch_{epoch:03d}.pt"))
    logger.info(f"Checkpoint saved to epoch_{epoch:03d}.pt")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PoC-3 A2 DDP Training")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--build-manifest-only", action="store_true")
    args = parser.parse_args()

    # Init distributed
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.device == "npu":
        import torch_npu
        torch_npu.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device(args.device)

    # Load config
    cfg = load_config(args.config)
    output_dir = args.output or cfg.get("experiment.output_dir", "outputs/poc3_edge/a2_8npu_ddp")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger (rank 0 only)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s R{rank}] %(message)s" if rank == 0 else "%(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False

    # Build manifest
    manifest_path = cfg.get("data.manifest_path", str(Path(output_dir) / "coastline_manifest.json"))
    if rank == 0:
        manifest = build_coastline_manifest(
            data_roots=list(cfg.get("data.roots", [])),
            output_path=manifest_path,
            val_ratio=float(cfg.get("data.val_ratio", 0.2)),
            seed=int(cfg.get("data.val_split_seed", 42)),
        )
    dist.barrier()
    if rank != 0:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    if args.build_manifest_only:
        if rank == 0:
            logger.info("Manifest built. Exiting.")
        return

    if rank == 0:
        logger.info(f"Device: {device}, World: {world_size}")

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
        logger.info("Vision encoder frozen ✓")

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

    # Warm-start FPN+EdgeHead from A2 checkpoint
    a1_ckpt = cfg.get("model.a0_checkpoint")
    if a1_ckpt and Path(a1_ckpt).exists():
        a1 = torch.load(a1_ckpt, map_location="cpu")
        # Support both formats: nested {fpn:..., edge_head:...} and flat {model: {...}}
        if "model" in a1:
            model_sd = a1["model"]
            fpn_sd = {k[4:]: v for k, v in model_sd.items() if k.startswith("fpn.")}
            edge_sd = {k[10:]: v for k, v in model_sd.items() if k.startswith("edge_head.")}
        elif "fpn" in a1:
            fpn_sd = a1["fpn"]
            edge_sd = a1.get("edge_head", {})
        else:
            fpn_sd = {}
            edge_sd = {}
        if fpn_sd:
            fpn.load_state_dict(fpn_sd)
            if rank == 0:
                logger.info(f"Warm-started FPN from {a1_ckpt}")
        if edge_sd:
            edge_head.load_state_dict(edge_sd)
            if rank == 0:
                logger.info(f"Warm-started EdgeHead from {a1_ckpt}")

    # Wrap in DDP
    model = EdgeTrainingModel(fpn, edge_head, is_multi_scale=is_multi_scale)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank] if args.device == "npu" else None,
                find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {n_params:,} params (multi_scale={is_multi_scale})")

    # Datasets
    line_width_train = int(cfg.get("data.line_width_train", 3))
    soft_sigma = float(cfg.get("loss.soft_edge_sigma", 0.0))
    soft_radius = int(cfg.get("loss.soft_edge_radius", 3))
    image_size = int(cfg.get("data.image_size", 224))

    train_ds = CoastlineEdgeDataset(
        manifest["train"], line_width=line_width_train,
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
        image_size=image_size,
    )
    val_ds = CoastlineEdgeDataset(
        manifest["val"], line_width=int(cfg.get("data.line_width_eval", 1)),
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
        image_size=image_size,
    )

    batch_size = args.batch_size or int(cfg.get("train.batch_size", 2))
    num_workers = args.workers

    train_sampler = torch.utils.data.DistributedSampler(train_ds, num_replicas=world_size,
                                                         rank=rank, shuffle=True)
    val_sampler = torch.utils.data.DistributedSampler(val_ds, num_replicas=world_size,
                                                       rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
        pin_memory=True,
    )
    if rank == 0:
        logger.info(f"Data: {len(train_ds)} train, {len(val_ds)} val")

    # Optimizer
    lr_fpn = float(cfg.get("train.lr_fpn", 2e-5))
    lr_edge = float(cfg.get("train.lr_edge_head", 5e-5))
    wd = float(cfg.get("train.weight_decay", 1e-4))
    optimizer = torch.optim.AdamW([
        {"params": model.module.fpn.parameters(), "lr": lr_fpn},
        {"params": model.module.edge_head.parameters(), "lr": lr_edge},
    ], weight_decay=wd)

    # Loss
    loss_type = str(cfg.get("loss.type", "deep_supervised"))
    focal_alpha = float(cfg.get("loss.focal_alpha", 0.75))
    focal_gamma = float(cfg.get("loss.focal_gamma", 2.0))
    deep_sup_loss_fn = None
    if loss_type == "deep_supervised":
        ds_fused_w = float(cfg.get("loss.fused_weight", 1.0))
        ds_side_w = tuple(cfg.get("loss.side_weights", [0.5, 0.3, 0.2, 0.1]))
        deep_sup_loss_fn = DeepSupervisedEdgeLoss(
            alpha=focal_alpha, gamma=focal_gamma,
            fused_weight=ds_fused_w, side_weights=ds_side_w,
        )

    epochs = args.epochs or int(cfg.get("train.epochs", 30))
    val_interval = int(cfg.get("train.val_interval", 5))
    save_interval = int(cfg.get("train.save_interval", 5))
    log_interval = int(cfg.get("train.log_interval", 20))
    threshold = float(cfg.get("postprocess.threshold", 0.5))
    max_val_batches = int(cfg.get("eval.max_val_batches", 0))
    accumulation_steps = args.accumulation_steps

    threshold_values = None
    sweep_metric = "buffered_f1_1px"
    if cfg.get("postprocess.mode") == "threshold_sweep":
        threshold_values = cfg.get("postprocess.threshold_values")
        if threshold_values is None:
            threshold = float(cfg.get("postprocess.threshold_default", 0.30))

    if rank == 0:
        eff_batch = batch_size * accumulation_steps * world_size
        logger.info(f"Training: {epochs} epochs, eff_batch={eff_batch}, "
                     f"lr_fpn={lr_fpn}, lr_edge={lr_edge}")

    best_f1 = 0.0
    optimizer.zero_grad()
    accum_count = 0

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        vision.eval()

        total_loss_sum = 0.0
        total_steps = 0
        raw_model = model.module

        for batch_idx, (images, targets, metas) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                _, g_grid, pyramid_raw = vision.encode_with_spatial(images)

            c4, c8, c16, c32 = pyramid_raw
            vit_feat = g_grid if raw_model.fpn.has_vit else None

            if is_multi_scale:
                outputs = model(c4, c8, c16, c32, vit_feat=vit_feat)
                loss_dict = deep_sup_loss_fn(
                    outputs["fused"],
                    [outputs["side1"], outputs["side2"], outputs["side3"], outputs["side4"]],
                    targets,
                )
                loss_val = loss_dict["total"] / accumulation_steps
            else:
                logits = model(c4, c8, c16, c32, vit_feat=vit_feat)
                if loss_type == "focal_dice":
                    loss_val, _, _ = edge_focal_dice_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
                else:
                    loss_val, _, _ = edge_bce_dice_loss(logits, targets)
                loss_val = loss_val / accumulation_steps

            loss_val.backward()
            accum_count += 1

            if accum_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss_sum += loss_val.item() * accumulation_steps
            total_steps += 1

            if rank == 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss_sum / total_steps
                logger.info(f"Epoch {epoch:3d} | Step {batch_idx+1:5d} | Loss: {avg_loss:.4f}")

        avg_loss = total_loss_sum / max(total_steps, 1)
        if rank == 0:
            logger.info(f"Epoch {epoch:3d} done | Avg Loss: {avg_loss:.4f}")

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
                    save_checkpoint(model, optimizer, 0, output_dir, best_f1)
                    logger.info(f"  New best F1@1px: {best_f1:.4f}")

        if epoch % save_interval == 0:
            save_checkpoint(model, optimizer, epoch, output_dir, best_f1)

    save_checkpoint(model, optimizer, epochs, output_dir, best_f1)
    if rank == 0:
        logger.info(f"Done. Best F1@1px: {best_f1:.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

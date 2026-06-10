#!/usr/bin/env python3
"""
PoC-2 Landcover Semantic Head Training Script.

Orchestrates the PoC-2 training pipeline:
  - Loads frozen DualVisionEncoder from checkpoint
  - Builds FPN + LandcoverSemanticHead
  - Trains on partial-label land cover data with CE + Dice loss
  - Evaluates observed-pixel mIoU and per-class IoU
  - Exports overlay visualizations and polygonized GeoJSON predictions

Single NPU (or CPU), no DeepSpeed.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ml_collections import ConfigDict
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.semantic_head import LandcoverSemanticHead
from Dataset.landcover_dataset import (
    LandcoverSemanticDataset,
    landcover_collate_fn,
    split_by_source_image,
)
from Dataset.landcover_tile_grouping import (
    scan_landcover_directories,
    group_tiles_by_spatial_key,
    build_merged_samples,
)
from Dataset.landcover_label_map import (
    IGNORE_INDEX,
    BACKGROUND_ID,
    dlmc_to_train_id,
    train_id_to_dlmc,
    num_classes as get_num_classes,
)
from utils.georef_transform import pixel_to_wgs84
from utils.mask_utils import mask_to_polygon, filter_small_polygons
from utils.geojson_builder import (
    polygon_pixel_to_geojson_feature,
    build_feature_collection,
)
from utils.semantic_bg_prior import (
    scheduled_background_lambda,
    selective_pseudo_background_bce_loss,
)
from utils.semantic_overlay import save_semantic_overlays


# =============================================================================
# Config loading
# =============================================================================


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg


# =============================================================================
# Vision encoder
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
    print("Building DualVisionEncoder...")
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
    model_keys = set(vision.state_dict().keys())
    matched = len(model_keys & set(state_dict.keys()))
    ratio = matched / max(len(model_keys), 1)

    print(f"  Matched {matched}/{len(model_keys)} vision params ({ratio:.1%})")
    vision.load_state_dict(state_dict, strict=False)
    return vision


# =============================================================================
# Device
# =============================================================================


def resolve_device(device_str: str) -> torch.device:
    if device_str == "npu":
        try:
            import torch_npu  # noqa: F401
            return torch.device("npu:0")
        except (ImportError, RuntimeError):
            return torch.device("cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# =============================================================================
# Loss functions
# =============================================================================


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary Dice loss for a single class.

    Args:
        pred: [B, H, W] predicted probabilities (after sigmoid/softmax).
        target: [B, H, W] binary ground truth.
    """
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def observed_class_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Dice loss computed only for classes present in the current batch.

    Background (0) is excluded by default since we lack reliable bg labels.

    Args:
        logits: [B, C, H, W] raw logits.
        target: [B, H, W] int64 labels.
        ignore_index: label value to ignore.

    Returns:
        Scalar loss, or 0.0 if no observed foreground classes.
    """
    probs = torch.softmax(logits, dim=1)

    observed = torch.unique(target)
    observed = observed[(observed != ignore_index) & (observed != BACKGROUND_ID)]

    if len(observed) == 0:
        return logits.sum() * 0.0

    loss = 0.0
    for c in observed:
        pred_c = probs[:, int(c)]
        gt_c = (target == int(c)).float()
        loss = loss + dice_loss(pred_c, gt_c)

    return loss / len(observed)


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    lambda_bg: float = 0.0,
    min_labeled_ratio: float = 0.5,
    bg_type: str = "penalty",
    max_ignore_ratio: float = 0.25,
    max_labeled_for_bg: float = 0.3,
    pseudo_bg_threshold: float = 0.7,
    pseudo_bg_entropy_threshold: float = 0.25,
    pseudo_bg_protect_radius: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute combined CE + Dice loss with optional background prior.

    bg_type:
      - "penalty": penalize p(fg) on all ignore pixels (weak gradient)
      - "bce": sampled BCE on ignore pixels treated as background (stronger)
      - "selective_pseudo_bce": BCE only on confident, low-entropy, protected pseudo-bg pixels
      - "none": no background prior

    Conditional BCE (bg_type="bce"):
      Only applies background prior on tiles with labeled_ratio ≤ max_labeled_for_bg.
      On high-label tiles, the ignore region may contain unlabeled foreground,
      so we skip the BCE prior to protect thin/rare classes.

    Returns:
        total_loss, loss_ce, loss_dice, loss_bg, stats
    """
    valid_mask = target != ignore_index
    if valid_mask.sum() == 0:
        z = logits.sum() * 0.0
        return z, z, z, z, {}

    loss_ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    loss_dice = observed_class_dice_loss(logits, target, ignore_index)

    loss_bg = logits.sum() * 0.0
    bg_stats = {}
    if lambda_bg > 0:
        if bg_type == "hybrid":
            # Base: conditional BCE (maintains background signal throughout)
            loss_bg_base, base_stats = _sampled_background_bce_loss(
                logits, target, ignore_index, max_ignore_ratio, max_labeled_for_bg,
            )
            # Refinement: selective pseudo-bg on high-confidence pixels
            loss_bg_pseudo, pseudo_stats = selective_pseudo_background_bce_loss(
                logits,
                target,
                ignore_index=ignore_index,
                background_id=BACKGROUND_ID,
                p_bg_threshold=pseudo_bg_threshold,
                entropy_threshold=pseudo_bg_entropy_threshold,
                protect_radius_px=pseudo_bg_protect_radius,
                max_ignore_ratio=max_ignore_ratio,
            )
            loss_bg = loss_bg_base + loss_bg_pseudo
            # Merge stats with prefixes
            bg_stats = base_stats.copy()
            for k, v in pseudo_stats.items():
                bg_stats[f"pseudo_{k}"] = v
            total = loss_ce + loss_dice + lambda_bg * loss_bg
        elif bg_type == "selective_pseudo_bce":
            loss_bg, bg_stats = selective_pseudo_background_bce_loss(
                logits,
                target,
                ignore_index=ignore_index,
                background_id=BACKGROUND_ID,
                p_bg_threshold=pseudo_bg_threshold,
                entropy_threshold=pseudo_bg_entropy_threshold,
                protect_radius_px=pseudo_bg_protect_radius,
                max_ignore_ratio=max_ignore_ratio,
            )
            total = loss_ce + loss_dice + lambda_bg * loss_bg
        elif bg_type == "bce":
            loss_bg, bg_stats = _sampled_background_bce_loss(
                logits, target, ignore_index, max_ignore_ratio, max_labeled_for_bg,
            )
            total = loss_ce + loss_dice + lambda_bg * loss_bg
        else:
            loss_bg, bg_stats = _background_prior_loss(
                logits, target, ignore_index, min_labeled_ratio,
            )
            total = loss_ce + loss_dice + lambda_bg * loss_bg
    else:
        total = loss_ce + loss_dice

    # Per-size foreground prediction ratio
    probs = torch.softmax(logits, dim=1)
    pred_bg = probs[:, 0]  # [B, H, W]
    pred_fg = 1.0 - pred_bg
    bg_stats["background_pred_ratio"] = float(pred_bg.mean().item())
    bg_stats["pred_fg_ratio_per_tile"] = float(pred_fg.mean().item())  # batch-level, will be combined with meta later

    return total, loss_ce, loss_dice, loss_bg, bg_stats


def _background_prior_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    min_labeled_ratio: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-tile gated background prior: penalize p(foreground) on ignore pixels.

    Gate: skip tiles with labeled_ratio < min_labeled_ratio to avoid
    penalizing genuinely unannotated foreground.
    """
    probs = torch.softmax(logits, dim=1)
    p_fg = 1.0 - probs[:, 0]  # [B, H, W]

    batch_size = target.shape[0]
    losses = []
    active_tiles = 0
    for b in range(batch_size):
        tgt = target[b]
        ignore_mask = tgt == ignore_index
        labeled_mask = tgt != ignore_index

        if ignore_mask.sum() == 0:
            continue

        labeled_ratio = labeled_mask.float().mean()
        if labeled_ratio < min_labeled_ratio:
            continue

        active_tiles += 1
        losses.append(p_fg[b][ignore_mask].mean())

    stats = {"bg_prior_active_tile_ratio": active_tiles / batch_size if batch_size > 0 else 0.0}

    if len(losses) == 0:
        return logits.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats


def _sampled_background_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    max_ignore_ratio: float = 0.25,
    max_labeled_for_bg: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conditional sampled BCE: apply background prior only on low-label tiles.

    Gate: SKIP tiles with labeled_ratio > max_labeled_for_bg.
    On high-label tiles (e.g. 80%+ labeled), the ignore region may contain
    unlabeled foreground — applying BCE prior there crushes thin classes.
    On low-label tiles (e.g. ≤30% labeled), the ignore region is likely
    true background — BCE prior is safe.

    For the selected tiles, sample up to max_ignore_ratio * N_labeled
    ignore pixels and apply BCE loss treating them as background (class 0).

    Returns:
        loss: scalar tensor
        stats: dict with active_tile_ratio, sampled_pixel_count, active_labeled_ratio_mean
    """
    log_p = torch.log_softmax(logits, dim=1)  # [B, C, H, W]
    log_p_bg = log_p[:, 0]  # [B, H, W]

    batch_size = target.shape[0]
    losses = []
    active_tiles = 0
    total_sampled = 0
    active_labeled_ratios = []

    for b in range(batch_size):
        tgt = target[b]
        ignore_mask = tgt == ignore_index
        labeled_mask = tgt != ignore_index

        n_ignore = ignore_mask.sum().item()
        if n_ignore == 0:
            continue

        n_labeled = labeled_mask.sum().item()
        total_pixels = n_labeled + n_ignore
        labeled_ratio = n_labeled / total_pixels if total_pixels > 0 else 0.0

        # Conditional gate: skip if tile is highly labeled
        if labeled_ratio > max_labeled_for_bg:
            continue

        active_tiles += 1
        active_labeled_ratios.append(labeled_ratio)

        # Sample ignore pixels (at most max_ignore_ratio * n_labeled)
        max_sample = max(1, int(n_labeled * max_ignore_ratio)) if n_labeled > 0 else int(n_ignore * 0.1)
        n_sample = min(max_sample, n_ignore)
        total_sampled += n_sample

        ignore_indices = ignore_mask.nonzero(as_tuple=False)  # [N, 2]
        perm = torch.randperm(n_ignore, device=logits.device)[:n_sample]
        sampled = ignore_indices[perm]  # [n_sample, 2]

        # BCE: -log(p_bg) for sampled ignore pixels → pushes p_bg → 1
        loss_b = -log_p_bg[b][sampled[:, 0], sampled[:, 1]].mean()
        losses.append(loss_b)

    stats = {
        "bg_prior_active_tile_ratio": active_tiles / batch_size if batch_size > 0 else 0.0,
        "bg_prior_sampled_pixel_count": float(total_sampled),
        "active_bg_prior_labeled_ratio_mean": float(np.mean(active_labeled_ratios)) if active_labeled_ratios else 0.0,
    }

    if len(losses) == 0:
        return logits.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats


# =============================================================================
# Metrics
# =============================================================================


@torch.no_grad()
def compute_observed_pixel_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Compute observed-pixel IoU and per-class metrics.

    Only pixels where target != ignore_index are evaluated.

    Args:
        logits: [B, C, H, W].
        target: [B, H, W].
        num_classes: C (25 for PoC-2).

    Returns:
        Dict with mIoU, per_class_IoU, per_class_recall, pixel_accuracy.
    """
    pred = logits.argmax(dim=1)  # [B, H, W]
    valid_mask = target != ignore_index

    if valid_mask.sum() == 0:
        return {
            "mIoU": 0.0,
            "per_class_IoU": {},
            "per_class_recall": {},
            "pixel_accuracy": 0.0,
            "labeled_pixel_ratio": 0.0,
            "pred_foreground_ratio": float((pred != 0).float().mean().item()),
        }

    ious = {}
    recalls = {}

    for c in range(num_classes):
        pred_c = (pred == c) & valid_mask
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        gt_total = target_c.sum().item()

        ious[int(c)] = intersection / max(union, 1)
        recalls[int(c)] = intersection / max(gt_total, 1)

    # mIoU over classes with at least one GT pixel present
    active_classes = [
        c for c in range(num_classes)
        if ((target == c) & valid_mask).sum() > 0
    ]
    miou = sum(ious[c] for c in active_classes) / max(len(active_classes), 1)

    pixel_acc = (pred[valid_mask] == target[valid_mask]).float().mean().item()

    # Partial-label diagnostics: foreground spill ratios
    labeled_pixel_ratio = valid_mask.float().mean().item()
    pred_foreground_ratio = (pred != 0).float().mean().item()

    return {
        "mIoU": miou,
        "per_class_IoU": {train_id_to_dlmc(c): ious[c] for c in active_classes if c > 0},
        "per_class_recall": {train_id_to_dlmc(c): recalls[c] for c in active_classes if c > 0},
        "pixel_accuracy": pixel_acc,
        "labeled_pixel_ratio": labeled_pixel_ratio,
        "pred_foreground_ratio": pred_foreground_ratio,
    }


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    fpn: FPNNeck,
    sem_head: LandcoverSemanticHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
    num_classes: int,
) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "fpn": fpn.state_dict(),
            "sem_head": sem_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "num_classes": num_classes,
            "ignore_index": IGNORE_INDEX,
        },
        str(ckpt_path),
    )
    print(f"Checkpoint saved to {ckpt_path}")
    return str(ckpt_path)


# =============================================================================
# Training loop
# =============================================================================


def train_epoch(
    fpn: FPNNeck,
    sem_head: LandcoverSemanticHead,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    max_grad_norm: float = 1.0,
    lambda_bg: float = 0.0,
    min_labeled_ratio: float = 0.5,
    warmup_epochs: int = 0,
    lambda_ramp_epochs: int = 0,
    bg_type: str = "penalty",
    max_ignore_ratio: float = 0.25,
    max_labeled_for_bg: float = 0.3,
    pseudo_bg_threshold: float = 0.7,
    pseudo_bg_threshold_start: float = 0.3,
    pseudo_bg_threshold_ramp_epochs: int = 5,
    pseudo_bg_threshold_ramp_start_epoch: int = 1,
    pseudo_bg_entropy_threshold: float = 0.25,
    pseudo_bg_protect_radius: int = 8,
) -> float:
    vision.eval()      # frozen
    fpn.train()
    sem_head.train()
    total_loss_sum = 0.0
    total_steps = 0

    # Accumulate ablation stats over epoch
    epoch_bg_stats: Dict[str, List[float]] = {}

    # Warmup: disable background prior for early epochs
    effective_lambda = scheduled_background_lambda(
        lambda_bg,
        epoch=epoch,
        warmup_epochs=warmup_epochs,
        ramp_epochs=lambda_ramp_epochs,
    )
    if lambda_bg > 0 and effective_lambda == 0.0:
        print(f"  [bg_prior] warmup (epoch {epoch} <= {warmup_epochs}), lambda=0")
    elif lambda_bg > 0 and effective_lambda < lambda_bg:
        print(f"  [bg_prior] ramp epoch {epoch}: lambda={effective_lambda:.6f}/{lambda_bg:.6f}")

    # Dynamic pseudo-bg threshold ramping (for warm-start from BCE checkpoint)
    if bg_type == "selective_pseudo_bce" and pseudo_bg_threshold_ramp_epochs > 0:
        ramp_progress = min(1.0, max(0.0, (epoch - pseudo_bg_threshold_ramp_start_epoch) / pseudo_bg_threshold_ramp_epochs))
        dynamic_pseudo_bg_threshold = (
            pseudo_bg_threshold_start
            + (pseudo_bg_threshold - pseudo_bg_threshold_start) * ramp_progress
        )
        if epoch == pseudo_bg_threshold_ramp_start_epoch:
            print(f"  [bg_prior] pseudo_bg threshold ramp start: {pseudo_bg_threshold_start:.2f} -> {pseudo_bg_threshold:.2f} over {pseudo_bg_threshold_ramp_epochs} epochs")
    else:
        dynamic_pseudo_bg_threshold = pseudo_bg_threshold

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
        logits = sem_head(p1, p2, p3, p4)

        total_loss, loss_ce, loss_dice, loss_bg, bg_stats = compute_loss(
            logits, targets,
            lambda_bg=effective_lambda,
            min_labeled_ratio=min_labeled_ratio,
            bg_type=bg_type,
            max_ignore_ratio=max_ignore_ratio,
            max_labeled_for_bg=max_labeled_for_bg,
            pseudo_bg_threshold=dynamic_pseudo_bg_threshold,
            pseudo_bg_entropy_threshold=pseudo_bg_entropy_threshold,
            pseudo_bg_protect_radius=pseudo_bg_protect_radius,
        )

        # Accumulate stats
        for k, v in bg_stats.items():
            if v is not None:
                epoch_bg_stats.setdefault(k, []).append(v)

        total_loss.backward()

        if max_grad_norm > 0:
            trainable = list(fpn.parameters()) + list(sem_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        optimizer.step()

        total_loss_sum += total_loss.item()
        total_steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            bg_str = f" BgPrior: {loss_bg.item():.4f}" if effective_lambda > 0 else ""
            # Conditional BCE ablation stats
            if bg_stats.get("bg_prior_active_tile_ratio", 0.0) > 0:
                bg_str += (
                    f" | base_active={bg_stats['bg_prior_active_tile_ratio']:.2f}"
                    f" base_sampled={bg_stats.get('bg_prior_sampled_pixel_count', 0):.0f}"
                )
            # Hybrid mode: pseudo-bg stats prefixed with "pseudo_"
            pseudo_active_key = "pseudo_bg_active_tile_ratio" if "pseudo_bg_active_tile_ratio" in bg_stats else "pseudo_pseudo_bg_active_tile_ratio"
            pseudo_cand_key = "pseudo_bg_candidate_pixel_count" if "pseudo_bg_candidate_pixel_count" in bg_stats else "pseudo_pseudo_bg_candidate_pixel_count"
            pseudo_active_val = bg_stats.get(pseudo_active_key, 0.0)
            if pseudo_active_val > 0:
                bg_str += (
                    f" | pseudo_active={pseudo_active_val:.2f}"
                    f" pseudo_cand={bg_stats.get(pseudo_cand_key, 0):.0f}"
                )
            bg_str += f" | bg_pred={bg_stats.get('background_pred_ratio', 0):.3f}"
            print(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | CE: {loss_ce.item():.3f} Dice: {loss_dice.item():.3f}{bg_str}"
            )

    # Epoch summary stats
    avg_loss = total_loss_sum / max(total_steps, 1)
    summary_parts = []
    for k in [
        "bg_prior_active_tile_ratio",
        "pseudo_bg_active_tile_ratio",
        "background_pred_ratio",
        "pred_fg_ratio_per_tile",
    ]:
        vals = epoch_bg_stats.get(k, [])
        if vals:
            summary_parts.append(f"{k}={np.mean(vals):.3f}")
    summary_str = " (" + ", ".join(summary_parts) + ")" if summary_parts else ""
    print(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f}{summary_str}")
    return avg_loss


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(
    vision: DualVisionEncoder,
    fpn: FPNNeck,
    sem_head: LandcoverSemanticHead,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    num_classes: int,
    max_batches: int = 0,
) -> dict:
    vision.eval()
    fpn.eval()
    sem_head.eval()

    val_output_dir = Path(output_dir) / f"val_epoch_{epoch:03d}"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = val_output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    geojson_dir = val_output_dir / "geojson_pred"
    geojson_dir.mkdir(exist_ok=True)

    all_metrics: List[dict] = []
    by_size_metrics: Dict[str, List[dict]] = {}
    overlay_state = {"seen_keys": set(), "saved_sample_ids": set(), "saved_count": 0}
    max_overlay_samples = 12

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
        logits = sem_head(p1, p2, p3, p4)

        # Per-image metrics
        for i in range(images.shape[0]):
            m = compute_observed_pixel_metrics(
                logits[i:i + 1], targets[i:i + 1], num_classes
            )
            all_metrics.append(m)

            # By original size
            orig_size = metas[i].get("original_size", [128, 128])
            size_key = f"{orig_size[0]}x{orig_size[1]}"
            by_size_metrics.setdefault(size_key, []).append(m)

        # Save a small, diverse set of validation overlays across batches.
        if overlay_state["saved_count"] < max_overlay_samples:
            _save_overlays(
                images,
                targets,
                logits,
                metas,
                epoch,
                batch_idx,
                overlay_dir,
                max_overlay_samples,
                overlay_state,
            )

        # GeoJSON export (first batch only)
        if batch_idx == 0:
            _export_geojson_samples(
                logits, metas, epoch, batch_idx, geojson_dir
            )

    # Aggregate metrics
    global_miou = float(np.mean([m["mIoU"] for m in all_metrics]))
    global_pix_acc = float(np.mean([m["pixel_accuracy"] for m in all_metrics]))

    # Per-class average IoU
    per_class_ious: Dict[str, List[float]] = {}
    for m in all_metrics:
        for cls_name, iou in m["per_class_IoU"].items():
            per_class_ious.setdefault(cls_name, []).append(iou)
    avg_per_class = {k: float(np.mean(v)) for k, v in sorted(per_class_ious.items())}

    # By-size aggregation
    by_size_summary = {}
    for size_key, metrics_list in sorted(by_size_metrics.items()):
        by_size_summary[size_key] = {
            "observed_pixel_mIoU": float(np.mean([m["mIoU"] for m in metrics_list])),
            "n_samples": len(metrics_list),
            "pred_foreground_ratio": float(np.mean([
                m.get("pred_foreground_ratio", 0.0) for m in metrics_list
            ])),
        }

    # Foreground vs background confusion
    fg_bg_conf = _compute_fg_bg_confusion(
        [m for m in all_metrics], num_classes
    )

    print(
        f"Validation epoch {epoch}: "
        f"mIoU={global_miou:.4f}, pix_acc={global_pix_acc:.4f}, "
        f"batches={batch_idx + 1}"
    )

    # Over-prediction diagnostics for partial-label monitoring
    # Critical since unlabeled pixels (255) might attract foreground predictions
    diag = _compute_overprediction_diagnostics(all_metrics)

    return {
        "overall": {
            "observed_pixel_mIoU": global_miou,
            "per_class_IoU": avg_per_class,
            "per_class_recall": {},  # populated below
            "pixel_accuracy": global_pix_acc,
            "fg_bg_confusion_rate": fg_bg_conf,
        },
        "by_original_size": by_size_summary,
        "partial_label_diagnostics": diag,
    }


# =============================================================================
# Overlay + GeoJSON helpers
# =============================================================================


@torch.no_grad()
def _save_overlays(
    images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    metas: List[dict],
    epoch: int,
    batch_idx: int,
    output_dir: Path,
    max_samples: int = 12,
    state: Optional[dict] = None,
):
    """Save image/GT/pred overlay PNGs with metadata."""
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    save_semantic_overlays(
        images_np,
        targets_np,
        preds,
        metas,
        epoch=epoch,
        batch_idx=batch_idx,
        output_dir=output_dir,
        max_total=max_samples,
        state=state,
    )


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-id mask to a color visualization."""
    # Use a fixed colormap (tab20-like)
    colors = np.array([
        [0, 0, 0],       # 0: bg (black)
        [31, 119, 180],   # 1
        [255, 127, 14],   # 2
        [44, 160, 44],   # 3
        [148, 103, 189],  # 4
        [140, 86, 75],   # 5
        [227, 119, 194],  # 6
        [127, 127, 127],  # 7
        [188, 189, 34],  # 8
        [23, 190, 207],  # 9
        [174, 199, 232], # 10
        [255, 187, 120], # 11
        [152, 223, 138], # 12
        [197, 176, 213], # 13
        [196, 156, 148], # 14
        [247, 182, 210], # 15
        [199, 199, 199], # 16
        [219, 219, 141], # 17
        [158, 218, 229], # 18
        [255, 152, 150], # 19
        [255, 255, 51],  # 20
        [158, 202, 225], # 21
        [107, 174, 214], # 22
        [66, 146, 198],  # 23
        [33, 113, 181],  # 24
    ], dtype=np.uint8)
    ignore_color = np.array([64, 64, 64], dtype=np.uint8)

    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(colors.shape[0]):
        colored[mask == c] = colors[c]
    colored[mask == IGNORE_INDEX] = ignore_color
    return colored


@torch.no_grad()
def _export_geojson_samples(
    logits: torch.Tensor,
    metas: List[dict],
    epoch: int,
    batch_idx: int,
    output_dir: Path,
    num_samples: int = 2,
    min_area_px: float = 8.0,
    score_thresh: float = 0.5,
):
    """Export polygonized predictions as GeoJSON FeatureCollections."""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

    for i in range(min(num_samples, len(preds))):
        sample_id = metas[i].get("sample_id", f"s_{batch_idx}_{i}")
        georef = {
            "source_crs": metas[i].get("source_crs", "EPSG:4326"),
            "model_transform": metas[i].get("model_transform"),
        }

        features = []
        pred_mask = preds[i]

        for c in range(1, logits.shape[1]):  # skip background
            class_mask = (pred_mask == c).astype(np.uint8)
            if class_mask.sum() < min_area_px:
                continue

            # Per-class confidence: mean softmax prob in predicted region
            conf = float(probs[i, c][class_mask > 0].mean())

            polygons = mask_to_polygon(class_mask, simplify_epsilon=0.5)
            polygons = filter_small_polygons(polygons, min_area_px)

            for poly in polygons:
                if georef["model_transform"] is None:
                    continue
                try:
                    feat = polygon_pixel_to_geojson_feature(
                        poly, georef,
                        class_name=train_id_to_dlmc(c),
                        confidence=conf,
                    )
                    features.append(feat)
                except Exception:
                    continue

        fc = build_feature_collection(features)
        out_path = output_dir / f"{sample_id}.geojson"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False, indent=2)


def _compute_fg_bg_confusion(metrics_list: List[dict], num_classes: int) -> float:
    """Compute approximate fg→bg confusion rate.

    Measures what fraction of foreground pixels (class > 0) are predicted
    as background (class 0). Since we don't have reliable bg labels, this is
    estimated from per-class recall of class 0 on foreground pixels.
    """
    # For partial labels, a clean fg/bg confusion metric is not straightforward.
    # We report the average of (1 - recall) for foreground classes as a proxy.
    recalls = []
    for m in metrics_list:
        for c, r in m.get("per_class_recall", {}).items():
            if isinstance(c, int) and c > 0:
                continue
            recalls.append(r)
    if not recalls:
        return 0.0
    mean_recall = float(np.mean(recalls))
    return max(0.0, 1.0 - mean_recall)


def _compute_overprediction_diagnostics(metrics_list: List[dict]) -> dict:
    """Aggregate over-prediction risk indicators from per-sample metrics.

    For partial-label data, we monitor whether the model is spilling foreground
    predictions into unlabeled (ignore) regions.
    """
    if not metrics_list:
        return {}

    # Extract per-sample diagnostics collected during metric computation
    pred_fg_ratios = []
    labeled_ratios = []

    for m in metrics_list:
        if "pred_foreground_ratio" in m:
            pred_fg_ratios.append(m["pred_foreground_ratio"])
        if "labeled_pixel_ratio" in m:
            labeled_ratios.append(m["labeled_pixel_ratio"])

    result = {}
    if labeled_ratios:
        result["mean_labeled_pixel_ratio"] = float(np.mean(labeled_ratios))
    if pred_fg_ratios:
        result["mean_pred_foreground_ratio"] = float(np.mean(pred_fg_ratios))
    if labeled_ratios and pred_fg_ratios:
        mean_labeled = np.mean(labeled_ratios)
        mean_pred = np.mean(pred_fg_ratios)
        result["pred_to_labeled_area_ratio"] = float(
            mean_pred / max(mean_labeled, 1e-6)
        )
        # Approximate: foreground predicted in ignore regions
        result["ignore_region_foreground_rate"] = float(
            max(0, mean_pred - mean_labeled) / max(1 - mean_labeled, 1e-6)
        )

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PoC-2 Landcover Semantic Training")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "poc2_landcover_semantic.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--device", default="npu", help="Target device")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--eval-every", type=int, default=None, help="Override val interval")
    parser.add_argument("--save-every", type=int, default=None, help="Override save interval")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="Max val batches (0=full, useful for smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # CLI overrides
    output_dir = Path(args.output_dir or cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = args.epochs or cfg["train"]["epochs"]
    val_interval = args.eval_every or cfg["train"].get("val_interval", 5)
    save_interval = args.save_every or cfg["train"].get("save_interval", 5)
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    max_val_batches = args.max_val_batches

    # Setup logging directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = cfg["model"]["semantic_head"]["num_classes"]

    # ---- Resume or fresh start ----
    start_epoch = 1
    best_miou = 0.0

    # ---- Build vision encoder ----
    print("\n--- Building DualVisionEncoder ---")
    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"])
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False
    print("DualVisionEncoder frozen.")

    # ---- Build FPN + Semantic Head ----
    print("\n--- Building FPN + LandcoverSemanticHead ---")
    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)

    sem_head = LandcoverSemanticHead(
        in_channels=cfg["model"]["fpn"]["out_channels"],
        num_classes=num_classes,
        output_size=tuple(cfg["model"]["semantic_head"]["output_size"]),
    ).to(device)

    fpn_params = sum(p.numel() for p in fpn.parameters())
    sem_params = sum(p.numel() for p in sem_head.parameters())
    print(f"FPN params: {fpn_params:,}  |  Semantic Head params: {sem_params:,}")

    trainable_params = list(fpn.parameters()) + list(sem_head.parameters())

    # ---- Build dataset ----
    print("\n--- Building land cover dataset ---")
    data_cfg = cfg["data"]
    raw = scan_landcover_directories(
        patches_root=data_cfg["patches_root"],
        sizes=data_cfg.get("sizes", ["Size_128", "Size_256", "Size_512"]),
    )
    print(f"  Scanned {len(raw)} raw samples")
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)
    print(f"  Merged into {len(merged)} tile samples")

    train_samples, val_samples = split_by_source_image(
        merged,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        seed=data_cfg.get("val_split_seed", 42),
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    target_cache = str(_REPO_ROOT / "data" / "landcover_target_cache")
    train_ds = LandcoverSemanticDataset(train_samples, image_size=data_cfg["image_size"], cache_dir=target_cache)
    val_ds = LandcoverSemanticDataset(val_samples, image_size=data_cfg["image_size"], cache_dir=target_cache)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=landcover_collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=landcover_collate_fn,
    )

    # ---- Optimizer ----
    print("\n--- Setting up optimizer ---")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0001),
    )
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Learning rate: {cfg['train']['lr']}")

    # ---- Resume ----
    if args.resume:
        print(f"\n--- Resuming from {args.resume} ---")
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        fpn.load_state_dict(resume_ckpt["fpn"])
        sem_head.load_state_dict(resume_ckpt["sem_head"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_miou = resume_ckpt.get("best_miou", 0.0)
        print(f"  Resumed to epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Starting training: epoch {start_epoch} to {num_epochs}")
    print(f"Batch size: {batch_size}, Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Background prior config
    bg_prior_cfg = cfg["train"].get("background_prior", {})
    lambda_bg = float(bg_prior_cfg.get("lambda", 0.0) if bg_prior_cfg.get("enabled", False) else 0.0)
    min_labeled_ratio = float(bg_prior_cfg.get("min_labeled_ratio", 0.5))
    warmup_epochs = int(bg_prior_cfg.get("warmup_epochs", 1))
    lambda_ramp_epochs = int(bg_prior_cfg.get("lambda_ramp_epochs", 0))
    bg_type = str(bg_prior_cfg.get("type", "penalty"))
    max_ignore_ratio = float(bg_prior_cfg.get("max_ignore_ratio", 0.25))
    max_labeled_for_bg = float(bg_prior_cfg.get("max_labeled_for_bg", 0.3))
    pseudo_bg_threshold = float(bg_prior_cfg.get("pseudo_bg_threshold", 0.7))
    pseudo_bg_threshold_start = float(bg_prior_cfg.get("pseudo_bg_threshold_start", 0.3))
    pseudo_bg_threshold_ramp_epochs = int(bg_prior_cfg.get("pseudo_bg_threshold_ramp_epochs", 5))
    pseudo_bg_threshold_ramp_start = int(bg_prior_cfg.get("pseudo_bg_threshold_ramp_start_epoch", start_epoch))
    pseudo_bg_entropy_threshold = float(bg_prior_cfg.get("pseudo_bg_entropy_threshold", 0.25))
    pseudo_bg_protect_radius = int(bg_prior_cfg.get("pseudo_bg_protect_radius", 8))
    if lambda_bg > 0:
        print(f"Background prior enabled: type={bg_type}, lambda={lambda_bg}, "
              f"min_labeled={min_labeled_ratio}, warmup={warmup_epochs}, "
              f"lambda_ramp_epochs={lambda_ramp_epochs}, "
              f"max_ignore_ratio={max_ignore_ratio}, max_labeled_for_bg={max_labeled_for_bg}, "
              f"pseudo_bg_threshold={pseudo_bg_threshold}, "
              f"pseudo_bg_threshold_start={pseudo_bg_threshold_start}, "
              f"pseudo_bg_threshold_ramp_epochs={pseudo_bg_threshold_ramp_epochs}, "
              f"pseudo_bg_entropy_threshold={pseudo_bg_entropy_threshold}, "
              f"pseudo_bg_protect_radius={pseudo_bg_protect_radius}")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_epoch(
            fpn, sem_head, vision,
            train_loader, optimizer, device, epoch,
            log_interval=cfg["train"].get("log_interval", 20),
            max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
            lambda_bg=lambda_bg,
            min_labeled_ratio=min_labeled_ratio,
            warmup_epochs=warmup_epochs,
            lambda_ramp_epochs=lambda_ramp_epochs,
            bg_type=bg_type,
            max_ignore_ratio=max_ignore_ratio,
            max_labeled_for_bg=max_labeled_for_bg,
            pseudo_bg_threshold=pseudo_bg_threshold,
            pseudo_bg_threshold_start=pseudo_bg_threshold_start,
            pseudo_bg_threshold_ramp_epochs=pseudo_bg_threshold_ramp_epochs,
            pseudo_bg_threshold_ramp_start_epoch=pseudo_bg_threshold_ramp_start,
            pseudo_bg_entropy_threshold=pseudo_bg_entropy_threshold,
            pseudo_bg_protect_radius=pseudo_bg_protect_radius,
        )

        # Clear NPU cache after training to prevent memory fragmentation crash
        if device.type == "npu":
            torch.npu.empty_cache()

        if epoch == 1 or epoch % val_interval == 0:
            metrics = validate(
                vision, fpn, sem_head,
                val_loader, device,
                str(output_dir), epoch,
                num_classes,
                max_batches=max_val_batches,
            )

            # Save metrics
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            metrics_path = metrics_dir / f"metrics_epoch_{epoch:03d}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # Track best_miou
            current_miou = metrics["overall"]["observed_pixel_mIoU"]
            if current_miou > best_miou:
                best_miou = current_miou
                best_path = output_dir / "checkpoints" / "best_miou.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "fpn": fpn.state_dict(),
                    "sem_head": sem_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "ignore_index": IGNORE_INDEX,
                    "best_miou": best_miou,
                }, str(best_path))
                print(f"  New best mIoU: {best_miou:.4f} → saved to {best_path}")

            # Print diagnostics
            diag = metrics.get("partial_label_diagnostics", {})
            if diag:
                print(f"  [diag] labeled_ratio={diag.get('mean_labeled_pixel_ratio', 0):.3f} "
                      f"pred_fg_ratio={diag.get('mean_pred_foreground_ratio', 0):.3f} "
                      f"pred/labeled={diag.get('pred_to_labeled_area_ratio', 0):.2f} "
                      f"ignore_fg_rate={diag.get('ignore_region_foreground_rate', 0):.4f}")

            # Per-size pred_fg
            by_size = metrics.get("by_original_size", {})
            if by_size:
                size_fg_parts = []
                for sk in sorted(by_size.keys()):
                    pf = by_size[sk].get("pred_foreground_ratio", None)
                    if pf is not None:
                        size_fg_parts.append(f"{sk}={pf:.3f}")
                if size_fg_parts:
                    print(f"  [per-size pred_fg] {' '.join(size_fg_parts)}")

            # Stop-condition summary
            current_miou = metrics["overall"]["observed_pixel_mIoU"]
            pcls = metrics["overall"]["per_class_IoU"]
            pred_fg = diag.get("mean_pred_foreground_ratio", 0)
            pred_labeled = diag.get("pred_to_labeled_area_ratio", 0)
            ignore_fg = diag.get("ignore_region_foreground_rate", 0)

            checks = []
            if current_miou < best_miou - 0.02:
                checks.append(f"⚠ mIoU dropped {best_miou - current_miou:.3f} from best {best_miou:.4f}")
            if pred_labeled < 0.95:
                checks.append(f"⚠ pred/labeled={pred_labeled:.2f} < 0.95")
            if pcls.get("沿海滩涂", 0) < 0.70:
                checks.append(f"⚠ 沿海滩涂={pcls.get('沿海滩涂', 0):.3f} < 0.70")
            if checks:
                print(f"  [STOP CHECK] {' | '.join(checks)}")
            else:
                print(f"  [STOP CHECK] OK")

        if epoch % save_interval == 0:
            save_checkpoint(fpn, sem_head, optimizer, epoch, str(output_dir), num_classes)
            # Also save config alongside checkpoint for reproducibility
            config_copy = output_dir / "configs"
            config_copy.mkdir(exist_ok=True)
            import shutil
            shutil.copy(args.config, str(config_copy / Path(args.config).name))

        # Always save an epoch checkpoint after validation (for crash recovery)
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "fpn": fpn.state_dict(),
            "sem_head": sem_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "num_classes": num_classes,
            "ignore_index": IGNORE_INDEX,
            "best_miou": best_miou,
        }, str(latest_path))

        # Clear NPU cache after validation before next epoch
        if device.type == "npu":
            torch.npu.empty_cache()

    # ---- Final evaluation ----
    print("\n--- Final evaluation ---")
    if device.type == "npu":
        torch.npu.empty_cache()
    final_metrics = validate(
        vision, fpn, sem_head,
        val_loader, device,
        str(output_dir), num_epochs + 1,
        num_classes,
        max_batches=max_val_batches,
    )

    # Ensure best_miou checkpoint is up to date
    current_miou = final_metrics["overall"]["observed_pixel_mIoU"]
    if current_miou > best_miou:
        best_miou = current_miou
        best_path = output_dir / "checkpoints" / "best_miou.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": num_epochs,
            "fpn": fpn.state_dict(),
            "sem_head": sem_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "num_classes": num_classes,
            "ignore_index": IGNORE_INDEX,
            "best_miou": best_miou,
        }, str(best_path))

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print(f"Final metrics saved to {metrics_path}")

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}. Outputs in {output_dir}")


if __name__ == "__main__":
    main()

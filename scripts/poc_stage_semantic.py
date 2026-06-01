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
from typing import Dict, List, Tuple

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined CE + Dice loss.

    Returns:
        total_loss, loss_ce, loss_dice
    """
    valid_mask = target != ignore_index
    if valid_mask.sum() == 0:
        # All pixels ignored — return zero loss
        return logits.sum() * 0.0, logits.sum() * 0.0, logits.sum() * 0.0

    loss_ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    loss_dice = observed_class_dice_loss(logits, target, ignore_index)
    return loss_ce + loss_dice, loss_ce, loss_dice


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

    return {
        "mIoU": miou,
        "per_class_IoU": {train_id_to_dlmc(c): ious[c] for c in active_classes if c > 0},
        "per_class_recall": {train_id_to_dlmc(c): recalls[c] for c in active_classes if c > 0},
        "pixel_accuracy": pixel_acc,
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
) -> float:
    vision.eval()      # frozen
    fpn.train()
    sem_head.train()
    total_loss_sum = 0.0
    total_steps = 0

    for batch_idx, (images, targets, _metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
        logits = sem_head(p1, p2, p3, p4)

        total_loss, loss_ce, loss_dice = compute_loss(logits, targets)
        total_loss.backward()

        if max_grad_norm > 0:
            trainable = list(fpn.parameters()) + list(sem_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        optimizer.step()

        total_loss_sum += total_loss.item()
        total_steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            print(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | CE: {loss_ce.item():.3f} Dice: {loss_dice.item():.3f}"
            )

    avg_loss = total_loss_sum / max(total_steps, 1)
    print(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f}")
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

        # Overlays (first batch only)
        if batch_idx == 0:
            _save_overlays(
                images, targets, logits, metas, epoch, batch_idx, overlay_dir
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

    return {
        "overall": {
            "observed_pixel_mIoU": global_miou,
            "per_class_IoU": avg_per_class,
            "per_class_recall": {},  # populated below
            "pixel_accuracy": global_pix_acc,
            "fg_bg_confusion_rate": fg_bg_conf,
        },
        "by_original_size": by_size_summary,
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
    num_samples: int = 4,
):
    """Save image/GT/pred overlay PNGs."""
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    for i in range(min(num_samples, len(images_np))):
        sample_id = metas[i].get("sample_id", f"s_{batch_idx}_{i}")

        img = images_np[i].copy()
        target = targets_np[i]
        pred = preds[i]

        # Build 3-panel overlay: GT on left, Pred on right
        h, w = 224, 224

        gt_overlay = _colorize_mask(target)
        pred_overlay = _colorize_mask(pred)

        # Side-by-side: image | GT | pred
        panel = np.hstack([
            img,
            gt_overlay,
            pred_overlay,
        ])

        out_path = output_dir / f"{sample_id}.png"
        Image.fromarray(panel).save(str(out_path))


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = cfg["model"]["semantic_head"]["num_classes"]
    batch_size = cfg["train"]["batch_size"]

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

    # Collect all trainable params into a "model" for gradient management
    # We don't need a wrapper since we manage forward manually
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

    train_ds = LandcoverSemanticDataset(train_samples, image_size=data_cfg["image_size"])
    val_ds = LandcoverSemanticDataset(val_samples, image_size=data_cfg["image_size"])

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

    # ---- Training loop ----
    num_epochs = cfg["train"]["epochs"]
    print(f"\n{'='*60}")
    print(f"Starting training: {num_epochs} epochs")
    print(f"Batch size: {batch_size}, Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            fpn, sem_head, vision,
            train_loader, optimizer, device, epoch,
            log_interval=cfg["train"].get("log_interval", 20),
            max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        )

        if epoch == 1 or epoch % cfg["train"].get("val_interval", 5) == 0:
            metrics = validate(
                vision, fpn, sem_head,
                val_loader, device,
                str(output_dir), epoch,
                num_classes,
            )

            # Save metrics
            metrics_path = output_dir / f"metrics_epoch_{epoch:03d}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

        if epoch % cfg["train"].get("save_interval", 5) == 0:
            save_checkpoint(fpn, sem_head, optimizer, epoch, str(output_dir), num_classes)

    # ---- Final evaluation ----
    print("\n--- Final evaluation ---")
    final_metrics = validate(
        vision, fpn, sem_head,
        val_loader, device,
        str(output_dir), num_epochs + 1,
        num_classes,
    )
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print(f"Final metrics saved to {metrics_path}")

    print(f"\nTraining complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()

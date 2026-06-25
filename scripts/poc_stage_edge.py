#!/usr/bin/env python3
"""
PoC-3 Coastline Edge Head Training Script.

Orchestrates the PoC-3 training pipeline:
  - Builds coastline manifest from data directories
  - Loads frozen DualVisionEncoder from checkpoint
  - Builds ViT-FPN + Edge Head
  - Trains on coastline edge data with BCE+Dice loss
  - Evaluates with pixel metrics and buffered-F1
  - Exports overlay visualizations and GeoJSON predictions

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
import yaml
from ml_collections import ConfigDict
from PIL import Image

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


# =============================================================================
# Config loading
# =============================================================================


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg


def load_or_build_manifest(
    manifest_json: Optional[str],
    manifest_path: str,
    data_roots: List[str],
    val_ratio: float,
    seed: int,
    builder=build_coastline_manifest,
) -> dict:
    """Load an explicit manifest, or build one from data roots."""
    if manifest_json:
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or "train" not in manifest or "val" not in manifest:
            raise ValueError("manifest_json must contain 'train' and 'val' lists")
        print(
            f"Loaded manifest from {manifest_json}: "
            f"{len(manifest['train'])} train, {len(manifest['val'])} val"
        )
        return manifest

    return builder(
        data_roots=data_roots,
        output_path=manifest_path,
        val_ratio=val_ratio,
        seed=seed,
    )


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
# Checkpointing
# =============================================================================


def save_checkpoint(
    fpn: FPNNeck,
    edge_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
    best_f1: float = 0.0,
) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "fpn": fpn.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1": best_f1,
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
    edge_head: nn.Module,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    max_grad_norm: float = 1.0,
    loss_type: str = "bce_dice",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    deep_sup_loss_fn: Optional[DeepSupervisedEdgeLoss] = None,
) -> float:
    vision.eval()
    fpn.train()
    edge_head.train()

    total_loss_sum = 0.0
    total_steps = 0
    is_multi_scale = isinstance(edge_head, MultiScaleEdgeHead)

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=g_grid if fpn.has_vit else None)

        if is_multi_scale:
            outputs = edge_head(p1, p2, p3, p4)  # dict with 'fused', 'side1'..'side4'
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
            logits = edge_head(p1, p2, p3, p4)
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

        total_loss.backward()

        if max_grad_norm > 0:
            trainable = list(fpn.parameters()) + list(edge_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        optimizer.step()

        total_loss_sum += float(total_loss.item())
        total_steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            fg_ratio = float(targets.mean().item()) if targets.numel() > 0 else 0.0
            print(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | {comp_name}: {loss_comp1_val:.3f} "
                f"Dice: {loss_dice_val:.3f} | GT_fg: {fg_ratio:.4f}"
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
    edge_head: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    threshold: float = 0.5,
    max_batches: int = 0,
    threshold_values: Optional[List[float]] = None,
    sweep_metric: str = "buffered_f1_1px",
) -> dict:
    vision.eval()
    fpn.eval()
    edge_head.eval()

    is_multi_scale = isinstance(edge_head, MultiScaleEdgeHead)
    do_sweep = threshold_values is not None and len(threshold_values) > 1

    val_output_dir = Path(output_dir) / "vis" / f"val_epoch_{epoch:03d}"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir = val_output_dir / "geojson"
    geojson_dir.mkdir(exist_ok=True)

    # First pass: collect all heatmaps
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
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=g_grid if fpn.has_vit else None)

        if is_multi_scale:
            outputs = edge_head(p1, p2, p3, p4)
            logits = outputs["fused"]  # Use fused output for metrics
        else:
            logits = edge_head(p1, p2, p3, p4)

        probs = torch.sigmoid(logits)
        probs_np = probs[:, 0].cpu().numpy()

        for i in range(images.shape[0]):
            all_probs.append(probs_np[i])
            all_targets.append(targets_np[i])
            all_metas.append(metas[i])
            all_images.append(images[i].cpu())

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
        print(f"Threshold sweep (epoch {epoch}):")
        for t in threshold_values:
            marker = " <-- BEST" if t == best_t else ""
            print(f"  t={t:.2f}: {sweep_metric}={sweep_results[t]:.4f}{marker}")
        threshold = best_t
    else:
        best_t = threshold
        best_f1 = 0.0

    # Compute final metrics at best threshold
    all_metrics = [compute_all_edge_metrics(p, g, threshold=threshold)
                   for p, g in zip(all_probs, all_targets)]
    for m, meta in zip(all_metrics, all_metas):
        m["sample_id"] = meta.get("sample_id", "")

    # Save overlays at best threshold (first 12 samples)
    overlay_count = 0
    max_overlay = 12
    for idx in range(min(len(all_images), len(all_metas))):
        if overlay_count >= max_overlay:
            break

        meta = all_metas[idx]
        sid = meta.get("sample_id", "")
        _save_edge_overlay(
            all_images[idx], all_targets[idx], all_probs[idx],
            meta, epoch, val_output_dir, threshold=threshold,
        )
        overlay_count += 1

        # Export GeoJSON (first 5)
        if overlay_count <= 5:
            georef = {
                "source_crs": meta.get("source_crs", "EPSG:4326"),
                "model_transform": meta.get("model_transform", [1e-5, 0, 0, 0, -1e-5, 0]),
            }
            fc = postprocess_edge(
                all_probs[idx], georef,
                threshold=threshold, min_length=10,
                max_components=5, simplify_epsilon=1.0,
                sample_id=sid,
            )
            geo_path = geojson_dir / f"{sid}.geojson"
            with open(geo_path, 'w', encoding='utf-8') as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)

    agg = _aggregate_metrics(all_metrics)
    agg["best_threshold"] = float(threshold)
    if do_sweep:
        agg["threshold_sweep"] = sweep_results

    print(f"Validation epoch {epoch} (thresh={threshold:.2f}): "
          f"pixel_f1={agg.get('pixel_f1', 0):.4f}, "
          f"buffered_f1_1px={agg.get('buffered_f1_1px', 0):.4f}, "
          f"buffered_f1_3px={agg.get('buffered_f1_3px', 0):.4f}, "
          f"chamfer={agg.get('chamfer_distance_px', 0):.2f}px, "
          f"n_samples={len(all_metrics)}")

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
    """Save GT, prediction, and skeleton overlay images."""
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
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PoC-3 Coastline Edge Head Training")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (npu, cuda, cpu)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--build-manifest-only", action="store_true",
                        help="Build manifest and exit")
    parser.add_argument("--manifest-json", type=str, default=None,
                        help="Use an existing manifest JSON with train/val splits")
    parser.add_argument("--a0-checkpoint", type=str, default=None,
                        help="Warm-start FPN+Edge Head from A0 checkpoint")
    parser.add_argument("--unified-labels", type=str, default=None,
                        help="P3-L: Path to unified label directory (GeoJSON-derived .npy labels)")
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    cfg = ConfigDict(cfg_raw)

    output_dir = args.output or cfg.get("experiment.output_dir", "outputs/poc3_edge/a0")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device_str = args.device or cfg.get("train.device", "cpu")
    device = resolve_device(device_str)
    print(f"Device: {device}")

    # Build manifest
    print("Building coastline manifest...")
    manifest_path = cfg.get("data.manifest_path",
                             str(Path(output_dir) / "coastline_manifest.json"))
    manifest = load_or_build_manifest(
        manifest_json=args.manifest_json,
        manifest_path=manifest_path,
        data_roots=cfg.get("data.roots", []),
        val_ratio=cfg.get("data.val_ratio", 0.2),
        seed=cfg.get("data.val_split_seed", 42),
    )

    if args.build_manifest_only:
        print("Manifest built. Exiting.")
        return

    # Build vision encoder
    model_cfg = ConfigDict(cfg_raw.get("model", {}))
    vision = build_vision_encoder(
        model_cfg,
        cfg.get("model.vision_checkpoint",
                "./output/stage2/checkpoints/iter_2879_consolidated.pt")
    )
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    # Build FPN
    fpn_cfg = cfg.get("model.fpn", {})
    fpn = FPNNeck(
        in_channels=fpn_cfg.get("in_channels", [128, 256, 512, 1024]),
        out_channels=fpn_cfg.get("out_channels", 256),
        vit_in_channels=fpn_cfg.get("vit_in_channels", 1024),
    )
    fpn = fpn.to(device)
    print(f"FPN params: {sum(p.numel() for p in fpn.parameters()):,}")

    # Build Edge Head
    edge_head_type = cfg.get("model.edge_head.type", "single_scale")
    if edge_head_type == "multi_scale":
        edge_head = MultiScaleEdgeHead(
            in_channels=cfg.get("model.edge_head.in_channels", 256),
            output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
        )
        print(f"MultiScaleEdgeHead params: {sum(p.numel() for p in edge_head.parameters()):,}")
    else:
        edge_head = SingleScaleEdgeHead(
            in_channels=cfg.get("model.edge_head.in_channels", 256),
            decoder_channels=tuple(cfg.get("model.edge_head.decoder_channels", [256, 128])),
            output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
        )
        print(f"SingleScaleEdgeHead params: {sum(p.numel() for p in edge_head.parameters()):,}")
    edge_head = edge_head.to(device)

    # Warm-start from A0 checkpoint if specified
    a0_ckpt = args.a0_checkpoint or cfg.get("model.a0_checkpoint")
    if a0_ckpt:
        print(f"Warm-starting FPN + Edge Head from {a0_ckpt}")
        a0 = torch.load(a0_ckpt, map_location="cpu")
        fpn.load_state_dict(a0["fpn"])
        edge_head.load_state_dict(a0["edge_head"])
        print(f"  Loaded epoch {a0.get('epoch', '?')}, best_f1={a0.get('best_f1', '?')}")

    # Datasets
    line_width_train = cfg.get("data.line_width_train", 3)
    soft_sigma = cfg.get("loss.soft_edge_sigma", 0.0)
    soft_radius = cfg.get("loss.soft_edge_radius", 3)
    train_ds = CoastlineEdgeDataset(
        manifest["train"], line_width=line_width_train,
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
        unified_label_dir=args.unified_labels,
    )
    val_ds = CoastlineEdgeDataset(
        manifest["val"], line_width=cfg.get("data.line_width_eval", 1),
        soft_edge_sigma=soft_sigma, soft_edge_radius=soft_radius,
        unified_label_dir=args.unified_labels,
    )

    batch_size = args.batch_size or cfg.get("train.batch_size", 2)
    num_workers = cfg.get("data.num_workers", 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
        pin_memory=(device_str != "cpu"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=coastline_collate_fn,
    )
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": fpn.parameters(), "lr": cfg.get("train.lr_fpn", 5e-5)},
            {"params": edge_head.parameters(), "lr": cfg.get("train.lr_edge_head", 1e-4)},
        ],
        weight_decay=cfg.get("train.weight_decay", 1e-4),
    )

    epochs = args.epochs or cfg.get("train.epochs", 20)
    val_interval = cfg.get("train.val_interval", 5)
    save_interval = cfg.get("train.save_interval", 5)
    log_interval = cfg.get("train.log_interval", 20)
    max_grad_norm = cfg.get("train.max_grad_norm", 1.0)
    threshold = cfg.get("postprocess.threshold", 0.5)
    max_val_batches = cfg.get("eval.max_val_batches", 0)

    # Loss config
    loss_type = cfg.get("loss.type", "bce_dice")
    focal_alpha = cfg.get("loss.focal_alpha", 0.75)
    focal_gamma = cfg.get("loss.focal_gamma", 2.0)

    # Deep supervision loss (A2) — created once, reused per epoch
    deep_sup_loss_fn: Optional[DeepSupervisedEdgeLoss] = None
    if loss_type == "deep_supervised":
        ds_side_weights = tuple(cfg.get("loss.side_weights", [0.5, 0.3, 0.2, 0.1]))
        ds_fused_weight = cfg.get("loss.fused_weight", 1.0)
        deep_sup_loss_fn = DeepSupervisedEdgeLoss(
            alpha=focal_alpha, gamma=focal_gamma,
            fused_weight=ds_fused_weight, side_weights=ds_side_weights,
        )
        print(f"DeepSupervisedEdgeLoss: fused_weight={ds_fused_weight}, "
              f"side_weights={ds_side_weights}")

    # Threshold sweep config
    threshold_values = None
    if cfg.get("postprocess.mode", "minimal") == "threshold_sweep":
        threshold_values = cfg.get("postprocess.threshold_values", None)
        sweep_metric = cfg.get("postprocess.threshold_select_metric", "buffered_f1_1px")
        if threshold_values is None:
            threshold = cfg.get("postprocess.threshold_default", 0.30)

    loss_desc_map = {
        "bce_dice": "BCE+Dice",
        "focal_dice": "Focal+Dice",
        "deep_supervised": "DeepSup Focal+Dice (multi-scale)",
    }
    print(f"Training: {epochs} epochs, batch={batch_size}, "
          f"lr_fpn={cfg.get('train.lr_fpn', 5e-5)}, lr_edge={cfg.get('train.lr_edge_head', 1e-4)}")
    print(f"Loss: {loss_desc_map.get(loss_type, loss_type)} "
          f"(alpha={focal_alpha}, gamma={focal_gamma})"
          f"{' soft_sigma=' + str(soft_sigma) if soft_sigma > 0 else ''}")
    if threshold_values:
        print(f"Threshold sweep: {len(threshold_values)} values [{threshold_values[0]:.2f}..{threshold_values[-1]:.2f}]")

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(
            fpn, edge_head, vision, train_loader, optimizer, device,
            epoch=epoch, log_interval=log_interval, max_grad_norm=max_grad_norm,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            deep_sup_loss_fn=deep_sup_loss_fn,
        )

        if epoch % val_interval == 0:
            metrics = validate(
                vision, fpn, edge_head, val_loader, device,
                output_dir=output_dir, epoch=epoch,
                threshold=threshold, max_batches=max_val_batches,
                threshold_values=threshold_values, sweep_metric=sweep_metric,
            )
            f1_1px = metrics.get("buffered_f1_1px", 0)
            if f1_1px > best_f1:
                best_f1 = f1_1px
                best_thresh = metrics.get("best_threshold", threshold)
                save_checkpoint(fpn, edge_head, optimizer, 0, output_dir, best_f1)
                print(f"  New best buffered-F1@1px: {best_f1:.4f} @ thresh={best_thresh:.2f}")

        if epoch % save_interval == 0:
            save_checkpoint(fpn, edge_head, optimizer, epoch, output_dir, best_f1)

    # Final checkpoint
    save_checkpoint(fpn, edge_head, optimizer, epochs, output_dir, best_f1)
    print(f"Training complete. Best buffered-F1@1px: {best_f1:.4f}")


if __name__ == "__main__":
    main()

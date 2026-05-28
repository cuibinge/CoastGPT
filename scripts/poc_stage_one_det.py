#!/usr/bin/env python3
"""
PoC-1 Stage One Detection Training Script.

Orchestrates the complete PoC-1 training pipeline:
  - Loads frozen DualVisionEncoder from checkpoint
  - Builds FPN + Mask R-CNN detection head
  - Runs AquaPoCDataset training with torchvision Mask R-CNN
  - Exports GeoJSON predictions and overlay visualizations during validation

Single NPU (or CPU), no DeepSpeed, no EpochBasedTrainer.
"""
import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from ml_collections import ConfigDict

# --- Repo root path ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
from Models.det_head import (
    DualVisionFPNBackboneAdapter,
    FPNNeck,
    build_aqua_maskrcnn,
)
from Models.dual_vision_encoder import DualVisionEncoder
from utils.geojson_builder import outputs_to_geojson, validate_geojson
from utils.vis_overlay import save_overlay_grid


# =============================================================================
# Config loading
# =============================================================================


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg


# =============================================================================
# Vision checkpoint utilities
# =============================================================================


def clean_vision_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' (DDP/DeepSpeed) and 'vision.' prefixes from checkpoint keys."""
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("vision."):
            new_k = new_k[len("vision."):]
        cleaned[new_k] = v
    return cleaned


def build_vision_encoder(
    model_cfg: ConfigDict,
    ckpt_path: str,
) -> DualVisionEncoder:
    """Construct DualVisionEncoder and load pretrained weights.

    Args:
        model_cfg: ConfigDict with ``rgb_vision`` key for vision encoder config.
        ckpt_path: Path to consolidated checkpoint (may contain ``vision_ckpt``
            key, or be a raw vision state dict).

    Returns:
        DualVisionEncoder with weights loaded (strict=False).
    """
    print(f"Building DualVisionEncoder...")
    vision = DualVisionEncoder(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract vision weights from checkpoint
    if isinstance(ckpt, dict):
        if "vision_ckpt" in ckpt:
            state_dict = ckpt["vision_ckpt"]
            print("  Using 'vision_ckpt' key from checkpoint")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print("  Using 'model' key from checkpoint")
        else:
            state_dict = ckpt
            print("  Using full checkpoint as state dict")
    else:
        state_dict = ckpt

    state_dict = clean_vision_state_dict(state_dict)

    # Check parameter overlap ratio
    model_keys = set(vision.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    matched = len(model_keys & ckpt_keys)
    ratio = matched / len(model_keys) if len(model_keys) > 0 else 0.0

    if ratio < 0.5:
        warnings.warn(
            f"Only {ratio:.1%} of vision encoder parameters matched from checkpoint "
            f"({matched}/{len(model_keys)}). Check checkpoint path and config."
        )

    incompat = vision.load_state_dict(state_dict, strict=False)
    n_missing = len(incompat.missing_keys)
    n_unexpected = len(incompat.unexpected_keys)

    print(
        f"  Loaded {matched}/{len(model_keys)} vision parameters "
        f"({n_missing} missing, {n_unexpected} unexpected)"
    )
    return vision


# =============================================================================
# Device setup
# =============================================================================


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string (supports 'npu' and 'cuda'). Falls back to CPU."""
    if device_str == "npu":
        try:
            import torch_npu
            return torch.device("npu:0")
        except (ImportError, RuntimeError):
            warnings.warn("NPU not available, falling back to CPU")
            return torch.device("cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# =============================================================================
# Smoke tests
# =============================================================================


def smoke_test_npu_ops(device: torch.device) -> None:
    """Test torchvision detection ops on the target device."""
    print(f"Smoke-testing torchvision ops on {device}...")

    # nms
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=device)
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    assert len(keep) > 0, "nms returned empty result"
    print("  nms: OK")

    # batched_nms
    keep = torchvision.ops.batched_nms(
        boxes,
        scores,
        torch.tensor([0, 0, 1], dtype=torch.int64, device=device),
        0.5,
    )
    print("  batched_nms: OK")

    # roi_align
    feat = torch.randn(1, 256, 14, 14, device=device)
    rois = torch.tensor([[0, 0.0, 0.0, 5.0, 5.0]], dtype=torch.float32, device=device)
    pooled = torchvision.ops.roi_align(feat, rois, output_size=7, spatial_scale=1.0)
    assert pooled.shape == (1, 256, 7, 7), f"Unexpected roi_align shape: {pooled.shape}"
    print("  roi_align: OK")

    # MultiScaleRoIAlign
    msroi = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    ).to(device)
    pooled = msroi(OrderedDict([("0", feat)]), rois, [(14, 14)])
    print("  MultiScaleRoIAlign: OK")

    print("All torchvision ops OK on", device)


def smoke_test_maskrcnn(model: nn.Module, device: torch.device) -> None:
    """Full forward/backward smoke test with a synthetic image and one instance."""
    print("Smoke-testing Mask R-CNN forward/backward...")
    model.train()

    image = torch.randn(3, 224, 224, device=device)
    target_mask = torch.zeros(1, 224, 224, dtype=torch.uint8, device=device)
    target_mask[0, 50:150, 50:150] = 1

    images = [image]
    targets = [
        {
            "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32, device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
            "masks": target_mask,
            "image_id": torch.tensor([0], dtype=torch.int64, device=device),
            "area": torch.tensor([10000.0], dtype=torch.float32, device=device),
            "iscrowd": torch.tensor([0], dtype=torch.int64, device=device),
        }
    ]

    loss_dict = model(images, targets)
    assert isinstance(loss_dict, dict), f"Expected dict, got {type(loss_dict)}"

    required_keys = {
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    missing_keys = required_keys - set(loss_dict.keys())
    assert not missing_keys, f"Missing loss keys: {missing_keys}"

    total_loss = sum(v for v in loss_dict.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Loss is not finite: {total_loss}")

    total_loss.backward()

    # Verify gradients flowed to trainable parameters
    has_grad = any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    assert has_grad, "No gradients after backward pass"

    loss_items = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
    print(f"  Losses: {loss_items}")
    print(f"  Total: {total_loss.item():.4f}")
    print("Mask R-CNN smoke test passed.")


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    fpn: FPNNeck,
    maskrcnn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
) -> str:
    """Save FPN + Mask R-CNN weights and optimizer state.

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "fpn_state_dict": fpn.state_dict(),
            "maskrcnn_state_dict": maskrcnn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(ckpt_path),
    )
    print(f"Checkpoint saved to {ckpt_path}")
    return str(ckpt_path)


# =============================================================================
# Training loop
# =============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    max_grad_norm: float = 1.0,
) -> float:
    """Run a single training epoch.

    Returns:
        Average total loss over the epoch.
    """
    model.train()
    total_loss_sum = 0.0
    total_steps = 0

    for batch_idx, (images, targets, _metas) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        total_loss.backward()

        if max_grad_norm > 0:
            trainable_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

        optimizer.step()

        total_loss_sum += total_loss.item()
        total_steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss_sum / total_steps
            loss_str = "  ".join(
                f"{k}: {v.item():.3f}" for k, v in loss_dict.items()
            )
            print(
                f"Epoch {epoch:3d} | Step {batch_idx + 1:5d} | "
                f"Avg Loss: {avg_loss:.4f} | {loss_str}"
            )

    avg_loss = total_loss_sum / max(total_steps, 1)
    print(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    min_area_px: float = 8.0,
    max_batches: int = 3,
) -> dict:
    """Run validation: inference, GeoJSON export, overlay generation.

    Args:
        model: Mask R-CNN model in eval mode.
        dataloader: Validation DataLoader.
        device: Target device.
        output_dir: Root output directory.
        epoch: Current epoch number (for subdirectory naming).
        score_thresh: Minimum confidence score to keep a detection.
        mask_thresh: Binary threshold for mask logits.
        min_area_px: Minimum polygon area in pixels.
        max_batches: Maximum number of validation batches to process.

    Returns:
        Statistics dict with total_gt_boxes, total_pred_boxes, num_batches,
        geojson_files.
    """
    model.eval()

    val_output_dir = Path(output_dir) / f"val_epoch_{epoch:03d}"
    val_output_dir.mkdir(parents=True, exist_ok=True)

    all_geojson_paths: List[str] = []
    total_gt_boxes = 0
    total_pred_boxes = 0
    num_batches_processed = 0

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images_device = [img.to(device) for img in images]
        outputs = model(images_device)

        # Move outputs to CPU for post-processing
        outputs_cpu = [
            {k: v.cpu() for k, v in out.items()} for out in outputs
        ]

        total_gt_boxes += sum(len(t["boxes"]) for t in targets)
        total_pred_boxes += sum(len(o["boxes"]) for o in outputs_cpu)

        # ---- GeoJSON export ----
        geojson_results = outputs_to_geojson(
            outputs_cpu,
            metas,
            score_thresh=score_thresh,
            mask_thresh=mask_thresh,
            min_area_px=min_area_px,
        )

        for i, (fc, meta) in enumerate(zip(geojson_results, metas)):
            sample_id = meta.get("sample_id", f"batch_{batch_idx:02d}_{i:02d}")
            geojson_path = val_output_dir / f"{sample_id}.geojson"
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)
            all_geojson_paths.append(str(geojson_path))

            # Validate GeoJSON (warn on issues, don't abort)
            validation = validate_geojson(
                fc, tile_bounds_wgs84=meta.get("tile_bounds_wgs84")
            )
            if not validation["valid"]:
                warnings.warn(
                    f"GeoJSON validation issue for {sample_id}: "
                    f"{validation.get('errors', [])[:3]}"
                )

        # ---- Overlay images ----
        images_np = [
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for img in images
        ]

        # Build pred list for save_overlay_grid, filtering by score_thresh
        # to match GeoJSON export threshold
        pred_dicts = []
        for out in outputs_cpu:
            keep = out["scores"] >= score_thresh
            masks_sq = (
                out["masks"].squeeze(1)
                if out["masks"].ndim == 4
                else out["masks"]
            )
            pred_dicts.append(
                {
                    "boxes": out["boxes"][keep],
                    "masks": masks_sq[keep],
                    "scores": out["scores"][keep],
                }
            )

        sample_ids = [
            meta.get("sample_id", f"batch_{batch_idx:02d}_{i:02d}")
            for i, meta in enumerate(metas)
        ]
        save_overlay_grid(images_np, targets, pred_dicts, str(val_output_dir), sample_ids)

        num_batches_processed += 1

    stats = {
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "num_batches": num_batches_processed,
        "geojson_files": len(all_geojson_paths),
        "val_output_dir": str(val_output_dir),
    }

    print(
        f"Validation epoch {epoch}: "
        f"GT boxes={total_gt_boxes}, Pred boxes={total_pred_boxes}, "
        f"Batches={num_batches_processed}, GeoJSONs={len(all_geojson_paths)}"
    )
    return stats


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PoC-1 Stage One Detection Training"
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "poc_aqua_instance.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        default="npu",
        help="Target device: npu, cuda, or cpu",
    )
    args = parser.parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    # ---- Step 1: Build frozen vision encoder ----
    print("\n--- Step 1: Building DualVisionEncoder ---")
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

    # ---- Step 2: Build FPN + Mask R-CNN ----
    print("\n--- Step 2: Building FPN + Mask R-CNN ---")
    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)

    adapter = DualVisionFPNBackboneAdapter(vision, fpn).to(device)

    mrcnn_cfg = cfg["model"]["mask_rcnn"]
    model = build_aqua_maskrcnn(
        adapter,
        num_classes=mrcnn_cfg["num_classes"],
        anchor_sizes=tuple(tuple(s) for s in cfg["model"]["anchors"]["sizes"]),
        aspect_ratios=tuple(
            tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"]
        ),
        rpn_pre_nms_top_n_train=mrcnn_cfg.get("rpn_pre_nms_top_n_train", 512),
        rpn_post_nms_top_n_train=mrcnn_cfg.get("rpn_post_nms_top_n_train", 128),
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=mrcnn_cfg.get("box_score_thresh", 0.05),
        box_nms_thresh=mrcnn_cfg.get("box_nms_thresh", 0.5),
        box_detections_per_img=mrcnn_cfg.get("box_detections_per_img", 50),
        image_mean=mrcnn_cfg.get("image_mean", [0.0, 0.0, 0.0]),
        image_std=mrcnn_cfg.get("image_std", [1.0, 1.0, 1.0]),
        min_size=mrcnn_cfg.get("min_size", 224),
        max_size=mrcnn_cfg.get("max_size", 224),
    ).to(device)

    fpn_params = sum(p.numel() for p in fpn.parameters())
    maskrcnn_params = sum(
        p.numel() for p in model.parameters()
    ) - fpn_params - sum(p.numel() for p in vision.parameters())
    print(
        f"FPN params: {fpn_params:,}  |  "
        f"Mask R-CNN head params: {maskrcnn_params:,}"
    )

    # ---- Step 3: Smoke tests ----
    print("\n--- Step 3: Running smoke tests ---")
    smoke_test_npu_ops(device)
    smoke_test_maskrcnn(model, device)

    # ---- Step 4: Datasets ----
    print("\n--- Step 4: Loading datasets ---")
    data_cfg = cfg["data"]
    train_manifest = data_cfg["train_manifest"]
    val_manifest = data_cfg["val_manifest"]

    # Resolve manifest paths relative to repo root if not absolute
    for key, path_str in [("train", train_manifest), ("val", val_manifest)]:
        p = Path(path_str)
        if not p.is_absolute():
            resolved = str(_REPO_ROOT / p)
            print(f"  Resolved {key} manifest: {path_str} -> {resolved}")
            if key == "train":
                train_manifest = resolved
            else:
                val_manifest = resolved

    train_ds = AquaPoCDataset(
        manifest_path=train_manifest,
        data_root=data_cfg.get("data_root", "/home/ma-user/work/GeoJsonData"),
        image_size=data_cfg.get("image_size", 224),
    )
    val_ds = AquaPoCDataset(
        manifest_path=val_manifest,
        data_root=data_cfg.get("data_root", "/home/ma-user/work/GeoJsonData"),
        image_size=data_cfg.get("image_size", 224),
    )
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=poc_collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=poc_collate_fn,
    )

    # ---- Step 5: Optimizer (only trainable parameters) ----
    print("\n--- Step 5: Setting up optimizer ---")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0001),
    )
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Learning rate:    {cfg['train']['lr']}")

    # ---- Step 6: Training loop ----
    num_epochs = cfg["train"]["epochs"]
    print(f"\n{'='*60}")
    print(f"Starting training: {num_epochs} epochs")
    print(f"Batch size: {cfg['train']['batch_size']}, Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            log_interval=cfg["train"].get("log_interval", 10),
            max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        )

        if epoch % cfg["train"].get("val_interval", 1) == 0:
            val_stats = validate(
                model,
                val_loader,
                device,
                str(output_dir),
                epoch,
                score_thresh=cfg.get("eval", {}).get("score_thresh", 0.5),
                mask_thresh=cfg.get("eval", {}).get("mask_thresh", 0.5),
                min_area_px=cfg.get("eval", {}).get("min_area_px", 8.0),
            )

        if epoch % cfg["train"].get("save_interval", 1) == 0:
            save_checkpoint(fpn, model, optimizer, epoch, str(output_dir))

    print(f"\nTraining complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()

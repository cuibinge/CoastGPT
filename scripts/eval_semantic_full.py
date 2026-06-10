#!/usr/bin/env python3
"""Full validation of landcover semantic checkpoint on all val samples."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Dataset.landcover_tile_grouping import (
    scan_landcover_directories, group_tiles_by_spatial_key, build_merged_samples,
)
from Dataset.landcover_dataset import LandcoverSemanticDataset, split_by_source_image
from Dataset.landcover_label_map import IGNORE_INDEX, train_id_to_dlmc, active_class_names
from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.semantic_head import LandcoverSemanticHead


def clean_vision_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("vision."):
            new_k = new_k[len("vision."):]
        cleaned[new_k] = v
    return cleaned


def build_vision(cfg: ConfigDict, ckpt_path: str, device: torch.device) -> DualVisionEncoder:
    print("Building DualVisionEncoder...")
    vision = DualVisionEncoder(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("vision_ckpt", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt
    state_dict = clean_vision_state_dict(state_dict)

    # Filter out size-mismatched keys (PyTorch strict=False does not skip these)
    model_sd = vision.state_dict()
    filtered_sd = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            skipped.append(k)
            continue
        filtered_sd[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} size-mismatched keys: {skipped}")

    model_keys = set(model_sd.keys())
    matched = len(model_keys & set(filtered_sd.keys()))
    print(f"  Matched {matched}/{len(model_keys)} vision params ({matched/max(len(model_keys),1):.1%})")
    vision.load_state_dict(filtered_sd, strict=False)
    for p in vision.parameters():
        p.requires_grad = False
    vision.to(device)
    vision.eval()
    return vision


def compute_metrics_one(logits, target):
    # logits: [C, H, W] per sample, argmax over class dim
    pred = logits.argmax(dim=0)
    target_np = target.cpu().numpy()
    pred_np = pred.cpu().numpy()

    valid_mask = target_np != IGNORE_INDEX
    if valid_mask.sum() == 0:
        return None

    y_true = target_np[valid_mask]
    y_pred = pred_np[valid_mask]

    # IoU only for classes present in GT (matches training validation)
    num_classes = logits.shape[0]
    per_class_iou = {}
    for c in range(num_classes):
        if c == IGNORE_INDEX:
            continue
        gt_c = (target_np == c)
        if not gt_c.any():
            continue  # skip classes absent from this sample's GT
        pred_c = (pred_np == c) & valid_mask
        target_c = gt_c & valid_mask
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        per_class_iou[int(c)] = float(inter / max(union, 1))

    acc = (y_true == y_pred).mean()
    miou = np.mean(list(per_class_iou.values())) if per_class_iou else 0.0
    labeled_ratio = valid_mask.mean()
    pred_fg = (pred_np != 0).mean()

    return {
        "mIoU": miou,
        "pixel_accuracy": float(acc),
        "per_class_IoU": per_class_iou,
        "labeled_pixel_ratio": float(labeled_ratio),
        "pred_foreground_ratio": float(pred_fg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default=None, help="Path to stratified val split JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.npu.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config for vision encoder params
    config_path = _REPO_ROOT / "configs" / "poc2_landcover_semantic.yaml"
    with open(config_path) as f:
        cfg = ConfigDict(yaml.safe_load(f))
    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    ckpt_path = str(_REPO_ROOT / cfg["model"]["vision_checkpoint"])
    vision = build_vision(vis_cfg, ckpt_path, device)

    # FPN + Semantic Head
    print("Loading FPN + Semantic Head...")
    fpn = FPNNeck(in_channels=[128, 256, 512, 1024], out_channels=256)
    sem_head = LandcoverSemanticHead(in_channels=256, num_classes=25)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    fpn.load_state_dict(ckpt["fpn"])
    sem_head.load_state_dict(ckpt["sem_head"])
    print(f"  epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_miou','N/A')}")

    fpn.to(device)
    sem_head.to(device)
    fpn.eval()
    sem_head.eval()

    # Val dataset
    print("Building val dataset...")
    raw = scan_landcover_directories()
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)

    if args.split:
        with open(args.split) as f:
            split_data = json.load(f)
        val_ids = set(split_data["val_sample_ids"])
        val_samples = [s for s in merged if s["sample_id"] in val_ids]
        print(f"  Using stratified split: {args.split}")
    else:
        _, val_samples = split_by_source_image(merged, val_ratio=0.2, seed=42)
    print(f"  Val: {len(val_samples)} samples")

    val_ds = LandcoverSemanticDataset(
        val_samples, image_size=224,
        cache_dir=str(_REPO_ROOT / "data" / "landcover_target_cache"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=False,
        collate_fn=lambda batch: (
            torch.stack([b["image"] for b in batch]),
            torch.stack([b["target"] for b in batch]),
            [b["meta"] for b in batch],
        ),
    )

    print(f"Running full validation ({len(val_loader)} batches)...")
    all_metrics = []
    per_class_ious = defaultdict(list)
    by_size_metrics = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (images, targets, metas) in enumerate(val_loader):
            images = images.to(device)
            _, _, pyramid_raw = vision.encode_with_spatial(images)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
            logits = sem_head(p1, p2, p3, p4)

            for i in range(images.shape[0]):
                m = compute_metrics_one(logits[i], targets[i])
                if m is None:
                    continue
                all_metrics.append(m)
                for cid, iou in m["per_class_IoU"].items():
                    per_class_ious[train_id_to_dlmc(cid)].append(iou)
                sz = metas[i].get("original_size", [128, 128])
                by_size_metrics[f"{sz[0]}x{sz[1]}"].append(m)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(val_loader)}", flush=True)

    # --- Aggregate ---
    global_miou = float(np.mean([m["mIoU"] for m in all_metrics]))
    global_acc = float(np.mean([m["pixel_accuracy"] for m in all_metrics]))

    avg_per_class = {}
    for cls_name in active_class_names():
        if cls_name in per_class_ious:
            avg_per_class[cls_name] = {
                "IoU": float(np.mean(per_class_ious[cls_name])),
                "n_present": len(per_class_ious[cls_name]),
            }
        else:
            avg_per_class[cls_name] = {"IoU": 0.0, "n_present": 0}

    by_size = {}
    for sz, metrics in sorted(by_size_metrics.items()):
        by_size[sz] = {
            "mIoU": float(np.mean([m["mIoU"] for m in metrics])),
            "pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in metrics])),
            "n": len(metrics),
        }

    labeled_ratios = [m["labeled_pixel_ratio"] for m in all_metrics]
    pred_fg_ratios = [m["pred_foreground_ratio"] for m in all_metrics]
    mean_labeled = np.mean(labeled_ratios) if labeled_ratios else 0
    mean_pred = np.mean(pred_fg_ratios) if pred_fg_ratios else 0

    result = {
        "checkpoint": args.checkpoint,
        "n_val_samples": len(all_metrics),
        "n_val_batches": len(val_loader),
        "overall": {
            "mIoU": global_miou,
            "pixel_accuracy": global_acc,
            "per_class_IoU": avg_per_class,
        },
        "by_size": by_size,
        "partial_label_diagnostics": {
            "mean_labeled_pixel_ratio": float(mean_labeled),
            "mean_pred_foreground_ratio": float(mean_pred),
            "pred_to_labeled_area_ratio": float(mean_pred / max(mean_labeled, 1e-6)),
            "ignore_region_foreground_rate": float(
                max(0, mean_pred - mean_labeled) / max(1 - mean_labeled, 1e-6)
            ),
        },
    }

    print(f"\n=== Full Validation ({len(all_metrics)} samples) ===")
    print(f"  mIoU: {global_miou:.4f}, Pixel Acc: {global_acc:.4f}")
    print(f"\n  Per-class IoU:")
    for cls_name, info in sorted(avg_per_class.items(), key=lambda x: -x[1]["IoU"]):
        marker = "" if info["n_present"] > 0 else " (NO VAL SAMPLES)"
        print(f"    {cls_name:12s}: IoU={info['IoU']:.4f}  (n={info['n_present']}{marker})")

    print(f"\n  By size:")
    for sz, info in sorted(by_size.items()):
        print(f"    {sz}: mIoU={info['mIoU']:.4f}, pixel_acc={info['pixel_accuracy']:.4f}, n={info['n']}")

    diag = result["partial_label_diagnostics"]
    print(f"\n  Diagnostics:")
    print(f"    labeled: {diag['mean_labeled_pixel_ratio']*100:.1f}%")
    print(f"    pred_fg: {diag['mean_pred_foreground_ratio']*100:.1f}%")
    print(f"    pred/labeled: {diag['pred_to_labeled_area_ratio']:.2f}x")
    print(f"    ignore_fg_rate: {diag['ignore_region_foreground_rate']*100:.1f}%")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

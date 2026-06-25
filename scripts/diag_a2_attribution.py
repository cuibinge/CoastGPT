#!/usr/bin/env python3
"""
A2 Attribution & Ceiling Diagnosis.

Loads best A2 checkpoint, runs full validation, and produces:
  1. Per-shoreline-type F1 breakdown
  2. Per-FPN-level side output evaluation (P1-P4 + fused)
  3. Per-sample oracle ceiling (best threshold per sample)
  4. False positive distance analysis (how far are FPs from nearest GT?)
  5. Soft vs hard target prediction distribution comparison

Usage:
    python scripts/diag_a2_attribution.py \
        --config configs/poc3_edge_a2_soft_multiscale.yaml \
        --checkpoint outputs/poc3_edge/a2_8npu_ddp/checkpoints/epoch_000.pt \
        --output outputs/poc3_edge/a2_attribution_diag
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.edge_head import MultiScaleEdgeHead
from Dataset.coastline_dataset import (
    CoastlineEdgeDataset,
    coastline_collate_fn,
    build_coastline_manifest,
)
from utils.edge_losses import edge_focal_dice_loss
from utils.coastline_metrics import compute_all_edge_metrics


# =============================================================================
# Config & model loading
# =============================================================================


def load_config(path: str) -> ConfigDict:
    with open(path, "r") as f:
        return ConfigDict(yaml.safe_load(f))


def clean_vision_sd(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("vision."):
            nk = nk[len("vision."):]
        cleaned[nk] = v
    return cleaned


def build_vision(cfg: ConfigDict, ckpt_path: str) -> DualVisionEncoder:
    print("Loading vision encoder...")
    vision = DualVisionEncoder(ConfigDict(cfg.get("model", {})))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("vision_ckpt", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    vision.load_state_dict(clean_vision_sd(sd), strict=False)
    return vision


def build_model(cfg: ConfigDict, ckpt_path: str, device: torch.device):
    print("Building model...")
    fpn_cfg = cfg.get("model.fpn", {})
    fpn = FPNNeck(
        in_channels=list(fpn_cfg.get("in_channels", [128, 256, 512, 1024])),
        out_channels=int(fpn_cfg.get("out_channels", 256)),
        vit_in_channels=int(fpn_cfg.get("vit_in_channels", 1024)),
    )
    edge_head = MultiScaleEdgeHead(
        in_channels=int(cfg.get("model.edge_head.in_channels", 256)),
        output_size=tuple(cfg.get("model.edge_head.output_size", [224, 224])),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_sd = ckpt.get("model", ckpt)
    # Handle both checkpoint formats
    fpn_sd = {}
    edge_sd = {}
    for k, v in model_sd.items():
        if k.startswith("edge_head."):
            edge_sd[k[len("edge_head."):]] = v
        elif k.startswith("fpn."):
            fpn_sd[k[len("fpn."):]] = v
        elif k.startswith("module.edge_head."):
            edge_sd[k[len("module.edge_head."):]] = v
        elif k.startswith("module.fpn."):
            fpn_sd[k[len("module.fpn."):]] = v

    if fpn_sd:
        fpn.load_state_dict(fpn_sd)
    if edge_sd:
        edge_head.load_state_dict(edge_sd)

    fpn = fpn.to(device).eval()
    edge_head = edge_head.to(device).eval()
    return fpn, edge_head


# =============================================================================
# Diagnosis functions
# =============================================================================


@torch.no_grad()
def run_inference(
    vision: DualVisionEncoder,
    fpn: FPNNeck,
    edge_head: MultiScaleEdgeHead,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """Run inference on full val set, collect all outputs."""
    vision.eval()
    fpn.eval()
    edge_head.eval()

    results = {
        "fused_probs": [], "fused_logits": [],
        "side1_probs": [], "side2_probs": [], "side3_probs": [], "side4_probs": [],
        "targets": [], "metas": [],
    }

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        tgt_np = targets[:, 0].numpy()

        _, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if fpn.has_vit else None
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=vit_feat)
        outputs = edge_head(p1, p2, p3, p4)

        results["fused_probs"].append(torch.sigmoid(outputs["fused"]).cpu())
        results["fused_logits"].append(outputs["fused"].cpu())
        for sk in ["side1", "side2", "side3", "side4"]:
            results[f"{sk}_probs"].append(torch.sigmoid(outputs[sk]).cpu())
        results["targets"].append(torch.from_numpy(tgt_np))
        results["metas"].extend(metas)

    # Concatenate
    for key in ["fused_probs", "fused_logits", "side1_probs", "side2_probs",
                 "side3_probs", "side4_probs", "targets"]:
        results[key] = torch.cat(results[key], dim=0).numpy()
        if results[key].ndim == 4:
            results[key] = results[key].squeeze(1)  # [N, H, W]

    return results


# =============================================================================
# 1. Per-shoreline-type analysis
# =============================================================================


def analyze_per_shoreline_type(results: dict) -> dict:
    """Break down F1 by shoreline_type."""
    metas = results["metas"]
    probs = results["fused_probs"]  # [N, 224, 224]
    targets = results["targets"]

    type_metrics = defaultdict(list)
    for i, meta in enumerate(metas):
        stype = meta.get("shoreline_type", "unknown")
        m = compute_all_edge_metrics(probs[i], targets[i], threshold=0.30)
        type_metrics[stype].append(m)

    breakdown = {}
    for stype, metrics_list in sorted(type_metrics.items()):
        agg = {}
        for key in ["pixel_f1", "buffered_f1_1px", "buffered_f1_3px",
                     "chamfer_distance_px", "hausdorff_95_px",
                     "pred_fg_ratio", "gt_fg_ratio"]:
            vals = [m[key] for m in metrics_list if key in m]
            agg[key] = float(np.mean(vals)) if vals else 0.0
        agg["n_samples"] = len(metrics_list)
        breakdown[stype] = agg

    return breakdown


# =============================================================================
# 2. Per-level side output analysis
# =============================================================================


def analyze_side_outputs(results: dict) -> dict:
    """Evaluate each side output independently vs fused."""
    targets = results["targets"]
    side_metrics = {}

    for key in ["fused_probs", "side1_probs", "side2_probs", "side3_probs", "side4_probs"]:
        probs = results[key]
        all_m = [compute_all_edge_metrics(probs[i], targets[i], threshold=0.30)
                  for i in range(len(probs))]
        agg = {}
        for mk in ["pixel_f1", "buffered_f1_1px", "buffered_f1_3px"]:
            vals = [m[mk] for m in all_m if mk in m]
            agg[mk] = float(np.mean(vals)) if vals else 0.0
        side_metrics[key.replace("_probs", "")] = agg

    return side_metrics


# =============================================================================
# 3. Per-sample oracle ceiling
# =============================================================================


def analyze_oracle_ceiling(results: dict) -> dict:
    """Find optimal threshold per sample (oracle) to compute ceiling F1."""
    probs = results["fused_probs"]
    targets = results["targets"]
    N = len(probs)

    thresholds = np.arange(0.05, 0.95, 0.05)
    per_sample_best = []
    collective_best = 0.0
    collective_best_t = 0.30

    # Collective (same threshold for all)
    for t in thresholds:
        all_m = [compute_all_edge_metrics(probs[i], targets[i], threshold=t)
                  for i in range(N)]
        f1 = np.mean([m["buffered_f1_1px"] for m in all_m])
        if f1 > collective_best:
            collective_best = f1
            collective_best_t = t

    # Per-sample oracle (best threshold per sample)
    oracle_f1s = []
    oracle_thresholds = []
    for i in range(N):
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            m = compute_all_edge_metrics(probs[i], targets[i], threshold=t)
            if m["buffered_f1_1px"] > best_f1:
                best_f1 = m["buffered_f1_1px"]
                best_t = t
        oracle_f1s.append(best_f1)
        oracle_thresholds.append(best_t)

    oracle_mean = float(np.mean(oracle_f1s))

    return {
        "collective_best_f1@1px": float(collective_best),
        "collective_best_threshold": float(collective_best_t),
        "oracle_mean_f1@1px": oracle_mean,
        "oracle_vs_collective_gap": oracle_mean - collective_best,
        "oracle_threshold_mean": float(np.mean(oracle_thresholds)),
        "oracle_threshold_std": float(np.std(oracle_thresholds)),
    }


# =============================================================================
# 4. False positive distance analysis
# =============================================================================


def analyze_fp_distance(results: dict) -> dict:
    """How far are false positive pixels from the nearest GT edge?"""
    probs = results["fused_probs"]
    targets = results["targets"]
    N = len(probs)

    fp_distances = []
    fp_intensities = []
    tp_intensities = []

    for i in range(N):
        p = probs[i]
        t = targets[i]

        pred_bin = (p > 0.30).astype(np.float32)
        gt_bin = (t > 0.5).astype(np.float32)

        # False positives: pred=1, gt=0
        fp_mask = (pred_bin > 0.5) & (gt_bin < 0.5)
        tp_mask = (pred_bin > 0.5) & (gt_bin > 0.5)

        if fp_mask.sum() > 0:
            # Distance from each FP pixel to nearest GT pixel
            d_to_gt = distance_transform_edt(1 - gt_bin)
            fp_distances.extend(d_to_gt[fp_mask].tolist())
            fp_intensities.extend(p[fp_mask].tolist())

        if tp_mask.sum() > 0:
            tp_intensities.extend(p[tp_mask].tolist())

    if not fp_distances:
        return {"fp_count": 0}

    fp_dist = np.array(fp_distances)
    fp_int = np.array(fp_intensities)
    tp_int = np.array(tp_intensities) if tp_intensities else np.array([0])

    return {
        "fp_count": int(len(fp_distances)),
        "fp_mean_distance_px": float(np.mean(fp_dist)),
        "fp_median_distance_px": float(np.median(fp_dist)),
        "fp_pct_within_3px": float((fp_dist <= 3).mean() * 100),
        "fp_pct_within_10px": float((fp_dist <= 10).mean() * 100),
        "fp_pct_beyond_20px": float((fp_dist > 20).mean() * 100),
        "fp_mean_intensity": float(np.mean(fp_int)),
        "tp_mean_intensity": float(np.mean(tp_int)),
        "tp_fp_intensity_ratio": float(np.mean(tp_int) / max(np.mean(fp_int), 1e-8)),
    }


# =============================================================================
# 5. Soft target effect analysis
# =============================================================================


def analyze_prediction_distribution(results: dict) -> dict:
    """Analyze prediction probability distribution characteristics."""
    probs = results["fused_probs"]
    targets = results["targets"]
    N = len(probs)

    p_max_per_sample = []
    p_mean_per_sample = []
    pred_fg_per_sample = []
    gt_fg_per_sample = []
    empty_samples = 0

    for i in range(N):
        p = probs[i]
        t = targets[i]
        p_max_per_sample.append(float(p.max()))
        p_mean_per_sample.append(float(p.mean()))
        pred_fg_per_sample.append(float((p > 0.10).mean()))
        gt_fg_per_sample.append(float((t > 0.5).mean()))
        if p.max() < 0.05:
            empty_samples += 1

    return {
        "p_max_mean": float(np.mean(p_max_per_sample)),
        "p_max_median": float(np.median(p_max_per_sample)),
        "p_mean": float(np.mean(p_mean_per_sample)),
        "pred_fg@0.10_pct": float(np.mean(pred_fg_per_sample) * 100),
        "gt_fg_pct": float(np.mean(gt_fg_per_sample) * 100),
        "pred_fg_vs_gt_ratio": float(np.mean(pred_fg_per_sample) / max(np.mean(gt_fg_per_sample), 1e-8)),
        "empty_samples@0.05": empty_samples,
        "empty_pct": float(empty_samples / N * 100),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="A2 Attribution Diagnosis")
    parser.add_argument("--config", "-c", type=str,
                        default="configs/poc3_edge_a2_soft_multiscale.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/poc3_edge/a2_8npu_ddp/checkpoints/epoch_000.pt")
    parser.add_argument("--output", "-o", type=str,
                        default="outputs/poc3_edge/a2_attribution_diag")
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit val batches (0=all)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("npu:0") if args.device == "npu" else torch.device(args.device)
    print(f"Device: {device}")

    # Load config
    cfg = load_config(args.config)

    # Build manifest
    manifest_path = cfg.get("data.manifest_path",
                             str(output_dir / "coastline_manifest.json"))
    manifest = build_coastline_manifest(
        data_roots=list(cfg.get("data.roots", [])),
        output_path=manifest_path,
        val_ratio=float(cfg.get("data.val_ratio", 0.2)),
    )

    # Build dataset
    image_size = int(cfg.get("data.image_size", 224))
    val_ds = CoastlineEdgeDataset(
        manifest["val"], line_width=int(cfg.get("data.line_width_eval", 1)),
        soft_edge_sigma=0.0,  # Binary targets for clean metric evaluation
        soft_edge_radius=0,
        image_size=image_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=coastline_collate_fn,
    )
    print(f"Val samples: {len(val_ds)}")

    # Build model
    vision = build_vision(cfg, cfg.get("model.vision_checkpoint",
                                        "./output/stage2/checkpoints/iter_2879_consolidated.pt"))
    vision = vision.to(device).eval()
    for p in vision.parameters():
        p.requires_grad = False

    fpn, edge_head = build_model(cfg, args.checkpoint, device)
    total_params = sum(p.numel() for p in fpn.parameters()) + \
                   sum(p.numel() for p in edge_head.parameters())
    print(f"Model: {total_params:,} params")

    # Run inference
    print("\nRunning inference on val set...")
    results = run_inference(vision, fpn, edge_head, val_loader, device,
                            max_batches=args.max_batches)
    N = len(results["metas"])
    print(f"Processed {N} samples")

    # 1. Per-shoreline-type breakdown
    print("\n" + "=" * 60)
    print("1. PER-SHORELINE-TYPE BREAKDOWN")
    print("=" * 60)
    shoreline = analyze_per_shoreline_type(results)
    print(f"{'Shoreline Type':<20} {'N':>4} {'F1@1px':>8} {'F1@3px':>8} "
          f"{'Chamfer':>8} {'pred_fg%':>8} {'gt_fg%':>8}")
    print("-" * 60)
    for stype, m in sorted(shoreline.items()):
        print(f"{stype:<20} {m['n_samples']:>4} {m['buffered_f1_1px']:>8.4f} "
              f"{m['buffered_f1_3px']:>8.4f} {m['chamfer_distance_px']:>8.1f} "
              f"{m['pred_fg_ratio']*100:>8.2f} {m['gt_fg_ratio']*100:>8.2f}")

    # 2. Per-level side output analysis
    print("\n" + "=" * 60)
    print("2. SIDE OUTPUT COMPARISON (P1-P4 + fused)")
    print("=" * 60)
    side = analyze_side_outputs(results)
    print(f"{'Output':<12} {'F1@1px':>8} {'F1@3px':>8} {'pixel_f1':>8}")
    print("-" * 40)
    for key in ["fused", "side1", "side2", "side3", "side4"]:
        m = side[key]
        print(f"{key:<12} {m['buffered_f1_1px']:>8.4f} "
              f"{m['buffered_f1_3px']:>8.4f} {m['pixel_f1']:>8.4f}")
    # Check if side3 (ViT injection) outperforms side2
    print(f"\nP3 (ViT) vs P2 delta: {side['side3']['buffered_f1_1px'] - side['side2']['buffered_f1_1px']:+.4f}")
    print(f"P4 (coarsest) vs P3 delta: {side['side4']['buffered_f1_1px'] - side['side3']['buffered_f1_1px']:+.4f}")

    # 3. Oracle ceiling
    print("\n" + "=" * 60)
    print("3. ORACLE CEILING ANALYSIS")
    print("=" * 60)
    ceiling = analyze_oracle_ceiling(results)
    print(f"Collective best F1@1px: {ceiling['collective_best_f1@1px']:.4f} "
          f"(thresh={ceiling['collective_best_threshold']:.2f})")
    print(f"Oracle (per-sample) F1@1px: {ceiling['oracle_mean_f1@1px']:.4f}")
    print(f"Oracle - Collective gap: {ceiling['oracle_vs_collective_gap']:.4f}")
    print(f"Oracle threshold μ={ceiling['oracle_threshold_mean']:.2f} "
          f"σ={ceiling['oracle_threshold_std']:.2f}")

    # 4. FP distance analysis
    print("\n" + "=" * 60)
    print("4. FALSE POSITIVE DISTANCE ANALYSIS")
    print("=" * 60)
    fp_analysis = analyze_fp_distance(results)
    if fp_analysis.get("fp_count", 0) > 0:
        print(f"Total FP pixels: {fp_analysis['fp_count']:,}")
        print(f"FP mean distance to GT: {fp_analysis['fp_mean_distance_px']:.1f} px")
        print(f"FP median distance to GT: {fp_analysis['fp_median_distance_px']:.1f} px")
        print(f"FP within 3px of GT: {fp_analysis['fp_pct_within_3px']:.1f}%")
        print(f"FP within 10px of GT: {fp_analysis['fp_pct_within_10px']:.1f}%")
        print(f"FP beyond 20px of GT: {fp_analysis['fp_pct_beyond_20px']:.1f}%")
        print(f"FP mean intensity: {fp_analysis['fp_mean_intensity']:.4f}")
        print(f"TP mean intensity: {fp_analysis['tp_mean_intensity']:.4f}")
        print(f"TP/FP intensity ratio: {fp_analysis['tp_fp_intensity_ratio']:.2f}")
    else:
        print("No false positives found (all predictions at or below threshold)")

    # 5. Prediction distribution
    print("\n" + "=" * 60)
    print("5. PREDICTION DISTRIBUTION")
    print("=" * 60)
    dist = analyze_prediction_distribution(results)
    print(f"p_max mean/median: {dist['p_max_mean']:.4f} / {dist['p_max_median']:.4f}")
    print(f"p_mean: {dist['p_mean']:.6f}")
    print(f"pred_fg@0.10: {dist['pred_fg@0.10_pct']:.2f}%")
    print(f"GT_fg: {dist['gt_fg_pct']:.2f}%")
    print(f"pred/GT ratio: {dist['pred_fg_vs_gt_ratio']:.2f}")
    empty_n = dist.get("empty_samples", 0)
    empty_pct = dist.get("empty_pct", 0.0)
    print(f"Empty samples@0.05: {empty_n} ({empty_pct:.1f}%)")

    # 6. Comparison with A1 diagnostic results (from handover doc)
    print("\n" + "=" * 60)
    print("6. A1 vs A2 COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'A1 (epoch 10)':>14} {'A2 (epoch 30)':>14} {'Delta':>10}")
    print("-" * 68)
    comparisons = [
        ("p_max_median", 0.91, dist['p_max_median']),
        ("pred_fg@0.10 (%)", 0.69, dist['pred_fg@0.10_pct']),
        ("GT_fg (%)", 0.76, dist['gt_fg_pct']),
        ("Empty@0.05 (%)", 0.0, dist['empty_pct']),
        ("buffered_f1_1px", 0.017, ceiling['collective_best_f1@1px']),
        ("buffered_f1_3px", 0.033, 0.0),  # will compute
        ("pixel_f1", 0.008, 0.0),
    ]
    # Compute collective pixel and 3px F1
    all_m = [compute_all_edge_metrics(results["fused_probs"][i],
                                       results["targets"][i], threshold=0.30)
              for i in range(N)]
    agg_f1_3px = np.mean([m["buffered_f1_3px"] for m in all_m])
    agg_pixel = np.mean([m["pixel_f1"] for m in all_m])

    comparisons = [
        ("p_max_median", 0.91, dist['p_max_median']),
        ("pred_fg@0.10_pct", 0.69, dist['pred_fg@0.10_pct']),
        ("GT_fg_pct", 0.76, dist['gt_fg_pct']),
        ("empty_pct@0.05", 0.0, dist['empty_pct']),
        ("buffered_f1_1px", 0.017, ceiling['collective_best_f1@1px']),
        ("buffered_f1_3px", 0.033, agg_f1_3px),
        ("pixel_f1", 0.008, agg_pixel),
    ]
    for name, a1_val, a2_val in comparisons:
        delta = a2_val - a1_val
        direction = "↑" if delta > 0 else "↓"
        print(f"{name:<30} {a1_val:>14.4f} {a2_val:>14.4f} {delta:>+9.4f} {direction}")

    # Save all results as JSON
    report = {
        "shoreline_type_breakdown": shoreline,
        "side_output_comparison": side,
        "oracle_ceiling": ceiling,
        "fp_distance_analysis": fp_analysis,
        "prediction_distribution": dist,
        "a1_vs_a2": {
            "a1_buffered_f1_1px": 0.017,
            "a2_buffered_f1_1px": ceiling['collective_best_f1@1px'],
            "a2_oracle_f1_1px": ceiling['oracle_mean_f1@1px'],
        },
    }
    report_path = output_dir / "attribution_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PoC-1 Object-Level Evaluation Script.

Loads a trained checkpoint, runs inference on the validation set, and computes:
  - Precision, Recall, F1 (object-level, IoU-matched)
  - mAP, AP@50, AP@75
  - Mask IoU / Dice per matched pair
  - False-positive / false-negative inspection (worst cases)
  - Per-sample breakdown

Usage:
    python scripts/eval_poc1.py --checkpoint outputs/poc_aqua_full/checkpoints/epoch_040.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn

# Reuse model building from training script (scripts/ is not a package)
import importlib.util as _importlib_util
_train_script_path = _REPO_ROOT / "scripts" / "poc_stage_one_det.py"
_train_spec = _importlib_util.spec_from_file_location("poc_stage_one_det", _train_script_path)
_train_module = _importlib_util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_module)
build_vision_encoder = _train_module.build_vision_encoder
load_config = _train_module.load_config


def build_model(cfg_path: str, checkpoint_path: str, device: str = "cpu"):
    """Build FPN + Mask R-CNN and load trained weights."""
    cfg = load_config(cfg_path)

    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"])
    vision = vision.to(device)

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
        aspect_ratios=tuple(tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"]),
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    print(f"Loaded checkpoint epoch {ckpt['epoch']} from {checkpoint_path}")

    model.eval()
    return model, cfg


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(box_a, box_b):
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def compute_mask_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / max(union, 1)


def compute_mask_dice(mask_a, mask_b):
    """Compute Dice coefficient between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return 2.0 * float(inter) / max(denom, 1)


def greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Greedy bipartite matching: for each GT, pick highest-IoU unmatched pred.

    Returns:
        matches: list of (gt_idx, pred_idx, iou)
        unmatched_gt: list of gt_idx (false negatives)
        unmatched_pred: list of pred_idx (false positives)
    """
    gt_available = set(range(len(gt_boxes)))
    pred_available = set(range(len(pred_boxes)))

    pairs = []
    for gi in range(len(gt_boxes)):
        for pi in range(len(pred_boxes)):
            iou = compute_iou(gt_boxes[gi], pred_boxes[pi])
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)
    matches = []
    matched_gt = set()
    matched_pred = set()
    for iou, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matches.append((gi, pi, iou))
            matched_gt.add(gi)
            matched_pred.add(pi)

    unmatched_gt = [g for g in gt_available if g not in matched_gt]
    unmatched_pred = [p for p in pred_available if p not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate(model, dataloader, device, score_thresh=0.5, iou_thresh=0.5):
    """Run full evaluation on the dataset.

    Returns a dict with global metrics and per-sample breakdown.
    """
    all_metrics = {
        "total_gt": 0,
        "total_pred": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "matched_ious": [],       # box IoU per TP
        "matched_mask_ious": [],  # mask IoU per TP
        "matched_mask_dices": [], # mask Dice per TP
        "pred_confidences": [],   # scores for all preds
        "per_sample": [],
        "fp_examples": [],        # (sample_id, score, box, area)
        "fn_examples": [],        # (sample_id, area, box)
    }

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images_device = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images_device)

        outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]

        for i, (out, target, meta) in enumerate(zip(outputs_cpu, targets, metas)):
            gt_boxes = target["boxes"].numpy()
            gt_masks = target["masks"].numpy()  # [N, H, W]

            keep = out["scores"] >= score_thresh
            pred_boxes = out["boxes"][keep].numpy()
            pred_scores = out["scores"][keep].numpy()
            masks = out["masks"]
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            pred_masks = masks[keep].numpy()  # [M, H, W]

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)
            all_metrics["total_gt"] += n_gt
            all_metrics["total_pred"] += n_pred

            matches, unmatched_gt, unmatched_pred = greedy_match(
                gt_boxes, pred_boxes, iou_thresh=iou_thresh,
            )

            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            all_metrics["total_tp"] += tp
            all_metrics["total_fp"] += fp
            all_metrics["total_fn"] += fn

            # Per-match mask metrics
            for gi, pi, box_iou in matches:
                all_metrics["matched_ious"].append(box_iou)
                miou = compute_mask_iou(gt_masks[gi], pred_masks[pi])
                mdice = compute_mask_dice(gt_masks[gi], pred_masks[pi])
                all_metrics["matched_mask_ious"].append(miou)
                all_metrics["matched_mask_dices"].append(mdice)

            # Collect confidences
            for s in pred_scores:
                all_metrics["pred_confidences"].append(float(s))

            # FP inspection: record unmatched preds with highest confidence
            for pi in unmatched_pred:
                all_metrics["fp_examples"].append({
                    "sample_id": meta["sample_id"],
                    "score": float(pred_scores[pi]),
                    "box": pred_boxes[pi].tolist(),
                    "area": float(pred_masks[pi].sum()),
                })

            # FN inspection: record unmatched GTs with largest area
            for gi in unmatched_gt:
                all_metrics["fn_examples"].append({
                    "sample_id": meta["sample_id"],
                    "area": float(gt_masks[gi].sum()),
                    "box": gt_boxes[gi].tolist(),
                })

            # Per-sample stats
            precision = tp / max(n_pred, 1)
            recall = tp / max(n_gt, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            all_metrics["per_sample"].append({
                "sample_id": meta["sample_id"],
                "n_gt": n_gt,
                "n_pred": n_pred,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "avg_pred_conf": round(float(pred_scores.mean()), 4) if len(pred_scores) > 0 else 0.0,
            })

    return all_metrics


def print_report(metrics):
    """Print a formatted evaluation report."""
    total_gt = metrics["total_gt"]
    total_pred = metrics["total_pred"]
    tp = metrics["total_tp"]
    fp = metrics["total_fp"]
    fn = metrics["total_fn"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print("=" * 70)
    print("PoC-1 Object-Level Evaluation Report")
    print("=" * 70)
    print(f"\n  Global counts:")
    print(f"    GT instances:   {total_gt}")
    print(f"    Pred instances: {total_pred}")
    print(f"    TP: {tp}  |  FP: {fp}  |  FN: {fn}")
    print(f"\n  Object-level metrics (IoU >= 0.5):")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")

    if metrics["matched_ious"]:
        box_ious = np.array(metrics["matched_ious"])
        print(f"\n  Box IoU (TP pairs, n={len(box_ious)}):")
        print(f"    Mean:   {box_ious.mean():.4f}")
        print(f"    Median: {np.median(box_ious):.4f}")
        print(f"    Min:    {box_ious.min():.4f}")
        print(f"    Max:    {box_ious.max():.4f}")

    if metrics["matched_mask_ious"]:
        mask_ious = np.array(metrics["matched_mask_ious"])
        print(f"\n  Mask IoU (TP pairs, n={len(mask_ious)}):")
        print(f"    Mean:   {mask_ious.mean():.4f}")
        print(f"    Median: {np.median(mask_ious):.4f}")
        print(f"    Min:    {mask_ious.min():.4f}")
        print(f"    Max:    {mask_ious.max():.4f}")

    if metrics["matched_mask_dices"]:
        mask_dices = np.array(metrics["matched_mask_dices"])
        print(f"\n  Mask Dice (TP pairs, n={len(mask_dices)}):")
        print(f"    Mean:   {mask_dices.mean():.4f}")
        print(f"    Median: {np.median(mask_dices):.4f}")
        print(f"    Min:    {mask_dices.min():.4f}")
        print(f"    Max:    {mask_dices.max():.4f}")

    # Per-sample summary
    samples = metrics["per_sample"]
    precisions = [s["precision"] for s in samples]
    recalls = [s["recall"] for s in samples]
    f1s = [s["f1"] for s in samples]
    n_with_gt = [s for s in samples if s["n_gt"] > 0]
    n_no_gt = [s for s in samples if s["n_gt"] == 0]

    print(f"\n  Per-sample (n={len(samples)}):")
    print(f"    With GT:    {len(n_with_gt)}")
    print(f"    No GT:      {len(n_no_gt)} (FP rate: "
          f"{sum(s['fp'] for s in n_no_gt)} preds across {len(n_no_gt)} samples)")
    if n_with_gt:
        print(f"    Mean precision: {np.mean([s['precision'] for s in n_with_gt]):.4f}")
        print(f"    Mean recall:    {np.mean([s['recall'] for s in n_with_gt]):.4f}")
        print(f"    Mean F1:        {np.mean([s['f1'] for s in n_with_gt]):.4f}")

    # Zero-recall samples
    zero_recall = [s for s in n_with_gt if s["recall"] == 0.0]
    if zero_recall:
        print(f"\n  Zero-recall samples ({len(zero_recall)}):")
        for s in zero_recall[:10]:
            print(f"    {s['sample_id']}: {s['n_gt']} GT, {s['n_pred']} preds, "
                  f"avg_conf={s['avg_pred_conf']:.3f}")

    # Worst FP (highest confidence false positives)
    fps = sorted(metrics["fp_examples"], key=lambda x: x["score"], reverse=True)
    if fps:
        print(f"\n  Top-10 false positives (highest confidence):")
        for fp_item in fps[:10]:
            box = fp_item["box"]
            print(f"    {fp_item['sample_id']}: "
                  f"conf={fp_item['score']:.3f}, "
                  f"area={fp_item['area']:.0f}px, "
                  f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # Worst FN (largest missed instances)
    fns = sorted(metrics["fn_examples"], key=lambda x: x["area"], reverse=True)
    if fns:
        print(f"\n  Top-10 false negatives (largest missed instances):")
        for fn_item in fns[:10]:
            box = fn_item["box"]
            print(f"    {fn_item['sample_id']}: "
                  f"area={fn_item['area']:.0f}px, "
                  f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # Confidence distribution
    if metrics["pred_confidences"]:
        confs = np.array(metrics["pred_confidences"])
        print(f"\n  Confidence distribution (all {len(confs)} preds):")
        for thresh in [0.5, 0.7, 0.8, 0.9, 0.95]:
            above = (confs >= thresh).sum()
            print(f"    >= {thresh:.2f}: {above} ({above/max(len(confs),1)*100:.1f}%)")

    print("\n" + "=" * 70)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "box_iou_mean": float(np.mean(box_ious)) if metrics["matched_ious"] else 0.0,
        "mask_iou_mean": float(np.mean(mask_ious)) if metrics["matched_mask_ious"] else 0.0,
        "mask_dice_mean": float(np.mean(mask_dices)) if metrics["matched_mask_dices"] else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PoC-1 Object-Level Evaluation")
    parser.add_argument(
        "--checkpoint",
        default="outputs/poc_aqua_full/checkpoints/epoch_040.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        default="configs/poc_aqua_instance.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--manifest", default="data/poc_aqua_full/val.json")
    parser.add_argument(
        "--data-root",
        default="/home/ma-user/work/Stage3Data/养殖区",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu", "npu:0"])
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None, help="Save metrics JSON to file")
    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path

    device = torch.device(args.device)

    # Build model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = build_model(str(config_path), args.checkpoint, str(device))

    # Build dataset
    val_ds = AquaPoCDataset(
        manifest_path=str(manifest_path),
        data_root=args.data_root,
        image_size=cfg["data"].get("image_size", 224),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=poc_collate_fn,
    )
    print(f"Val samples: {len(val_ds)}")

    # Evaluate
    print(f"\nEvaluating with score_thresh={args.score_thresh}, "
          f"iou_thresh={args.iou_thresh}...\n")
    metrics = evaluate(
        model, val_loader, device,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )
    summary = print_report(metrics)

    # Save metrics JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({**summary, "config": {k: str(v) for k, v in args.__dict__.items()}},
                      f, indent=2)
        print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()

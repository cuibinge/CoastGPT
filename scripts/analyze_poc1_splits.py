#!/usr/bin/env python3
"""
PoC-1 Split Analysis: GF6 vs non-GF6 metrics + zero-recall tile forensics.

Loads epoch 40 checkpoint, runs inference on val set, and outputs
per-subset Precision/Recall/F1 with zero-recall breakdown.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn

# Import from training script
import importlib.util as _u
_s = _u.spec_from_file_location("poc", _REPO_ROOT / "scripts" / "poc_stage_one_det.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
build_vision_encoder = _m.build_vision_encoder
load_config = _m.load_config


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5):
    pairs = []
    for gi in range(len(gt_boxes)):
        for pi in range(len(pred_boxes)):
            iou = compute_iou(gt_boxes[gi], pred_boxes[pi])
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt, matched_pred = set(), set()
    matches = []
    for iou, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matches.append((gi, pi, iou))
            matched_gt.add(gi); matched_pred.add(pi)
    unmatched_gt = [g for g in range(len(gt_boxes)) if g not in matched_gt]
    unmatched_pred = [p for p in range(len(pred_boxes)) if p not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def main():
    device = torch.device("npu:0")
    cfg = load_config("configs/poc_aqua_instance.yaml")

    # Build model
    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"]).to(device)
    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)
    adapter = DualVisionFPNBackboneAdapter(vision, fpn).to(device)
    mrcnn_cfg = cfg["model"]["mask_rcnn"]
    model = build_aqua_maskrcnn(
        adapter, num_classes=mrcnn_cfg["num_classes"],
        anchor_sizes=tuple(tuple(s) for s in cfg["model"]["anchors"]["sizes"]),
        aspect_ratios=tuple(tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"]),
        rpn_pre_nms_top_n_train=mrcnn_cfg.get("rpn_pre_nms_top_n_train", 512),
        rpn_post_nms_top_n_train=mrcnn_cfg.get("rpn_post_nms_top_n_train", 128),
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        image_mean=[0.0,0.0,0.0], image_std=[1.0,1.0,1.0],
        min_size=224, max_size=224,
    ).to(device)
    ckpt = torch.load("outputs/poc_aqua_full/checkpoints/epoch_040.pt", map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    model.eval()

    # Load manifest for sensor/size lookup
    with open("data/poc_aqua_full/val.json") as f:
        manifest = json.load(f)
    sample_info = {}
    for s in manifest:
        sid = s["sample_id"]
        sz = s.get("original_size", [0, 0])
        sample_info[sid] = {
            "sensor": s.get("sensor", "?"),
            "size": sz[0] if isinstance(sz, list) else sz,
        }

    ds = AquaPoCDataset(
        manifest_path="data/poc_aqua_full/val.json",
        data_root="/home/ma-user/work/Stage3Data/养殖区",
        image_size=224,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=poc_collate_fn,
    )

    # Per-subset accumulators
    subsets = defaultdict(lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0,
                                     "n_samples": 0, "zero_recall": 0,
                                     "zero_recall_tiles": []})
    zero_recall_details = []

    score_thresh = 0.5
    for images, targets, metas in loader:
        images_dev = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images_dev)
        outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]

        for out, target, meta in zip(outputs_cpu, targets, metas):
            sid = meta["sample_id"]
            info = sample_info.get(sid, {"sensor": "?", "size": 0})
            key = f"{info['sensor']}/{info['size']}px"

            gt_boxes = target["boxes"].numpy()
            keep = out["scores"] >= score_thresh
            pred_boxes = out["boxes"][keep].numpy()
            pred_scores = out["scores"][keep].numpy()

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)
            matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5)
            tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gt)

            s = subsets[key]
            s["gt"] += n_gt; s["pred"] += n_pred
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn
            s["n_samples"] += 1
            if n_gt > 0 and tp == 0:
                s["zero_recall"] += 1
                s["zero_recall_tiles"].append({
                    "sample_id": sid, "n_gt": n_gt, "n_pred": n_pred,
                    "max_conf": round(float(pred_scores.max()), 4) if len(pred_scores) > 0 else 0.0,
                })
                zero_recall_details.append({
                    "sample_id": sid, "sensor": info["sensor"], "size": info["size"],
                    "n_gt": n_gt, "n_pred": n_pred,
                    "max_conf": round(float(pred_scores.max()), 4) if len(pred_scores) > 0 else 0.0,
                    "gt_areas": [float(target["masks"][i].sum()) for i in range(n_gt)],
                })

    # Print report
    total_gt = sum(s["gt"] for s in subsets.values())
    total_fn = sum(s["fn"] for s in subsets.values())

    print("=" * 70)
    print("PoC-1 Split Analysis: GF6 vs non-GF6")
    print("=" * 70)
    print(f"\n{'Subset':<20} {'Samples':>7} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>7} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'ZeroR':>5}")
    print("-" * 90)

    for key in sorted(subsets.keys()):
        s = subsets[key]
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        fn_pct = s["fn"] / max(total_fn, 1) * 100
        print(f"{key:<20} {s['n_samples']:>7} {s['gt']:>6} {s['pred']:>6} {s['tp']:>6} "
              f"{s['fp']:>6} {s['fn']:>7} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {s['zero_recall']:>5} "
              f"({fn_pct:.0f}% FN)")

    # GF6 vs non-GF6 aggregate
    gf6_keys = [k for k in subsets if "GF6" in k]
    non_gf6_keys = [k for k in subsets if "GF6" not in k]

    def aggregate(keys, label):
        gt = sum(subsets[k]["gt"] for k in keys)
        pred = sum(subsets[k]["pred"] for k in keys)
        tp = sum(subsets[k]["tp"] for k in keys)
        fp = sum(subsets[k]["fp"] for k in keys)
        fn = sum(subsets[k]["fn"] for k in keys)
        zr = sum(subsets[k]["zero_recall"] for k in keys)
        ns = sum(subsets[k]["n_samples"] for k in keys)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"\n{'─'*70}")
        print(f"{label:<20} {ns:>7} {gt:>6} {pred:>6} {tp:>6} {fp:>6} {fn:>7} "
              f"{prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {zr:>5}")
        return {"prec": prec, "rec": rec, "f1": f1, "fn": fn, "zr": zr}

    agg_gf6 = aggregate(gf6_keys, "GF6 (all sizes)")
    agg_non = aggregate(non_gf6_keys, "Non-GF6 (all sizes)")

    fn_ratio = agg_gf6["fn"] / max(total_fn, 1)
    print(f"\n  GF6 FN / total FN = {agg_gf6['fn']}/{total_fn} = {fn_ratio:.1%}")

    # Zero-recall tile details
    print(f"\n{'='*70}")
    print(f"Zero-Recall Tile Forensics ({len(zero_recall_details)} tiles)")
    print(f"{'='*70}")
    print(f"\n{'Sample ID':<75} {'Sensor':>6} {'Size':>5} {'GT':>4} {'Pred':>5} {'MaxConf':>8} {'GT Areas'}")
    print("-" * 130)

    for z in sorted(zero_recall_details, key=lambda x: x["n_gt"], reverse=True):
        areas_str = ", ".join(f"{a:.0f}" for a in sorted(z["gt_areas"], reverse=True)[:5])
        if len(z["gt_areas"]) > 5:
            areas_str += f", ... ({len(z['gt_areas'])} total)"
        print(f"{z['sample_id']:<75} {z['sensor']:>6} {z['size']:>5} {z['n_gt']:>4} "
              f"{z['n_pred']:>5} {z['max_conf']:>8.3f}  [{areas_str}]")

    # Per-sensor zero-recall summary
    print(f"\n  Zero-recall by sensor:")
    zr_by_sensor = defaultdict(int)
    for z in zero_recall_details:
        zr_by_sensor[z["sensor"]] += 1
    for sensor, count in sorted(zr_by_sensor.items()):
        print(f"    {sensor}: {count} tiles")


if __name__ == "__main__":
    main()

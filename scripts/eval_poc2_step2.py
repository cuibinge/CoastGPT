#!/usr/bin/env python3
"""
Evaluate PoC-2 Step 2: Compare baseline (epoch 40) vs oversampled (epoch 60).

Full evaluation with:
  - GF6 vs non-GF6 split: Precision, Recall, F1, TP/FP/FN
  - Full RPN proposal recall (all validation tiles, not subset)
  - Zero-recall tiles count
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn
from utils.mask_utils import binary_mask_to_instances

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
    for iou, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi); matched_pred.add(pi)
    tp = len(matched_gt)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(matched_pred)
    return tp, fp, fn


def build_model(device, ckpt_path):
    cfg = load_config("configs/poc_aqua_instance.yaml")
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
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
        min_size=224, max_size=224,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    model.eval()
    return model


def evaluate(model, device, manifest, data_root):
    """Full evaluation: final detection + RPN proposal recall on all tiles."""
    stats = {
        "gf6": {
            "tp": 0, "fp": 0, "fn": 0,
            "rpn_gt": 0, "rpn_covered": 0,
            "tile_count": 0, "zero_recall_tiles": 0,
        },
        "non_gf6": {
            "tp": 0, "fp": 0, "fn": 0,
            "rpn_gt": 0, "rpn_covered": 0,
            "tile_count": 0, "zero_recall_tiles": 0,
        },
        "all": {"tp": 0, "fp": 0, "fn": 0},
    }

    for sample in manifest:
        sid = sample["sample_id"]
        is_gf6 = "GF6" in sample.get("sensor", "")
        key = "gf6" if is_gf6 else "non_gf6"
        img_path = data_root / sample["image_path"]
        binary_path = (
            data_root / sample["binary_label_path"]
            if sample.get("binary_label_path") else None
        )
        if not binary_path or not binary_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        image_224 = image.resize((224, 224), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(image_224, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).to(device)

        binary = Image.open(binary_path).convert("L")
        binary_224 = binary.resize((224, 224), Image.NEAREST)
        binary_224_np = np.array(binary_224)
        _, gt_boxes, _ = binary_mask_to_instances(binary_224_np, min_area=8, connectivity=4)
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        if len(gt_boxes) == 0:
            continue

        with torch.no_grad():
            # Final detections
            out = model([img_t])[0]
            keep = out["scores"] >= 0.5
            det_boxes = out["boxes"][keep].cpu().numpy()

            # RPN proposals
            images_list, _ = model.transform([img_t], None)
            features = model.backbone(images_list.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, _ = model.rpn(images_list, features, None)
            props = proposals[0].cpu().numpy()

        # Final detection metrics
        tp, fp, fn = greedy_match(gt_boxes, det_boxes)
        stats[key]["tp"] += tp
        stats[key]["fp"] += fp
        stats[key]["fn"] += fn
        stats["all"]["tp"] += tp
        stats["all"]["fp"] += fp
        stats["all"]["fn"] += fn

        # RPN proposal recall
        covered = 0
        for gt in gt_boxes:
            for prop in props:
                if compute_iou(gt, prop) >= 0.5:
                    covered += 1
                    break
        stats[key]["rpn_gt"] += len(gt_boxes)
        stats[key]["rpn_covered"] += covered
        stats[key]["tile_count"] += 1
        if tp == 0 and len(gt_boxes) > 0:
            stats[key]["zero_recall_tiles"] += 1

    return stats


def print_metrics(label, s):
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    has_rpn = "rpn_covered" in s
    print(f"\n  {label}:")
    if has_rpn:
        rpn_rec = s["rpn_covered"] / max(s["rpn_gt"], 1)
        print(f"    Tiles: {s['tile_count']}  |  Zero-recall tiles: {s['zero_recall_tiles']}")
        print(f"    RPN proposal recall: {rpn_rec:.4f} ({s['rpn_covered']}/{s['rpn_gt']})")
    print(f"    Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"    TP: {tp}  FP: {fp}  FN: {fn}")
    result = {"precision": prec, "recall": rec, "f1": f1}
    if has_rpn:
        result["rpn_recall"] = rpn_rec
    return result


def main():
    device = torch.device("npu:0")

    print("=" * 70)
    print("PoC-2 Step 2: Full Evaluation — Baseline (Epoch 40) vs Oversampled (Epoch 60)")
    print("=" * 70)

    with open("data/poc_aqua_full/val.json") as f:
        manifest = json.load(f)
    data_root = Path("/home/ma-user/work/Stage3Data/养殖区")

    gf6_samples = [s for s in manifest if s.get("sensor") == "GF6"]
    non_gf6_samples = [s for s in manifest if s.get("sensor") != "GF6"]
    print(f"\nVal set: {len(gf6_samples)} GF6 + {len(non_gf6_samples)} non-GF6 = {len(manifest)} total")

    results = {}

    for name, ckpt in [
        ("Baseline Epoch 40", "outputs/poc_aqua_full/checkpoints/epoch_040.pt"),
        ("Oversampled Epoch 60", "outputs/poc2_gf6_oversample/checkpoints/epoch_060.pt"),
    ]:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt}")
        model = build_model(device, ckpt)
        stats = evaluate(model, device, manifest, data_root)
        results[name] = stats

        for subset_key, label in [("all", "OVERALL"), ("gf6", "GF6"), ("non_gf6", "non-GF6")]:
            print_metrics(label, stats[subset_key])

    # Comparison
    print(f"\n{'='*70}")
    print("DELTA: Oversampled Epoch 60 — Baseline Epoch 40")
    print(f"{'='*70}")

    for subset_key, label in [("all", "OVERALL"), ("gf6", "GF6"), ("non_gf6", "non-GF6")]:
        b = results["Baseline Epoch 40"][subset_key]
        o = results["Oversampled Epoch 60"][subset_key]
        delta_f1 = (o["tp"] / max(o["tp"] + o["fn"], 1)) - (b["tp"] / max(b["tp"] + b["fn"], 1))
        has_rpn = "rpn_covered" in b
        print(f"\n  {label}:")
        print(f"    Delta final recall: {delta_f1:+.4f}")
        if has_rpn:
            delta_rpn = (o["rpn_covered"] / max(o["rpn_gt"], 1)) - (b["rpn_covered"] / max(b["rpn_gt"], 1))
            delta_zr = o["zero_recall_tiles"] - b["zero_recall_tiles"]
            print(f"    Delta RPN recall: {delta_rpn:+.4f}")
            print(f"    Delta zero-recall tiles: {delta_zr:+d}")

    # Save
    out = {}
    for name, stats in results.items():
        out[name] = {}
        for key in ["all", "gf6", "non_gf6"]:
            s = stats[key]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            entry = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            }
            if "rpn_covered" in s:
                entry["rpn_recall"] = s["rpn_covered"] / max(s["rpn_gt"], 1)
                entry["rpn_covered"] = s["rpn_covered"]
                entry["rpn_gt"] = s["rpn_gt"]
                entry["tile_count"] = s["tile_count"]
                entry["zero_recall_tiles"] = s["zero_recall_tiles"]
            out[name][key] = entry

    out_path = Path("outputs/poc2_gf6_oversample/step2_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

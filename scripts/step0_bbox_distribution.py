#!/usr/bin/env python3
"""
Step 0: GT bbox size distribution analysis in 224px model coordinate space.

Breaks down bbox width/height/area by subset:
  - GF6 all / GF6 FN / GF6 RPN-uncovered / GF6 zero-recall tiles
  - non-GF6 all / non-GF6 FN

Key question: are RPN-uncovered GT boxes systematically larger than covered ones?
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
    return matched_gt, matched_pred


def box_stats(arr_4col):
    """Compute percentile stats for pre-computed [w, h, sqrt_area, aspect] columns."""
    if len(arr_4col) == 0:
        return None
    w = arr_4col[:, 0]
    h = arr_4col[:, 1]
    sqrt_area = arr_4col[:, 2]
    aspect = arr_4col[:, 3]
    result = {}
    for name, arr in [("width", w), ("height", h), ("sqrt_area", sqrt_area), ("aspect_ratio", aspect)]:
        result[name] = {
            "n": len(arr),
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return result


def main():
    device = torch.device("npu:0")
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
    ckpt = torch.load("outputs/poc_aqua_full/checkpoints/epoch_040.pt", map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    model.eval()

    with open("data/poc_aqua_full/val.json") as f:
        manifest = json.load(f)

    data_root = Path("/home/ma-user/work/Stage3Data/养殖区")

    # Collectors: each entry is (w, h, sqrt_area, aspect)
    subsets = {
        "gf6_all": [],
        "gf6_fn": [],       # GF6 GT boxes not matched to any final detection
        "gf6_rpn_miss": [], # GF6 GT boxes with no RPN proposal coverage
        "gf6_zr_tiles": [], # GF6 GT boxes from zero-recall tiles
        "gf6_rpn_cov": [],  # GF6 GT boxes WITH RPN proposal coverage (for comparison)
        "non_gf6_all": [],
        "non_gf6_fn": [],
    }

    for sample in manifest:
        sid = sample["sample_id"]
        is_gf6 = "GF6" in sample.get("sensor", "")
        img_path = data_root / sample["image_path"]
        binary_path = (
            data_root / sample["binary_label_path"]
            if sample.get("binary_label_path") else None
        )
        if not binary_path or not binary_path.exists():
            continue

        # Load image and resize to 224
        image = Image.open(img_path).convert("RGB")
        image_224 = image.resize((224, 224), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(image_224, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).to(device)

        # GT boxes in 224px space
        binary = Image.open(binary_path).convert("L")
        binary_224 = binary.resize((224, 224), Image.NEAREST)
        binary_224_np = np.array(binary_224)
        _, gt_boxes, _ = binary_mask_to_instances(binary_224_np, min_area=8, connectivity=4)
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        if len(gt_boxes) == 0:
            continue

        # Get RPN proposals
        from collections import OrderedDict
        with torch.no_grad():
            images_list, _ = model.transform([img_t], None)
            features = model.backbone(images_list.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, _ = model.rpn(images_list, features, None)
        props = proposals[0].cpu().numpy()

        # Get final detections
        with torch.no_grad():
            out = model([img_t])[0]
        keep = out["scores"] >= 0.5
        det_boxes = out["boxes"][keep].cpu().numpy()

        # Per-box analysis
        rpn_covered = set()
        for gi, gt in enumerate(gt_boxes):
            for prop in props:
                if compute_iou(gt, prop) >= 0.5:
                    rpn_covered.add(gi)
                    break

        detected = greedy_match(gt_boxes, det_boxes, iou_thresh=0.5)[0]

        # Assign each GT box to subsets
        box_data = []
        for gi, box in enumerate(gt_boxes):
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            area = w * h
            row = (w, h, np.sqrt(area), w / max(h, 1e-8))
            box_data.append(row)

        is_zr_tile = len(gt_boxes) > 0 and len(detected) == 0

        for gi, row in enumerate(box_data):
            if is_gf6:
                subsets["gf6_all"].append(row)
                if gi not in detected:
                    subsets["gf6_fn"].append(row)
                if gi not in rpn_covered:
                    subsets["gf6_rpn_miss"].append(row)
                if gi in rpn_covered:
                    subsets["gf6_rpn_cov"].append(row)
                if is_zr_tile:
                    subsets["gf6_zr_tiles"].append(row)
            else:
                subsets["non_gf6_all"].append(row)
                if gi not in detected:
                    subsets["non_gf6_fn"].append(row)

    # Print report
    print("=" * 75)
    print("Step 0: GT Bbox Size Distribution (224px model coordinate space)")
    print("=" * 75)

    current_anchors = [(16, 32), (32, 64), (64, 96, 128)]
    print(f"\nCurrent anchor sizes: {current_anchors}")
    print(f"Max anchor (largest side): {max(max(a) for a in current_anchors)}")

    for subset_name, rows in subsets.items():
        if not rows:
            print(f"\n  {subset_name}: NO DATA")
            continue
        arr = np.array(rows, dtype=np.float32)  # [N, 4]: w, h, sqrt_area, aspect
        stats = box_stats(arr)
        if stats is None:
            continue
        print(f"\n{'─'*75}")
        print(f"  {subset_name}  (n={stats['width']['n']})")
        print(f"  {'':>12} {'min':>7} {'p10':>7} {'p25':>7} {'p50':>7} "
              f"{'p75':>7} {'p90':>7} {'p95':>7} {'max':>7} {'mean':>7}")
        print(f"  {'─'*80}")
        for metric in ["width", "height", "sqrt_area", "aspect_ratio"]:
            s = stats[metric]
            print(f"  {metric:>12} {s['min']:>7.1f} {s['p10']:>7.1f} {s['p25']:>7.1f} "
                  f"{s['p50']:>7.1f} {s['p75']:>7.1f} {s['p90']:>7.1f} "
                  f"{s['p95']:>7.1f} {s['max']:>7.1f} {s['mean']:>7.1f}")

    # Key comparison: RPN-covered vs RPN-missed for GF6
    if subsets["gf6_rpn_miss"] and subsets["gf6_rpn_cov"]:
        miss = np.array(subsets["gf6_rpn_miss"], dtype=np.float32)  # [N,4]: w, h, sqrt_area, aspect
        cov = np.array(subsets["gf6_rpn_cov"], dtype=np.float32)
        miss_stats = box_stats(miss)
        cov_stats = box_stats(cov)

        print(f"\n{'='*75}")
        print("GF6 RPN-COVERED vs RPN-MISSED Comparison")
        print(f"{'='*75}")
        print(f"\n{'Metric':>12} {'RPN-Covered':>12} {'RPN-Missed':>12} {'Ratio':>8}")
        print(f"{'':>12} {'(n='+str(cov_stats['width']['n'])+')':>12} "
              f"{'(n='+str(miss_stats['width']['n'])+')':>12}")
        print(f"  {'─'*50}")
        for metric in ["width", "height", "sqrt_area"]:
            cov_val = cov_stats[metric]["p50"]
            miss_val = miss_stats[metric]["p50"]
            ratio = miss_val / max(cov_val, 1e-8)
            print(f"  {metric:>12} (p50) {cov_val:>12.1f} {miss_val:>12.1f} {ratio:>8.2f}x")

        # What % of RPN-missed boxes exceed max anchor?
        miss_w = miss[:, 0]
        miss_h = miss[:, 1]
        max_anchor = 128.0
        exceed_w = (miss_w > max_anchor).sum()
        exceed_h = (miss_h > max_anchor).sum()
        exceed_either = ((miss_w > max_anchor) | (miss_h > max_anchor)).sum()
        print(f"\n  RPN-missed boxes with width > {max_anchor}: {exceed_w}/{len(miss_w)} ({exceed_w/len(miss_w)*100:.1f}%)")
        print(f"  RPN-missed boxes with height > {max_anchor}: {exceed_h}/{len(miss_h)} ({exceed_h/len(miss_h)*100:.1f}%)")
        print(f"  RPN-missed boxes with width OR height > {max_anchor}: {exceed_either}/{len(miss_w)} ({exceed_either/len(miss_w)*100:.1f}%)")

    # Save
    out = {}
    for name, rows in subsets.items():
        if rows:
            arr = np.array(rows, dtype=np.float32)
            out[name] = box_stats(arr)

    out_path = Path("outputs/poc_aqua_full/step0_bbox_distribution.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

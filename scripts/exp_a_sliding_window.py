#!/usr/bin/env python3
"""
Experiment A: GF6 512px sliding-window vs baseline 224px inference.

Tests whether native-resolution sliding windows recover recall on GF6 tiles
where 512→224 resize is suspected to cause the recall collapse.

Two methods compared on the same GF6 512px tiles:
  1. Baseline: 512→224 resize → inference → scale boxes back to 512
  2. Sliding window: 224x224 windows (stride 112) → inference → merge via NMS

GT is computed from Binary_WFQ.tif at native resolution (no resize).
"""
import json
import sys
from collections import defaultdict
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
    tp = 0
    for iou, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            tp += 1
            matched_gt.add(gi); matched_pred.add(pi)
    fn = len(gt_boxes) - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return tp, fp, fn


def generate_windows(image_np, window_size=224, stride=112):
    """Generate overlapping sliding windows covering the full image.

    Returns list of (tensor[C,H,W], (y_offset, x_offset)).
    Always includes the bottom-right corner window for full coverage.
    """
    h, w = image_np.shape[:2]
    if h <= window_size and w <= window_size:
        tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        return [(tensor, (0, 0))]

    positions = set()
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            positions.add((y, x))
    # Ensure bottom-right corner is covered
    if h > window_size:
        for x in range(0, w - window_size + 1, stride):
            positions.add((h - window_size, x))
    if w > window_size:
        for y in range(0, h - window_size + 1, stride):
            positions.add((y, w - window_size))
    if h > window_size and w > window_size:
        positions.add((h - window_size, w - window_size))

    windows = []
    for y, x in sorted(positions):
        crop = image_np[y:y+window_size, x:x+window_size]
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        windows.append((tensor, (y, x)))
    return windows


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
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=200,
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
    gf6_samples = [s for s in manifest if s.get("sensor") == "GF6"]
    print(f"GF6 val tiles: {len(gf6_samples)}")

    score_thresh = 0.5
    results_base = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
    results_sw = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
    per_tile = []

    for si, sample in enumerate(gf6_samples):
        sid = sample["sample_id"]
        img_path = data_root / sample["image_path"]
        binary_path = (
            data_root / sample["binary_label_path"]
            if sample.get("binary_label_path")
            else None
        )

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image, dtype=np.float32)
        orig_w, orig_h = image.size

        # GT at native resolution from binary mask
        if binary_path and binary_path.exists():
            binary = Image.open(binary_path).convert("L")
            binary_np = np.array(binary)
            instance_masks, gt_boxes, _ = binary_mask_to_instances(
                binary_np, min_area=8, connectivity=4
            )
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
        else:
            continue

        n_gt = len(gt_boxes)
        if n_gt == 0:
            continue

        # ---- Baseline: 512→224 resize ----
        image_224 = image.resize((224, 224), Image.BILINEAR)
        img_t_224 = torch.from_numpy(np.array(image_224, dtype=np.float32) / 255.0)
        img_t_224 = img_t_224.permute(2, 0, 1).to(device)

        with torch.no_grad():
            out_224 = model([img_t_224])[0]

        keep = out_224["scores"] >= score_thresh
        boxes_224 = out_224["boxes"][keep].cpu().numpy()
        scale_x = orig_w / 224.0
        scale_y = orig_h / 224.0
        if len(boxes_224) > 0:
            boxes_224[:, [0, 2]] *= scale_x
            boxes_224[:, [1, 3]] *= scale_y

        tp_b, fp_b, fn_b = greedy_match(gt_boxes, boxes_224)
        results_base["tp"] += tp_b; results_base["fp"] += fp_b; results_base["fn"] += fn_b
        results_base["gt"] += n_gt; results_base["pred"] += len(boxes_224)

        # ---- Sliding window ----
        windows = generate_windows(image_np, window_size=224, stride=112)

        all_boxes = []
        all_scores = []
        for batch_start in range(0, len(windows), 8):
            batch_win = windows[batch_start:batch_start + 8]
            batch_tensors = [w[0].to(device) for w in batch_win]

            with torch.no_grad():
                outputs = model(batch_tensors)

            for out, (_, (y_off, x_off)) in zip(outputs, batch_win):
                k = out["scores"] >= score_thresh
                boxes = out["boxes"][k].cpu().numpy()
                scores = out["scores"][k].cpu().numpy()
                if len(boxes) > 0:
                    boxes[:, [0, 2]] += x_off
                    boxes[:, [1, 3]] += y_off
                    all_boxes.append(boxes)
                    all_scores.append(scores)

        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
            keep_idx = torchvision.ops.nms(
                torch.from_numpy(all_boxes),
                torch.from_numpy(all_scores),
                iou_threshold=0.5,
            ).numpy()
            all_boxes = all_boxes[keep_idx]
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)

        tp_sw, fp_sw, fn_sw = greedy_match(gt_boxes, all_boxes)
        results_sw["tp"] += tp_sw; results_sw["fp"] += fp_sw; results_sw["fn"] += fn_sw
        results_sw["gt"] += n_gt; results_sw["pred"] += len(all_boxes)

        per_tile.append({
            "sample_id": sid, "n_gt": n_gt,
            "b_tp": tp_b, "b_fn": fn_b, "b_pred": len(boxes_224),
            "sw_tp": tp_sw, "sw_fn": fn_sw, "sw_pred": len(all_boxes),
        })

        if (si + 1) % 20 == 0:
            print(f"  [{si+1}/{len(gf6_samples)}] "
                  f"base rec={results_base['tp']/max(results_base['gt'],1):.3f}  "
                  f"sw rec={results_sw['tp']/max(results_sw['gt'],1):.3f}")

    # ---- Report ----
    def metrics(r):
        prec = r["tp"] / max(r["tp"] + r["fp"], 1)
        rec = r["tp"] / max(r["tp"] + r["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        return prec, rec, f1

    b_prec, b_rec, b_f1 = metrics(results_base)
    sw_prec, sw_rec, sw_f1 = metrics(results_sw)

    print("\n" + "=" * 75)
    print("Experiment A: GF6 Tiles — Baseline 224 vs Sliding Window")
    print("=" * 75)
    print(f"\n{'Method':<30} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 85)
    print(f"{'Baseline (512->224->512)':<30} {results_base['gt']:>6} {results_base['pred']:>6} "
          f"{results_base['tp']:>6} {results_base['fp']:>6} {results_base['fn']:>6} "
          f"{b_prec:>7.4f} {b_rec:>7.4f} {b_f1:>7.4f}")
    print(f"{'Sliding Window (3x3x224)':<30} {results_sw['gt']:>6} {results_sw['pred']:>6} "
          f"{results_sw['tp']:>6} {results_sw['fp']:>6} {results_sw['fn']:>6} "
          f"{sw_prec:>7.4f} {sw_rec:>7.4f} {sw_f1:>7.4f}")

    improved = sum(1 for t in per_tile if t["sw_fn"] < t["b_fn"])
    same = sum(1 for t in per_tile if t["sw_fn"] == t["b_fn"])
    worse = sum(1 for t in per_tile if t["sw_fn"] > t["b_fn"])
    zr_base = sum(1 for t in per_tile if t["n_gt"] > 0 and t["b_tp"] == 0)
    zr_sw = sum(1 for t in per_tile if t["n_gt"] > 0 and t["sw_tp"] == 0)

    print(f"\n  FN reduced: {improved}  |  FN same: {same}  |  FN worse: {worse}")
    print(f"  Zero-recall tiles: baseline={zr_base}  →  sliding window={zr_sw}")

    # Top improvements
    per_tile.sort(key=lambda t: t["b_fn"] - t["sw_fn"], reverse=True)
    print(f"\n{'='*75}")
    print(f"Top 15 Most Improved Tiles")
    print(f"{'='*75}")
    print(f"{'Sample ID':<75} {'GT':>4} {'B_FN':>5} {'SW_FN':>5} {'ΔFN':>5} "
          f"{'B_Pred':>6} {'SW_Pred':>6}")
    print("-" * 105)
    for t in per_tile[:15]:
        delta = t["b_fn"] - t["sw_fn"]
        print(f"{t['sample_id']:<75} {t['n_gt']:>4} {t['b_fn']:>5} {t['sw_fn']:>5} "
              f"{delta:>5} {t['b_pred']:>6} {t['sw_pred']:>6}")

    # Save detailed results
    out_path = Path("outputs/poc_aqua_full/exp_a_sliding_window.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "baseline": results_base,
            "sliding_window": results_sw,
            "per_tile": per_tile,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment D: RPN proposal recall diagnostic + NMS/threshold sweep.

Two-part analysis on GF6 512px tiles:
  1. RPN proposal recall: are GT boxes covered by RPN proposals?
     → Low proposal recall = problem in backbone/RPN/anchor scale
     → High proposal recall, low final recall = problem in ROI/NMS/score filtering

  2. Post-processing sweep: score_thresh × detections_per_img × nms_thresh
     → Find the pareto-optimal settings for GF6 recall
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


def proposal_recall(gt_boxes, proposals, iou_thresh=0.5):
    """Fraction of GT boxes covered by at least one proposal with IoU >= thresh."""
    if len(gt_boxes) == 0:
        return 1.0, 0
    covered = 0
    for gt in gt_boxes:
        for prop in proposals:
            if compute_iou(gt, prop) >= iou_thresh:
                covered += 1
                break
    return covered / len(gt_boxes), covered


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
        rpn_pre_nms_top_n_train=mrcnn_cfg.get("rpn_pre_nms_top_n_train", 512),
        rpn_post_nms_top_n_train=mrcnn_cfg.get("rpn_post_nms_top_n_train", 128),
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

    # Process GF6 tiles + sample of non-GF6
    gf6_samples = [s for s in manifest if s.get("sensor") == "GF6"]
    non_gf6_samples = [s for s in manifest if s.get("sensor") != "GF6"]
    print(f"GF6: {len(gf6_samples)}, non-GF6: {len(non_gf6_samples)}")

    # =========================================================================
    # Part 1: RPN Proposal Recall
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: RPN Proposal Recall Diagnostic")
    print("=" * 70)

    rpn_stats = {"gf6": {"total_gt": 0, "covered": 0, "n_proposals": []},
                 "non_gf6": {"total_gt": 0, "covered": 0, "n_proposals": []}}
    rpn_per_tile = []

    # Process a subset for RPN analysis (all GF6, sample non-GF6)
    samples_for_rpn = gf6_samples + non_gf6_samples[::4]  # every 4th non-GF6

    for si, sample in enumerate(samples_for_rpn):
        sid = sample["sample_id"]
        img_path = data_root / sample["image_path"]
        binary_path = (
            data_root / sample["binary_label_path"]
            if sample.get("binary_label_path") else None
        )
        if not binary_path or not binary_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # W, H
        image_224 = image.resize((224, 224), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(image_224, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).to(device)

        # GT boxes at 224px scale (resize binary mask)
        binary = Image.open(binary_path).convert("L")
        binary_224 = binary.resize((224, 224), Image.NEAREST)
        binary_224_np = np.array(binary_224)
        _, gt_boxes_224, _ = binary_mask_to_instances(
            binary_224_np, min_area=8, connectivity=4
        )
        gt_boxes_224 = np.array(gt_boxes_224, dtype=np.float32)
        if len(gt_boxes_224) == 0:
            continue

        # Run backbone + RPN only (not full model forward)
        model.eval()
        with torch.no_grad():
            # Use model.transform to normalize/resize
            images_list, _ = model.transform([img_t], None)
            features = model.backbone(images_list.tensors)

            # RPN forward
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, proposal_losses = model.rpn(images_list, features, None)

        # proposals is List[Tensor[N,4]] — post-NMS proposals for each image
        props = proposals[0].cpu().numpy()  # post-NMS proposals
        prop_rec, prop_cov = proposal_recall(gt_boxes_224, props, iou_thresh=0.5)

        is_gf6 = "GF6" in sample.get("sensor", "")
        key = "gf6" if is_gf6 else "non_gf6"
        rpn_stats[key]["total_gt"] += len(gt_boxes_224)
        rpn_stats[key]["covered"] += prop_cov
        rpn_stats[key]["n_proposals"].append(len(props))
        rpn_per_tile.append({
            "sample_id": sid, "sensor": sample.get("sensor"),
            "n_gt": len(gt_boxes_224), "n_proposals": len(props),
            "prop_recall": round(prop_rec, 4), "prop_covered": prop_cov,
        })

        if (si + 1) % 30 == 0:
            gf6_r = rpn_stats["gf6"]["covered"] / max(rpn_stats["gf6"]["total_gt"], 1)
            ngf6_r = rpn_stats["non_gf6"]["covered"] / max(rpn_stats["non_gf6"]["total_gt"], 1)
            print(f"  [{si+1}/{len(samples_for_rpn)}] GF6 RPN rec={gf6_r:.3f}  "
                  f"non-GF6 RPN rec={ngf6_r:.3f}")

    # Print RPN results
    from collections import OrderedDict

    for key, label in [("gf6", "GF6"), ("non_gf6", "non-GF6")]:
        s = rpn_stats[key]
        rec = s["covered"] / max(s["total_gt"], 1)
        avg_props = np.mean(s["n_proposals"]) if s["n_proposals"] else 0
        print(f"\n  {label}: RPN proposal recall = {rec:.4f} "
              f"({s['covered']}/{s['total_gt']} GT boxes covered)")
        print(f"         Avg proposals per image (post-NMS): {avg_props:.1f}")

    # Zero proposal-recall tiles
    zpr = [t for t in rpn_per_tile if t["prop_recall"] == 0.0 and t["n_gt"] > 0]
    print(f"\n  Tiles with 0% RPN proposal recall: {len(zpr)}")
    for t in sorted(zpr, key=lambda x: x["n_gt"], reverse=True)[:10]:
        print(f"    {t['sample_id'][:70]:<70} GT={t['n_gt']:>3}  Props={t['n_proposals']:>4}")

    # Low proposal recall tiles (0 < rec < 0.5)
    lpr = [t for t in rpn_per_tile if 0 < t["prop_recall"] < 0.5 and t["n_gt"] > 3]
    print(f"\n  GF6 tiles with RPN rec in (0, 0.5) with GT>3: {len(lpr)}")
    for t in sorted(lpr, key=lambda x: x["prop_recall"])[:10]:
        print(f"    {t['sample_id'][:70]:<70} GT={t['n_gt']:>3}  "
              f"Props={t['n_proposals']:>4}  RPNrec={t['prop_recall']:.3f}")

    # =========================================================================
    # Part 2: Post-processing Sweep (GF6 only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: Post-processing Sweep (GF6 only)")
    print("=" * 70)

    score_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    det_caps = [50, 100, 200, 300, 500]
    nms_thresholds = [0.3, 0.5, 0.7]

    # Rebuild model with high detections_per_img and low score_thresh for raw outputs
    model_sweep = build_aqua_maskrcnn(
        adapter, num_classes=mrcnn_cfg["num_classes"],
        anchor_sizes=tuple(tuple(s) for s in cfg["model"]["anchors"]["sizes"]),
        aspect_ratios=tuple(tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"]),
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=0.0,  # keep all boxes
        box_nms_thresh=0.5,
        box_detections_per_img=500,  # keep many
        image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
        min_size=224, max_size=224,
    ).to(device)
    model_sweep.load_state_dict(ckpt["maskrcnn_state_dict"])
    model_sweep.eval()

    # Collect raw outputs for GF6 tiles
    gf6_outputs = []
    gf6_gts = []
    for si, sample in enumerate(gf6_samples):
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
            out = model_sweep([img_t])[0]

        gf6_outputs.append({
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
            "labels": out["labels"].cpu().numpy(),
        })
        gf6_gts.append(gt_boxes)

        if (si + 1) % 50 == 0:
            print(f"  Collected raw outputs: {si+1}/{len(gf6_samples)}")

    print(f"\n  Collected {len(gf6_outputs)} GF6 tiles for sweep")

    # Sweep over post-processing params
    best_f1 = 0
    best_params = None

    print(f"\n{'score_thr':>10} {'det_cap':>7} {'nms_thr':>7} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)

    for score_thr in score_thresholds:
        for det_cap in det_caps:
            for nms_thr in nms_thresholds:
                total_tp, total_fp, total_fn = 0, 0, 0
                total_gt = 0

                for raw_out, gt_boxes in zip(gf6_outputs, gf6_gts):
                    # Apply score threshold
                    keep = raw_out["scores"] >= score_thr
                    boxes = raw_out["boxes"][keep]
                    scores = raw_out["scores"][keep]

                    # Apply class-specific NMS (only class 1 = aquaculture)
                    if len(boxes) > 0:
                        # Apply per-class NMS
                        labels = raw_out["labels"][keep]
                        keep_idx = torchvision.ops.batched_nms(
                            torch.from_numpy(boxes),
                            torch.from_numpy(scores),
                            torch.from_numpy(labels),
                            nms_thr,
                        ).numpy()
                        boxes = boxes[keep_idx]
                        scores = scores[keep_idx]

                    # Cap detections
                    if len(boxes) > det_cap:
                        top_idx = np.argsort(scores)[-det_cap:]
                        boxes = boxes[top_idx]

                    tp, fp, fn = greedy_match(gt_boxes, boxes)
                    total_tp += tp; total_fp += fp; total_fn += fn
                    total_gt += len(gt_boxes)

                prec = total_tp / max(total_tp + total_fp, 1)
                rec = total_tp / max(total_tp + total_fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)

                marker = ""
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (score_thr, det_cap, nms_thr, prec, rec, f1)
                    marker = " *"

                print(f"{score_thr:>10.2f} {det_cap:>7} {nms_thr:>7} "
                      f"{prec:>7.4f} {rec:>7.4f} {f1:>7.4f} "
                      f"{total_tp:>6} {total_fp:>6} {total_fn:>6}{marker}")

    print(f"\n  Best: score_thr={best_params[0]:.2f}  det_cap={best_params[1]}  "
          f"nms_thr={best_params[2]:.1f}  →  "
          f"Prec={best_params[3]:.4f}  Recall={best_params[4]:.4f}  F1={best_params[5]:.4f}")

    # Compare with current default (score_thr=0.5, det_cap=100, nms_thr=0.5)
    # Find this config in the sweep
    for score_thr, det_cap, nms_thr, prec, rec, f1 in []:  # placeholder
        pass

    # Save results
    out_path = Path("outputs/poc_aqua_full/exp_d_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "rpn_stats": {k: {"total_gt": v["total_gt"], "covered": v["covered"],
                              "avg_proposals": float(np.mean(v["n_proposals"]))}
                          for k, v in rpn_stats.items()},
            "best_params": list(best_params),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

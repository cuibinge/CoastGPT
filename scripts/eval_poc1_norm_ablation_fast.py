#!/usr/bin/env python3
"""Fast GF6 input-domain normalization ablation for PoC-1.

Runs stratified small-sample RPN recall first, with optional ROI/final metrics.
The script intentionally uses the same direct PIL + binary-mask path as the
existing RPN diagnostics to keep the experiment cheap and interpretable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from ml_collections import ConfigDict

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn  # noqa: E402
from utils.mask_utils import binary_mask_to_instances  # noqa: E402

import importlib.util as _u  # noqa: E402

_train_spec = _u.spec_from_file_location("poc_stage_one_det", _REPO_ROOT / "scripts" / "poc_stage_one_det.py")
_train_mod = _u.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)
build_vision_encoder = _train_mod.build_vision_encoder
load_config = _train_mod.load_config


def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / max(area_a + area_b - inter, 1e-8))


def greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5):
    pairs = []
    for gi in range(len(gt_boxes)):
        for pi in range(len(pred_boxes)):
            iou = compute_iou(gt_boxes[gi], pred_boxes[pi])
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt, matched_pred = set(), set()
    for _iou, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
    return len(matched_gt), len(pred_boxes) - len(matched_pred), len(gt_boxes) - len(matched_gt)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_records(
    records: Sequence[dict],
    max_gf6: int,
    max_non_gf6: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    gf6 = [r for r in records if str(r.get("sensor")) == "GF6"]
    non = [r for r in records if str(r.get("sensor")) != "GF6"]
    rng.shuffle(gf6)
    rng.shuffle(non)
    selected = gf6[: min(max_gf6, len(gf6))] + non[: min(max_non_gf6, len(non))]
    rng.shuffle(selected)
    return selected


def compute_stats(
    records: Sequence[dict],
    data_root: Path,
    image_size: int,
    max_per_sensor: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    by_sensor = defaultdict(list)
    for r in records:
        by_sensor[str(r.get("sensor") or "UNKNOWN")].append(r)

    chosen = []
    for sensor, items in sorted(by_sensor.items()):
        items = list(items)
        rng.shuffle(items)
        chosen.extend(items[: min(max_per_sensor, len(items))])

    groups = {"GF6": [], "non_GF6": [], "all": []}
    for sample in chosen:
        image = Image.open(data_root / sample["image_path"]).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        groups["all"].append(arr)
        if str(sample.get("sensor")) == "GF6":
            groups["GF6"].append(arr)
        else:
            groups["non_GF6"].append(arr)

    stats = {}
    for key, arrs in groups.items():
        stack = np.stack(arrs, axis=0)
        stats[key] = {
            "count": int(stack.shape[0]),
            "channel_mean": stack.mean(axis=(0, 1, 2)).tolist(),
            "channel_std": stack.std(axis=(0, 1, 2)).tolist(),
            "global_mean": float(stack.mean()),
            "global_std": float(stack.std()),
        }
    return stats


def apply_norm(image: torch.Tensor, method: str, stats: dict, is_gf6: bool) -> torch.Tensor:
    if method == "none":
        return image
    if not is_gf6 and not method.startswith("all_"):
        return image

    def clamp(x):
        return x.clamp(0.0, 1.0)

    if method == "gf6_channel_match_non_gf6":
        src_m = torch.tensor(stats["GF6"]["channel_mean"], dtype=image.dtype).view(3, 1, 1)
        src_s = torch.tensor(stats["GF6"]["channel_std"], dtype=image.dtype).view(3, 1, 1)
        dst_m = torch.tensor(stats["non_GF6"]["channel_mean"], dtype=image.dtype).view(3, 1, 1)
        dst_s = torch.tensor(stats["non_GF6"]["channel_std"], dtype=image.dtype).view(3, 1, 1)
        return clamp((image - src_m) / (src_s + 1e-6) * dst_s + dst_m)
    if method == "gf6_global_match_non_gf6":
        src_m = float(stats["GF6"]["global_mean"])
        src_s = float(stats["GF6"]["global_std"])
        dst_m = float(stats["non_GF6"]["global_mean"])
        dst_s = float(stats["non_GF6"]["global_std"])
        return clamp((image - src_m) / (src_s + 1e-6) * dst_s + dst_m)
    if method.startswith("gf6_gamma_"):
        gamma = float(method.rsplit("_", 1)[-1])
        return clamp(image).pow(gamma)
    if method == "gf6_percentile_stretch":
        flat = image.flatten(1)
        p02 = torch.quantile(flat, 0.02, dim=1).view(3, 1, 1)
        p98 = torch.quantile(flat, 0.98, dim=1).view(3, 1, 1)
        return clamp((image - p02) / (p98 - p02 + 1e-6))
    if method == "all_per_image_standardize_to_0_1":
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True)
        return clamp(((image - mean) / (std + 1e-6) + 2.0) / 4.0)
    raise ValueError(f"Unknown method: {method}")


def load_image_and_boxes(sample: dict, data_root: Path, image_size: int):
    image = Image.open(data_root / sample["image_path"]).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    img_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

    binary_rel = sample.get("binary_label_path")
    if not binary_rel:
        return img_t, np.zeros((0, 4), dtype=np.float32)
    binary_path = data_root / binary_rel
    if not binary_path.exists():
        return img_t, np.zeros((0, 4), dtype=np.float32)
    binary = Image.open(binary_path).convert("L")
    binary = binary.resize((image_size, image_size), Image.NEAREST)
    _masks, boxes, _areas = binary_mask_to_instances(np.asarray(binary), min_area=8, connectivity=4)
    return img_t, np.asarray(boxes, dtype=np.float32)


def build_model(cfg, checkpoint: str, device: torch.device):
    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"]).to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

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
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=0.0,
        box_nms_thresh=mrcnn_cfg.get("box_nms_thresh", 0.5),
        box_detections_per_img=100,
        image_mean=mrcnn_cfg.get("image_mean", [0.0, 0.0, 0.0]),
        image_std=mrcnn_cfg.get("image_std", [1.0, 1.0, 1.0]),
        min_size=mrcnn_cfg.get("min_size", 224),
        max_size=mrcnn_cfg.get("max_size", 224),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    model.eval()
    return model


def empty_metrics():
    return {
        "samples": 0,
        "gt": 0,
        "rpn_covered": 0,
        "final_pred": 0,
        "final_tp": 0,
        "final_fp": 0,
        "final_fn": 0,
        "zero_rpn_tiles": 0,
        "zero_final_tiles": 0,
    }


def add_group(metrics, group: str, gt_boxes, proposals, detections, score_thresh, iou_thresh, skip_final):
    m = metrics[group]
    m["samples"] += 1
    m["gt"] += int(len(gt_boxes))

    covered = 0
    for gt in gt_boxes:
        if any(compute_iou(gt, prop) >= iou_thresh for prop in proposals):
            covered += 1
    m["rpn_covered"] += covered
    if len(gt_boxes) > 0 and covered == 0:
        m["zero_rpn_tiles"] += 1

    if skip_final:
        return

    keep = detections["scores"].cpu().numpy() >= score_thresh
    pred_boxes = detections["boxes"].cpu().numpy()[keep]
    tp, fp, fn = greedy_match(gt_boxes, pred_boxes, iou_thresh=iou_thresh)
    m["final_pred"] += int(len(pred_boxes))
    m["final_tp"] += int(tp)
    m["final_fp"] += int(fp)
    m["final_fn"] += int(fn)
    if len(gt_boxes) > 0 and tp == 0:
        m["zero_final_tiles"] += 1


def finalize(metrics: dict, skip_final: bool):
    out = {}
    for group, m in sorted(metrics.items()):
        rpn_recall = m["rpn_covered"] / max(m["gt"], 1)
        item = dict(m)
        item["rpn_recall"] = rpn_recall
        if not skip_final:
            tp, fp, fn = m["final_tp"], m["final_fp"], m["final_fn"]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            item["precision"] = p
            item["recall"] = r
            item["f1"] = 2 * p * r / max(p + r, 1e-8)
        out[group] = item
    return out


@torch.no_grad()
def evaluate_method(
    model,
    samples: Sequence[dict],
    data_root: Path,
    image_size: int,
    stats: dict,
    method: str,
    device: torch.device,
    score_thresh: float,
    iou_thresh: float,
    skip_final: bool,
) -> dict:
    metrics = defaultdict(empty_metrics)
    for idx, sample in enumerate(samples):
        sensor = str(sample.get("sensor") or "UNKNOWN")
        is_gf6 = sensor == "GF6"
        image, gt_boxes = load_image_and_boxes(sample, data_root, image_size)
        image = apply_norm(image, method, stats, is_gf6=is_gf6).to(device)
        original_sizes = [tuple(image.shape[-2:])]

        images_list, _ = model.transform([image], None)
        features = model.backbone(images_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = model.rpn(images_list, features, None)
        props = proposals[0].detach().cpu().numpy()

        detections = {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))}
        if not skip_final:
            detections_list, _ = model.roi_heads(features, proposals, images_list.image_sizes, None)
            detections_list = model.transform.postprocess(
                detections_list,
                images_list.image_sizes,
                original_sizes,
            )
            detections = {k: v.detach().cpu() for k, v in detections_list[0].items()}

        groups = [sensor, "GF6" if is_gf6 else "non_GF6", "all"]
        for group in groups:
            add_group(metrics, group, gt_boxes, props, detections, score_thresh, iou_thresh, skip_final)

        if (idx + 1) % 20 == 0:
            print(f"    {method}: {idx + 1}/{len(samples)}", flush=True)
    return finalize(metrics, skip_final)


def print_method(method: str, results: dict, skip_final: bool):
    print(f"\n=== {method} ===", flush=True)
    for group in ["GF6", "non_GF6", "GF1", "GF2", "all"]:
        if group not in results:
            continue
        r = results[group]
        msg = (
            f"{group:7s} n={r['samples']:3d} gt={r['gt']:4d} "
            f"RPN={r['rpn_recall']:.4f} zeroRPN={r['zero_rpn_tiles']:3d}"
        )
        if not skip_final:
            msg += f" P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} zeroFinal={r['zero_final_tiles']:3d}"
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/poc_aqua_instance.yaml")
    p.add_argument("--checkpoint", default="outputs/poc_aqua_full/checkpoints/epoch_040.pt")
    p.add_argument("--train-manifest", default="data/poc_aqua_full/train.json")
    p.add_argument("--manifest", default="data/poc_aqua_full/val.json")
    p.add_argument("--data-root", default="/home/ma-user/work/Stage3Data/养殖区")
    p.add_argument("--output", default="outputs/diagnostics/gf6_norm_ablation/fast_rpn.json")
    p.add_argument("--methods", default="none,gf6_channel_match_non_gf6,gf6_global_match_non_gf6,gf6_gamma_0.75,gf6_gamma_0.65,gf6_percentile_stretch")
    p.add_argument("--max-gf6", type=int, default=80)
    p.add_argument("--max-non-gf6", type=int, default=80)
    p.add_argument("--stats-max-per-sensor", type=int, default=300)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--skip-final", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    image_size = int(cfg["data"].get("image_size", 224))
    data_root = Path(args.data_root)
    train_records = load_json(Path(args.train_manifest))
    val_records = load_json(Path(args.manifest))
    selected = sample_records(val_records, args.max_gf6, args.max_non_gf6, args.seed)
    stats = compute_stats(train_records, data_root, image_size, args.stats_max_per_sensor, args.seed)

    print(f"Selected {len(selected)} val tiles: {Counter(str(s.get('sensor')) for s in selected)}", flush=True)
    print("Stats:", json.dumps(stats, indent=2), flush=True)

    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    all_results = {
        "checkpoint": args.checkpoint,
        "selected_counts": dict(Counter(str(s.get("sensor")) for s in selected)),
        "stats": stats,
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "skip_final": args.skip_final,
        "methods": {},
    }
    for method in methods:
        print(f"\nRunning {method}", flush=True)
        results = evaluate_method(
            model,
            selected,
            data_root,
            image_size,
            stats,
            method,
            device,
            args.score_thresh,
            args.iou_thresh,
            args.skip_final,
        )
        print_method(method, results, args.skip_final)
        all_results["methods"][method] = results

    out = Path(args.output)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

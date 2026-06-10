#!/usr/bin/env python3
"""Inference-time normalization ablations for PoC-1 aquaculture detector.

This is a read-only experiment: it loads an existing checkpoint and evaluates
several lightweight image-domain transforms, stratified by sensor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn  # noqa: E402

import importlib.util as _u  # noqa: E402

_eval_path = _REPO_ROOT / "scripts" / "eval_poc1.py"
_eval_spec = _u.spec_from_file_location("eval_poc1", _eval_path)
_eval_mod = _u.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
build_model = _eval_mod.build_model
compute_iou = _eval_mod.compute_iou
compute_mask_iou = _eval_mod.compute_mask_iou
compute_mask_dice = _eval_mod.compute_mask_dice
greedy_match = _eval_mod.greedy_match


def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_image_stats(
    samples: Sequence[dict],
    data_root: Path,
    image_size: int,
) -> dict:
    from PIL import Image

    groups = {
        "GF6": [],
        "non_GF6": [],
        "all": [],
    }
    for sample in samples:
        sensor = str(sample.get("sensor") or "")
        image = Image.open(data_root / sample["image_path"]).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        groups["all"].append(arr)
        if "GF6" in sensor:
            groups["GF6"].append(arr)
        else:
            groups["non_GF6"].append(arr)

    stats = {}
    for key, arrs in groups.items():
        stack = np.stack(arrs, axis=0)
        channel_mean = stack.mean(axis=(0, 1, 2))
        channel_std = stack.std(axis=(0, 1, 2))
        stats[key] = {
            "count": int(stack.shape[0]),
            "channel_mean": channel_mean.tolist(),
            "channel_std": channel_std.tolist(),
            "global_mean": float(stack.mean()),
            "global_std": float(stack.std()),
        }
    return stats


class NormAblationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: AquaPoCDataset,
        method: str,
        stats: dict,
        gf6_only: bool = True,
    ):
        self.base = base
        self.method = method
        self.stats = stats
        self.gf6_only = gf6_only

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _clamp(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        method = self.method
        if method == "none":
            return image

        if method == "gf6_channel_match_non_gf6":
            src_m = torch.tensor(self.stats["GF6"]["channel_mean"], dtype=image.dtype).view(3, 1, 1)
            src_s = torch.tensor(self.stats["GF6"]["channel_std"], dtype=image.dtype).view(3, 1, 1)
            dst_m = torch.tensor(self.stats["non_GF6"]["channel_mean"], dtype=image.dtype).view(3, 1, 1)
            dst_s = torch.tensor(self.stats["non_GF6"]["channel_std"], dtype=image.dtype).view(3, 1, 1)
            return self._clamp((image - src_m) / (src_s + 1e-6) * dst_s + dst_m)

        if method == "gf6_global_match_non_gf6":
            src_m = float(self.stats["GF6"]["global_mean"])
            src_s = float(self.stats["GF6"]["global_std"])
            dst_m = float(self.stats["non_GF6"]["global_mean"])
            dst_s = float(self.stats["non_GF6"]["global_std"])
            return self._clamp((image - src_m) / (src_s + 1e-6) * dst_s + dst_m)

        if method.startswith("gf6_gamma_"):
            gamma = float(method.rsplit("_", 1)[-1])
            return self._clamp(image).pow(gamma)

        if method == "gf6_percentile_stretch":
            x = image.clone()
            flat = x.flatten(1)
            p02 = torch.quantile(flat, 0.02, dim=1).view(3, 1, 1)
            p98 = torch.quantile(flat, 0.98, dim=1).view(3, 1, 1)
            return self._clamp((x - p02) / (p98 - p02 + 1e-6))

        if method == "all_per_image_standardize_to_0_1":
            x = image.clone()
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True)
            y = (x - mean) / (std + 1e-6)
            # Map roughly standard-normal range [-2, 2] to [0, 1].
            return self._clamp((y + 2.0) / 4.0)

        raise ValueError(f"Unknown normalization method: {method}")

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        sensor = str(item["meta"].get("sensor") or "")
        apply = (not self.gf6_only) or ("GF6" in sensor)
        if apply:
            item = dict(item)
            item["image"] = self._apply(item["image"])
            item["meta"] = dict(item["meta"])
            item["meta"]["norm_ablation"] = self.method
        return item


def empty_bucket() -> dict:
    return {
        "total_gt": 0,
        "total_pred": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "matched_ious": [],
        "matched_mask_ious": [],
        "matched_mask_dices": [],
        "rpn_total_gt": 0,
        "rpn_covered": 0,
        "samples": 0,
        "zero_recall_samples": 0,
    }


def add_final_metrics(bucket: dict, target: dict, out: dict, score_thresh: float, iou_thresh: float) -> None:
    gt_boxes = target["boxes"].numpy()
    gt_masks = target["masks"].numpy()
    keep = out["scores"] >= score_thresh
    pred_boxes = out["boxes"][keep].numpy()
    masks = out["masks"]
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    pred_masks = masks[keep].numpy()

    matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes, iou_thresh=iou_thresh)

    bucket["samples"] += 1
    bucket["total_gt"] += len(gt_boxes)
    bucket["total_pred"] += len(pred_boxes)
    bucket["total_tp"] += len(matches)
    bucket["total_fp"] += len(unmatched_pred)
    bucket["total_fn"] += len(unmatched_gt)
    if len(gt_boxes) > 0 and len(matches) == 0:
        bucket["zero_recall_samples"] += 1

    for gi, pi, box_iou in matches:
        bucket["matched_ious"].append(float(box_iou))
        bucket["matched_mask_ious"].append(float(compute_mask_iou(gt_masks[gi], pred_masks[pi])))
        bucket["matched_mask_dices"].append(float(compute_mask_dice(gt_masks[gi], pred_masks[pi])))


def add_rpn_metrics(bucket: dict, target: dict, proposals: torch.Tensor, iou_thresh: float) -> None:
    gt_boxes = target["boxes"].numpy()
    props = proposals.cpu().numpy()
    covered = 0
    for gt in gt_boxes:
        for prop in props:
            if compute_iou(gt, prop) >= iou_thresh:
                covered += 1
                break
    bucket["rpn_total_gt"] += len(gt_boxes)
    bucket["rpn_covered"] += covered


def finalize_bucket(bucket: dict) -> dict:
    tp = bucket["total_tp"]
    fp = bucket["total_fp"]
    fn = bucket["total_fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    rpn_recall = bucket["rpn_covered"] / max(bucket["rpn_total_gt"], 1)

    return {
        "samples": bucket["samples"],
        "total_gt": bucket["total_gt"],
        "total_pred": bucket["total_pred"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rpn_total_gt": bucket["rpn_total_gt"],
        "rpn_covered": bucket["rpn_covered"],
        "rpn_recall": rpn_recall,
        "zero_recall_samples": bucket["zero_recall_samples"],
        "box_iou_mean": float(np.mean(bucket["matched_ious"])) if bucket["matched_ious"] else 0.0,
        "mask_iou_mean": float(np.mean(bucket["matched_mask_ious"])) if bucket["matched_mask_ious"] else 0.0,
        "mask_dice_mean": float(np.mean(bucket["matched_mask_dices"])) if bucket["matched_mask_dices"] else 0.0,
    }


@torch.no_grad()
def evaluate_stratified(
    model,
    dataloader,
    device: torch.device,
    score_thresh: float,
    iou_thresh: float,
) -> dict:
    buckets = defaultdict(empty_bucket)
    model.eval()

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images_device = [img.to(device) for img in images]

        outputs = model(images_device)
        outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]

        images_list, _ = model.transform(images_device, None)
        features = model.backbone(images_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = model.rpn(images_list, features, None)

        for target, out, proposal, meta in zip(targets, outputs_cpu, proposals, metas):
            sensor = str(meta.get("sensor") or "UNKNOWN")
            groups = [sensor, "GF6" if "GF6" in sensor else "non_GF6", "all"]
            for group in groups:
                add_final_metrics(buckets[group], target, out, score_thresh, iou_thresh)
                add_rpn_metrics(buckets[group], target, proposal, iou_thresh)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx + 1}/{len(dataloader)}", flush=True)

    return {k: finalize_bucket(v) for k, v in sorted(buckets.items())}


def print_compact(method: str, results: dict) -> None:
    print(f"\n=== {method} ===")
    for group in ["GF6", "non_GF6", "GF1", "GF2", "all"]:
        if group not in results:
            continue
        r = results[group]
        print(
            f"{group:7s} samples={r['samples']:3d} "
            f"RPN={r['rpn_recall']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
            f"zero={r['zero_recall_samples']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/poc_aqua_full/checkpoints/epoch_040.pt")
    parser.add_argument("--config", default="configs/poc_aqua_instance.yaml")
    parser.add_argument("--train-manifest", default="data/poc_aqua_full/train.json")
    parser.add_argument("--manifest", default="data/poc_aqua_full/val.json")
    parser.add_argument("--data-root", default="/home/ma-user/work/Stage3Data/养殖区")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/diagnostics/gf6_norm_ablation/norm_ablation_eval.json")
    parser.add_argument(
        "--methods",
        default="none,gf6_channel_match_non_gf6,gf6_global_match_non_gf6,gf6_gamma_0.75,gf6_gamma_0.65,gf6_percentile_stretch",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path
    train_manifest_path = Path(args.train_manifest)
    if not train_manifest_path.is_absolute():
        train_manifest_path = _REPO_ROOT / train_manifest_path

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    train_samples = load_manifest(train_manifest_path)

    print("Computing train image stats...", flush=True)
    # Config image size is 224 for this PoC.
    image_size = 224
    stats = compute_image_stats(train_samples, data_root, image_size)
    print(json.dumps(stats, indent=2), flush=True)

    print(f"Loading model from {args.checkpoint}...", flush=True)
    model, cfg = build_model(str(config_path), args.checkpoint, str(device))
    image_size = cfg["data"].get("image_size", 224)
    base_ds = AquaPoCDataset(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        image_size=image_size,
    )
    print(f"Val samples: {len(base_ds)} by_sensor={Counter(str(s.get('sensor')) for s in base_ds.samples)}", flush=True)

    all_results = {
        "checkpoint": args.checkpoint,
        "manifest": str(manifest_path),
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "train_image_stats": stats,
        "methods": {},
    }

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        ds = NormAblationDataset(base_ds, method=method, stats=stats, gf6_only=not method.startswith("all_"))
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=poc_collate_fn,
        )
        print(f"\nRunning method: {method}", flush=True)
        results = evaluate_stratified(model, loader, device, args.score_thresh, args.iou_thresh)
        print_compact(method, results)
        all_results["methods"][method] = results

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

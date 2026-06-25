#!/usr/bin/env python3
"""Batch inference, threshold sweep, and GeoJSON generation for PoC-3."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.edge_head import SingleScaleEdgeHead
from Models.fpn_neck import FPNNeck
from utils.coastline_eval_tools import (
    aggregate_metric_records,
    choose_best_thresholds,
    parse_threshold_values,
)
from utils.coastline_metrics import compute_all_edge_metrics
from utils.edge_postprocess import (
    binary_to_skeleton,
    heatmap_to_binary,
    paths_to_geojson,
    simplify_path,
    skeleton_to_paths,
)


def load_model(config_path, checkpoint_path, device):
    import yaml
    from ml_collections import ConfigDict

    with open(config_path, encoding="utf-8") as f:
        cfg = ConfigDict(yaml.safe_load(f))

    vision = DualVisionEncoder(ConfigDict(dict(cfg.get("model", {}))))
    sd = torch.load(cfg.get("model.vision_checkpoint"), map_location="cpu")
    sd = sd.get("vision_ckpt", sd.get("model", sd))
    clean = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("vision."):
            nk = nk[7:]
        clean[nk] = v

    vision.load_state_dict(clean, strict=False)
    vision = vision.to(device).eval()
    for p in vision.parameters():
        p.requires_grad = False

    fpn = FPNNeck([128, 256, 512, 1024], 256, vit_in_channels=1024).to(device).eval()
    head = SingleScaleEdgeHead(256, output_size=(224, 224)).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    fpn.load_state_dict(ckpt["fpn"])
    head.load_state_dict(ckpt["edge_head"])
    return vision, fpn, head


def predict_one(vision, fpn, head, img_tensor, device):
    with torch.no_grad():
        _, g_grid, pyr = vision.encode_with_spatial(img_tensor.to(device))
        p1, p2, p3, p4 = fpn(
            pyr[0], pyr[1], pyr[2], pyr[3], vit_feat=g_grid if fpn.has_vit else None
        )
        logits = head(p1, p2, p3, p4)
    return torch.sigmoid(logits).detach().cpu().numpy()


def load_image(img_path):
    arr = tifffile.imread(str(img_path))
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((224, 224), Image.BILINEAR)
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def build_georef_from_geojson(geojson_path, size=224):
    path = Path(geojson_path)
    if not path.exists():
        return {"source_crs": "EPSG:4326", "model_transform": [1e-5, 0, 0, 0, -1e-5, 0]}

    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    coords = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        geom_type = geom.get("type")
        raw = geom.get("coordinates", [])
        if geom_type == "LineString":
            coords.extend(raw)
        elif geom_type == "MultiLineString":
            for line in raw:
                coords.extend(line)

    if not coords:
        return {"source_crs": "EPSG:4326", "model_transform": [1e-5, 0, 0, 0, -1e-5, 0]}

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    xr = (max(lons) - min(lons)) / size
    yr = (max(lats) - min(lats)) / size
    return {"source_crs": "EPSG:4326", "model_transform": [xr, 0, min(lons), 0, -yr, max(lats)]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="outputs/poc3_edge/batch_infer")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--threshold-values",
        default="",
        help="Comma list or start:end:step range for threshold sweep",
    )
    parser.add_argument("--threshold-select-metric", default="buffered_f1_3px")
    parser.add_argument(
        "--threshold-by-category",
        action="store_true",
        help="Select best threshold independently for each tile category",
    )
    parser.add_argument("--config", default="configs/poc3_edge_a1_focal_dice.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/poc3_edge/a1_relabel_unified/checkpoints/epoch_010.pt",
    )
    parser.add_argument("--manifest", default="outputs/poc3_edge/coastline_manifest.json")
    parser.add_argument("--label-dir", default="outputs/poc3_labels_unified/labels_224")
    parser.add_argument(
        "--details-json",
        default="",
        help="Optional path for per-sample selected-threshold metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    out_dir = Path(args.output)
    (out_dir / "geojson").mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    vision, fpn, head = load_model(args.config, args.checkpoint, device)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    tiles = manifest["val"]
    if args.max_samples > 0:
        tiles = tiles[: args.max_samples]

    label_dir = Path(args.label_dir)
    print(f"Processing {len(tiles)} tiles...")

    samples = []
    for tile in tqdm(tiles):
        sid = tile["sample_id"]
        try:
            img_t = load_image(tile["image_path"]).unsqueeze(0)
            heatmap = predict_one(vision, fpn, head, img_t, device)[0, 0]
            gt_path = label_dir / "edge_center" / f"{sid}.npy"
            if not gt_path.exists():
                continue
            samples.append(
                {
                    "sample_id": sid,
                    "category": tile.get("category") or tile.get("shoreline_type") or "unknown",
                    "heatmap": heatmap,
                    "gt": (np.load(gt_path) > 0.5).astype(np.float32),
                    "georef": build_georef_from_geojson(tile.get("label_geojson_path", "")),
                }
            )
        except Exception as exc:
            print(f"WARNING: skipped {sid}: {exc}")

    thresholds = parse_threshold_values(args.threshold_values) or [float(args.threshold)]
    sweep_records = []
    for sample in samples:
        for threshold in thresholds:
            metrics = compute_all_edge_metrics(sample["heatmap"], sample["gt"], threshold=threshold)
            metrics["sample_id"] = sample["sample_id"]
            metrics["category"] = sample["category"]
            metrics["threshold"] = float(threshold)
            sweep_records.append(metrics)

    group_key = "category" if args.threshold_by_category and len(thresholds) > 1 else None
    best_thresholds = choose_best_thresholds(
        sweep_records,
        metric=args.threshold_select_metric,
        group_key=group_key,
    )

    fallback_threshold = best_thresholds.get("global", {}).get("threshold", float(args.threshold))
    selected_thresholds = defaultdict(lambda: fallback_threshold)
    if group_key:
        for group, row in best_thresholds.items():
            selected_thresholds[group] = row["threshold"]

    results = []
    for sample in samples:
        sid = sample["sample_id"]
        category = sample["category"]
        threshold = selected_thresholds[category]

        binary = heatmap_to_binary(sample["heatmap"], threshold=threshold, min_area=4)
        skeleton = binary_to_skeleton(binary)
        paths = skeleton_to_paths(skeleton)
        paths = [p for p in paths if len(p) >= 3]
        if paths:
            paths = [simplify_path(p, epsilon=0.1) for p in paths]
            paths = sorted(paths, key=len, reverse=True)[:10]
            paths = [p for p in paths if len(p) >= 5]
        fc = paths_to_geojson(paths, sample["georef"], sid, class_name=category)

        geo_path = out_dir / "geojson" / f"{sid}.geojson"
        with open(geo_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        metrics = compute_all_edge_metrics(sample["heatmap"], sample["gt"], threshold=threshold)
        metrics["sample_id"] = sid
        metrics["category"] = category
        metrics["selected_threshold"] = float(threshold)
        metrics["num_features"] = len(fc.get("features", []))
        results.append(metrics)

    if results:
        agg = aggregate_metric_records(results)
        by_category = {
            group: aggregate_metric_records(group_records)
            for group, group_records in _group_by(results, "category").items()
        }
        details_path = Path(args.details_json) if args.details_json else out_dir / "details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({"records": _json_ready(results)}, f, ensure_ascii=False, indent=2)

        print(f"\n=== Batch Inference Results ({len(results)} tiles) ===")
        print(f"  Threshold mode:       {'category' if group_key else 'global'}")
        for group, row in best_thresholds.items():
            print(
                f"  Best threshold[{group}]: {row['threshold']:.3f} "
                f"{args.threshold_select_metric}={row.get(args.threshold_select_metric, 0):.4f}"
            )
        print(f"  Avg F1@1px (buf):    {agg.get('buffered_f1_1px', 0):.4f}")
        print(f"  Avg F1@3px (buf):    {agg.get('buffered_f1_3px', 0):.4f}")
        print(f"  Avg Chamfer (px):    {agg.get('chamfer_distance_px', 0):.1f}")
        print(f"  Avg pred_fg (%):     {agg.get('pred_fg_ratio', 0) * 100:.2f}")
        print(f"  Avg num features:    {agg.get('num_features', 0):.1f}")
        for group, group_agg in by_category.items():
            print(
                f"  {group} F1@3px:        {group_agg.get('buffered_f1_3px', 0):.4f} "
                f"(threshold={group_agg.get('selected_threshold', 0):.3f})"
            )
        print(f"  GeoJSON: {out_dir}/geojson/")
        print(f"  Details: {details_path}")

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aggregate": _json_ready(agg),
                    "by_category": _json_ready(by_category),
                    "threshold_selection": _json_ready(best_thresholds),
                    "threshold_select_metric": args.threshold_select_metric,
                    "threshold_by_category": bool(group_key),
                    "n_samples": len(results),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def _group_by(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get(key, "unknown"))].append(record)
    return groups


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


if __name__ == "__main__":
    main()

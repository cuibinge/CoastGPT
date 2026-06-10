#!/usr/bin/env python3
"""Per-sensor feature and label diagnostics for PoC aquaculture data.

This script is intentionally read-only. It checks:
  1. Input normalization and per-sensor image statistics.
  2. Binary mask / dataset target / GeoJSON fallback alignment.
  3. Per-sensor embedding separability for DINO grid, ConvNeXt pyramid, and fused seq.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

from utils.georef_transform import resize_georef  # noqa: E402


def load_symbol_from_file(module_name: str, path: Path, symbol: str):
    """Load a symbol without executing package __init__.py side effects."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


AquaPoCDataset = load_symbol_from_file(
    "aqua_poc_dataset_direct",
    REPO_ROOT / "Dataset" / "aqua_poc_dataset.py",
    "AquaPoCDataset",
)
DualVisionEncoder = load_symbol_from_file(
    "dual_vision_encoder_direct",
    REPO_ROOT / "Models" / "dual_vision_encoder.py",
    "DualVisionEncoder",
)


def clean_vision_state_dict(state_dict: dict) -> dict:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        if new_k.startswith("vision."):
            new_k = new_k[len("vision.") :]
        cleaned[new_k] = v
    return cleaned


def build_vision_encoder(model_cfg, ckpt_path: str):
    print("[features] initializing DualVisionEncoder", flush=True)
    vision = DualVisionEncoder(model_cfg)
    print(f"[features] loading vision checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "vision_ckpt" in ckpt:
            state_dict = ckpt["vision_ckpt"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    state_dict = clean_vision_state_dict(state_dict)
    incompat = vision.load_state_dict(state_dict, strict=False)
    warnings.warn(
        f"Loaded vision checkpoint with missing={len(incompat.missing_keys)}, "
        f"unexpected={len(incompat.unexpected_keys)}"
    )
    return vision


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        if device_arg.startswith("npu"):
            import torch_npu  # noqa: F401

        return torch.device(device_arg)
    try:
        import torch_npu  # noqa: F401

        return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifests(paths: Sequence[Path]) -> List[dict]:
    records = []
    for manifest_path in paths:
        split = manifest_path.stem
        for i, sample in enumerate(load_json(manifest_path)):
            rec = dict(sample)
            rec["_manifest_path"] = str(manifest_path)
            rec["_manifest_index"] = i
            rec["_split"] = split
            records.append(rec)
    return records


def group_by_sensor(records: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get("sensor") or "UNKNOWN")].append(rec)
    return dict(grouped)


def sample_balanced(
    records: Sequence[dict],
    max_per_sensor: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    selected = []
    for sensor, items in sorted(group_by_sensor(records).items()):
        items = list(items)
        rng.shuffle(items)
        selected.extend(items[: min(max_per_sensor, len(items))])
    rng.shuffle(selected)
    return selected


def image_to_model_array(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def image_to_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    arr = image_to_model_array(image_path, image_size)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def summarize_numeric(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def safe_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def combined_target_mask(target: dict, size: int) -> np.ndarray:
    masks = target["masks"]
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    if masks.shape[0] == 0:
        return np.zeros((size, size), dtype=np.uint8)
    return (masks.sum(axis=0) > 0).astype(np.uint8)


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(mask, dtype=bool)
    up = np.zeros_like(m)
    up[:-1, :] = m[1:, :]
    down = np.zeros_like(m)
    down[1:, :] = m[:-1, :]
    left = np.zeros_like(m)
    left[:, :-1] = m[:, 1:]
    right = np.zeros_like(m)
    right[:, 1:] = m[:, :-1]
    interior = m & up & down & left & right
    return m & ~interior


def gradient_boundary_ratio(image_arr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not mask.any():
        return None
    gray = (
        0.299 * image_arr[:, :, 0]
        + 0.587 * image_arr[:, :, 1]
        + 0.114 * image_arr[:, :, 2]
    )
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    boundary = boundary_from_mask(mask)
    if boundary.sum() == 0:
        return None
    overall = float(grad.mean())
    if overall <= 1e-8:
        return None
    return float(grad[boundary].mean() / overall)


def make_overlay(image_arr: np.ndarray, mask: np.ndarray) -> Image.Image:
    img = (np.clip(image_arr, 0, 1) * 255).astype(np.uint8)
    overlay = img.copy()
    m = mask.astype(bool)
    overlay[m] = (0.55 * overlay[m] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    b = boundary_from_mask(mask)
    overlay[b] = np.array([255, 255, 0], dtype=np.uint8)
    return Image.fromarray(overlay)


def save_overlay_grid(
    records: Sequence[dict],
    data_root: Path,
    image_size: int,
    out_path: Path,
    max_items: int = 16,
) -> None:
    if not records:
        return
    cells = []
    n = min(max_items, len(records))
    for rec in records[:n]:
        image_arr = image_to_model_array(data_root / rec["image_path"], image_size)
        binary_path = data_root / rec.get("binary_label_path", "")
        if binary_path.exists():
            binary = Image.open(binary_path).convert("L")
            binary = binary.resize((image_size, image_size), Image.NEAREST)
            mask = (np.asarray(binary) > 0).astype(np.uint8)
        else:
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cells.append(make_overlay(image_arr, mask))
    cols = 4
    rows = int(math.ceil(len(cells) / cols))
    grid = Image.new("RGB", (cols * image_size, rows * image_size), "white")
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % cols) * image_size, (i // cols) * image_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def run_input_and_label_diagnostics(
    records: Sequence[dict],
    manifest_paths: Sequence[Path],
    data_root: Path,
    image_size: int,
    output_dir: Path,
    max_label_per_sensor: int,
    seed: int,
) -> dict:
    selected = sample_balanced(records, max_label_per_sensor, seed)
    by_sensor = group_by_sensor(selected)

    transform_summary = {
        "image_resize": f"PIL.Image.BILINEAR -> {image_size}x{image_size}",
        "image_scale": "float32 / 255.0",
        "mask_resize": f"PIL.Image.NEAREST -> {image_size}x{image_size}",
        "mask_connectivity": 4,
        "maskrcnn_image_mean": [0.0, 0.0, 0.0],
        "maskrcnn_image_std": [1.0, 1.0, 1.0],
        "sensor_specific_normalization": False,
    }

    # Build datasets once so target construction follows the same code path.
    datasets = {
        str(path): AquaPoCDataset(
            manifest_path=str(path),
            data_root=str(data_root),
            image_size=image_size,
        )
        for path in manifest_paths
    }

    rows = []
    sensor_acc = defaultdict(lambda: defaultdict(list))
    sensor_counts = defaultdict(Counter)

    for rec in selected:
        sensor = str(rec.get("sensor") or "UNKNOWN")
        img_path = data_root / rec["image_path"]
        binary_rel = rec.get("binary_label_path")
        binary_path = data_root / binary_rel if binary_rel else None

        image = Image.open(img_path).convert("RGB")
        image_size_orig = image.size
        image_arr = image_to_model_array(img_path, image_size)

        for ch, name in enumerate(["r", "g", "b"]):
            sensor_acc[sensor][f"image_{name}_mean"].append(float(image_arr[:, :, ch].mean()))
            sensor_acc[sensor][f"image_{name}_std"].append(float(image_arr[:, :, ch].std()))
        sensor_acc[sensor]["image_global_mean"].append(float(image_arr.mean()))
        sensor_acc[sensor]["image_global_std"].append(float(image_arr.std()))

        target = datasets[rec["_manifest_path"]][rec["_manifest_index"]]["target"]
        target_mask = combined_target_mask(target, image_size)
        target_area = int(target_mask.sum())
        sensor_acc[sensor]["target_area_px"].append(float(target_area))
        sensor_acc[sensor]["target_instances"].append(float(len(target["boxes"])))

        binary_exists = bool(binary_path and binary_path.exists())
        binary_size = None
        binary_iou = None
        binary_area_diff_ratio = None
        image_binary_size_match = None
        if binary_exists:
            binary = Image.open(binary_path).convert("L")
            binary_size = binary.size
            image_binary_size_match = image_size_orig == binary_size
            binary = binary.resize((image_size, image_size), Image.NEAREST)
            binary_mask = (np.asarray(binary) > 0).astype(np.uint8)
            binary_iou = safe_iou(binary_mask, target_mask)
            binary_area = int(binary_mask.sum())
            denom = max(binary_area, 1)
            binary_area_diff_ratio = float(abs(binary_area - target_area) / denom)
            sensor_acc[sensor]["binary_target_iou"].append(binary_iou)
            sensor_acc[sensor]["binary_target_area_diff_ratio"].append(binary_area_diff_ratio)
            sensor_counts[sensor]["binary_present"] += 1
            if image_binary_size_match:
                sensor_counts[sensor]["image_binary_size_match"] += 1
        else:
            sensor_counts[sensor]["binary_missing"] += 1

        # Compare GeoJSON fallback to binary target when both sources exist.
        geojson_iou = None
        geojson_area_diff_ratio = None
        label_path = data_root / rec.get("label_path", "")
        if binary_exists and label_path.exists():
            try:
                model_transform, _resize_scale = resize_georef(
                    tuple(rec["original_size"]),
                    (image_size, image_size),
                    rec["original_transform"],
                )
                georef = {"source_crs": rec["source_crs"], "model_transform": model_transform}
                ds = datasets[rec["_manifest_path"]]
                geo_target = ds._build_target_from_geojson(rec, georef, rec["_manifest_index"])
                geo_mask = combined_target_mask(geo_target, image_size)
                geojson_iou = safe_iou(target_mask, geo_mask)
                denom = max(int(target_mask.sum()), 1)
                geojson_area_diff_ratio = float(abs(int(geo_mask.sum()) - int(target_mask.sum())) / denom)
                sensor_acc[sensor]["geojson_binary_iou"].append(geojson_iou)
                sensor_acc[sensor]["geojson_binary_area_diff_ratio"].append(geojson_area_diff_ratio)
                sensor_counts[sensor]["geojson_compared"] += 1
            except Exception:
                sensor_counts[sensor]["geojson_compare_failed"] += 1

        ratio = gradient_boundary_ratio(image_arr, target_mask)
        if ratio is not None:
            sensor_acc[sensor]["boundary_gradient_ratio"].append(ratio)

        rows.append({
            "sample_id": rec.get("sample_id"),
            "split": rec.get("_split"),
            "sensor": sensor,
            "image_path": rec.get("image_path"),
            "binary_label_path": rec.get("binary_label_path"),
            "image_size": list(image_size_orig),
            "binary_size": list(binary_size) if binary_size else None,
            "image_binary_size_match": image_binary_size_match,
            "target_instances": int(len(target["boxes"])),
            "target_area_px": target_area,
            "binary_target_iou": binary_iou,
            "binary_target_area_diff_ratio": binary_area_diff_ratio,
            "geojson_binary_iou": geojson_iou,
            "geojson_binary_area_diff_ratio": geojson_area_diff_ratio,
            "boundary_gradient_ratio": ratio,
        })

    # CSV rows for spot checks.
    csv_path = output_dir / "input_label_sample_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overlay_dir = output_dir / "overlays"
    rng = random.Random(seed)
    for sensor, sensor_records in sorted(by_sensor.items()):
        sensor_records = list(sensor_records)
        rng.shuffle(sensor_records)
        save_overlay_grid(
            sensor_records,
            data_root=data_root,
            image_size=image_size,
            out_path=overlay_dir / f"{sensor}_mask_overlay_grid.png",
            max_items=16,
        )

    per_sensor = {}
    for sensor, metrics in sorted(sensor_acc.items()):
        per_sensor[sensor] = {
            "counts": dict(sensor_counts[sensor]),
            "metrics": {k: summarize_numeric(v) for k, v in sorted(metrics.items())},
        }

    return {
        "transform_summary": transform_summary,
        "sampled_records": len(selected),
        "manifest_counts": {
            "all": len(records),
            "by_sensor": {k: len(v) for k, v in sorted(group_by_sensor(records).items())},
        },
        "per_sensor": per_sensor,
        "csv_path": str(csv_path),
        "overlay_dir": str(overlay_dir),
    }


def plot_embedding(
    name: str,
    x: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    method: str = "pca",
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if x.shape[0] < 3:
        return None

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        z = StandardScaler().fit_transform(x)
        if method == "tsne":
            perplexity = max(5, min(30, (x.shape[0] - 1) // 3))
            coords = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            ).fit_transform(z)
        else:
            coords = PCA(n_components=2, random_state=42).fit_transform(z)
    except Exception:
        z = x - x.mean(axis=0, keepdims=True)
        u, s, _vh = np.linalg.svd(z, full_matrices=False)
        coords = u[:, :2] * s[:2]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_{method}.png"

    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, sensor in enumerate(unique):
        idx = np.asarray([lab == sensor for lab in labels])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=18,
            alpha=0.75,
            label=f"{sensor} (n={idx.sum()})",
            color=cmap(i % 10),
        )
        if idx.sum() > 0:
            center = coords[idx].mean(axis=0)
            ax.scatter(center[0], center[1], s=120, marker="x", color=cmap(i % 10))
    ax.set_title(f"{name} {method.upper()} by sensor")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def embedding_metrics(x: np.ndarray, labels: Sequence[str]) -> dict:
    out = {
        "n_samples": int(x.shape[0]),
        "n_dims": int(x.shape[1]),
        "sensors": dict(Counter(labels)),
    }
    if x.shape[0] < 3 or len(set(labels)) < 2:
        return out

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        z = StandardScaler().fit_transform(x)
        y = LabelEncoder().fit_transform(labels)
        out["silhouette"] = float(silhouette_score(z, y))
        out["davies_bouldin"] = float(davies_bouldin_score(z, y))

        min_count = min(Counter(labels).values())
        n_splits = max(2, min(5, min_count))
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="auto",
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, z, y, cv=cv)
        out["sensor_cv_accuracy_mean"] = float(scores.mean())
        out["sensor_cv_accuracy_std"] = float(scores.std(ddof=0))
        out["sensor_cv_splits"] = int(n_splits)
    except Exception as e:
        out["sklearn_metrics_error"] = repr(e)

    # Centroid distances on standardized vectors.
    z = x.astype(np.float64)
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    centroids = {}
    spreads = {}
    for sensor in sorted(set(labels)):
        idx = np.asarray([lab == sensor for lab in labels])
        centroids[sensor] = z[idx].mean(axis=0)
        spreads[sensor] = float(np.linalg.norm(z[idx] - centroids[sensor], axis=1).mean())
    pairwise = {}
    for a in sorted(centroids):
        for b in sorted(centroids):
            if a >= b:
                continue
            key = f"{a}__{b}"
            dist = float(np.linalg.norm(centroids[a] - centroids[b]))
            denom = (spreads[a] + spreads[b]) / 2.0 + 1e-8
            pairwise[key] = {
                "centroid_distance": dist,
                "spread_normalized_distance": float(dist / denom),
            }
    out["centroid_spreads"] = spreads
    out["pairwise_centroid_distances"] = pairwise
    return out


def run_feature_diagnostics(
    records: Sequence[dict],
    config_path: Path,
    data_root: Path,
    image_size: int,
    output_dir: Path,
    max_per_sensor: int,
    batch_size: int,
    device_arg: str,
    seed: int,
    skip_tsne: bool,
) -> dict:
    selected = sample_balanced(records, max_per_sensor, seed)
    labels = [str(rec.get("sensor") or "UNKNOWN") for rec in selected]
    sample_ids = [str(rec.get("sample_id")) for rec in selected]

    print(f"[features] selected={len(selected)} by_sensor={Counter(labels)}", flush=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    vision_ckpt = cfg["model"]["vision_checkpoint"]
    model_cfg = cfg["model"]

    try:
        from ml_collections import ConfigDict

        model_cfg = ConfigDict(model_cfg)
    except Exception:
        pass

    device = resolve_device(device_arg)
    print(f"[features] building vision encoder on {device}", flush=True)
    vision = build_vision_encoder(model_cfg, vision_ckpt)
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    embeddings = {
        "dino_grid_gap": [],
        "conv_pyramid_gap": [],
        "fused_seq_mean": [],
    }

    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_records = selected[start : start + batch_size]
            images = [
                image_to_tensor(data_root / rec["image_path"], image_size)
                for rec in batch_records
            ]
            images_t = torch.stack(images, dim=0).to(device)
            seq, g_grid, pyramid_raw = vision.encode_with_spatial(images_t)
            dino = g_grid.float().mean(dim=(2, 3)).detach().cpu().numpy()
            conv = torch.cat(
                [feat.float().mean(dim=(2, 3)) for feat in pyramid_raw],
                dim=1,
            ).detach().cpu().numpy()
            fused = seq.float().mean(dim=1).detach().cpu().numpy()
            embeddings["dino_grid_gap"].append(dino)
            embeddings["conv_pyramid_gap"].append(conv)
            embeddings["fused_seq_mean"].append(fused)
            print(f"[features] batch {start // batch_size + 1}/{math.ceil(len(selected) / batch_size)}", flush=True)

    emb_arrays = {k: np.concatenate(v, axis=0) for k, v in embeddings.items()}
    npz_path = output_dir / "sensor_embeddings.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        labels=np.asarray(labels),
        sample_ids=np.asarray(sample_ids),
        **emb_arrays,
    )

    plot_dir = output_dir / "feature_plots"
    results = {
        "selected_samples": len(selected),
        "by_sensor": dict(Counter(labels)),
        "embeddings_npz": str(npz_path),
        "feature_sets": {},
    }
    for name, arr in emb_arrays.items():
        item = embedding_metrics(arr, labels)
        item["pca_plot"] = plot_embedding(name, arr, labels, plot_dir, method="pca")
        if not skip_tsne:
            item["tsne_plot"] = plot_embedding(name, arr, labels, plot_dir, method="tsne")
        results["feature_sets"][name] = item
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc_aqua_instance.yaml")
    parser.add_argument("--train-manifest", default="data/poc_aqua_full/train.json")
    parser.add_argument("--val-manifest", default="data/poc_aqua_full/val.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="outputs/diagnostics/gf6_feature_label")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-feature-per-sensor", type=int, default=120)
    parser.add_argument("--max-label-per-sensor", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-input-label", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    config_path = (repo / args.config).resolve()
    train_manifest = (repo / args.train_manifest).resolve()
    val_manifest = (repo / args.val_manifest).resolve()
    output_dir = (repo / args.output_dir).resolve()

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(args.data_root or cfg["data"]["data_root"]).resolve()

    manifest_paths = [train_manifest, val_manifest]
    records = load_manifests(manifest_paths)
    print(f"[main] records={len(records)} by_sensor={Counter(str(r.get('sensor') or 'UNKNOWN') for r in records)}", flush=True)
    print(f"[main] data_root={data_root}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_label = None
    if not args.skip_input_label:
        input_label = run_input_and_label_diagnostics(
            records=records,
            manifest_paths=manifest_paths,
            data_root=data_root,
            image_size=args.image_size,
            output_dir=output_dir,
            max_label_per_sensor=args.max_label_per_sensor,
            seed=args.seed,
        )
        write_json(output_dir / "input_label_diagnostics.json", input_label)

    feature_results = None
    if not args.skip_features:
        feature_results = run_feature_diagnostics(
            records=records,
            config_path=config_path,
            data_root=data_root,
            image_size=args.image_size,
            output_dir=output_dir,
            max_per_sensor=args.max_feature_per_sensor,
            batch_size=args.batch_size,
            device_arg=args.device,
            seed=args.seed,
            skip_tsne=args.skip_tsne,
        )
        write_json(output_dir / "feature_separability.json", feature_results)

    summary = {
        "config": str(config_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "input_label_diagnostics": str(output_dir / "input_label_diagnostics.json") if input_label else None,
        "feature_separability": str(output_dir / "feature_separability.json") if feature_results else None,
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
P3-B1 Oracle: Sea/Land mask from coastline GeoJSON → upper-bound coastline.

For each validation tile:
  1. Read coastline GeoJSON LineString(s)
  2. Rasterize lines to binary edge at tile resolution
  3. Flood-fill from tile borders to determine sea side vs land side
  4. Extract boundary: morphological dilation - erosion
  5. Evaluate against binary edge GT (standard metrics)

This answers: "If we had a perfect sea/land prior, how good would coastline detection be?"
"""

import json
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.coastline_dataset import build_coastline_manifest, IMAGE_SIZE
from utils.coastline_metrics import compute_all_edge_metrics
from utils.georef_transform import resize_georef


# =============================================================================
# GeoJSON → Sea/Land mask via flood-fill
# =============================================================================


def geojson_to_edge_mask(
    geojson_path: str,
    georef: dict,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Rasterize GeoJSON LineStrings to binary edge mask.

    Args:
        geojson_path: Path to GeoJSON FeatureCollection.
        georef: Georeference dict with model_transform.
        size: Output (H, W).

    Returns:
        edge_mask: [H, W] uint8 binary edge mask (1=coastline, 0=background).
    """
    from utils.georef_transform import wgs84_to_pixel

    H, W = size
    edge_mask = np.zeros((H, W), dtype=np.uint8)

    with open(geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("LineString", "MultiLineString"):
            continue

        coords_wgs84 = geom.get("coordinates", [])
        if geom["type"] == "LineString":
            lines = [coords_wgs84]
        else:
            lines = coords_wgs84

        for line in lines:
            if len(line) < 2:
                continue
            # Convert to pixel coordinates
            coords_pixel = wgs84_to_pixel(
                [(lon, lat) for lon, lat in line], georef
            )
            # Rasterize line using Bresenham-like drawing
            for i in range(len(coords_pixel) - 1):
                c0, r0 = coords_pixel[i]
                c1, r1 = coords_pixel[i + 1]
                _draw_line(edge_mask, int(r0), int(c0), int(r1), int(c1))

    return edge_mask


def _draw_line(img: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    """Bresenham line drawing on image in-place."""
    H, W = img.shape
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        if 0 <= r0 < H and 0 <= c0 < W:
            img[r0, c0] = 1
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def flood_fill_sea_land(
    edge_mask: np.ndarray,
    sea_border_fraction: float = 0.5,
) -> np.ndarray:
    """Flood-fill from tile borders to determine sea vs land side.

    Strategy:
      - The coastline edge_mask separates sea and land.
      - We flood-fill from each border pixel, treating edge_mask as barrier.
      - The side with more border pixels connected to open water is "sea".
      - If a single flood-fill covers most of the border, we assume the other
        side of the edge is sea (coastline tiles typically have sea on one side).

    Args:
        edge_mask: [H, W] binary edge (1=barrier, 0=passable).
        sea_border_fraction: Fraction of border pixels that should be "sea".

    Returns:
        sea_mask: [H, W] uint8 (1=sea, 0=land).
    """
    H, W = edge_mask.shape

    # Dilate edge to ensure it's a continuous barrier
    kernel = np.ones((3, 3), dtype=np.uint8)
    barrier = binary_dilation(edge_mask > 0, structure=kernel, iterations=2)

    # Collect border pixels not blocked by barrier
    border_pixels = []
    for r in range(H):
        if not barrier[r, 0]:
            border_pixels.append((r, 0))
        if not barrier[r, W - 1]:
            border_pixels.append((r, W - 1))
    for c in range(1, W - 1):
        if not barrier[0, c]:
            border_pixels.append((0, c))
        if not barrier[H - 1, c]:
            border_pixels.append((H - 1, c))

    # BFS flood-fill from border pixels
    visited = np.zeros((H, W), dtype=np.uint8)
    for start_r, start_c in border_pixels:
        if visited[start_r, start_c]:
            continue
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = 1
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not visited[nr, nc] and not barrier[nr, nc]:
                        visited[nr, nc] = 1
                        q.append((nr, nc))

    # The flood-filled region is "land" (connected to border).
    # The non-flooded region (inside the coastline) is "sea".
    sea_mask = (1 - visited).astype(np.uint8)

    # Sanity check: if sea is > 50% of tile, it's probably land
    # (coastline tiles are typically ~50/50 or land-dominant)
    sea_frac = sea_mask.mean()
    if sea_frac > 0.90 or sea_frac < 0.02:
        # Edge case: tile might be all sea or all land
        # Use simpler heuristic: everything on the non-border side of the edge
        pass

    return sea_mask


def sea_land_to_coastline(sea_mask: np.ndarray) -> np.ndarray:
    """Extract coastline boundary from sea/land mask.

    Uses morphological boundary: dilation(sea) XOR erosion(sea).

    Args:
        sea_mask: [H, W] binary sea mask.

    Returns:
        coastline: [H, W] binary edge (1=coastline boundary).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = binary_dilation(sea_mask, structure=kernel)
    eroded = binary_erosion(sea_mask, structure=kernel)
    boundary = (dilated.astype(np.uint8) != eroded.astype(np.uint8)).astype(np.uint8)
    return boundary


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_oracle(
    manifest_tiles: List[dict],
    output_dir: str = "",
) -> dict:
    """Run oracle evaluation: GeoJSON → sea/land → coastline → metrics.

    Returns dict with aggregate metrics and per-sample details.
    """
    from tqdm import tqdm

    all_metrics = []
    all_details = []
    failed = 0
    no_geojson = 0

    for tile in tqdm(manifest_tiles, desc="Oracle eval"):
        geojson_path = tile.get("label_geojson_path")
        if not geojson_path or not Path(geojson_path).exists():
            no_geojson += 1
            continue

        try:
            # Build georef
            original_size = tile.get("original_size", [128, 128])
            original_transform = tile.get("original_transform",
                                           [1e-5, 0, 0, 0, -1e-5, 0])
            model_transform, _ = resize_georef(
                tuple(original_size), (IMAGE_SIZE, IMAGE_SIZE),
                original_transform,
            )
            georef = {
                "source_crs": tile.get("source_crs", "EPSG:4326"),
                "model_transform": model_transform,
            }

            # GeoJSON → edge mask
            edge = geojson_to_edge_mask(geojson_path, georef, (IMAGE_SIZE, IMAGE_SIZE))

            if edge.sum() == 0:
                failed += 1
                continue

            # Flood-fill → sea/land
            sea = flood_fill_sea_land(edge)

            # Sea/land → coastline boundary
            coastline = sea_land_to_coastline(sea)

            # Load GT edge (from binary TIF)
            gt = _load_gt_edge(tile)

            # Compute metrics
            metrics = compute_all_edge_metrics(
                coastline.astype(np.float32),
                gt.astype(np.float32),
                threshold=0.5,
            )
            metrics["sample_id"] = tile.get("sample_id", "")
            metrics["sea_frac"] = float(sea.mean())
            all_metrics.append(metrics)
            all_details.append({
                "sample_id": tile["sample_id"],
                "sea_frac": float(sea.mean()),
                "edge_px": int(edge.sum()),
                "coastline_px": int(coastline.sum()),
            })

        except Exception as e:
            failed += 1
            continue

    if not all_metrics:
        return {"error": "No valid samples", "no_geojson": no_geojson, "failed": failed}

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            agg[key] = float(np.mean(vals))

    # FP distance analysis
    fp_dists = []
    for i, m in enumerate(all_metrics):
        tile = manifest_tiles[i]
        gt = _load_gt_edge(tile)
        if gt.sum() == 0:
            continue
        pred = np.zeros_like(gt)
        if all_details and i < len(all_details):
            # Reconstruct from stored info (simplified — use the coastline from sea/land)
            pass

    agg["n_samples"] = len(all_metrics)
    agg["failed"] = failed
    agg["no_geojson"] = no_geojson

    return {
        "aggregate": agg,
        "details": all_details,
    }


def _load_gt_edge(tile: dict) -> np.ndarray:
    """Load binary edge GT for a tile."""
    from PIL import Image

    binary_path = tile.get("binary_label_path")
    if binary_path and Path(binary_path).exists():
        binary = Image.open(binary_path).convert("L")
        binary = binary.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        return (np.array(binary) > 128).astype(np.float32)

    # Fallback: generate from GeoJSON
    geojson_path = tile.get("label_geojson_path")
    if geojson_path and Path(geojson_path).exists():
        original_size = tile.get("original_size", [128, 128])
        original_transform = tile.get("original_transform",
                                       [1e-5, 0, 0, 0, -1e-5, 0])
        model_transform, _ = resize_georef(
            tuple(original_size), (IMAGE_SIZE, IMAGE_SIZE),
            original_transform,
        )
        georef = {"source_crs": tile.get("source_crs", "EPSG:4326"),
                   "model_transform": model_transform}
        return geojson_to_edge_mask(geojson_path, georef, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


# =============================================================================
# FP distance analysis helper
# =============================================================================


def compute_fp_distance_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
) -> dict:
    """Compute FP distance statistics across all samples."""
    all_fd = []
    for pred, tgt in zip(predictions, targets):
        pb = pred > 0.5
        tb = tgt > 0.5
        fp = pb & (~tb)
        if fp.sum() == 0:
            continue
        d = distance_transform_edt(1 - tb.astype(np.uint8))
        all_fd.extend(d[fp].tolist())

    if not all_fd:
        return {}

    fd = np.array(all_fd)
    return {
        "fp_count": int(len(fd)),
        "fp_mean_dist": float(fd.mean()),
        "fp_median_dist": float(np.median(fd)),
        "fp_within_3px_pct": float((fd <= 3).mean() * 100),
        "fp_within_10px_pct": float((fd <= 10).mean() * 100),
        "fp_beyond_20px_pct": float((fd > 20).mean() * 100),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P3-B1 Oracle Sea/Land Evaluation")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit eval samples (0=all)")
    parser.add_argument("--output", type=str, default="outputs/poc3_semantic/b1_oracle")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    print("Building coastline manifest...")
    manifest = build_coastline_manifest(
        data_roots=[
            "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches",
            "/home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches",
        ],
        output_path=str(output_dir / "coastline_manifest.json"),
        val_ratio=0.2,
        seed=42,
    )

    val_tiles = manifest["val"]
    if args.max_samples > 0:
        val_tiles = val_tiles[:args.max_samples]

    print(f"Evaluating {len(val_tiles)} validation tiles...")

    # Run oracle evaluation per sample
    predictions = []
    targets = []
    f1_values = []
    no_geojson = 0
    failed = 0

    for idx, tile in enumerate(val_tiles):
        geojson_path = tile.get("label_geojson_path")
        if not geojson_path or not Path(geojson_path).exists():
            no_geojson += 1
            continue

        try:
            original_size = tile.get("original_size", [128, 128])
            original_transform = tile.get("original_transform",
                                           [1e-5, 0, 0, 0, -1e-5, 0])
            model_transform, _ = resize_georef(
                tuple(original_size), (IMAGE_SIZE, IMAGE_SIZE),
                original_transform,
            )
            georef = {
                "source_crs": tile.get("source_crs", "EPSG:4326"),
                "model_transform": model_transform,
            }

            edge = geojson_to_edge_mask(geojson_path, georef, (IMAGE_SIZE, IMAGE_SIZE))
            if edge.sum() == 0:
                failed += 1
                continue

            sea = flood_fill_sea_land(edge)
            coastline = sea_land_to_coastline(sea).astype(np.float32)
            gt = _load_gt_edge(tile)

            predictions.append(coastline)
            targets.append(gt)

            m = compute_all_edge_metrics(coastline, gt, threshold=0.5)
            f1_values.append(m.get("buffered_f1_1px", 0.0))

            if (idx + 1) % 50 == 0:
                mean_f1 = np.mean(f1_values[-50:])
                print(f"  {idx+1}/{len(val_tiles)}: running F1@1px={mean_f1:.4f}")

        except Exception as e:
            failed += 1
            continue

    N = len(predictions)
    if N == 0:
        print("No valid samples to evaluate.")
        return

    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"  P3-B1 ORACLE: Sea/Land → Coastline Upper Bound")
    print(f"  {'='*60}")
    print(f"  Valid samples: {N}")
    print(f"  No GeoJSON: {no_geojson}")

    mean_f1 = np.mean(f1_values)
    oracle_ceiling = np.mean([max(f1_values[i:i+1]) for i in range(N)])  # same as mean here since single threshold

    print(f"\n  Coastline F1@1px: {mean_f1:.4f}")

    # FP distance
    fp_stats = compute_fp_distance_metrics(predictions, targets)
    if fp_stats:
        print(f"  FP median distance: {fp_stats['fp_median_dist']:.1f} px")
        print(f"  FP beyond 20px: {fp_stats['fp_beyond_20px_pct']:.1f}%")
        print(f"  FP within 3px: {fp_stats['fp_within_3px_pct']:.1f}%")

    # Comparison with A2 baseline
    print(f"\n  {'Metric':<30} {'A2 edge':>12} {'Oracle SL':>12}")
    print(f"  {'-'*54}")
    print(f"  {'F1@1px':<30} {0.039:>12.4f} {mean_f1:>12.4f}")
    if fp_stats:
        print(f"  {'FP median dist (px)':<30} {51.5:>12.1f} {fp_stats['fp_median_dist']:>12.1f}")
        print(f"  {'FP beyond 20px (%)':<30} {79.4:>12.1f} {fp_stats['fp_beyond_20px_pct']:>12.1f}")
        print(f"  {'FP within 3px (%)':<30} {3.0:>12.1f} {fp_stats['fp_within_3px_pct']:>12.1f}")

    improvement = (mean_f1 - 0.039) / 0.039 * 100
    print(f"\n  Improvement over A2: {improvement:+.1f}%")
    if mean_f1 > 0.10:
        print("  VERDICT: Semantic direction strongly validated (F1 > 0.10)")
    elif mean_f1 > 0.06:
        print("  VERDICT: Semantic direction validated (F1 > 0.06)")
    else:
        print("  VERDICT: Sea/land prior alone insufficient — need better boundary extraction")

    # Save results
    results = {
        "oracle_f1_1px": float(mean_f1),
        "f1_values": [float(x) for x in f1_values],
        "fp_stats": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in fp_stats.items()} if fp_stats else {},
        "n_valid": N,
        "no_geojson": no_geojson,
        "failed": failed,
        "a2_baseline_f1": 0.039,
        "improvement_pct": float(improvement),
    }
    with open(output_dir / "b1_oracle_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_dir / 'b1_oracle_results.json'}")


if __name__ == "__main__":
    main()

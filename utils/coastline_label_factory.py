"""Unified coastline label factory — GeoJSON as single source of truth.

All label types derived from coastline GeoJSON LineString/MultiLineString:
  - Edge centerline (anti-aliased, width=1)
  - Soft edge target (Gaussian distance, sigma/radius configurable)
  - GDF target (truncated gravitational distance field)
  - Sea/land oracle mask (flood-fill from GeoJSON barrier)

Coordinate convention:
  row = y, col = x
  WGS84 [lon, lat] → pixel (col, row) via georef model_transform
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)


# =============================================================================
# GeoJSON → pixel lines
# =============================================================================


def wgs84_to_pixel_coords(
    coords_wgs84: List[Tuple[float, float]],
    model_transform: List[float],
) -> List[Tuple[float, float]]:
    """Convert WGS84 [lon, lat] → pixel (col, row).

    model_transform: [x_res, 0, x_origin, 0, -y_res, y_origin] (GDAL convention).
    col = (lon - x_origin) / x_res
    row = (y_origin - lat) / y_res
    """
    x_res, _, x_origin, _, y_res_neg, y_origin = model_transform
    y_res = -y_res_neg  # y_res is negative in GDAL

    pixels = []
    for lon, lat in coords_wgs84:
        col = (lon - x_origin) / x_res
        row = (y_origin - lat) / y_res
        pixels.append((col, row))
    return pixels


def geojson_to_pixel_lines(
    geojson_path: str,
    georef: dict,
    target_size: Tuple[int, int],
    densify_step: float = 0.5,
) -> List[List[Tuple[float, float]]]:
    """Extract all LineString/MultiLineString from GeoJSON, convert to pixel coords.

    Returns list of polylines, each a list of (col, row) float pixel coordinates.
    """
    H, W = target_size
    model_transform = georef.get("model_transform", [1e-5, 0, 0, 0, -1e-5, 0])

    with open(geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)

    all_lines = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("LineString", "MultiLineString"):
            continue
        coords = geom.get("coordinates", [])
        lines = [coords] if geom["type"] == "LineString" else coords
        for line in lines:
            if len(line) < 2:
                continue
            px = wgs84_to_pixel_coords(
                [(float(lon), float(lat)) for lon, lat in line],
                model_transform,
            )
            px = densify_polyline(px, max_step=densify_step)
            # Clip to valid range
            px = [(max(0.0, min(W - 0.001, c)), max(0.0, min(H - 0.001, r)))
                   for c, r in px]
            all_lines.append(px)

    return all_lines


def densify_polyline(
    points: List[Tuple[float, float]],
    max_step: float = 0.5,
) -> List[Tuple[float, float]]:
    """Insert intermediate points so consecutive points are within max_step."""
    if len(points) < 2:
        return points
    dense = []
    for i in range(len(points) - 1):
        c0, r0 = points[i]
        c1, r1 = points[i + 1]
        dense.append((c0, r0))
        dc, dr = c1 - c0, r1 - r0
        dist = math.sqrt(dc * dc + dr * dr)
        n = max(1, int(math.ceil(dist / max_step)))
        for j in range(1, n):
            t = j / n
            dense.append((c0 + t * dc, r0 + t * dr))
    dense.append(points[-1])
    return dense


# =============================================================================
# Edge centerline rasterization (anti-aliased)
# =============================================================================


def rasterize_centerline(
    pixel_lines: List[List[Tuple[float, float]]],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Rasterize polylines to anti-aliased edge centerline.

    Uses skimage.draw.line_aa for anti-aliased 1px lines.
    Returns [H, W] float32 in [0, 1].
    """
    from skimage.draw import line_aa

    H, W = target_size
    edge_map = np.zeros((H, W), dtype=np.float32)

    for line in pixel_lines:
        for i in range(len(line) - 1):
            c0, r0 = line[i]
            c1, r1 = line[i + 1]
            rr, cc, val = line_aa(int(r0), int(c0), int(r1), int(c1))
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            edge_map[rr[valid], cc[valid]] = np.maximum(
                edge_map[rr[valid], cc[valid]], val[valid]
            )

    return np.clip(edge_map, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Soft edge target
# =============================================================================


def build_soft_edge_target(
    centerline: np.ndarray,
    sigma: float = 1.0,
    radius: int = 3,
) -> np.ndarray:
    """Build soft edge target from centerline via Gaussian distance.

    soft = exp(-dist² / (2σ²)), truncated to 0 beyond `radius`.
    Edge pixels get value 1.0.
    """
    hard = (centerline > 0.5).astype(np.uint8)

    if hard.sum() == 0:
        return np.zeros_like(centerline, dtype=np.float32)

    d_fg = distance_transform_edt(1 - hard)
    soft = np.exp(-(d_fg ** 2) / (2.0 * sigma * sigma + 1e-8))
    soft[d_fg > radius] = 0.0
    soft[hard > 0] = 1.0

    return soft.astype(np.float32)


# =============================================================================
# GDF target
# =============================================================================


def build_gdf_target(
    centerline: np.ndarray,
    max_radius: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build truncated GDF target from edge centerline.

    Returns:
        target: [4, H, W] (dx_norm, dy_norm, log_dist_norm, valid)
        loss_mask: [1, H, W]
    """
    H, W = centerline.shape
    hard = (centerline > 0.5).astype(np.uint8)

    if hard.sum() == 0:
        target = np.zeros((4, H, W), dtype=np.float32)
        target[2] = 1.0
        return target, np.zeros((1, H, W), dtype=np.float32)

    dist, indices = distance_transform_edt(1 - hard, return_indices=True)
    nr = indices[0].astype(np.float32)
    nc = indices[1].astype(np.float32)

    rr, cc = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    dx = nc - cc
    dy = nr - rr
    valid = (dist <= max_radius).astype(np.float32)

    dx_norm = np.clip(dx / max_radius, -1.0, 1.0)
    dy_norm = np.clip(dy / max_radius, -1.0, 1.0)
    log_dist = np.log1p(np.minimum(dist, max_radius)) / math.log1p(max_radius)

    dx_norm[hard > 0] = 0.0
    dy_norm[hard > 0] = 0.0
    log_dist[hard > 0] = 0.0

    target = np.stack([dx_norm, dy_norm, log_dist, valid], axis=0).astype(np.float32)
    mask = valid[None, :, :].astype(np.float32)
    return target, mask


# =============================================================================
# Sea/Land oracle mask via flood-fill
# =============================================================================


def build_sealand_oracle_mask(
    pixel_lines: List[List[Tuple[float, float]]],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Build sea/land oracle mask from coastline GeoJSON via flood-fill.

    Rasterizes lines as barrier, flood-fills from tile borders.
    The non-flooded region is "sea". This is oracle-quality — not for production.

    Returns [H, W] uint8 (1=sea, 0=land).
    """
    H, W = target_size

    # Rasterize thick barrier
    barrier = np.zeros((H, W), dtype=np.uint8)
    for line in pixel_lines:
        for i in range(len(line) - 1):
            c0, r0 = int(line[i][0]), int(line[i][1])
            c1, r1 = int(line[i + 1][0]), int(line[i + 1][1])
            _bresenham_line(barrier, r0, c0, r1, c1)

    if barrier.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Dilate to ensure continuous barrier
    barrier = binary_dilation(barrier > 0, iterations=3).astype(np.uint8)

    # Flood-fill from borders
    visited = np.zeros((H, W), dtype=np.uint8)
    q = deque()
    for r in range(H):
        if not barrier[r, 0]:
            q.append((r, 0))
            visited[r, 0] = 1
        if not barrier[r, W - 1]:
            q.append((r, W - 1))
            visited[r, W - 1] = 1
    for c in range(1, W - 1):
        if not barrier[0, c]:
            q.append((0, c))
            visited[0, c] = 1
        if not barrier[H - 1, c]:
            q.append((H - 1, c))
            visited[H - 1, c] = 1

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr, nc] and not barrier[nr, nc]:
                    visited[nr, nc] = 1
                    q.append((nr, nc))

    # Non-visited region = sea (inside coastline polygon)
    sea = (1 - visited).astype(np.uint8)
    return sea


def _bresenham_line(img: np.ndarray, r0: int, c0: int, r1: int, c1: int):
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


# =============================================================================
# Sea/Land → coastline boundary
# =============================================================================


def sealand_to_coastline_boundary(sea_mask: np.ndarray) -> np.ndarray:
    """Extract coastline boundary from sea/land mask via morphological edge."""
    dilated = binary_dilation(sea_mask, structure=np.ones((3, 3), dtype=np.uint8))
    eroded = binary_erosion(sea_mask, structure=np.ones((3, 3), dtype=np.uint8))
    return (dilated.astype(np.uint8) != eroded.astype(np.uint8)).astype(np.float32)


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    print("=== Unified Label Factory Tests ===")

    # Test with simple horizontal line GeoJSON
    import tempfile, os

    H, W = 64, 64
    georef = {
        "source_crs": "EPSG:4326",
        "model_transform": [1.0 / W, 0, 0, 0, -1.0 / H, 0.5],
    }

    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[0.0, 0.0], [1.0, 0.0]]
            },
            "properties": {}
        }]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(gj, f)
        tmp = f.name

    # 1. Pixel lines
    lines = geojson_to_pixel_lines(tmp, georef, (H, W))
    print(f"[1] Pixel lines: {len(lines)} polylines, first has {len(lines[0])} pts")
    assert len(lines) > 0 and len(lines[0]) > 1

    # 2. Edge centerline
    edge = rasterize_centerline(lines, (H, W))
    print(f"[2] Edge centerline: {edge.sum():.1f} total intensity (expect >0)")
    assert edge.sum() > 0

    # 3. Soft edge
    soft = build_soft_edge_target(edge, sigma=1.0, radius=3)
    print(f"[3] Soft edge: max={soft.max():.1f}, fg_ratio={soft.mean():.4f}")
    assert soft.max() > 0.9

    # 4. GDF
    gdf, mask = build_gdf_target(edge, max_radius=32)
    print(f"[4] GDF: target={gdf.shape}, mask={mask.shape}, valid_pct={gdf[3].mean():.3f}")
    assert gdf.shape == (4, H, W)

    # 5. Sea/land oracle
    sea = build_sealand_oracle_mask(lines, (H, W))
    print(f"[5] Sea/land: sea_frac={sea.mean():.3f}")
    boundary = sealand_to_coastline_boundary(sea)
    print(f"    Boundary px: {boundary.sum():.0f}")

    os.unlink(tmp)
    print("\n=== ALL LABEL FACTORY TESTS PASSED ===")

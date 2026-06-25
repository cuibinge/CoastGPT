"""GDF postprocessing for PoC-3 A3: endpoint voting → vote map → GeoJSON.

Provides:
  - decode_field: Convert raw field predictions to interpretable values.
  - endpoint_voting: Accumulate pixel votes into a vote_map.
  - vote_map_to_edge: Extract binary edge from vote map (threshold + skeleton).
  - vote_map_to_geojson: Full pipeline to GeoJSON FeatureCollection.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Field decoding
# =============================================================================


def decode_field(
    field: np.ndarray,
    max_radius: int = 32,
) -> Dict[str, np.ndarray]:
    """Decode raw field prediction to physical quantities.

    Args:
        field: [4, H, W] or [B, 4, H, W] raw field prediction.
        max_radius: Truncation radius used in training.

    Returns:
        Dict with:
          'dx': [H, W] or [B, H, W] normalized [-1, 1]
          'dy': [H, W] or [B, H, W] normalized [-1, 1]
          'log_dist': [H, W] or [B, H, W] in [0, 1]
          'valid_prob': [H, W] or [B, H, W] in [0, 1]
          'vector_r': [H, W] or [B, H, W] unscaled vector in pixels
          'vector_c': [H, W] or [B, H, W] unscaled vector in pixels
    """
    squeeze = field.ndim == 3
    if squeeze:
        field = field[None, ...]  # [1, 4, H, W]

    B = field.shape[0]
    dx_raw = field[:, 0]   # [B, H, W]
    dy_raw = field[:, 1]
    ld_raw = field[:, 2]
    v_raw = field[:, 3]

    dx = np.tanh(dx_raw)                     # [-1, 1]
    dy = np.tanh(dy_raw)                     # [-1, 1]
    log_dist = 1.0 / (1.0 + np.exp(-ld_raw))  # sigmoid → [0, 1]
    valid_prob = 1.0 / (1.0 + np.exp(-v_raw))  # sigmoid → [0, 1]

    # Unnormalized vectors in pixel units
    vector_r = dy * max_radius
    vector_c = dx * max_radius

    result = {
        "dx": dx,
        "dy": dy,
        "log_dist": log_dist,
        "valid_prob": valid_prob,
        "vector_r": vector_r,
        "vector_c": vector_c,
    }

    if squeeze:
        result = {k: v[0] for k, v in result.items()}

    return result


# =============================================================================
# Endpoint voting
# =============================================================================


def endpoint_voting(
    field: np.ndarray,
    max_radius: int = 32,
    valid_threshold: float = 0.5,
    tau: float = 16.0,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Accumulate endpoint votes into a vote map.

    Each valid pixel votes for its endpoint: endpoint = pixel + vector * max_radius.
    Votes are bilinearly scattered and weighted by: valid_prob * exp(-dist / tau).

    Args:
        field: [4, H, W] raw field prediction (single sample).
        max_radius: Truncation radius used in training.
        valid_threshold: Minimum valid_prob to participate in voting.
        tau: Distance weight temperature (smaller = more local).
        smooth_sigma: Gaussian blur sigma for vote_map smoothing.

    Returns:
        vote_map: [H, W] float32 vote accumulation map.
    """
    decoded = decode_field(field, max_radius=max_radius)
    H, W = field.shape[1], field.shape[2]

    valid_prob = decoded["valid_prob"]       # [H, W]
    vector_r = decoded["vector_r"]            # [H, W]
    vector_c = decoded["vector_c"]            # [H, W]
    log_dist = decoded["log_dist"]            # [H, W]

    vote_map = np.zeros((H, W), dtype=np.float32)

    # Only vote from valid pixels
    valid_mask = valid_prob > valid_threshold

    if valid_mask.sum() == 0:
        return vote_map

    # Get coordinates of valid pixels
    rr, cc = np.where(valid_mask)
    vr = vector_r[rr, cc]
    vc = vector_c[rr, cc]
    vp = valid_prob[rr, cc]
    ld = log_dist[rr, cc]

    # Endpoint coordinates (floating point)
    ep_r = rr.astype(np.float32) + vr
    ep_c = cc.astype(np.float32) + vc

    # Vote weight
    weight = vp * np.exp(-ld * max_radius / max(tau, 1.0))

    # Bilinear scatter
    for i in range(len(rr)):
        r_float, c_float = ep_r[i], ep_c[i]
        w = weight[i]

        r0 = int(np.floor(r_float))
        c0 = int(np.floor(c_float))
        r1 = r0 + 1
        c1 = c0 + 1

        dr = r_float - r0
        dc = c_float - c0

        # 4-neighbor bilinear weights
        if 0 <= r0 < H and 0 <= c0 < W:
            vote_map[r0, c0] += w * (1 - dr) * (1 - dc)
        if 0 <= r0 < H and 0 <= c1 < W:
            vote_map[r0, c1] += w * (1 - dr) * dc
        if 0 <= r1 < H and 0 <= c0 < W:
            vote_map[r1, c0] += w * dr * (1 - dc)
        if 0 <= r1 < H and 0 <= c1 < W:
            vote_map[r1, c1] += w * dr * dc

    # Smooth
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        vote_map = gaussian_filter(vote_map, sigma=smooth_sigma)

    # Normalize to [0, 1]
    vmax = vote_map.max()
    if vmax > 0:
        vote_map = vote_map / vmax

    return vote_map.astype(np.float32)


# =============================================================================
# Vote map → edge
# =============================================================================


def vote_map_to_edge(
    vote_map: np.ndarray,
    threshold: float = 0.3,
    min_area: int = 8,
) -> np.ndarray:
    """Extract binary edge from vote map.

    Args:
        vote_map: [H, W] vote accumulation map in [0, 1].
        threshold: Threshold for binary edge.
        min_area: Minimum connected component area in pixels.

    Returns:
        binary_edge: [H, W] uint8 binary edge map.
    """
    from scipy.ndimage import label

    binary = (vote_map > threshold).astype(np.uint8)

    if min_area > 1:
        labeled, n_features = label(binary)
        for i in range(1, n_features + 1):
            if (labeled == i).sum() < min_area:
                binary[labeled == i] = 0

    return binary


def vote_map_to_skeleton(vote_map: np.ndarray, **kwargs) -> np.ndarray:
    """Convert vote_map to skeleton.

    Pipeline: threshold → remove small components → skeletonize.
    """
    from skimage.morphology import skeletonize

    binary = vote_map_to_edge(vote_map, **kwargs)
    if binary.sum() == 0:
        return binary
    return skeletonize(binary.astype(bool)).astype(np.uint8)


# =============================================================================
# Full pipeline → GeoJSON
# =============================================================================


def vote_map_to_geojson(
    vote_map: np.ndarray,
    georef: dict,
    threshold: float = 0.3,
    min_length: int = 10,
    max_components: int = 5,
    simplify_epsilon: float = 1.0,
    sample_id: str = "",
) -> dict:
    """Convert vote_map to GeoJSON FeatureCollection.

    Pipeline: vote_map → threshold → skeleton → path → simplify → GeoJSON.

    Args:
        vote_map: [H, W] vote accumulation map.
        georef: Georeference dict with 'model_transform', 'source_crs'.
        threshold: Vote map threshold.
        min_length: Minimum polyline length in pixels.
        max_components: Maximum number of polyline components.
        simplify_epsilon: Douglas-Peucker epsilon in pixels.
        sample_id: Sample identifier for properties.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    try:
        import sys
        from pathlib import Path
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from utils.edge_postprocess import (
            heatmap_to_binary,
            binary_to_skeleton,
            skeleton_to_paths,
            simplify_paths,
        )
    except ImportError:
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": "edge_postprocess not available",
        }

    binary = heatmap_to_binary(vote_map, threshold=threshold, min_area=8)
    skeleton = binary_to_skeleton(binary)
    paths = skeleton_to_paths(skeleton)
    paths = simplify_paths(paths, epsilon=simplify_epsilon)
    paths = sorted(paths, key=len, reverse=True)[:max_components]
    paths = [p for p in paths if len(p) >= min_length]

    # Convert pixel paths to WGS84 coordinates
    model_transform = georef.get("model_transform", [1e-5, 0, 0, 0, -1e-5, 0])

    features = []
    for path in paths:
        coords = []
        for col, row in path:
            lon = model_transform[0] * col + model_transform[2]
            lat = model_transform[4] * row + model_transform[5]
            coords.append([float(lon), float(lat)])

        if len(coords) < 2:
            continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "sample_id": sample_id,
                "source": "gdf_voting",
                "num_points": len(coords),
                "threshold": threshold,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": georef.get("source_crs", "EPSG:4326")}},
    }


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    print("=== GDF Postprocess Unit Tests ===")

    from scipy.ndimage import distance_transform_edt

    H, W = 64, 64
    R = 32

    # Create perfect GDF target for a vertical line
    edge = np.zeros((H, W), dtype=np.float32)
    edge[:, 32] = 1.0
    non_edge = 1 - edge.astype(np.uint8)
    dist, indices = distance_transform_edt(non_edge, return_indices=True)
    nr, nc = indices[0].astype(np.float32), indices[1].astype(np.float32)
    rr, cc = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    dx = np.clip((nc - cc) / R, -1, 1)
    dy = np.clip((nr - rr) / R, -1, 1)
    ld = np.log1p(np.minimum(dist, R)) / np.log1p(R)
    valid = (dist <= R).astype(np.float32)
    dx[edge > 0] = 0; dy[edge > 0] = 0; ld[edge > 0] = 0

    # Convert to raw field (apply inverse activations)
    eps = 1e-6
    dx_r = np.clip(dx, -1 + eps, 1 - eps)
    dy_r = np.clip(dy, -1 + eps, 1 - eps)
    ld_r = np.clip(ld, eps, 1 - eps)
    field_raw = np.stack([
        np.arctanh(dx_r),
        np.arctanh(dy_r),
        np.log(ld_r / (1 - ld_r)),
        10.0 * valid - 10.0 * (1 - valid),  # logit for valid
    ], axis=0).astype(np.float32)

    # Test 1: Decode field
    print("\n[1] Decode field...")
    decoded = decode_field(field_raw, max_radius=R)
    assert abs(decoded["dx"][16, 32]).max() < 0.05, "Edge dx should be ~0"
    assert abs(decoded["dy"][16, 32]).max() < 0.05, "Edge dy should be ~0"
    print(f"  Edge vector: ({decoded['dx'][16,32]:.3f}, {decoded['dy'][16,32]:.3f})")
    print("  ✓")

    # Test 2: Perfect field voting (Gate 1)
    print("\n[2] Endpoint voting...")
    vote_map = endpoint_voting(field_raw, max_radius=R, valid_threshold=0.5, tau=16)
    # Peak should be along column 32
    col_profile = vote_map.sum(axis=0)
    peak_col = np.argmax(col_profile)
    print(f"  Vote map peak column: {peak_col} (expected ~32)")
    assert abs(peak_col - 32) <= 2, f"Peak offset: {peak_col}"
    print("  ✓")

    # Test 3: Vote map to edge
    print("\n[3] Vote map → binary edge...")
    binary = vote_map_to_edge(vote_map, threshold=0.1, min_area=4)
    print(f"  Binary foreground: {binary.sum()} px")
    assert binary.sum() > 0, "Should have foreground pixels"
    print("  ✓")

    # Test 4: Vote map to skeleton
    print("\n[4] Vote map → skeleton...")
    skeleton = vote_map_to_skeleton(vote_map, threshold=0.1, min_area=4)
    print(f"  Skeleton foreground: {skeleton.sum()} px")
    print("  ✓")

    # Test 5: Vote map to GeoJSON
    print("\n[5] Vote map → GeoJSON...")
    georef = {"source_crs": "EPSG:4326", "model_transform": [1e-5, 0, 0, 0, -1e-5, 0]}
    fc = vote_map_to_geojson(vote_map, georef, threshold=0.1, sample_id="test_vline")
    print(f"  Features: {len(fc.get('features', []))}")
    print("  ✓")

    print("\n=== ALL GDF POSTPROCESS TESTS PASSED ===")

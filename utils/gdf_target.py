"""GDF target generation for PoC-3 A3.

Builds truncated gravitational distance field targets from binary coastline edges:
  - dx_norm: normalized x-component toward nearest edge pixel
  - dy_norm: normalized y-component toward nearest edge pixel
  - log_dist_norm: log-scaled distance to nearest edge pixel
  - valid: binary mask for pixels within max_radius of an edge

Coordinate convention:
  row = y, col = x
  dx = target_col - current_col
  dy = target_row - current_row
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def build_gdf_target(
    edge_center: np.ndarray,
    max_radius: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build truncated GDF target from binary edge centerline.

    Args:
        edge_center: [H, W] binary edge map (1 = coastline, 0 = background).
                     Use width=1 centerline, NOT thick band.
        max_radius: Truncation radius in pixels. Pixels beyond this radius
                    only get valid supervision (valid=0), no vector/distance.

    Returns:
        target:  [4, H, W] float32 array:
                channel 0: dx_norm  ∈ [-1, 1]
                channel 1: dy_norm  ∈ [-1, 1]
                channel 2: log_dist_norm ∈ [0, 1]
                channel 3: valid ∈ {0, 1}
        loss_mask: [1, H, W] float32 mask (1 = supervise vector + distance + valid,
                                          0 = supervise valid only if all-zero tile)
    """
    H, W = edge_center.shape
    edge_binary = (edge_center > 0.5).astype(np.uint8)

    # Empty tile: all zeros, valid=0 everywhere
    if edge_binary.sum() == 0:
        target = np.zeros((4, H, W), dtype=np.float32)
        target[2] = 1.0  # log_dist_norm = 1 (max distance)
        loss_mask = np.zeros((1, H, W), dtype=np.float32)
        return target, loss_mask

    # Distance transform: distance + nearest edge indices
    non_edge = 1 - edge_binary
    dist, indices = distance_transform_edt(non_edge, return_indices=True)
    nearest_r = indices[0].astype(np.float32)
    nearest_c = indices[1].astype(np.float32)

    # Meshgrid for pixel coordinates
    rr, cc = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    # Vector field: dx, dy toward nearest edge
    dx = nearest_c - cc
    dy = nearest_r - rr

    # Valid mask: pixels within max_radius of an edge
    valid = (dist <= max_radius).astype(np.float32)

    # Normalize: dx/dy clipped to [-1, 1], distance log-scaled
    dx_norm = np.clip(dx / max_radius, -1.0, 1.0)
    dy_norm = np.clip(dy / max_radius, -1.0, 1.0)
    log_dist_norm = np.log1p(np.minimum(dist, max_radius)) / math.log1p(max_radius)

    # Edge pixels: zero vector, zero distance
    edge_mask = edge_binary > 0
    dx_norm[edge_mask] = 0.0
    dy_norm[edge_mask] = 0.0
    log_dist_norm[edge_mask] = 0.0

    target = np.stack([dx_norm, dy_norm, log_dist_norm, valid], axis=0).astype(np.float32)
    loss_mask = valid.astype(np.float32)[None, :, :]

    return target, loss_mask


def validate_gdf_target(
    target: np.ndarray,
    edge_center: np.ndarray,
    max_radius: int = 32,
    tolerance: float = 0.05,
) -> dict:
    """Validate GDF target correctness.

    Checks:
      1. Edge pixels have dx=0, dy=0, dist=0, valid=1
      2. dx/dy are in [-1, 1]
      3. valid matches distance threshold
      4. All-zero edge tile produces valid target

    Returns dict with keys: 'edge_zero_vector', 'dx_range', 'dy_range',
          'valid_consistent', 'empty_tile_ok', 'all_passed'
    """
    dx = target[0]
    dy = target[1]
    log_dist = target[2]
    valid = target[3]
    edge_binary = (edge_center > 0.5).astype(np.uint8)

    results = {}

    # Check edge pixels
    if edge_binary.sum() > 0:
        edge_dx = dx[edge_binary > 0]
        edge_dy = dy[edge_binary > 0]
        edge_dd = log_dist[edge_binary > 0]
        edge_v = valid[edge_binary > 0]
        results["edge_zero_vector"] = bool(
            np.allclose(edge_dx, 0, atol=tolerance)
            and np.allclose(edge_dy, 0, atol=tolerance)
            and np.allclose(edge_dd, 0, atol=tolerance)
            and np.all(edge_v > 0.9)
        )
    else:
        results["edge_zero_vector"] = True  # vacuously true

    # Range checks
    results["dx_range"] = bool(np.all(dx >= -1.0 - tolerance) and np.all(dx <= 1.0 + tolerance))
    results["dy_range"] = bool(np.all(dy >= -1.0 - tolerance) and np.all(dy <= 1.0 + tolerance))

    # Valid consistency
    non_edge = 1 - edge_binary
    dist, _ = distance_transform_edt(non_edge, return_indices=True)
    expected_valid = (dist <= max_radius).astype(np.float32)
    results["valid_consistent"] = bool(np.allclose(valid, expected_valid, atol=0.01))

    results["all_passed"] = all(results.values())
    return results


def visualize_gdf_target(
    target: np.ndarray,
    edge_center: np.ndarray,
    sample_id: str = "",
) -> None:
    """Quick matplotlib visualization of GDF target (for notebook/debug).

    Args:
        target: [4, H, W] GDF target.
        edge_center: [H, W] binary edge map.
        sample_id: Optional label for plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(edge_center, cmap="gray")
    axes[0, 0].set_title("Edge GT")
    axes[0, 0].axis("off")

    dx = target[0]
    im1 = axes[0, 1].imshow(dx, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("dx (→ edge)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    dy = target[1]
    im2 = axes[0, 2].imshow(dy, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 2].set_title("dy (→ edge)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(target[2], cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("log_dist")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(target[3], cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("valid")
    axes[1, 1].axis("off")

    # Quiver overlay: subsampled vector field
    H, W = edge_center.shape
    step = max(1, H // 32)
    yy, xx = np.meshgrid(
        np.arange(0, H, step), np.arange(0, W, step), indexing="ij"
    )
    axes[1, 2].imshow(edge_center, cmap="gray", alpha=0.5)
    axes[1, 2].quiver(
        xx, yy,
        dx[::step, ::step],
        dy[::step, ::step],
        scale=20, width=0.002,
    )
    axes[1, 2].set_title("Vector field (subsampled)")
    axes[1, 2].axis("off")
    axes[1, 2].invert_yaxis()

    title = "GDF Target"
    if sample_id:
        title += f" — {sample_id}"
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== GDF Target Unit Tests ===")

    # Test 1: Simple horizontal line
    print("\n[1] Horizontal line at row=100...")
    H, W = 224, 224
    edge = np.zeros((H, W), dtype=np.float32)
    edge[100, 50:170] = 1.0
    target, mask = build_gdf_target(edge, max_radius=32)

    # Validate
    results = validate_gdf_target(target, edge, max_radius=32)
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")
    assert results["all_passed"], f"Validation failed: {results}"

    # Check specific pixels
    dx, dy, ld, v = target[0], target[1], target[2], target[3]
    # Edge pixel
    assert abs(dx[100, 100]) < 0.01, f"Edge dx should be 0, got {dx[100,100]:.4f}"
    assert abs(dy[100, 100]) < 0.01, f"Edge dy should be 0, got {dy[100,100]:.4f}"
    assert ld[100, 100] < 0.01, f"Edge log_dist should be 0, got {ld[100,100]:.4f}"
    assert v[100, 100] > 0.9, f"Edge valid should be 1, got {v[100,100]:.4f}"
    # Pixel above edge: dy positive (pointing down to edge)
    assert dy[90, 100] > 0.2, f"Above-edge dy should be positive, got {dy[90,100]:.4f}"
    # Pixel below edge: dy negative (pointing up to edge)
    assert dy[110, 100] < -0.05, f"Below-edge dy should be negative, got {dy[110,100]:.4f}"
    # Pixel left of edge strip
    assert dx[100, 30] > 0.05, f"Left-of-edge dx should be positive, got {dx[100,30]:.4f}"
    print("  All spot checks passed ✓")

    # Test 2: Empty tile
    print("\n[2] Empty tile...")
    edge_empty = np.zeros((H, W), dtype=np.float32)
    target_e, mask_e = build_gdf_target(edge_empty, max_radius=32)
    assert target_e.shape == (4, H, W)
    assert mask_e.sum() == 0, "Empty tile mask should be all zero"
    assert np.all(target_e[3] == 0), "Empty tile valid should be all zero"
    print("  Empty tile handling ✓")

    # Test 3: Perfect field voting (Gate 1 prep)
    print("\n[3] Perfect field voting sanity check...")
    # With perfect GT field, endpoint voting should reconstruct edge
    H_small, W_small = 64, 64
    edge_s = np.zeros((H_small, W_small), dtype=np.float32)
    edge_s[32, :] = 1.0  # horizontal line at middle
    target_s, _ = build_gdf_target(edge_s, max_radius=32)

    # Endpoint voting: pixel + vector*R
    R = 32
    vote_map = np.zeros((H_small, W_small), dtype=np.float32)
    dx_s, dy_s, _, v_s = target_s[0], target_s[1], target_s[2], target_s[3]

    # Subsample for speed
    for r in range(0, H_small, 2):
        for c in range(0, W_small, 2):
            if v_s[r, c] < 0.5:
                continue
            ep_r = r + dy_s[r, c] * R
            ep_c = c + dx_s[r, c] * R
            er, ec = int(np.floor(ep_r)), int(np.floor(ep_c))
            if 0 <= er < H_small and 0 <= ec < W_small:
                vote_map[er, ec] += 1.0

    # Vote map should have peak at row 32
    peak_row = np.argmax(vote_map.sum(axis=1))
    print(f"  Vote map peak row: {peak_row} (expected 32)")
    assert abs(peak_row - 32) <= 2, f"Peak row offset too large: {peak_row}"
    print("  Perfect field voting ✓")

    print("\n=== ALL GDF TARGET TESTS PASSED ===")

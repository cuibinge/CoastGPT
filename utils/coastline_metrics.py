"""Coastline edge detection metrics for PoC-3.

Provides:
  - pixel_edge_metrics: pixel-level precision/recall/F1 on edge maps.
  - buffered_f1: geometry-buffered F1 score.
  - chamfer_distance: mean nearest-neighbor distance.
  - hausdorff_distance: max nearest-neighbor distance.
"""

import numpy as np
from typing import Dict, List, Tuple


def pixel_edge_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> Dict[str, float]:
    """Pixel-level edge detection metrics.

    Args:
        pred_binary: [H, W] uint8 binary edge prediction.
        gt_binary: [H, W] uint8 binary edge GT.

    Returns:
        Dict with precision, recall, f1, foreground_ratio_pred, foreground_ratio_gt.
    """
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pred_fg_ratio": float(pred.mean()),
        "gt_fg_ratio": float(gt.mean()),
    }


def _coords_from_binary(binary: np.ndarray) -> np.ndarray:
    """Extract (row, col) coordinates of foreground pixels."""
    return np.argwhere(binary.astype(bool)).astype(np.float32)


def chamfer_distance(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
) -> float:
    """Chamfer distance: mean of nearest-neighbor distances.

    Args:
        pred_coords: [N, 2] predicted (row, col) coordinates.
        gt_coords: [M, 2] GT (row, col) coordinates.

    Returns:
        Mean Chamfer distance in pixels.
    """
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 0.0
    if len(pred_coords) == 0:
        return float(np.inf)
    if len(gt_coords) == 0:
        return float(np.inf)

    diff_p2g = pred_coords[:, None, :] - gt_coords[None, :, :]  # [N, M, 2]
    dist_p2g = np.sqrt((diff_p2g ** 2).sum(axis=2)).min(axis=1)  # [N]

    diff_g2p = gt_coords[:, None, :] - pred_coords[None, :, :]  # [M, N, 2]
    dist_g2p = np.sqrt((diff_g2p ** 2).sum(axis=2)).min(axis=1)  # [M]

    return float((dist_p2g.mean() + dist_g2p.mean()) / 2.0)


def hausdorff_distance(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Hausdorff distance (default: 95th percentile, i.e. robust HD).

    Args:
        pred_coords: [N, 2] predicted coordinates.
        gt_coords: [M, 2] GT coordinates.
        percentile: Percentile for robust HD. 100 = standard HD.

    Returns:
        Hausdorff distance in pixels.
    """
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float(np.inf)

    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))  # [N, M]
    p2g = dists.min(axis=1)  # [N]
    g2p = dists.min(axis=0)  # [M]
    all_dists = np.concatenate([p2g, g2p])

    if percentile >= 100:
        return float(all_dists.max())
    return float(np.percentile(all_dists, percentile))


def buffered_f1(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    buffer_px: int = 1,
) -> Dict[str, float]:
    """Compute buffered-F1 by dilating GT and prediction.

    Args:
        pred_binary: [H, W] uint8 edge prediction.
        gt_binary: [H, W] uint8 edge GT.
        buffer_px: Buffer radius in pixels.

    Returns:
        Dict with precision, recall, f1.
    """
    from scipy.ndimage import binary_dilation

    kernel = _disk_kernel(buffer_px)

    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    gt_buf = binary_dilation(gt, structure=kernel) if buffer_px > 0 else gt
    pred_buf = binary_dilation(pred, structure=kernel) if buffer_px > 0 else pred

    tp = (pred & gt_buf).sum()
    fp = (pred & ~gt_buf).sum()
    fn = (~pred_buf & gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        f"buffered_precision_{buffer_px}px": float(precision),
        f"buffered_recall_{buffer_px}px": float(recall),
        f"buffered_f1_{buffer_px}px": float(f1),
    }


def _disk_kernel(radius: int) -> np.ndarray:
    """Create a disk-shaped structuring element."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def compute_all_edge_metrics(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """Compute all edge metrics from heatmaps.

    Args:
        pred_heatmap: [H, W] sigmoid prediction.
        gt_heatmap: [H, W] binary GT edge map (0/1 or 0/255).
        threshold: Binarization threshold for pred.

    Returns:
        Combined metrics dict.
    """
    pred_bin = (pred_heatmap >= threshold).astype(np.uint8)
    gt_bin = (gt_heatmap > 0.5).astype(np.uint8) if gt_heatmap.max() > 1 else gt_heatmap.astype(np.uint8)

    metrics = {}
    metrics.update(pixel_edge_metrics(pred_bin, gt_bin))

    pred_coords = _coords_from_binary(pred_bin)
    gt_coords = _coords_from_binary(gt_bin)

    metrics["chamfer_distance_px"] = chamfer_distance(pred_coords, gt_coords)
    metrics["hausdorff_95_px"] = hausdorff_distance(pred_coords, gt_coords, percentile=95)
    metrics["hausdorff_100_px"] = hausdorff_distance(pred_coords, gt_coords, percentile=100)

    for buf in [1, 3]:
        metrics.update(buffered_f1(pred_bin, gt_bin, buffer_px=buf))

    metrics["pred_coord_count"] = len(pred_coords)
    metrics["gt_coord_count"] = len(gt_coords)

    return metrics


if __name__ == "__main__":
    # Smoke test on synthetic data
    gt = np.zeros((224, 224), dtype=np.uint8)
    gt[100:120, 50:170] = 1

    pred = np.zeros((224, 224), dtype=np.float32)
    pred[100:120, 50:170] = 0.8
    pred[102:118, 55:165] = 0.9

    metrics = compute_all_edge_metrics(pred, gt, threshold=0.3)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Metrics smoke test passed.")

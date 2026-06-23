"""Edge heatmap postprocessing for PoC-3 coastline detection.

Converts edge logits → binary mask → skeleton → polylines → GeoJSON.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.measure import label as connected_label
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def heatmap_to_binary(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 8,
) -> np.ndarray:
    """Convert sigmoid heatmap to binary mask.

    Args:
        heatmap: [H, W] float32 in [0, 1].
        threshold: Binarization threshold.
        min_area: Minimum connected component area in pixels.

    Returns:
        binary: [H, W] uint8 binary mask.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    if min_area > 0 and HAS_SKIMAGE:
        binary = remove_small_objects(binary.astype(bool), min_size=min_area, connectivity=2)
        binary = binary.astype(np.uint8)
    return binary


def binary_to_skeleton(binary: np.ndarray) -> np.ndarray:
    """Skeletonize binary edge mask.

    Args:
        binary: [H, W] uint8 binary mask.

    Returns:
        skeleton: [H, W] uint8 skeleton (1px wide).
    """
    if HAS_SKIMAGE:
        return skeletonize(binary.astype(bool)).astype(np.uint8)
    else:
        raise ImportError("scikit-image required: pip install scikit-image")


def skeleton_to_paths(
    skeleton: np.ndarray,
    min_length: int = 10,
    max_components: int = 5,
) -> List[List[Tuple[float, float]]]:
    """Extract polylines from skeleton image via connected components.

    Each connected component is traced into an ordered path.
    Short components (< min_length) are dropped.

    Args:
        skeleton: [H, W] uint8 skeleton.
        min_length: Minimum path length in pixels.
        max_components: Maximum number of paths to return (top-k by length).

    Returns:
        List of paths, each path is a list of (row, col) pixel coordinates.
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required")

    labels = connected_label(skeleton, connectivity=2, background=0)
    paths = []

    for label_id in range(1, labels.max() + 1):
        mask = (labels == label_id)
        coords = np.argwhere(mask)
        if len(coords) < min_length:
            continue
        path = _trace_component(mask)
        if len(path) >= 2:
            paths.append(path)

    paths.sort(key=len, reverse=True)
    paths = paths[:max_components]
    paths = [p for p in paths if len(p) >= min_length]
    return paths


def _trace_component(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Trace a single skeleton component into an ordered path.

    Finds an endpoint (pixel with 1 neighbor), then follows the path.

    Args:
        mask: [H, W] boolean mask of the component.

    Returns:
        Ordered list of (row, col) coordinates.
    """
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return [(float(r), float(c)) for r, c in coords]

    h, w = mask.shape
    neighbors = {}
    for r, c in coords:
        key = (int(r), int(c))
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    nbrs.append((nr, nc))
        neighbors[key] = nbrs

    if not neighbors:
        return []

    endpoints = [k for k, v in neighbors.items() if len(v) == 1]
    if not endpoints:
        endpoints = [min(neighbors.keys(), key=lambda k: len(neighbors[k]))]

    start = endpoints[0]
    path = [start]
    visited = {start}

    current = start
    while True:
        nbrs = [n for n in neighbors.get(current, []) if n not in visited]
        if not nbrs:
            break
        if len(path) >= 2:
            prev = path[-2]
            dr_prev = current[0] - prev[0]
            dc_prev = current[1] - prev[1]
            nbrs.sort(
                key=lambda n: abs((n[0] - current[0]) - dr_prev)
                + abs((n[1] - current[1]) - dc_prev)
            )
        next_px = nbrs[0]
        path.append(next_px)
        visited.add(next_px)
        current = next_px

    return [(float(r), float(c)) for r, c in path]


def simplify_path(
    path: List[Tuple[float, float]],
    epsilon: float = 1.0,
) -> List[Tuple[float, float]]:
    """Douglas-Peucker simplification.

    Args:
        path: List of (row, col) pixel coordinates.
        epsilon: Maximum distance in pixels.

    Returns:
        Simplified path.
    """
    if len(path) <= 2:
        return path

    dmax = 0.0
    index = 0
    end = len(path) - 1

    for i in range(1, end):
        d = _perpendicular_distance(path[i], path[0], path[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = simplify_path(path[:index + 1], epsilon)
        right = simplify_path(path[index:], epsilon)
        return left[:-1] + right
    else:
        return [path[0], path[-1]]


def _perpendicular_distance(
    pt: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Distance from pt to line segment (line_start, line_end)."""
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    if dx == 0 and dy == 0:
        return math.sqrt((pt[0] - line_start[0]) ** 2 + (pt[1] - line_start[1]) ** 2)

    t = ((pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj = (line_start[0] + t * dx, line_start[1] + t * dy)
    return math.sqrt((pt[0] - proj[0]) ** 2 + (pt[1] - proj[1]) ** 2)


def paths_to_geojson(
    paths: List[List[Tuple[float, float]]],
    georef: dict,
    sample_id: str = "",
    class_name: str = "海岸线",
) -> dict:
    """Convert pixel paths to GeoJSON FeatureCollection.

    Args:
        paths: List of paths, each a list of (row, col) pixel coords.
        georef: Dict with 'model_transform', 'source_crs'.
        sample_id: Sample identifier.
        class_name: Feature class name.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    from utils.georef_transform import pixel_to_wgs84

    if not paths:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for path in paths:
        pixel_coords = [(col, row) for row, col in path]
        wgs84_coords = pixel_to_wgs84(pixel_coords, georef)
        coords = [[lon, lat] for lon, lat in wgs84_coords]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "class": class_name,
                "sample_id": sample_id,
                "length_px": len(path),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def postprocess_edge(
    heatmap: np.ndarray,
    georef: dict,
    threshold: float = 0.5,
    min_area: int = 8,
    min_length: int = 10,
    max_components: int = 5,
    simplify_epsilon: float = 1.0,
    sample_id: str = "",
) -> dict:
    """Full edge postprocessing pipeline: heatmap → GeoJSON FeatureCollection.

    Args:
        heatmap: [H, W] sigmoid probabilities.
        georef: Georeference dict for pixel→WGS84.
        threshold: Binarization threshold.
        min_area: Minimum component area.
        min_length: Minimum LineString length in pixels.
        max_components: Maximum number of paths.
        simplify_epsilon: Douglas-Peucker epsilon in pixels.
        sample_id: Sample identifier.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    binary = heatmap_to_binary(heatmap, threshold=threshold, min_area=min_area)
    skeleton = binary_to_skeleton(binary)
    paths = skeleton_to_paths(
        skeleton, min_length=min_length, max_components=max_components
    )
    paths = [simplify_path(p, epsilon=simplify_epsilon) for p in paths]
    paths = [p for p in paths if len(p) >= 2]

    return paths_to_geojson(paths, georef, sample_id=sample_id)


if __name__ == "__main__":
    print("Testing edge_postprocess...")
    if not HAS_SKIMAGE:
        print("WARNING: scikit-image not installed, skeletonize disabled")

    # Create a synthetic heatmap with a diagonal line
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for i in range(50, 170):
        heatmap[i, i] = 0.9
        heatmap[i, i + 1] = 0.3

    if HAS_SKIMAGE:
        binary = heatmap_to_binary(heatmap, threshold=0.5, min_area=4)
        print(f"Binary: fg_px={binary.sum()}, max={binary.max()}")

        skeleton = binary_to_skeleton(binary)
        print(f"Skeleton: fg_px={skeleton.sum()}")

        paths = skeleton_to_paths(skeleton, min_length=5, max_components=3)
        print(f"Paths: {len(paths)} components")
        for i, p in enumerate(paths):
            print(f"  Path {i}: {len(p)} points, start={p[0]}, end={p[-1]}")

        simplified = [simplify_path(p, epsilon=1.0) for p in paths]
        for i, p in enumerate(simplified):
            print(f"  Simplified {i}: {len(p)} points")

    print("Done.")

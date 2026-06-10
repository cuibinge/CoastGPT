"""Pixel-space geometry utilities: rasterization, bbox, polygon extraction.

All coordinates in model pixel space (224x224). No WGS84/CRS logic here.
"""
import numpy as np
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def polygon_to_bbox(polygon: List[Tuple[float, float]]) -> Optional[List[int]]:
    """
    Compute xyxy bbox from polygon vertices.
    Args:
        polygon: [(col, row), ...] in pixel space
    Returns:
        [x1, y1, x2, y2] or None if degenerate
    """
    if len(polygon) < 3:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, x2 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y1, y2 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    return [x1, y1, x2, y2]


def bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    """
    Compute xyxy bbox from binary mask (preferred over polygon_to_bbox).
    Args:
        mask: np.ndarray[H, W], binary
    Returns:
        [x1, y1, x2, y2] or None if mask empty
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1  # exclusive
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def rasterize_polygon(
    polygon: List[Tuple[float, float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> np.ndarray:
    """
    Rasterize a single polygon to binary mask using PIL.
    Args:
        polygon: outer ring [(col, row), ...]
        width, height: output mask size. Defaults to 224.
        holes: optional list of hole rings
    Returns:
        np.ndarray[H, W] of dtype uint8
    """
    if width is None:
        width = 224
    if height is None:
        height = 224
    if Image is None:
        raise ImportError("PIL required for rasterization")

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # PIL takes (x, y) tuples
    xy = [(float(p[0]), float(p[1])) for p in polygon]
    draw.polygon(xy, fill=1)

    if holes:
        for hole in holes:
            xy_hole = [(float(p[0]), float(p[1])) for p in hole]
            draw.polygon(xy_hole, fill=0)

    return np.array(mask, dtype=np.uint8)


def rasterize_multipolygon(
    polygons: List[List[Tuple[float, float]]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """
    Rasterize multiple polygons to a single mask.
    For MultiPolygon -> individual instances, call rasterize_polygon per polygon.
    This merges all into one mask.
    """
    if width is None:
        width = 224
    if height is None:
        height = 224
    if Image is None:
        raise ImportError("PIL required for rasterization")
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        xy = [(float(p[0]), float(p[1])) for p in poly]
        draw.polygon(xy, fill=1)
    return np.array(mask, dtype=np.uint8)


def mask_to_polygon(
    mask: np.ndarray,
    simplify_epsilon: float = 0.5,
) -> List[List[Tuple[float, float]]]:
    """
    Extract polygon(s) from binary mask via contour detection.
    Uses OpenCV if available, otherwise falls back to a bbox-rectangle polygon.
    Returns list of polygons (outer rings only, sorted by area desc).
    """
    try:
        import cv2 as cv
        result = cv.findContours(
            mask.astype(np.uint8),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contours = result[0] if len(result) == 2 else result[1]

        def _cnt_to_poly(cnt):
            """Convert OpenCV contour to [(col, row), ...] polygon."""
            return [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]

        polygons = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            poly = _cnt_to_poly(cnt)
            if simplify_epsilon > 0:
                approx = cv.approxPolyDP(
                    cnt, simplify_epsilon, closed=True
                )
                if len(approx) < 3:
                    continue
                poly = _cnt_to_poly(approx)
            polygons.append(poly)
        return sorted(polygons, key=lambda p: _polygon_area(p), reverse=True)
    except ImportError:
        return _pil_mask_to_polygon(mask)


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _pil_mask_to_polygon(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Fallback polygon extraction using bbox (no OpenCV, no scipy)."""
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    return [[
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    ]]


def binary_mask_to_instances(
    binary_mask: "np.ndarray",
    min_area: int = 8,
    connectivity: int = 4,
):
    """
    Convert a binary semantic mask into instance masks by connected components.

    Args:
        binary_mask: np.ndarray[H, W], values can be 0/255 or 0/1.
        min_area: minimum component area in pixels.
        connectivity: 4 or 8. Use 4 if diagonal touching should not merge
            instances, 8 if they should.

    Returns:
        instance_masks: list[np.ndarray[H, W]], each mask is 0/1 uint8.
        boxes: list[[x1, y1, x2, y2]], xyxy with x2/y2 exclusive.
        areas: list[float], mask area in pixels.
    """
    import cv2

    mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=connectivity,
    )

    instance_masks = []
    boxes = []
    areas_list = []

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        if area < min_area:
            continue

        inst_mask = (labels == label_id).astype(np.uint8)

        instance_masks.append(inst_mask)
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
        areas_list.append(float(area))

    return instance_masks, boxes, areas_list


def filter_small_polygons(
    polygons: List[List[Tuple[float, float]]],
    min_area: float = 8.0,
) -> List[List[Tuple[float, float]]]:
    """Filter out polygons with area < min_area (in pixel^2)."""
    return [p for p in polygons if _polygon_area(p) >= min_area]


if __name__ == "__main__":
    import numpy as np

    # Create a test polygon
    poly = [(40.0, 50.0), (160.0, 50.0), (160.0, 180.0), (40.0, 180.0)]

    # Rasterize
    mask = rasterize_polygon(poly, 224, 224)
    assert mask.sum() > 0, "Mask should not be empty"

    # Bbox from mask
    bbox = bbox_from_mask(mask)
    assert bbox is not None
    assert bbox[0] <= 40 and bbox[2] >= 160

    # Mask -> polygon (with OpenCV if available)
    try:
        polys = mask_to_polygon(mask, simplify_epsilon=0.5)
        if polys:
            print(f"Extracted {len(polys)} polygon(s), area={_polygon_area(polys[0]):.0f}")
    except Exception as e:
        print(f"mask_to_polygon skipped: {e}")

    # Empty mask
    empty_mask = np.zeros((224, 224), dtype=np.uint8)
    assert bbox_from_mask(empty_mask) is None
    assert len(mask_to_polygon(empty_mask)) == 0

    print("All mask_utils checks passed.")

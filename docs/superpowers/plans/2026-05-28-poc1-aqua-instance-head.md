# PoC-1: 养殖区 Instance Head 闭环 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 DualVisionEncoder 特征上实现 FPN + Mask R-CNN，完成 GeoJSON→pixel GT→训练→pixel polygon→WGS84→GeoJSON→overlay 的最小闭环。

**Architecture:** 9 个独立模块，按依赖关系自底向上构建。`poc_stage_one_det.py` 组合所有模块。model/dataset/geojson/vis 完全解耦。

**Tech Stack:** PyTorch 2.1, torchvision 0.16, DeepSpeed (仅权重加载), Huawei Ascend NPU, affine, shapely, PIL

---

## 文件结构总览

| 文件 | 创建/修改 | 职责 |
|------|----------|------|
| `utils/georef_transform.py` | Create | WGS84↔pixel 坐标变换 |
| `utils/mask_utils.py` | Create | 像素几何操作 (rasterize/bbox/polygon) |
| `Dataset/aqua_poc_dataset.py` | Create | 数据集 + GT 在线生成 |
| `scripts/build_poc_aqua_data.py` | Create | 离线 manifest 构建 |
| `Models/det_head.py` | Create | FPN + Adapter + MaskRCNN builder |
| `configs/poc_aqua_instance.yaml` | Create | PoC 配置 |
| `utils/geojson_builder.py` | Create | GeoJSON 输出 + validation |
| `utils/vis_overlay.py` | Create | GT/Pred overlay 可视化 |
| `scripts/poc_stage_one_det.py` | Create | 主训练脚本 (orchestration) |

---

### Task 1: `utils/georef_transform.py` — 坐标变换工具

**Files:**
- Create: `utils/georef_transform.py`
- Create: `utils/__init__.py`

- [ ] **Step 1: 创建模块文件与核心函数**

```python
"""Coordinate transform utilities for WGS84 <-> model pixel space.

Conventions:
- coords_wgs84: (lon, lat) tuples in EPSG:4326
- coords_pixel: (col, row) in model pixel space (224x224), pixel corner 0-indexed
- transform: GDAL-order affine [a, b, c, d, e, f]
"""
from typing import List, Tuple
import numpy as np

try:
    from affine import Affine
except ImportError:
    Affine = None


def resize_georef(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    original_transform: List[float],
):
    """
    Compute model_transform after image resize.
    Uses orig_affine * Affine.scale(sx, sy) for rotation/shear compatibility.

    Args:
        original_size: [width, height] of original image
        target_size: [width, height] of model input
        original_transform: GDAL affine [a, b, c, d, e, f]

    Returns:
        model_transform: list[float] of 6 affine coefficients
        resize_scale: (sx, sy)
    """
    if Affine is None:
        raise ImportError("affine library required: pip install affine")

    sx = original_size[0] / target_size[0]
    sy = original_size[1] / target_size[1]

    orig_affine = Affine(*original_transform)
    model_affine = orig_affine * Affine.scale(sx, sy)

    return list(model_affine)[:6], (sx, sy)


def wgs84_to_pixel(
    coords_wgs84: List[Tuple[float, float]],
    georef: dict,
) -> List[Tuple[float, float]]:
    """
    Convert WGS84 (lon, lat) -> model pixel (col, row).

    georef must contain:
        model_transform: list[float] of 6 affine coefficients
        source_crs: str, e.g. "EPSG:4326"
    """
    if georef.get("source_crs") == "EPSG:4326":
        # Direct WGS84 -> pixel (no CRS transform needed)
        model_affine = Affine(*georef["model_transform"])
        inv_affine = ~model_affine
        return [(inv_affine * (lon, lat)) for lon, lat in coords_wgs84]
    else:
        raise NotImplementedError(
            f"CRS transform not implemented for {georef.get('source_crs')}"
        )


def pixel_to_wgs84(
    coords_pixel: List[Tuple[float, float]],
    georef: dict,
) -> List[Tuple[float, float]]:
    """
    Convert model pixel (col, row) -> WGS84 (lon, lat).
    """
    if georef.get("source_crs") == "EPSG:4326":
        model_affine = Affine(*georef["model_transform"])
        return [model_affine * (col, row) for col, row in coords_pixel]
    else:
        raise NotImplementedError(
            f"CRS transform not implemented for {georef.get('source_crs')}"
        )


def round_trip_check(
    coords_wgs84: List[Tuple[float, float]],
    georef: dict,
) -> float:
    """
    WGS84 -> pixel -> WGS84 round-trip error.
    Returns mean error in degrees.
    Target: < 1e-6 degree.
    """
    coords_pixel = wgs84_to_pixel(coords_wgs84, georef)
    coords_back = pixel_to_wgs84(coords_pixel, georef)

    errors = []
    for (lon1, lat1), (lon2, lat2) in zip(coords_wgs84, coords_back):
        err = np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)
        errors.append(err)

    return float(np.mean(errors))


def clip_pixel_coords(
    coords_pixel: List[Tuple[float, float]],
    width: int = 224,
    height: int = 224,
) -> List[Tuple[float, float]]:
    """Clip pixel coordinates to image bounds [0, width) x [0, height)."""
    return [
        (max(0.0, min(width - 1e-9, col)), max(0.0, min(height - 1e-9, row)))
        for col, row in coords_pixel
    ]
```

- [ ] **Step 2: 验证 round_trip 对 EPSG:4326 直通情况**

```python
# Smoke test in same file or run as script
if __name__ == "__main__":
    # GF2-style transform: ~1m/pixel at this latitude
    # 256x256 original, 224x224 model
    original_transform = [1e-5, 0.0, 119.300000, 0.0, -1e-5, 35.072560]
    model_transform, (sx, sy) = resize_georef(
        (256, 256), (224, 224), original_transform
    )

    georef = {
        "source_crs": "EPSG:4326",
        "model_transform": model_transform,
    }

    # Test round trip on tile corners
    test_coords = [
        (119.300000, 35.072560),  # top-left
        (119.302560, 35.072560),  # top-right
        (119.302560, 35.070000),  # bottom-right
        (119.300000, 35.070000),  # bottom-left
    ]

    err = round_trip_check(test_coords, georef)
    print(f"Round-trip error: {err:.2e} degrees")
    assert err < 1e-6, f"Round-trip error too large: {err}"

    # Test pixel conversion
    pixel_coords = wgs84_to_pixel(test_coords, georef)
    print(f"Pixel coords: {pixel_coords}")
    print("All checks passed.")
```

Run: `python utils/georef_transform.py`
Expected: error < 1e-6, pixel coords roughly [0,0], [223,0], [223,223], [0,223]

- [ ] **Step 3: Commit**

```bash
git add utils/__init__.py utils/georef_transform.py
git commit -m "feat(poc): add georef_transform.py with WGS84-pixel round-trip"
```

---

### Task 2: `utils/mask_utils.py` — 像素几何操作

**Files:**
- Create: `utils/mask_utils.py`

- [ ] **Step 1: 创建模块**

```python
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
    width: int = 224,
    height: int = 224,
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> np.ndarray:
    """
    Rasterize a single polygon to binary mask using PIL.

    Args:
        polygon: outer ring [(col, row), ...]
        width, height: output mask size
        holes: optional list of hole rings

    Returns:
        np.ndarray[H, W] of dtype uint8
    """
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
    width: int = 224,
    height: int = 224,
) -> np.ndarray:
    """
    Rasterize multiple polygons to a single mask.
    For MultiPolygon -> individual instances, call rasterize_polygon per polygon.
    This merges all into one mask.
    """
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

    Uses OpenCV if available, otherwise PIL edge tracing.
    Returns list of polygons (outer rings only, sorted by area desc).

    Args:
        mask: np.ndarray[H, W] uint8 binary
        simplify_epsilon: Douglas-Peucker epsilon in pixels (0 = no simplify)

    Returns:
        List of polygons, each is [(col, row), ...]
    """
    try:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            # cv2 returns (row, col) -> convert to (col, row)
            poly = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
            if simplify_epsilon > 0:
                import cv2 as cv
                approx = cv.approxPolyDP(
                    cnt, simplify_epsilon, closed=True
                )
                poly = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            polygons.append(poly)
        return sorted(polygons, key=lambda p: _polygon_area(p), reverse=True)
    except ImportError:
        # Fallback: PIL-based contour via connected components
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


def filter_small_polygons(
    polygons: List[List[Tuple[float, float]]],
    min_area: float = 8.0,
) -> List[List[Tuple[float, float]]]:
    """Filter out polygons with area < min_area (in pixel^2)."""
    return [p for p in polygons if _polygon_area(p) >= min_area]
```

- [ ] **Step 2: 验证 rasterize + bbox + mask_to_polygon round-trip**

```python
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
```

Run: `python utils/mask_utils.py`
Expected: all assertions pass

- [ ] **Step 3: Commit**

```bash
git add utils/mask_utils.py
git commit -m "feat(poc): add mask_utils.py for pixel geometry operations"
```

---

### Task 3: `scripts/build_poc_aqua_data.py` — 离线 Manifest 构建

**Files:**
- Create: `scripts/build_poc_aqua_data.py`

- [ ] **Step 1: 创建 GF2 数据扫描脚本**

GF2 tile 的关键事实:
- 瓦片命名: `海水养殖区_GF2_PMS2_E119.4_N35.1_..._R017C006_256_Label_WFQ.geojson`
  - `R017C006`: Row 17, Col 6
  - `E119.4_N35.1`: 源图像左下角 WGS84 原点
  - 256: 瓦片尺寸
- GF2 PMS ~1m 分辨率 → ~1e-5 degree/pixel
- 每片 256×256 → ~0.00256 degree
- **tile bounds** = 源图像原点 + row/col offset × 像素分辨率 × 瓦片尺寸

```python
#!/usr/bin/env python3
"""Build PoC-1 manifest from GF aquaculture tiles.

Derives tile-level georef from the tile grid system encoded in filenames.
NEVER derives tile_transform from object polygon min/max.

Usage:
    python scripts/build_poc_aqua_data.py --sensor GF2 --split 0.8
    python scripts/build_poc_aqua_data.py --sensor GF2,GF1 --max-samples 100 --split 0.8
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Build PoC-1 aquaculture manifest")
    p.add_argument("--data-root", default="/home/ma-user/work/GeoJsonData")
    p.add_argument("--sensor", default="GF2", help="Comma-separated: GF2,GF1,GF6")
    p.add_argument("--size", default="256", help="Tile size to include (Size_256)")
    p.add_argument("--max-samples", type=int, default=0, help="0 = use all")
    p.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    p.add_argument("--output", default=None, help="Output directory for manifest JSONs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def gf_tile_bounds_from_tile_name(
    label_filename: str,
    original_size: int = 256,
    gsd: float = 1e-5,  # ~1m at equator, adjust for latitude
) -> Optional[Dict]:
    """
    Derive tile bounds from GF tile naming convention.

    Filename pattern:
        ..._E<lon>_N<lat>_..._R<row>C<col>_<size>_...

    The tile origin (E, N) represents the source image corner.
    Tile bounds = origin + row/col * 256 * GSD.

    This filename-derived transform is only a provisional estimate.
    It must pass feature pixel in-bounds validation and GT overlay inspection
    before being used for training.
    """
    # Extract E/lon and N/lat origin
    m_origin = re.search(r'_E([\d.]+)_N([\d.]+)_', label_filename)
    if not m_origin:
        return None

    origin_lon = float(m_origin.group(1))
    origin_lat = float(m_origin.group(2))

    # Extract row/col
    m_rc = re.search(r'_R(\d+)C(\d+)_', label_filename)
    if not m_rc:
        return None

    row = int(m_rc.group(1))
    col = int(m_rc.group(2))

    # Extract tile size from filename
    m_sz = re.search(r'_(\d+)_Label_', label_filename)
    tile_size = int(m_sz.group(1)) if m_sz else original_size

    # Tile extent in degrees
    tile_extent = tile_size * gsd

    # Tile origin in the grid (top-left of the tile)
    # GF convention: col increases eastward, row increases southward from origin
    tile_min_lon = origin_lon + col * tile_size * gsd
    tile_max_lat = origin_lat - row * tile_size * gsd  # origin is SW corner, rows go north
    tile_max_lon = tile_min_lon + tile_extent
    tile_min_lat = tile_max_lat - tile_extent

    # Build affine: pixel (col, row) -> WGS84 (lon, lat)
    x_res = gsd
    y_res = -gsd  # negative: row increases downward, lat decreases

    return {
        "source_crs": "EPSG:4326",
        "original_size": [tile_size, tile_size],
        "model_input_size": [224, 224],
        "original_transform": [
            x_res, 0.0, tile_min_lon,
            0.0, y_res, tile_max_lat,
        ],
        "tile_bounds_wgs84": [
            tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat,
        ],
        "tile_row": row,
        "tile_col": col,
        "gsd": gsd,
    }


def scan_sensor(data_root: Path, sensor: str, size: str) -> List[Dict]:
    """Scan one sensor directory for matching image-label pairs."""
    sensor_dir = data_root / sensor
    img_dir = sensor_dir / f"Size_{size}" / "Image_TrueColor"
    label_dir = sensor_dir / f"Size_{size}" / "Label_GeoJSON"

    if not img_dir.exists() or not label_dir.exists():
        print(f"  [SKIP] {sensor}/Size_{size}: missing dirs")
        return []

    samples = []
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        image_paths.extend(img_dir.glob(ext))
    for img_path in sorted(image_paths):
        stem = img_path.stem
        # Find matching label: strip _True_WFQ suffix, add _Label_WFQ
        label_stem = stem.replace("_True_WFQ", "_Label_WFQ")
        label_path = label_dir / f"{label_stem}.geojson"

        if not label_path.exists():
            print(f"  [WARN] No label for {img_path.name}")
            continue

        # Parse label
        try:
            with open(label_path) as fh:
                label_data = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"  [BAD] {label_path.name}: JSON error: {e}")
            continue

        features = label_data.get("features", [])
        num_features = len(features)

        # Check geometry types in features
        geom_types = set()
        for feat in features:
            geom = feat.get("geometry", {})
            t = geom.get("type", "")
            if t:
                geom_types.add(t)

        # Validate features
        bad_reason = None
        if not isinstance(features, list):
            bad_reason = "features_not_list"
        elif any(
            feat.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon")
            for feat in features
            if feat.get("geometry", {}).get("type")
        ):
            # Warn but don't reject - could have mixed types
            pass

        # Derive tile georef
        tile_georef = gf_tile_bounds_from_tile_name(
            label_path.name, original_size=int(size)
        )

        if tile_georef is None:
            bad_reason = "missing_tile_bounds"

        source_image_id = re.sub(r'_R\d+C\d+_.*', '', stem)
        # Clean up: remove _True_WFQ or similar suffixes
        source_image_id = re.sub(r'_(True|False|Orig)_WFQ$', '', source_image_id)

        sample = {
            "sample_id": f"{sensor}_{stem}",
            "sensor": sensor,
            "source_image_id": source_image_id,
            "image_path": str(img_path.relative_to(data_root)),
            "label_path": str(label_path.relative_to(data_root)),
            "task": "DET",
            "branch": "instance",
            "known_classes": ["海水养殖区"],
            "has_object": num_features > 0,
            "num_features": num_features,
            "geom_types": sorted(geom_types),
            "split_group": f"{sensor}_{source_image_id}",
            "label_source": "geojson_answer",
            "bad_reason": bad_reason,
        }

        if tile_georef:
            sample.update(tile_georef)

        samples.append(sample)

    print(f"  [{sensor}/Size_{size}] Found {len(samples)} tiles")
    return samples


def split_train_val(samples: List[Dict], split: float, seed: int):
    """Split by unique split_groups, not random tiles."""
    import random
    rng = random.Random(seed)

    groups = list(set(s["split_group"] for s in samples if not s.get("bad_reason")))
    rng.shuffle(groups)

    n_train = max(1, int(len(groups) * split))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:])

    train = [s for s in samples if s["split_group"] in train_groups and not s.get("bad_reason")]
    val = [s for s in samples if s["split_group"] in val_groups and not s.get("bad_reason")]
    bad = [s for s in samples if s.get("bad_reason")]

    return train, val, bad


def validate_feature_pixels(label_data: dict, georef: dict, width: int = 224, height: int = 224):
    """
    Validate that GeoJSON feature coordinates fall within image bounds after
    WGS84 -> pixel conversion. Returns statistics for sanity checking.
    """
    total = 0
    inside = 0
    min_col, min_row = float("inf"), float("inf")
    max_col, max_row = float("-inf"), float("-inf")

    for feat in label_data.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            rings = coords
        elif gtype == "MultiPolygon":
            rings = [ring for poly in coords for ring in poly]
        else:
            continue

        for ring in rings:
            pixels = wgs84_to_pixel([(lon, lat) for lon, lat in ring], georef)
            for col, row in pixels:
                total += 1
                min_col = min(min_col, col)
                max_col = max(max_col, col)
                min_row = min(min_row, row)
                max_row = max(max_row, row)
                if 0 <= col < width and 0 <= row < height:
                    inside += 1

    in_bounds_ratio = inside / max(total, 1)

    return {
        "total_vertices": total,
        "in_bounds_ratio": in_bounds_ratio,
        "min_col": min_col,
        "max_col": max_col,
        "min_row": min_row,
        "max_row": max_row,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sensors = [s.strip() for s in args.sensor.split(",")]

    all_samples = []
    for sensor in sensors:
        all_samples.extend(scan_sensor(data_root, sensor, args.size))

    if args.max_samples > 0 and len(all_samples) > args.max_samples:
        all_samples = all_samples[:args.max_samples]

    # Validate non-bad samples have tile_bounds
    good = [s for s in all_samples if not s.get("bad_reason")]
    print(f"\nTotal: {len(all_samples)}, Good: {len(good)}, Bad: {len(all_samples) - len(good)}")

    # Feature pixel in-bounds validation against original_size
    print("\n=== Feature Pixel In-Bounds Validation ===")
    sys.path.insert(0, str(_REPO_ROOT))
    from utils.georef_transform import wgs84_to_pixel as _wgs84_to_pixel

    suspicious_count = 0
    for s in good:
        if not s.get("has_object"):
            continue
        label_path = data_root / s["label_path"]
        with open(label_path) as fh:
            label_data = json.load(fh)
        georef = {
            "source_crs": s["source_crs"],
            "model_transform": s["original_transform"],
        }
        orig_w, orig_h = s["original_size"]
        stats = validate_feature_pixels(label_data, georef, orig_w, orig_h)
        ratio = stats["in_bounds_ratio"]
        if ratio < 0.2:
            print(f"  [BAD] {s['sample_id']}: in_bounds={ratio:.1%} (likely bad georef)")
            s["bad_reason"] = "pixel_in_bounds_failed"
            suspicious_count += 1
        elif ratio < 0.8:
            print(f"  [SUSPICIOUS] {s['sample_id']}: in_bounds={ratio:.1%} (needs manual check)")
            suspicious_count += 1

    if suspicious_count == 0:
        print("  All samples passed in-bounds validation.")
    else:
        print(f"  {suspicious_count} sample(s) flagged for georef review.")

    # Re-split after potential new bad samples
    train, val, bad = split_train_val(all_samples, args.split, args.seed)

    output_dir = Path(args.output) if args.output else Path("data/poc_aqua")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("bad_samples", bad)]:
        path = output_dir / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"  Wrote {path} ({len(data)} samples)")

    # Summary
    pos_train = sum(1 for s in train if s["has_object"])
    neg_train = len(train) - pos_train
    if neg_train > 0:
        ratio = pos_train / max(neg_train, 1)
        print(f"  Train: {pos_train} positive, {neg_train} negative (ratio {ratio:.1f}:1)")

    if bad:
        reasons = {}
        for s in bad:
            reasons[s.get("bad_reason", "unknown")] = reasons.get(s.get("bad_reason", "unknown"), 0) + 1
        print(f"  Bad samples by reason: {reasons}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 执行构建 GF2/256 manifest 并验证**

```bash
python scripts/build_poc_aqua_data.py --sensor GF2 --split 0.8
```

Expected output:
- `data/poc_aqua/train.json` (~31 samples)
- `data/poc_aqua/val.json` (~8 samples)
- `data/poc_aqua/bad_samples.json` (0 or few samples)
- Console shows positive/negative ratio >= 2:1

Verify: check one entry has correct `original_transform` by computing tile_bounds_wgs84 back from it:
```python
import json
with open('data/poc_aqua/train.json') as f:
    data = json.load(f)
s = data[0]
print(s['sample_id'])
print(s['tile_bounds_wgs84'])
# Verify: original_transform maps pixel (0,0) -> (tile_min_lon, tile_max_lat)
# And pixel (256,256) -> (tile_max_lon, tile_min_lat)
```

- [ ] **Step 3: 验证 GSD 近似值**

GF2 PMS 在纬度 35°N 的实际 GSD 约为:
- 8m GSD × cos(35°) adjustment... Actually GF2 PMS native resolution is 2m pan-sharpened, ~2e-5 deg.
- For PoC-1, the exact GSD value matters less than consistency between WGS84→pixel and pixel→WGS84.
- Validate by checking that pixel coordinates of features fall within [0, 256).

If tile_bounds are systematically off, the dry-run overlay (Task 6) will reveal it.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_poc_aqua_data.py
git commit -m "feat(poc): add build_poc_aqua_data.py for GF tile manifest"
```

---

### Task 4: `configs/poc_aqua_instance.yaml` — PoC 配置

**Files:**
- Create: `configs/poc_aqua_instance.yaml`

- [ ] **Step 1: 创建配置模板**

```yaml
# PoC-1: Aquaculture Instance Head Configuration
# Single NPU, no DeepSpeed, frozen DualVisionEncoder

experiment:
  name: poc_aqua_instance
  output_dir: outputs/poc_aqua_instance
  seed: 42

data:
  data_root: /home/ma-user/work/GeoJsonData
  train_manifest: data/poc_aqua/train.json
  val_manifest: data/poc_aqua/val.json
  image_size: 224
  max_samples_train: 0
  num_workers: 2

model:
  # Vision encoder config (needed to construct DualVisionEncoder)
  rgb_vision:
    arch: dual
    global_encoder_name: dinov3_vitl16
    local_source: dino
    local_encoder_name: convnext_base
    local_ckpt_path: ./dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
    freeze_global: true
    freeze_local: true
    global_ckpt_path: ./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    input_size: [224, 224]
    physical_prompt_dim: 4096
    patch_dropout: 0.0
    input_patchnorm: false
    tune_pooler: false
    attn_pooler:
      num_query: 144
      num_attn_heads: 16
      num_layers: 6

  # Checkpoint to load vision weights from
  vision_checkpoint: ./FINAL.pt

  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256

  mask_rcnn:
    num_classes: 2
    min_size: 224
    max_size: 224
    image_mean: [0.0, 0.0, 0.0]
    image_std: [1.0, 1.0, 1.0]
    rpn_pre_nms_top_n_train: 512
    rpn_post_nms_top_n_train: 128
    rpn_pre_nms_top_n_test: 256
    rpn_post_nms_top_n_test: 64
    rpn_nms_thresh: 0.7
    box_score_thresh: 0.05
    box_nms_thresh: 0.5
    box_detections_per_img: 50

  anchors:
    sizes:
      - [16, 32]
      - [32, 64]
      - [64, 128, 192]
    aspect_ratios:
      - [0.5, 1.0, 2.0, 3.0]
      - [0.5, 1.0, 2.0, 3.0]
      - [0.5, 1.0, 2.0, 3.0]

train:
  device: npu
  epochs: 20
  batch_size: 2
  lr: 0.0001
  weight_decay: 0.0001
  max_grad_norm: 1.0
  log_interval: 10
  val_interval: 1
  save_interval: 1

eval:
  score_thresh: 0.5
  mask_thresh: 0.5
  min_area_px: 8
  export_geojson: true
  export_overlay: true
```

- [ ] **Step 2: Commit**

```bash
git add configs/poc_aqua_instance.yaml
git commit -m "feat(poc): add poc_aqua_instance.yaml configuration"
```

---

### Task 5: `Models/det_head.py` — FPN + Mask R-CNN

**Files:**
- Create: `Models/det_head.py`

- [ ] **Step 1: 创建 FPN Neck**

```python
"""
FPN Neck + DualVisionFPNBackboneAdapter + Aqua Mask R-CNN builder.

Provides:
- FPNNeck: c4/c8/c16/c32 -> P1/P2/P3/P4 (all 256-channel)
- DualVisionFPNBackboneAdapter: wraps frozen DualVisionEncoder + FPNNeck
  as torchvision Mask R-CNN backbone.
- build_aqua_maskrcnn(): constructs the complete Mask R-CNN model.
"""
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the existing DualVisionEncoder
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck for ConvNeXt pyramid features.

    Input:
        c4:  [B, 128, 56, 56]
        c8:  [B, 256, 28, 28]
        c16: [B, 512, 14, 14]
        c32: [B, 1024, 7, 7]

    Processing:
        1x1 Conv to unify channels to 256
        Top-down upsampling with lateral connections

    Output:
        P1: [B, 256, 56, 56]
        P2: [B, 256, 28, 28]
        P3: [B, 256, 14, 14]
        P4: [B, 256, 7, 7]
    """

    def __init__(
        self,
        in_channels: List[int] = [128, 256, 512, 1024],
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 Conv to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])

        # Post-merge smoothing convolutions
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels) - 1)  # P4 doesn't get smoothed
        ])
        self.smooth_convs.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
        c32: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lateral: 1x1 conv on each level
        lat4 = self.lateral_convs[0](c4)   # [B, 256, 56, 56]
        lat8 = self.lateral_convs[1](c8)   # [B, 256, 28, 28]
        lat16 = self.lateral_convs[2](c16) # [B, 256, 14, 14]
        lat32 = self.lateral_convs[3](c32) # [B, 256, 7, 7]

        # Top-down
        P4 = self.smooth_convs[3](lat32)  # [B, 256, 7, 7]
        P3 = self.smooth_convs[2](
            lat16 + F.interpolate(P4, size=lat16.shape[-2:], mode='nearest')
        )  # [B, 256, 14, 14]
        P2 = self.smooth_convs[1](
            lat8 + F.interpolate(P3, size=lat8.shape[-2:], mode='nearest')
        )  # [B, 256, 28, 28]
        P1 = self.smooth_convs[0](
            lat4 + F.interpolate(P2, size=lat4.shape[-2:], mode='nearest')
        )  # [B, 256, 56, 56]

        return P1, P2, P3, P4
```

- [ ] **Step 2: 创建 DualVisionFPNBackboneAdapter**

```python
class DualVisionFPNBackboneAdapter(nn.Module):
    """
    Wrap frozen DualVisionEncoder + FPNNeck as torchvision Mask R-CNN backbone.

    torchvision's GeneralizedRCNNTransform batches List[Tensor] into ImageList,
    then passes batched Tensor[B, 3, H, W] to backbone.forward().

    Input:  Tensor[B, 3, 224, 224]    (NOT List[Tensor])
    Output: OrderedDict[str, Tensor]
            {"0": P2, "1": P3, "2": P4}   # for Mask R-CNN featmap_names
    """

    def __init__(self, vision_encoder: DualVisionEncoder, fpn_neck: FPNNeck):
        super().__init__()
        self.vision = vision_encoder
        self.fpn = fpn_neck
        self.out_channels = fpn_neck.out_channels

        # Freeze vision
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> OrderedDict:
        """
        Args:
            images: Tensor[B, 3, 224, 224]  (batched by GeneralizedRCNNTransform)
        """
        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = self.vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = self.fpn(c4, c8, c16, c32)

        return OrderedDict({
            "0": p2,  # [B, 256, 28, 28]
            "1": p3,  # [B, 256, 14, 14]
            "2": p4,  # [B, 256, 7, 7]
        })
```

- [ ] **Step 3: 创建 build_aqua_maskrcnn()**

```python
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def build_aqua_maskrcnn(
    backbone_adapter: DualVisionFPNBackboneAdapter,
    num_classes: int = 2,
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ((16, 32), (32, 64), (64, 128, 192)),
    aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0, 3.0),) * 3,
    rpn_pre_nms_top_n_train: int = 512,
    rpn_post_nms_top_n_train: int = 128,
    rpn_pre_nms_top_n_test: int = 256,
    rpn_post_nms_top_n_test: int = 64,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 50,
    image_mean: List[float] = [0.0, 0.0, 0.0],
    image_std: List[float] = [1.0, 1.0, 1.0],
    min_size: int = 224,
    max_size: int = 224,
) -> MaskRCNN:
    """
    Build a torchvision Mask R-CNN with a custom FPN backbone adapter.

    Args:
        backbone_adapter: DualVisionFPNBackboneAdapter instance
        num_classes: including background (2 for aquaculture + bg)
        ...
    Returns:
        MaskRCNN model ready for training
    """
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    # featmap_names must match OrderedDict keys from adapter
    featmap_names = ["0", "1", "2"]

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=7,
        sampling_ratio=2,
    )

    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=14,
        sampling_ratio=2,
    )

    model = MaskRCNN(
        backbone_adapter,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        image_mean=image_mean,
        image_std=image_std,
        min_size=min_size,
        max_size=max_size,
    )

    return model
```

- [ ] **Step 4: 验证 FPN forward shape**

```python
if __name__ == "__main__":
    # Test FPN shapes only (no DualVisionEncoder needed)
    fpn = FPNNeck(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
    )
    print(f"FPN params: {sum(p.numel() for p in fpn.parameters()):,}")

    c4 = torch.randn(2, 128, 56, 56)
    c8 = torch.randn(2, 256, 28, 28)
    c16 = torch.randn(2, 512, 14, 14)
    c32 = torch.randn(2, 1024, 7, 7)

    p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
    print(f"P1: {list(p1.shape)}")
    print(f"P2: {list(p2.shape)}")
    print(f"P3: {list(p3.shape)}")
    print(f"P4: {list(p4.shape)}")

    assert list(p1.shape) == [2, 256, 56, 56]
    assert list(p2.shape) == [2, 256, 28, 28]
    assert list(p3.shape) == [2, 256, 14, 14]
    assert list(p4.shape) == [2, 256, 7, 7]
    print("FPN shape check passed.")
```

Run: `python Models/det_head.py`
Expected: all shape assertions pass, ~200K params

- [ ] **Step 5: Commit**

```bash
git add Models/det_head.py
git commit -m "feat(poc): add FPNNeck, DualVisionFPNBackboneAdapter, build_aqua_maskrcnn"
```

---

### Task 6: `Dataset/aqua_poc_dataset.py` — 数据集

**Files:**
- Create: `Dataset/aqua_poc_dataset.py`

- [ ] **Step 1: 创建 AquaPoCDataset**

```python
"""
AquaPoCDataset: loads GF tiles, converts GeoJSON -> pixel GT online.

Output contract per sample:
    {
        "image": Tensor[3, 224, 224],
        "target": {
            "boxes": FloatTensor[N, 4],       # xyxy, model pixel
            "labels": LongTensor[N],           # all 1
            "masks": UInt8Tensor[N, 224, 224],
            "image_id": LongTensor[1],
            "area": FloatTensor[N],
            "iscrowd": LongTensor[N],          # all 0
        },
        "meta": {
            "image_path": str,
            "source_crs": str,
            "original_transform": list[float],
            "model_transform": list[float],
            "original_size": [int, int],
            "model_input_size": [224, 224],
            "resize_scale": [float, float],
            "sample_id": str,
        }
    }
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.georef_transform import (
    resize_georef,
    wgs84_to_pixel,
    clip_pixel_coords,
)
from utils.mask_utils import (
    rasterize_polygon,
    bbox_from_mask,
)


class AquaPoCDataset(torch.utils.data.Dataset):
    """
    Dataset for PoC-1 aquaculture instance detection.

    Reads manifest JSON, loads images, converts GeoJSON labels
    to torchvision detection targets in model pixel space.

    MultiPolygon features are split into separate instances.
    Empty features[] tiles are retained as negative samples.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str = "/home/ma-user/work/GeoJsonData",
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size

        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"AquaPoCDataset: loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load image
        img_path = self.data_root / sample["image_path"]
        image = Image.open(img_path).convert("RGB")

        original_size = sample["original_size"]
        original_w, original_h = original_size[0], original_size[1]

        # Resize to model input
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(
            np.array(image, dtype=np.float32).transpose(2, 0, 1)
        ) / 255.0

        # Compute model_transform
        original_transform = sample["original_transform"]
        model_transform, resize_scale = resize_georef(
            tuple(original_size), (self.image_size, self.image_size), original_transform
        )

        georef = {
            "source_crs": sample["source_crs"],
            "model_transform": model_transform,
        }

        # Load and parse GeoJSON label
        label_path = self.data_root / sample["label_path"]
        with open(label_path) as f:
            label_data = json.load(f)

        features = label_data.get("features", [])

        # Generate GT per feature
        boxes = []
        masks_list = []
        labels = []

        for feature in features:
            geom = feature.get("geometry", {})
            geom_type = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if geom_type == "Polygon":
                rings = coords  # [outer, hole1, ...]
                pixel_rings = self._polygon_to_pixel(rings, georef)
            elif geom_type == "MultiPolygon":
                # Split MultiPolygon into separate instances
                for polygon_coords in coords:
                    pixel_rings = self._polygon_to_pixel(polygon_coords, georef)
                    self._add_instance(pixel_rings, boxes, masks_list, labels)
                continue  # already added from inner loop
            else:
                continue  # skip unsupported types (Point, LineString, etc.)

            self._add_instance(pixel_rings, boxes, masks_list, labels)

        # Filter instances where mask rasterization produced empty mask
        raw_num_features = len(features)
        valid_indices = [i for i, m in enumerate(masks_list) if m.sum() > 0]
        boxes = [boxes[i] for i in valid_indices] if valid_indices else []
        masks_list = [masks_list[i] for i in valid_indices] if valid_indices else []
        labels = [labels[i] for i in valid_indices] if valid_indices else []

        # Non-empty features but all masks empty -> bad sample, not negative
        valid_num_instances = len(boxes)
        if raw_num_features > 0 and valid_num_instances == 0:
            raise ValueError(
                f"Sample {sample['sample_id']} has {raw_num_features} features "
                "but no valid rasterized instances. Check georef / polygon coordinates."
            )

        # Build target dict
        if len(boxes) > 0:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.stack(masks_list) if masks_list else torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.tensor(
                    [float(m.sum().item()) for m in masks_list],
                    dtype=torch.float32,
                ),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        meta = {
            "image_path": str(img_path),
            "source_crs": sample["source_crs"],
            "original_transform": original_transform,
            "model_transform": model_transform,
            "original_size": original_size,
            "model_input_size": [self.image_size, self.image_size],
            "resize_scale": list(resize_scale),
            "sample_id": sample["sample_id"],
            "tile_bounds_wgs84": sample.get("tile_bounds_wgs84"),
            "sensor": sample.get("sensor"),
            "source_image_id": sample.get("source_image_id"),
        }

        return {
            "image": image_tensor,
            "target": target,
            "meta": meta,
        }

    def _polygon_to_pixel(self, rings, georef):
        """Convert GeoJSON Polygon rings from WGS84 to pixel space with clipping."""
        pixel_rings = []
        for ring in rings:
            coords_wgs84 = [(lon, lat) for lon, lat in ring]
            coords_pixel = wgs84_to_pixel(coords_wgs84, georef)
            coords_pixel = clip_pixel_coords(coords_pixel, self.image_size, self.image_size)
            pixel_rings.append(coords_pixel)
        return pixel_rings

    def _add_instance(self, pixel_rings, boxes, masks_list, labels):
        """Rasterize a polygon (outer ring + holes) and compute bbox."""
        outer_ring = pixel_rings[0]
        holes = pixel_rings[1:] if len(pixel_rings) > 1 else None

        mask = rasterize_polygon(outer_ring, self.image_size, self.image_size, holes)
        mask_tensor = torch.from_numpy(mask)

        bbox = bbox_from_mask(mask)
        if bbox is None:
            return  # skip empty instances

        boxes.append(bbox)
        masks_list.append(mask_tensor)
        labels.append(1)  # aquaculture


def poc_collate_fn(batch: List[Dict]) -> Tuple[List[torch.Tensor], List[Dict], List[Dict]]:
    """
    Collate function for AquaPoCDataset.
    Returns (images, targets, metas) matching torchvision MaskRCNN expected input.
    """
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    metas = [item["meta"] for item in batch]
    return images, targets, metas


def dataset_dry_run(dataset: AquaPoCDataset, output_dir: str = None, num_samples: int = 3):
    """
    Validate dataset pipeline on a few samples.
    Checks: image shape, target tensors, round-trip, GT overlay.
    """
    from utils.georef_transform import round_trip_check, pixel_to_wgs84
    import random

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        item = dataset[idx]
        print(f"\n--- Sample {item['meta']['sample_id']} ---")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Boxes: {item['target']['boxes'].shape[0]}")
        print(f"  Labels: {item['target']['labels'].tolist()}")

        # Round-trip check on first bbox center
        if item['target']['boxes'].shape[0] > 0:
            box = item['target']['boxes'][0]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            wgs84_back = pixel_to_wgs84([(float(cx), float(cy))], item['meta'])
            print(f"  First bbox center WGS84: ({wgs84_back[0][0]:.6f}, {wgs84_back[0][1]:.6f})")

            # Full round-trip on label features
            label_path = Path(dataset.data_root) / dataset.samples[idx]['label_path']
            with open(label_path) as f:
                label_data = json.load(f)
            for feat in label_data['features'][:1]:
                if feat['geometry']['type'] == 'Polygon':
                    coords = feat['geometry']['coordinates'][0]
                    err = round_trip_check(
                        [(lon, lat) for lon, lat in coords],
                        item['meta'],
                    )
                    print(f"  Round-trip error: {err:.2e} degrees")

        print(f"  Resize scale: {item['meta']['resize_scale']}")
        print(f"  has_object: {item['target']['boxes'].shape[0] > 0}")

    print("\nDry run complete.")
```

- [ ] **Step 2: 验证 Dataset dry run**

```python
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/poc_aqua/train.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        ds = AquaPoCDataset(args.manifest)
        dataset_dry_run(ds)
```

Run:
```bash
python Dataset/aqua_poc_dataset.py --manifest data/poc_aqua/train.json --dry-run
```

Expected: image shape [3, 224, 224], positive boxes > 0, round-trip error < 1e-6

- [ ] **Step 3: Commit**

```bash
git add Dataset/aqua_poc_dataset.py
git commit -m "feat(poc): add AquaPoCDataset with online GeoJSON->pixel GT conversion"
```

---

### Task 7: `utils/geojson_builder.py` — GeoJSON 输出与校验

**Files:**
- Create: `utils/geojson_builder.py`

- [ ] **Step 1: 创建模块**

```python
"""
GeoJSON Builder: convert pixel polygon predictions to WGS84 GeoJSON.

Pipeline:
    mask -> mask_to_polygon -> pixel polygons
    -> pixel_to_wgs84 -> WGS84 polygons
    -> GeoJSON Feature -> FeatureCollection
    -> validate
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .georef_transform import pixel_to_wgs84
from .mask_utils import mask_to_polygon, filter_small_polygons

try:
    from shapely.geometry import Polygon, mapping, shape
    from shapely.validation import explain_validity
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def polygon_pixel_to_geojson_feature(
    polygon_pixel: List[Tuple[float, float]],
    georef: dict,
    class_name: str = "海水养殖区",
    confidence: float = 1.0,
) -> dict:
    """
    Convert a pixel-space polygon to a GeoJSON Feature with WGS84 coordinates.
    """
    coords_wgs84 = pixel_to_wgs84(polygon_pixel, georef)

    # Close the ring if not already
    if coords_wgs84[0] != coords_wgs84[-1]:
        coords_wgs84 = list(coords_wgs84) + [coords_wgs84[0]]

    # GeoJSON order: [lon, lat] (not [x, y])
    ring = [[lon, lat] for lon, lat in coords_wgs84]

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [ring],
        },
        "properties": {
            "class": class_name,
            "confidence": round(float(confidence), 4),
        },
    }


def build_feature_collection(
    features: List[dict],
) -> dict:
    """Build a GeoJSON FeatureCollection from feature list."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def validate_geojson(
    feature_collection: dict,
    tile_bounds_wgs84: Optional[List[float]] = None,
    min_area_px: float = 8.0,
    bounds_eps: float = 1e-6,
) -> Dict:
    """
    Validate a GeoJSON FeatureCollection.

    Returns:
        {
            "valid": bool,
            "parse_ok": bool,
            "schema_ok": bool,
            "geometry_valid_rate": float,
            "empty_output": bool,
            "errors": [str, ...],
            "repaired_features": int,
        }
    """
    result = {
        "valid": True,
        "parse_ok": True,
        "schema_ok": True,
        "geometry_valid_rate": 1.0,
        "empty_output": False,
        "errors": [],
        "repaired_features": 0,
    }

    # Schema check
    if feature_collection.get("type") != "FeatureCollection":
        result["valid"] = False
        result["schema_ok"] = False
        result["errors"].append("Not a FeatureCollection")
        return result

    if not isinstance(feature_collection.get("features"), list):
        result["valid"] = False
        result["schema_ok"] = False
        result["errors"].append("Missing features array")
        return result

    features = feature_collection["features"]
    if len(features) == 0:
        result["empty_output"] = True
        return result

    # Per-feature validation and repair
    valid_count = 0
    for feat in features:
        errors = []
        if feat.get("type") != "Feature":
            errors.append("Not a Feature")
            continue
        if "geometry" not in feat:
            errors.append("Missing geometry")
            continue

        geom = feat["geometry"]
        if geom.get("type") != "Polygon":
            errors.append(f"Expected Polygon, got {geom.get('type')}")
            continue

        coords = geom.get("coordinates", [])
        if not coords or not coords[0] or len(coords[0]) < 3:
            errors.append("Invalid polygon coordinates")
            continue

        # Coordinate range check
        if tile_bounds_wgs84:
            min_lon, min_lat, max_lon, max_lat = tile_bounds_wgs84
            for lon, lat in coords[0]:
                if lon < min_lon - bounds_eps or lon > max_lon + bounds_eps:
                    errors.append(f"Coordinate lon={lon} outside tile bounds")
                    break
                if lat < min_lat - bounds_eps or lat > max_lat + bounds_eps:
                    errors.append(f"Coordinate lat={lat} outside tile bounds")
                    break

        # Shapely validation (if available)
        if HAS_SHAPELY and not errors:
            try:
                poly = shape(geom)
                if not poly.is_valid:
                    reason = explain_validity(poly)
                    errors.append(f"Invalid geometry: {reason}")
                    # Attempt repair
                    repaired = poly.buffer(0)
                    if repaired.is_valid and not repaired.is_empty:
                        feat["geometry"] = mapping(repaired)
                        result["repaired_features"] += 1
                        errors.clear()
            except Exception as e:
                errors.append(f"Shapely error: {e}")

        if not errors:
            valid_count += 1
        else:
            result["errors"].extend(errors)

    result["geometry_valid_rate"] = valid_count / max(len(features), 1)
    result["valid"] = (
        result["parse_ok"]
        and result["schema_ok"]
        and result["geometry_valid_rate"] >= 0.9
    )

    return result


def outputs_to_geojson(
    outputs: List[dict],
    metas: List[dict],
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    min_area_px: float = 8.0,
) -> List[dict]:
    """
    Convert Mask R-CNN outputs to GeoJSON FeatureCollections.

    Args:
        outputs: MaskRCNN output list [{boxes, labels, scores, masks}, ...]
        metas: meta dict per sample
        score_thresh: minimum score for a detection
        mask_thresh: binary threshold for masks
        min_area_px: minimum polygon area

    Returns:
        List of FeatureCollection dicts (one per image)
    """
    results = []
    for output, meta in zip(outputs, metas):
        features = []
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        masks = output["masks"].cpu().numpy()  # [N, 1, H, W]

        for i in range(len(scores)):
            if scores[i] < score_thresh:
                continue

            mask = (masks[i, 0] > mask_thresh).astype(np.uint8)
            polygons = mask_to_polygon(mask, simplify_epsilon=0.5)
            polygons = filter_small_polygons(polygons, min_area_px)

            for poly in polygons:
                feat = polygon_pixel_to_geojson_feature(
                    poly,
                    meta,
                    class_name="海水养殖区",
                    confidence=float(scores[i]),
                )
                features.append(feat)

        fc = build_feature_collection(features)
        results.append(fc)

    return results
```

- [ ] **Step 2: Commit**

```bash
git add utils/geojson_builder.py
git commit -m "feat(poc): add geojson_builder.py for pixel->GeoJSON conversion and validation"
```

---

### Task 8: `utils/vis_overlay.py` — 可视化

**Files:**
- Create: `utils/vis_overlay.py`

- [ ] **Step 1: 创建模块**

```python
"""Overlay visualization for GT and predicted instances in pixel space."""
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _draw_boxes(draw: ImageDraw.ImageDraw, boxes: np.ndarray, color: str, width: int = 2):
    """Draw bounding boxes on PIL ImageDraw."""
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_masks(image: np.ndarray, masks: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.4):
    """Overlay masks on image with alpha blending. Modifies image in-place."""
    for mask in masks:
        mask_bool = mask > 0.5
        if mask_bool.any():
            overlay = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
            image[mask_bool] = (image[mask_bool] * (1 - alpha) + overlay * alpha).astype(np.uint8)


def _draw_polygons(draw: ImageDraw.ImageDraw, polygons: List[List[Tuple[float, float]]], color: str, width: int = 1):
    """Draw polygon outlines."""
    for poly in polygons:
        if len(poly) >= 3:
            xy = [(float(p[0]), float(p[1])) for p in poly]
            draw.polygon(xy, outline=color, width=width)


def overlay_gt_pixel(
    image_tensor: np.ndarray,
    target: dict,
    output_path: str,
    mask_color: Tuple[int, int, int] = (0, 255, 0),  # green
    box_color: str = "green",
):
    """Overlay GT masks and boxes on the original image (pixel space)."""
    _overlay_single(
        image_tensor, target.get("boxes", None), target.get("masks", None),
        output_path, mask_color, box_color, label="GT"
    )


def overlay_pred_pixel(
    image_tensor: np.ndarray,
    boxes: Optional[np.ndarray],
    masks: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    output_path: str,
    mask_color: Tuple[int, int, int] = (255, 0, 0),  # red
    box_color: str = "red",
):
    """Overlay predicted masks and boxes."""
    _overlay_single(
        image_tensor, boxes, masks, output_path, mask_color, box_color,
        scores=scores, label="Pred"
    )


def overlay_gt_pred_pixel(
    image_tensor: np.ndarray,
    target: dict,
    pred_boxes: Optional[np.ndarray],
    pred_masks: Optional[np.ndarray],
    pred_scores: Optional[np.ndarray],
    output_path: str,
):
    """Side-by-side: GT (green) vs Pred (red)."""
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
        img = (image_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.copy()

    # Draw masks on numpy array FIRST
    gt_masks = target["masks"].cpu().numpy() if hasattr(target["masks"], 'cpu') else np.array(target["masks"])
    if len(gt_masks) > 0:
        _draw_masks(img, gt_masks[:5], (0, 255, 0), alpha=0.3)
    if pred_masks is not None and len(pred_masks) > 0:
        _draw_masks(img, pred_masks[:5], (255, 0, 0), alpha=0.3)

    # Then create PIL image and draw boxes
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # GT: green boxes
    gt_boxes = target["boxes"].cpu().numpy() if hasattr(target["boxes"], 'cpu') else np.array(target["boxes"])
    if len(gt_boxes) > 0:
        _draw_boxes(draw, gt_boxes, "green", width=2)

    # Pred: red boxes
    if pred_boxes is not None and len(pred_boxes) > 0:
        _draw_boxes(draw, pred_boxes, "red", width=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(output_path)


def _overlay_single(
    image_tensor, boxes, masks, output_path, mask_color, box_color, scores=None, label="",
):
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
        img = (image_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = image_tensor.copy()

    boxes_np = None
    masks_np = None
    scores_np = None

    if boxes is not None and len(boxes) > 0:
        boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)
    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)

    # Draw masks on numpy array FIRST, before creating PIL image
    if masks_np is not None:
        _draw_masks(img, masks_np[:5], mask_color, alpha=0.3)

    # Now create PIL image and draw boxes / text
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    if boxes_np is not None:
        _draw_boxes(draw, boxes_np, box_color)

        if scores is not None:
            scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
            for box, score in zip(boxes_np, scores_np):
                draw.text((box[0], max(0, box[1] - 8)), f"{score:.2f}", fill=box_color)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(output_path)
    print(f"  [{label}] Saved overlay to {output_path}")


def save_overlay_grid(
    images: List[np.ndarray],
    targets: List[dict],
    pred_outputs: Optional[List[dict]],
    output_dir: str,
    sample_ids: List[str] = None,
):
    """Generate overlay images for a batch of samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, target) in enumerate(zip(images, targets)):
        sid = sample_ids[i] if sample_ids else f"sample_{i:04d}"

        # GT overlay
        overlay_gt_pixel(img, target, str(output_dir / f"{sid}_gt.png"))

        if pred_outputs:
            pred = pred_outputs[i]
            pred_boxes = pred["boxes"].cpu().numpy()
            pred_masks = pred["masks"].cpu().numpy()[:, 0]
            pred_scores = pred["scores"].cpu().numpy()
            overlay_pred_pixel(
                img, pred_boxes, pred_masks, pred_scores,
                str(output_dir / f"{sid}_pred.png"),
            )
            overlay_gt_pred_pixel(
                img, target, pred_boxes, pred_masks, pred_scores,
                str(output_dir / f"{sid}_gt_pred.png"),
            )
```

- [ ] **Step 2: Commit**

```bash
git add utils/vis_overlay.py
git commit -m "feat(poc): add vis_overlay.py for pixel-space GT/Pred visualization"
```

---

### Task 9: `scripts/poc_stage_one_det.py` — 主训练脚本

**Files:**
- Create: `scripts/poc_stage_one_det.py`

- [ ] **Step 1: 创建训练脚本 (orchestration)**

```python
#!/usr/bin/env python3
"""
PoC-1: Aquaculture Instance Head Training Script.

Minimal training loop: single NPU, no DeepSpeed, no EpochBasedTrainer.
Orchestrates all PoC modules.

Usage:
    python scripts/poc_stage_one_det.py --config configs/poc_aqua_instance.yaml
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import ml_collections

from Models.det_head import (
    FPNNeck,
    DualVisionFPNBackboneAdapter,
    build_aqua_maskrcnn,
)
from Models.dual_vision_encoder import DualVisionEncoder
from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
from utils.geojson_builder import outputs_to_geojson, validate_geojson
from utils.vis_overlay import save_overlay_grid


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def dict_to_ml_collections(d: dict) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(d)


def clean_vision_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' and 'vision.' prefixes from checkpoint keys."""
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("vision."):
            key = key[len("vision."):]
        cleaned[key] = v
    return cleaned


def build_vision_encoder(
    model_cfg: ml_collections.ConfigDict,
    ckpt_path: str,
) -> DualVisionEncoder:
    """Construct DualVisionEncoder and load weights from checkpoint."""
    model = DualVisionEncoder(model_cfg)

    if os.path.exists(ckpt_path):
        print(f"Loading vision weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vision_ckpt = ckpt.get("vision_ckpt", ckpt)
        vision_ckpt = clean_vision_state_dict(vision_ckpt)

        missing, unexpected = model.load_state_dict(vision_ckpt, strict=False)

        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print(f"  First missing keys: {missing[:20]}")
        if len(unexpected) > 0:
            print(f"  First unexpected keys: {unexpected[:20]}")

        loaded_param_names = set(vision_ckpt.keys())
        model_param_names = set(model.state_dict().keys())
        matched = loaded_param_names & model_param_names

        if len(matched) < 0.5 * len(model_param_names):
            raise RuntimeError(
                f"Too few vision params matched: {len(matched)} / {len(model_param_names)}"
            )

        print(f"  Matched params: {len(matched)} / {len(model_param_names)}")
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt_path}, using random init")

    return model


def smoke_test_npu_ops(device: torch.device):
    """Test torchvision ops on target device."""
    import torchvision
    print("\n=== NPU Ops Smoke Test ===")
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [30.0, 30.0, 80.0, 80.0]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)

    # NMS
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"  nms: {boxes.shape[0]} -> {len(keep)} boxes kept")

    # Batched NMS
    idxs = torch.tensor([0, 0], device=device)
    keep = torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)
    print(f"  batched_nms: {len(keep)} boxes kept")

    # RoIAlign
    feat = torch.randn(1, 256, 14, 14, device=device)
    proposals = [torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)]
    roi_out = torchvision.ops.roi_align(feat, proposals, output_size=7, spatial_scale=1.0)
    print(f"  roi_align: {list(roi_out.shape)}")

    # MultiScaleRoIAlign
    ms_roi = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )
    feat_dict = {"0": feat}
    ms_out = ms_roi(feat_dict, proposals, [(14, 14)])
    print(f"  MultiScaleRoIAlign: {list(ms_out.shape)}")

    print("  All ops OK.\n")


def smoke_test_maskrcnn(model, device: torch.device):
    """Full forward/backward smoke test with synthetic data."""
    print("=== Mask R-CNN Forward/Backward Smoke Test ===")
    model.train()

    images = [torch.rand(3, 224, 224, device=device)]
    mask = torch.zeros((1, 224, 224), dtype=torch.uint8, device=device)
    mask[0, 50:180, 40:160] = 1

    targets = [{
        "boxes": torch.tensor([[40.0, 50.0, 160.0, 180.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
        "masks": mask,
        "image_id": torch.tensor([0], dtype=torch.int64, device=device),
        "area": torch.tensor([120.0 * 130.0], device=device),
        "iscrowd": torch.tensor([0], dtype=torch.int64, device=device),
    }]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())
    print(f"  Total loss: {loss.item():.4f}")

    required = ["loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg", "loss_mask"]
    for name in required:
        val = loss_dict.get(name)
        ok = val is not None and torch.isfinite(val)
        print(f"  {name}: {val.item() if ok else 'MISSING/NaN'} {'OK' if ok else 'FAIL'}")
        assert ok, f"Loss {name} invalid"

    loss.backward()
    print("  backward: OK")

    # Gradient check
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"Frozen param {name} has gradient!"
        else:
            if p.grad is None:
                print(f"  [WARN] Trainable param {name} has no gradient")

    model.zero_grad()
    print("  All smoke tests passed.\n")


def save_checkpoint(
    fpn: nn.Module, maskrcnn: nn.Module, optimizer, epoch: int, output_dir: str
):
    """Save FPN + Mask R-CNN weights."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "fpn": fpn.state_dict(),
            "maskrcnn": maskrcnn.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(path),
    )
    print(f"  Checkpoint saved to {path}")


def train_epoch(
    model, dataloader, optimizer, device, epoch: int, log_interval: int
):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            loss_str = " ".join([f"{k}={v.item():.3f}" for k, v in loss_dict.items()])
            print(f"  Epoch {epoch:3d} [{batch_idx:3d}] {loss_str}")

    avg_loss = total_loss / max(len(dataloader), 1)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch:3d} complete: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, output_dir: str, epoch: int):
    """Validation: inference + GeoJSON output + overlay export."""
    model.eval()
    vis_dir = Path(output_dir) / "vis" / f"epoch_{epoch:03d}"
    geojson_dir = Path(output_dir) / "geojson" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total_samples": 0, "valid_geojson": 0, "parse_errors": 0, "empty_outputs": 0}

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        if batch_idx >= 3:  # Limit val to 3 batches
            break

        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        # Post-process: outputs to GeoJSON
        metas_np = metas if isinstance(metas, list) else [metas]
        feature_collections = outputs_to_geojson(outputs, metas_np)

        for i, fc in enumerate(feature_collections):
            stats["total_samples"] += 1
            tile_bounds = metas[i].get("tile_bounds_wgs84")

            validation = validate_geojson(fc, tile_bounds)
            if validation["valid"]:
                stats["valid_geojson"] += 1
            if validation.get("empty_output"):
                stats["empty_outputs"] += 1
            if not validation.get("parse_ok"):
                stats["parse_errors"] += 1

            # Save GeoJSON
            sid = metas[i].get("sample_id", f"sample_{i:04d}")
            with open(geojson_dir / f"{sid}.geojson", "w") as f:
                json.dump(fc, f, ensure_ascii=False, indent=2)

        # Save overlay
        imgs_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
        save_overlay_grid(
            imgs_np, targets, outputs,
            str(vis_dir),
            sample_ids=[m.get("sample_id", "") for m in metas],
        )

    valid_rate = stats["valid_geojson"] / max(stats["total_samples"], 1)
    print(f"  Validation: {stats['total_samples']} samples, "
          f"valid_geojson_rate={valid_rate:.1%}, "
          f"empty_outputs={stats['empty_outputs']}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc_aqua_instance.yaml")
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # Build output dirs
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Vision encoder
    print("Building DualVisionEncoder...")
    vis_cfg = dict_to_ml_collections({"rgb_vision": cfg["model"]["rgb_vision"]})
    vision = build_vision_encoder(
        vis_cfg,
        ckpt_path=cfg["model"]["vision_checkpoint"],
    )
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    # Step 2: FPN + Mask R-CNN
    print("Building FPN + Mask R-CNN...")
    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)

    adapter = DualVisionFPNBackboneAdapter(vision, fpn).to(device)

    model = build_aqua_maskrcnn(
        adapter,
        num_classes=cfg["model"]["mask_rcnn"]["num_classes"],
        min_size=cfg["model"]["mask_rcnn"]["min_size"],
        max_size=cfg["model"]["mask_rcnn"]["max_size"],
        image_mean=cfg["model"]["mask_rcnn"]["image_mean"],
        image_std=cfg["model"]["mask_rcnn"]["image_std"],
        rpn_nms_thresh=cfg["model"]["mask_rcnn"]["rpn_nms_thresh"],
        box_score_thresh=cfg["model"]["mask_rcnn"]["box_score_thresh"],
        box_nms_thresh=cfg["model"]["mask_rcnn"]["box_nms_thresh"],
        box_detections_per_img=cfg["model"]["mask_rcnn"]["box_detections_per_img"],
    ).to(device)

    # Step 3: Smoke tests
    smoke_test_npu_ops(device)
    smoke_test_maskrcnn(model, device)

    # Step 4: Data
    print("Loading datasets...")
    train_ds = AquaPoCDataset(
        manifest_path=cfg["data"]["train_manifest"],
        data_root=cfg["data"]["data_root"],
        image_size=cfg["data"]["image_size"],
    )
    val_ds = AquaPoCDataset(
        manifest_path=cfg["data"]["val_manifest"],
        data_root=cfg["data"]["data_root"],
        image_size=cfg["data"]["image_size"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=poc_collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=poc_collate_fn,
    )

    # Step 5: Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # Step 6: Training loop
    print(f"\nStarting training: {cfg['train']['epochs']} epochs")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            log_interval=cfg["train"]["log_interval"],
        )

        if epoch % cfg["train"]["val_interval"] == 0:
            val_stats = validate(
                model, val_loader, device, str(output_dir), epoch,
            )

        if epoch % cfg["train"]["save_interval"] == 0:
            save_checkpoint(fpn, model, optimizer, epoch, str(output_dir))

    print(f"\nTraining complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本可导入**

```bash
python -c "
from Models.det_head import FPNNeck, DualVisionFPNBackboneAdapter, build_aqua_maskrcnn
from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
print('All imports OK')
"
```

Expected: all imports succeed (DualVisionEncoder may warn if NPU not available)

- [ ] **Step 3: Commit**

```bash
git add scripts/poc_stage_one_det.py
git commit -m "feat(poc): add poc_stage_one_det.py training orchestration script"
```

---

## Self-Review

### 1. Spec Coverage

| Spec Section | Task |
|---|---|
| 2. 文件结构与模块边界 | Tasks 1-9 cover all 9 files |
| 3.1 FPN Neck | Task 5 Step 1 |
| 3.2 DualVisionFPNBackboneAdapter | Task 5 Step 2 |
| 3.3 Mask R-CNN 配置 | Task 5 Step 3 |
| 4.2 离线阶段 build_poc_aqua_data.py | Task 3 |
| 4.3 tile_transform 来源 | Task 3 gf_tile_bounds_from_tile_name() |
| 4.4 Manifest 字段 | Task 3 sample dict |
| 4.5 在线阶段 AquaPoCDataset | Task 6 |
| 4.6 Dataset 输出 Contract | Task 6 __getitem__ |
| 4.7 MultiPolygon 处理 | Task 6 _add_instance split |
| 4.8 空标注 tile | Task 6 empty target handling |
| 5. 坐标转换 | Task 1 |
| 6. GeoJSON Builder & Validation | Task 7 |
| 7. 可视化 | Task 8 |
| 8.1 模型加载 | Task 9 build_vision_encoder() |
| 8.2 NPU Smoke Test | Task 9 smoke_test_npu_ops() + smoke_test_maskrcnn() |
| 8.3 训练参数 | Task 4 config yaml |
| 9. 执行顺序 | Task 9 main() orchestrates steps |
| 10. 通过标准 | Embedded in smoke tests + validation |

### 2. Placeholder Scan

No "TBD", "TODO", or ambiguous references found.

### 3. Type Consistency

- `georef` dict shape consistent across Task 1, 6, 7, 8, 9
- `OrderedDict` keys "0", "1", "2" consistent between Task 5 (FPN output) and Task 5 (Mask R-CNN featmap_names)
- `target` dict schema consistent between Task 6 (Dataset output) and Task 9 (training loop)
- `meta` dict fields consistent between Task 3 (manifest), Task 6 (Dataset), Task 7 (GeoJSON builder), Task 9 (training loop)
- All module imports match the file structure from the spec

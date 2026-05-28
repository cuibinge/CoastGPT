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


def gf_tile_bounds_from_geotiff(
    orig_tif_path: Path,
    original_size: int = 256,
) -> Optional[Dict]:
    """
    Read tile georef from a GeoTIFF file using tifffile.

    Reads ModelPixelScaleTag (GSD) and ModelTiepointTag (origin).
    These are the authoritative tile bounds — NOT derived from filename or object polygons.
    """
    try:
        import tifffile
    except ImportError:
        return None

    try:
        with tifffile.TiffFile(str(orig_tif_path)) as tif:
            page = tif.pages[0]
            scale_tag = page.tags.get('ModelPixelScaleTag')
            tiepoint_tag = page.tags.get('ModelTiepointTag')

            if scale_tag is None or tiepoint_tag is None:
                return None

            gsd_x = float(scale_tag.value[0])
            gsd_y = float(scale_tag.value[1])
            tie_x = float(tiepoint_tag.value[3])  # top-left lon
            tie_y = float(tiepoint_tag.value[4])  # top-left lat

            w = page.shape[1]
            h = page.shape[0]
    except Exception:
        return None

    tile_min_lon = tie_x
    tile_max_lat = tie_y
    tile_max_lon = tie_x + w * gsd_x
    tile_min_lat = tie_y - h * abs(gsd_y)

    return {
        "source_crs": "EPSG:4326",
        "original_size": [w, h],
        "model_input_size": [224, 224],
        "original_transform": [
            gsd_x, 0.0, tile_min_lon,
            0.0, -abs(gsd_y), tile_max_lat,
        ],
        "tile_bounds_wgs84": [
            tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat,
        ],
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "georef_source": "geotiff",
    }


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

    This is a FALLBACK used when GeoTIFF (ModelPixelScaleTag) is not available.
    The filename-derived transform is only a provisional estimate.
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

        # Derive tile georef: prefer GeoTIFF, fall back to filename
        orig_stem = stem.replace("_True_WFQ", "_Orig_WFQ")
        orig_tif_path = sensor_dir / f"Size_{size}" / "Image_Orig" / f"{orig_stem}.tif"

        if orig_tif_path.exists():
            tile_georef = gf_tile_bounds_from_geotiff(orig_tif_path, original_size=int(size))
        else:
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


def validate_feature_pixels(label_data: dict, georef: dict, wgs84_to_pixel_fn,
                            width: int = 224, height: int = 224):
    """
    Validate that GeoJSON feature coordinates fall within image bounds after
    WGS84 -> pixel conversion. Returns statistics for sanity checking.

    Args:
        label_data: Parsed GeoJSON dict.
        georef: Dict with 'source_crs' and 'model_transform' keys.
        wgs84_to_pixel_fn: Function (coords_wgs84, georef) -> List[(col, row)].
        width: Image width in pixels.
        height: Image height in pixels.
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
            pixels = wgs84_to_pixel_fn([(lon, lat) for lon, lat in ring], georef)
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
        stats = validate_feature_pixels(label_data, georef, _wgs84_to_pixel, orig_w, orig_h)
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

"""
Tile grouping: scan land cover class directories and merge GeoJSON features
that belong to the same spatial tile.

A tile_group_key is built from (quantized_tile_bounds, original_size).
Tiles from different class directories sharing the same key have their
GeoJSON features merged into a multi-class partial label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.landcover_label_map import dlmc_to_train_id, dir_name_to_dlmc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PATCHES_ROOT = Path("/home/ma-user/work/Stage3Data/土地分类/Patches")
_IMAGE_SUBDIR = "Image_TrueColor"
_LABEL_SUBDIR = "Label_GeoJSON"
_SIZES = ["Size_128", "Size_256", "Size_512"]

# Quantization tolerance: ~1cm at this latitude, well below pixel size.
_COORD_DECIMALS = 7


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_tile_group_key(
    tile_bounds_wgs84: Tuple[float, float, float, float],
    original_size: Tuple[int, int],
    source_image_id: Optional[str] = None,
) -> Tuple:
    """Build a tile group key from quantized spatial metadata.

    Args:
        tile_bounds_wgs84: (min_lon, min_lat, max_lon, max_lat).
        original_size: (width, height) of the original tile.
        source_image_id: Optional source image identifier. If provided,
            used to disambiguate tiles with identical bounds.

    Returns:
        Hashable tuple suitable as a dict key.
    """
    q_bounds = tuple(round(float(v), _COORD_DECIMALS) for v in tile_bounds_wgs84)
    if source_image_id:
        return (source_image_id, q_bounds, original_size)
    return (q_bounds, original_size)


def compute_tile_bounds_from_geojson(features: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Compute WGS84 bounding box from a list of GeoJSON features.

    Returns (min_lon, min_lat, max_lon, max_lat) or None if no coordinates.
    """
    all_lons: List[float] = []
    all_lats: List[float] = []

    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else feat.get("geometry")
        if geom is None:
            continue
        coords = _extract_all_coords(geom)
        for lon, lat in coords:
            all_lons.append(lon)
            all_lats.append(lat)

    if not all_lons:
        return None

    return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))


def scan_landcover_directories(
    patches_root: Optional[str] = None,
    sizes: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
) -> List[dict]:
    """Scan class directories and return a list of raw tile sample dicts.

    Each sample dict contains:
        class_dir: directory name (e.g. "公路用地")
        dlmc: canonical DLMC class name
        size: "Size_128" | "Size_256" | "Size_512"
        image_path: Path to the JPEG image
        geojson_path: Path to the GeoJSON label
        features: list of parsed GeoJSON feature dicts
        tile_bounds_wgs84: (min_lon, min_lat, max_lon, max_lat) from features
        sensor: "GF1"

    Args:
        patches_root: Root of the land cover patches data.
        sizes: Which size directories to scan.
        classes: Which class directories to scan (None = all).

    Returns:
        List of sample dicts, one per discovered GeoJSON file.
    """
    root = Path(patches_root or _PATCHES_ROOT)
    size_dirs = sizes or _SIZES
    samples: List[dict] = []

    # Determine class directories
    if classes is None:
        class_dirs = sorted(
            d.name for d in root.iterdir() if d.is_dir()
        )
    else:
        class_dirs = list(classes)

    for cls_dir_name in class_dirs:
        cls_path = root / cls_dir_name / "GF1"
        if not cls_path.exists():
            continue

        dlmc = dir_name_to_dlmc(cls_dir_name)

        for size_dir_name in size_dirs:
            geojson_dir = cls_path / size_dir_name / _LABEL_SUBDIR
            image_dir = cls_path / size_dir_name / _IMAGE_SUBDIR
            if not geojson_dir.exists():
                continue

            for geojson_path in sorted(geojson_dir.glob("*.geojson")):
                # Derive image path from geojson path: Label_GeoJSON → Image_TrueColor,
                # _Label_CK.geojson → _True_CK.jpg
                stem = geojson_path.stem  # e.g., "xxx_128_Label_CK"
                img_stem = stem.replace("_Label_CK", "_True_CK")
                image_path = image_dir / f"{img_stem}.jpg"

                if not image_path.exists():
                    # Try alternate naming pattern
                    img_stem2 = stem.replace("_Label_CK", "_True")
                    alt_path = image_dir / f"{img_stem2}.jpg"
                    if alt_path.exists():
                        image_path = alt_path
                    else:
                        continue  # skip samples with missing image

                # Parse GeoJSON features
                try:
                    with open(geojson_path, "r", encoding="utf-8") as f:
                        geojson = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                features = geojson.get("features", [])
                if not features:
                    continue

                # Derive tile bounds from the feature coordinates
                bounds = compute_tile_bounds_from_geojson(features)
                if bounds is None:
                    continue

                sample = {
                    "class_dir": cls_dir_name,
                    "dlmc": dlmc,
                    "size": size_dir_name,
                    "original_width": int(size_dir_name.split("_")[1]),
                    "original_height": int(size_dir_name.split("_")[1]),
                    "image_path": str(image_path),
                    "geojson_path": str(geojson_path),
                    "features": features,
                    "tile_bounds_wgs84": bounds,
                    "sensor": "GF1",
                }
                samples.append(sample)

    return samples


def group_tiles_by_spatial_key(
    samples: List[dict],
) -> Dict[Tuple, List[dict]]:
    """Group samples by spatial tile_group_key.

    Returns a dict mapping group_key → list of samples (from different
    class directories covering the same tile area).
    """
    groups: Dict[Tuple, List[dict]] = {}

    for s in samples:
        original_size = (s["original_width"], s["original_height"])
        key = make_tile_group_key(
            tile_bounds_wgs84=s["tile_bounds_wgs84"],
            original_size=original_size,
        )
        groups.setdefault(key, []).append(s)

    return groups


def build_merged_samples(
    samples: List[dict],
    groups: Optional[Dict[Tuple, List[dict]]] = None,
) -> List[dict]:
    """Build merged sample list from grouped raw samples.

    For each group, merges GeoJSON features from all constituent class
    directories into a single sample dict:

        sample_id: "{group_index:06d}_{n_classes}cls"
        known_classes: [list of DLMC names]
        features: concatenated GeoJSON feature list
        image_path: from the first sample in group
        tile_bounds_wgs84: same for all in group
        original_size: [width, height]
        sensor: "GF1"
        train_ids: [list of train_id values]

    Args:
        samples: Output from scan_landcover_directories().
        groups: Optional pre-computed grouping. If None, computed from samples.

    Returns:
        List of merged sample dicts, one per spatial tile.
    """
    if groups is None:
        groups = group_tiles_by_spatial_key(samples)

    merged: List[dict] = []

    for group_idx, (key, group_samples) in enumerate(sorted(groups.items())):
        all_features: List[dict] = []
        known_classes: List[str] = []
        train_ids: List[int] = []

        for s in group_samples:
            dlmc = s["dlmc"]
            if dlmc not in known_classes:
                known_classes.append(dlmc)
                train_ids.append(dlmc_to_train_id(dlmc))
            for feat in s["features"]:
                # Tag feature with its source DLMC for rasterization
                feat_copy = dict(feat)
                feat_copy.setdefault("properties", {})
                feat_copy["properties"]["DLMC"] = dlmc
                all_features.append(feat_copy)

        first = group_samples[0]
        sample_id = f"{group_idx:06d}_{len(known_classes)}cls"

        merged.append({
            "sample_id": sample_id,
            "image_path": first["image_path"],
            "known_classes": known_classes,
            "train_ids": train_ids,
            "features": all_features,
            "tile_bounds_wgs84": first["tile_bounds_wgs84"],
            "original_size": [first["original_width"], first["original_height"]],
            "sensor": first["sensor"],
        })

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_all_coords(geom: dict) -> List[Tuple[float, float]]:
    """Recursively extract all (lon, lat) pairs from a GeoJSON geometry."""
    coords: List[Tuple[float, float]] = []
    geom_type = geom.get("type")
    raw = geom.get("coordinates")

    if geom_type == "Point":
        coords.append((float(raw[0]), float(raw[1])))
    elif geom_type in ("MultiPoint", "LineString"):
        for pt in raw:
            coords.append((float(pt[0]), float(pt[1])))
    elif geom_type in ("MultiLineString", "Polygon"):
        for ring in raw:
            for pt in ring:
                coords.append((float(pt[0]), float(pt[1])))
    elif geom_type == "MultiPolygon":
        for polygon in raw:
            for ring in polygon:
                for pt in ring:
                    coords.append((float(pt[0]), float(pt[1])))

    return coords


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    samples = scan_landcover_directories()
    print(f"Scanned {len(samples)} raw GeoJSON samples")

    groups = group_tiles_by_spatial_key(samples)
    print(f"Grouped into {len(groups)} unique spatial tiles")

    # Stats
    group_sizes = [len(g) for g in groups.values()]
    single = sum(1 for s in group_sizes if s == 1)
    multi = sum(1 for s in group_sizes if s > 1)
    max_merge = max(group_sizes) if group_sizes else 0
    print(f"  Single-class tiles: {single}")
    print(f"  Multi-class tiles:  {multi}")
    print(f"  Max classes per tile: {max_merge}")

    # Group sizes distribution
    from collections import Counter
    dist = Counter(group_sizes)
    for size, count in sorted(dist.items()):
        print(f"    {size} class(es): {count} tiles")

    # By original size
    by_size = {}
    for s in samples:
        by_size.setdefault(s["size"], []).append(s)
    for sz, items in sorted(by_size.items()):
        print(f"  {sz}: {len(items)} samples")

    # Build merged samples
    merged = build_merged_samples(samples, groups)
    print(f"\nMerged: {len(merged)} tile samples")

    total_classes = sum(m["train_ids"][-1] for m in merged)  # just a check
    avg_classes = sum(len(m["known_classes"]) for m in merged) / max(len(merged), 1)
    print(f"  Avg classes per tile: {avg_classes:.2f}")

    # Show first multi-class tile
    for m in merged:
        if len(m["known_classes"]) > 1:
            print(f"\nExample multi-class tile:")
            print(f"  sample_id: {m['sample_id']}")
            print(f"  known_classes: {m['known_classes']}")
            print(f"  train_ids: {m['train_ids']}")
            print(f"  bounds: {m['tile_bounds_wgs84']}")
            print(f"  original_size: {m['original_size']}")
            print(f"  n_features: {len(m['features'])}")
            break

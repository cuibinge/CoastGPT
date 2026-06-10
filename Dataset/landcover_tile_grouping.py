"""
Tile grouping: scan land cover class directories and merge per-class binary TIF
labels that belong to the same spatial tile.

A tile_group_key is built from (source_image_key, grid, size).
Tiles from different class directories sharing the same key have their
binary TIFs merged into a multi-class partial label target.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.landcover_label_map import dlmc_to_train_id, dir_name_to_dlmc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PATCHES_ROOT = Path("/home/ma-user/work/Stage3Data/土地分类/Patches")
_IMAGE_SUBDIR = "Image_TrueColor"
_BINARY_SUBDIR = "Label_Binary"
_SIZES = ["Size_128", "Size_256", "Size_512"]

# Filename pattern for grid and source: ..._R071C040_128_...
_GRID_PATTERN = re.compile(r"(R\d+C\d+)")


# ---------------------------------------------------------------------------
# Source image key extraction
# ---------------------------------------------------------------------------


def extract_source_image_key(filename_stem: str) -> str:
    """Extract source image key from filename stem.

    Example: "公路用地_GF1_PMS2_E119.6_N34.6_20251109_连云区_R071C040_128_Binary_CK"
    Returns: "GF1_PMS2_E119.6_N34.6_20251109_连云区"
    """
    parts = filename_stem.split("_")
    sensor_idx = next(
        (i for i, p in enumerate(parts) if p in ("GF1", "GF2", "GF6")), None
    )
    if sensor_idx is None:
        return filename_stem

    key_parts = []
    for p in parts[sensor_idx:]:
        if _GRID_PATTERN.match(p):
            break
        key_parts.append(p)
    return "_".join(key_parts)


def extract_grid(filename_stem: str) -> Optional[str]:
    """Extract grid coordinate like 'R071C040' from filename stem."""
    m = _GRID_PATTERN.search(filename_stem)
    return m.group(1) if m else None


def extract_size_int(filename_stem: str) -> Optional[int]:
    """Extract size integer (128, 256, 512) from filename stem."""
    parts = filename_stem.split("_")
    for p in parts:
        if p in ("128", "256", "512"):
            return int(p)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_landcover_directories(
    patches_root: Optional[str] = None,
    sizes: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
) -> List[dict]:
    """Scan class directories for binary TIF labels and return raw tile samples.

    Each sample dict contains:
        class_dir: directory name (e.g. "公路用地")
        dlmc: canonical DLMC class name
        size: "Size_128" | "Size_256" | "Size_512"
        original_width, original_height: tile dimensions
        image_path: Path to the JPEG image
        tif_path: Path to the binary TIF label
        grid: grid coordinate string (e.g. "R071C040")
        source_image_key: unique key per satellite acquisition
        sensor: "GF1"

    Args:
        patches_root: Root of the land cover patches data.
        sizes: Which size directories to scan.
        classes: Which class directories to scan (None = all).

    Returns:
        List of sample dicts, one per discovered TIF file.
    """
    root = Path(patches_root or _PATCHES_ROOT)
    size_dirs = sizes or _SIZES
    samples: List[dict] = []

    if classes is None:
        class_dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    else:
        class_dirs = list(classes)

    for cls_dir_name in class_dirs:
        cls_path = root / cls_dir_name / "GF1"
        if not cls_path.exists():
            continue

        dlmc = dir_name_to_dlmc(cls_dir_name)

        for size_dir_name in size_dirs:
            binary_dir = cls_path / size_dir_name / _BINARY_SUBDIR
            image_dir = cls_path / size_dir_name / _IMAGE_SUBDIR
            if not binary_dir.exists():
                continue

            orig_size = int(size_dir_name.split("_")[1])

            for tif_path in sorted(binary_dir.glob("*.tif")):
                stem = tif_path.stem

                # Extract grid and source key from filename
                grid = extract_grid(stem)
                if grid is None:
                    continue

                source_key = extract_source_image_key(stem)

                # Derive image path: _Binary_CK.tif → _True_CK.jpg
                img_stem = stem.replace("_Binary_CK", "_True_CK")
                image_path = image_dir / f"{img_stem}.jpg"
                if not image_path.exists():
                    img_stem2 = stem.replace("_Binary_CK", "_True")
                    alt_path = image_dir / f"{img_stem2}.jpg"
                    if alt_path.exists():
                        image_path = alt_path
                    else:
                        continue

                # Derive GeoJSON path for tile bounds lookup: _Binary_CK.tif → _Label_CK.geojson
                geojson_dir = cls_path / size_dir_name / "Label_GeoJSON"
                gj_stem = stem.replace("_Binary_CK", "_Label_CK")
                geojson_path = geojson_dir / f"{gj_stem}.geojson"

                samples.append({
                    "class_dir": cls_dir_name,
                    "dlmc": dlmc,
                    "size": size_dir_name,
                    "original_width": orig_size,
                    "original_height": orig_size,
                    "image_path": str(image_path),
                    "tif_path": str(tif_path),
                    "geojson_path": str(geojson_path) if geojson_path.exists() else None,
                    "grid": grid,
                    "source_image_key": source_key,
                    "sensor": "GF1",
                })

    return samples


def group_tiles_by_spatial_key(
    samples: List[dict],
) -> Dict[Tuple, List[dict]]:
    """Group samples by (source_image_key, grid, size).

    Tiles from different class directories that share the same satellite
    acquisition, grid cell, and resolution are merged.
    """
    groups: Dict[Tuple, List[dict]] = {}

    for s in samples:
        key = (
            s["source_image_key"],
            s["grid"],
            (s["original_width"], s["original_height"]),
        )
        groups.setdefault(key, []).append(s)

    return groups


def build_merged_samples(
    samples: List[dict],
    groups: Optional[Dict[Tuple, List[dict]]] = None,
) -> List[dict]:
    """Build merged sample list from grouped raw samples.

    For each group, collects per-class TIF paths into a single sample:

        sample_id: "{group_index:06d}_{n_classes}cls"
        known_classes: [list of DLMC names]
        class_tifs: [{"tif_path": str, "train_id": int}, ...]
        image_path: from the first sample in group
        original_size: [width, height]
        sensor: "GF1"
        train_ids: [list of train_id values]

    Args:
        samples: Output from scan_landcover_directories().
        groups: Optional pre-computed grouping.

    Returns:
        List of merged sample dicts, one per spatial tile.
    """
    if groups is None:
        groups = group_tiles_by_spatial_key(samples)

    merged: List[dict] = []

    for group_idx, (_key, group_samples) in enumerate(sorted(groups.items())):
        known_classes: List[str] = []
        train_ids: List[int] = []
        class_tifs: List[dict] = []

        for s in group_samples:
            dlmc = s["dlmc"]
            if dlmc not in known_classes:
                known_classes.append(dlmc)
                train_ids.append(dlmc_to_train_id(dlmc))
            class_tifs.append({
                "tif_path": s["tif_path"],
                "train_id": dlmc_to_train_id(dlmc),
                "dlmc": dlmc,
            })

        first = group_samples[0]
        sample_id = f"{group_idx:06d}_{len(known_classes)}cls"

        # Derive tile bounds from any GeoJSON in the group (for export)
        tile_bounds = _load_tile_bounds_from_group(group_samples)

        merged.append({
            "sample_id": sample_id,
            "image_path": first["image_path"],
            "known_classes": known_classes,
            "train_ids": train_ids,
            "class_tifs": class_tifs,
            "original_size": [first["original_width"], first["original_height"]],
            "grid": first["grid"],
            "sensor": first["sensor"],
            "tile_bounds_wgs84": tile_bounds,
        })

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tile_bounds_from_group(
    group_samples: List[dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Derive WGS84 tile bounds from any GeoJSON file in a group.

    Only used for export (GeoJSON prediction output). Returns None if
    no GeoJSON file is available.
    """
    import json
    for s in group_samples:
        gj_path = s.get("geojson_path")
        if gj_path is None:
            continue
        try:
            with open(gj_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            features = gj.get("features", [])
            if features:
                bounds = compute_tile_bounds_from_geojson(features)
                if bounds is not None:
                    return bounds
        except (json.JSONDecodeError, OSError):
            continue
    return None


# ---------------------------------------------------------------------------
# Compatibility: compute_tile_bounds_from_geojson
# ---------------------------------------------------------------------------


def compute_tile_bounds_from_geojson(
    features: List[dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Compute WGS84 bounding box from GeoJSON features. Deprecated."""
    all_lons, all_lats = [], []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if geom is None:
            continue
        for lon, lat in _extract_all_coords(geom):
            all_lons.append(lon)
            all_lats.append(lat)
    if not all_lons:
        return None
    return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))


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
    print(f"Scanned {len(samples)} raw TIF samples")

    groups = group_tiles_by_spatial_key(samples)
    print(f"Grouped into {len(groups)} unique spatial tiles")

    group_sizes = [len(g) for g in groups.values()]
    from collections import Counter
    dist = Counter(group_sizes)
    for size, count in sorted(dist.items()):
        print(f"  {size} class(es): {count} tiles")

    by_size = {}
    for s in samples:
        by_size.setdefault(s["size"], []).append(s)
    for sz, items in sorted(by_size.items()):
        print(f"  {sz}: {len(items)} samples")

    merged = build_merged_samples(samples, groups)
    print(f"\nMerged: {len(merged)} tile samples")
    single = sum(1 for m in merged if len(m["known_classes"]) == 1)
    multi = sum(1 for m in merged if len(m["known_classes"]) > 1)
    print(f"  Single-class: {single}, Multi-class: {multi}")

    if multi > 0:
        for m in merged:
            if len(m["known_classes"]) > 1:
                print(f"\nExample multi-class tile:")
                print(f"  sample_id: {m['sample_id']}")
                print(f"  known_classes: {m['known_classes']}")
                print(f"  train_ids: {m['train_ids']}")
                print(f"  original_size: {m['original_size']}")
                for ct in m["class_tifs"]:
                    print(f"    {ct['dlmc']}: {ct['tif_path']}")
                break

"""
Rasterize GeoJSON features to a 224x224 target mask in model pixel space.

Conventions:
- WGS84 coords → model pixel via georef (EPSG:4326 direct affine).
- target initialized to IGNORE_INDEX=255.
- Known polygon regions set to their train_id.
- Unlabeled regions stay 255 (ignore).
- Overlapping polygons from different classes are logged as conflicts;
  last write wins in the mask.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.georef_transform import wgs84_to_pixel, clip_pixel_coords
from utils.mask_utils import rasterize_polygon
from Dataset.landcover_label_map import IGNORE_INDEX, BACKGROUND_ID, dlmc_to_train_id


def compute_model_transform_from_bounds(
    tile_bounds_wgs84: Tuple[float, float, float, float],
    model_size: Tuple[int, int] = (224, 224),
) -> List[float]:
    """Compute a GDAL-order affine transform from tile bounds.

    Maps model pixel space → WGS84 (EPSG:4326). Since source is WGS84,
    this is a simple linear mapping:
        lon = col * (width_deg / width_px) + min_lon
        lat = max_lat - row * (height_deg / height_px)

    Args:
        tile_bounds_wgs84: (min_lon, min_lat, max_lon, max_lat).
        model_size: (width, height) of model input.

    Returns:
        GDAL affine [a, b, c, d, e, f] = [px_w, 0, min_lon, 0, -px_h, max_lat].
    """
    min_lon, min_lat, max_lon, max_lat = tile_bounds_wgs84
    px_w = (max_lon - min_lon) / model_size[0]
    px_h = (max_lat - min_lat) / model_size[1]
    return [px_w, 0.0, min_lon, 0.0, -px_h, max_lat]


def rasterize_features_to_target(
    features: List[dict],
    tile_bounds_wgs84: Tuple[float, float, float, float],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, List[str]]:
    """Rasterize a list of GeoJSON features to a partial-label target mask.

    Args:
        features: List of GeoJSON Feature dicts. Each feature must have
            ``geometry`` (Polygon or MultiPolygon in WGS84) and
            ``properties.DLMC`` (class name string).
        tile_bounds_wgs84: (min_lon, min_lat, max_lon, max_lat).
        target_size: (width, height) output mask dimensions.

    Returns:
        target: np.ndarray[uint8] of shape (H, W). Values:
            0 = background, 1-24 = active class, 255 = ignore.
        conflicts: List of conflict descriptions (same pixel, different classes).
    """
    model_transform = compute_model_transform_from_bounds(
        tile_bounds_wgs84, target_size
    )
    georef = {
        "source_crs": "EPSG:4326",
        "model_transform": model_transform,
    }

    H, W = target_size[1], target_size[0]
    target = np.full((H, W), IGNORE_INDEX, dtype=np.uint8)

    # Track per-pixel source class for conflict detection
    pixel_source: Optional[np.ndarray] = None

    conflicts: List[str] = []

    for feat in features:
        geom = feat.get("geometry")
        if geom is None:
            continue

        props = feat.get("properties", {})
        dlmc = props.get("DLMC", "")
        if not dlmc:
            continue

        try:
            train_id = dlmc_to_train_id(dlmc)
        except KeyError:
            warnings.warn(f"Unknown DLMC '{dlmc}', skipping feature")
            continue

        geom_type = geom.get("type")
        coords = geom.get("coordinates")

        if geom_type == "Polygon":
            _rasterize_polygon(
                coords, georef, target, train_id, target_size, pixel_source, conflicts
            )
        elif geom_type == "MultiPolygon":
            for poly_coords in coords:
                _rasterize_polygon(
                    poly_coords, georef, target, train_id, target_size,
                    pixel_source, conflicts
                )
        else:
            warnings.warn(f"Unsupported geometry type: {geom_type}")
            continue

    return target, conflicts


def _rasterize_polygon(
    coords: list,
    georef: dict,
    target: np.ndarray,
    train_id: int,
    target_size: Tuple[int, int],
    pixel_source: Optional[np.ndarray],
    conflicts: List[str],
) -> None:
    """Rasterize one GeoJSON polygon ring set into the target mask."""
    W, H = target_size

    # Outer ring WGS84 → pixel
    outer_wgs84 = [(float(lon), float(lat)) for lon, lat in coords[0]]
    outer_pixel = wgs84_to_pixel(outer_wgs84, georef)
    outer_pixel = clip_pixel_coords(outer_pixel, W, H)

    # Hole rings
    holes_pixel = None
    if len(coords) > 1:
        holes_pixel = []
        for hole_ring in coords[1:]:
            hole_wgs84 = [(float(lon), float(lat)) for lon, lat in hole_ring]
            hole_pixel = wgs84_to_pixel(hole_wgs84, georef)
            hole_pixel = clip_pixel_coords(hole_pixel, W, H)
            holes_pixel.append(hole_pixel)

    mask = rasterize_polygon(outer_pixel, W, H, holes_pixel)
    mask_bool = mask > 0

    # Conflict detection: check if mask region already has a different class
    if pixel_source is not None and mask_bool.any():
        existing = pixel_source[mask_bool]
        existing_valid = existing[existing > 0]
        diff = existing_valid[existing_valid != train_id]
        if len(diff) > 0:
            conflicts.append(
                f"train_id={train_id} overlaps train_id={int(diff[0])} "
                f"({len(diff)} px affected)"
            )

    # Write class id to target
    target[mask_bool] = train_id


if __name__ == "__main__":
    print("Testing rasterize_features_to_target with synthetic data...")

    # Synthetic features: two rectangles at different locations
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [119.474, 34.7018],
                    [119.475, 34.7018],
                    [119.475, 34.7020],
                    [119.474, 34.7020],
                    [119.474, 34.7018],
                ]],
            },
            "properties": {"DLMC": "水田"},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [119.4755, 34.7013],
                    [119.476, 34.7013],
                    [119.476, 34.7015],
                    [119.4755, 34.7015],
                    [119.4755, 34.7013],
                ]],
            },
            "properties": {"DLMC": "公路用地"},
        },
    ]

    bounds = (119.474, 34.7012, 119.476, 34.7020)
    target, conflicts = rasterize_features_to_target(features, bounds, (224, 224))

    print(f"Target shape: {target.shape}")
    print(f"Target dtype: {target.dtype}")
    unique = np.unique(target)
    print(f"Unique values: {unique}")

    # Verify ignore is 255
    assert 255 in unique, "255 (ignore) must be present"
    # Verify two class IDs are present
    water_id = dlmc_to_train_id("水田")
    road_id = dlmc_to_train_id("公路用地")
    assert water_id in unique, f"train_id {water_id} (水田) not found in target"
    assert road_id in unique, f"train_id {road_id} (公路用地) not found in target"
    assert len(conflicts) == 0, f"Unexpected conflicts: {conflicts}"

    # Each class should have some pixels
    assert (target == water_id).sum() > 0, "No water pixels"
    assert (target == road_id).sum() > 0, "No road pixels"
    # Background should be 0 only if not labeled
    assert (target == BACKGROUND_ID).sum() == 0, "No bg should be set in partial labels"

    print(f"  Water pixels: {(target == water_id).sum()}")
    print(f"  Road pixels: {(target == road_id).sum()}")
    print(f"  Ignore pixels: {(target == IGNORE_INDEX).sum()}")
    print("All rasterize checks passed.")

"""GeoJSON coordinate helpers used by Stage-3 data preparation.

The utilities here keep tile-relative coordinate normalization and mojibake
repair separate from tokenizer vocabulary management.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


TileTransform = Dict[str, float]


def make_tile_transform(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> TileTransform:
    width = float(max_x) - float(min_x)
    height = float(max_y) - float(min_y)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid tile bounds: ({min_x}, {min_y}, {max_x}, {max_y})")
    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width": width,
        "height": height,
    }


def normalize_xy(x: float, y: float, transform: TileTransform) -> Tuple[float, float]:
    nx = (float(x) - transform["min_x"]) / transform["width"]
    ny = (transform["max_y"] - float(y)) / transform["height"]
    return nx, ny


def denormalize_xy(nx: float, ny: float, transform: TileTransform) -> Tuple[float, float]:
    x = transform["min_x"] + float(nx) * transform["width"]
    y = transform["max_y"] - float(ny) * transform["height"]
    return x, y


def _is_coord_leaf(coords) -> bool:
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return False
    return all(
        not isinstance(v, bool) and isinstance(v, (int, float, str))
        for v in coords[:2]
    )


def _walk_coordinates(coords: Any, fn) -> Any:
    if isinstance(coords, (list, tuple)):
        if _is_coord_leaf(coords):
            return list(fn(coords))
        return [_walk_coordinates(item, fn) for item in coords]
    return coords


def transform_geometry_coordinates(geometry: Dict, fn) -> Dict:
    if not isinstance(geometry, dict):
        return geometry
    out = dict(geometry)
    if "coordinates" in out:
        out["coordinates"] = _walk_coordinates(out["coordinates"], fn)
    if isinstance(out.get("geometries"), list):
        out["geometries"] = [
            transform_geometry_coordinates(g, fn) for g in out["geometries"]
        ]
    return out


def encode_feature_collection(
    feature_collection: Dict,
    transform: Optional[TileTransform] = None,
) -> Dict:
    """Rewrite coordinates in a FeatureCollection to normalized floats."""
    if transform is None:
        return feature_collection

    def _convert(point):
        x, y = point[0], point[1]
        nx, ny = normalize_xy(x, y, transform)
        return [round(nx, 4), round(ny, 4)]

    new_features = []
    for feature in feature_collection.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            geometry = transform_geometry_coordinates(geometry, _convert)
        new_features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": feature.get("properties", {}),
        })
    out = dict(feature_collection)
    out["features"] = new_features
    return out


def decode_feature_collection(
    feature_collection: Dict,
    transform: TileTransform,
) -> Dict:
    """Inverse of encode_feature_collection for normalized float coordinates."""

    def _convert(point):
        rx, ry = denormalize_xy(float(point[0]), float(point[1]), transform)
        return [rx, ry]

    new_features = []
    for feature in feature_collection.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            geometry = transform_geometry_coordinates(geometry, _convert)
        new_features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": feature.get("properties", {}),
        })
    out = dict(feature_collection)
    out["features"] = new_features
    return out


def feature_collection_bbox(feature_collection: Dict) -> Optional[Tuple[float, float, float, float]]:
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    found = False

    def _scan(coords):
        nonlocal min_x, min_y, max_x, max_y, found
        if isinstance(coords, (list, tuple)):
            if (
                len(coords) >= 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in coords[:2])
            ):
                x = float(coords[0])
                y = float(coords[1])
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                found = True
            else:
                for item in coords:
                    _scan(item)

    for feature in feature_collection.get("features", []):
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if isinstance(geom, dict):
            _scan(geom.get("coordinates"))
            for sub in geom.get("geometries", []) or []:
                _scan(sub.get("coordinates"))
    if not found:
        return None
    return (min_x, min_y, max_x, max_y)


def tile_transform_from_raster(image_path) -> Optional[TileTransform]:
    try:
        import rasterio  # type: ignore
        from rasterio.warp import transform_bounds  # type: ignore
    except Exception:
        return None
    try:
        with rasterio.open(str(image_path)) as ds:
            if ds.crs is None or ds.transform is None:
                return None
            bounds = ds.bounds
            if ds.crs.to_epsg() != 4326:
                left, bottom, right, top = transform_bounds(
                    ds.crs, "EPSG:4326", *bounds, densify_pts=21
                )
            else:
                left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
            return make_tile_transform(left, bottom, right, top)
    except Exception:
        return None


def tile_transform_from_feature_collection(
    feature_collection: Dict,
    pad_ratio: float = 0.0,
) -> Optional[TileTransform]:
    bbox = feature_collection_bbox(feature_collection)
    if bbox is None:
        return None
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return None
    if pad_ratio > 0:
        min_x -= width * pad_ratio
        max_x += width * pad_ratio
        min_y -= height * pad_ratio
        max_y += height * pad_ratio
    return make_tile_transform(min_x, min_y, max_x, max_y)


_MOJIBAKE_HINTS = (
    "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?, "generic?,
)


def looks_mojibake(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return any(ch in text for ch in _MOJIBAKE_HINTS)


def repair_mojibake_str(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    if not looks_mojibake(text):
        return text
    try:
        repaired = text.encode("gbk", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        try:
            repaired = text.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return text
    if not repaired or looks_mojibake(repaired):
        return text
    return repaired


def repair_mojibake_in_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return repair_mojibake_str(obj)
    if isinstance(obj, dict):
        return {
            repair_mojibake_str(k) if isinstance(k, str) else k: repair_mojibake_in_obj(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [repair_mojibake_in_obj(v) for v in obj]
    return obj

"""GeoJSON builder: pixel -> WGS84 FeatureCollection, validation, and Mask R-CNN output conversion.

Conventions:
- Pixel coordinates: (col, row) in model pixel space (224x224).
- WGS84 coordinates: (lon, lat) tuples, written to GeoJSON as [lon, lat].
- GeoJSON Polygon rings: closed (first == last).
"""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from .georef_transform import pixel_to_wgs84
    from .mask_utils import filter_small_polygons, mask_to_polygon
except ImportError:
    from georef_transform import pixel_to_wgs84
    from mask_utils import filter_small_polygons, mask_to_polygon

try:
    from shapely.geometry import shape as shapely_shape
    from shapely.validation import explain_validity

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _close_ring(ring: List[Tuple[float, float]], eps: float = 1e-12) -> List[Tuple[float, float]]:
    """Ensure the ring is closed (first == last)."""
    if len(ring) < 1:
        return ring
    first = ring[0]
    last = ring[-1]
    if abs(first[0] - last[0]) > eps or abs(first[1] - last[1]) > eps:
        return ring + [first]
    return ring


def _flatten_geojson_coords(geom: dict) -> List[Tuple[float, float]]:
    """Extract all (lon, lat) coordinate pairs from a GeoJSON Polygon geometry."""
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) == 0:
        return []
    # Polygon coords are [[ring], ...]; flatten outer ring
    outer = coords[0]
    if not isinstance(outer, list):
        return []
    return [tuple(pt) for pt in outer if isinstance(pt, (list, tuple)) and len(pt) >= 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def polygon_pixel_to_geojson_feature(
    polygon_pixel: List[Tuple[float, float]],
    georef: dict,
    class_name: str = "海水养殖区",
    confidence: float = 1.0,
) -> dict:
    """Convert a pixel-space polygon [(col, row), ...] to a GeoJSON Feature.

    Args:
        polygon_pixel: Polygon vertices in model pixel space.
        georef: Dict with ``source_crs`` and ``model_transform`` (GDAL affine).
        class_name: Value for ``properties.class``.
        confidence: Value for ``properties.confidence``.

    Returns:
        GeoJSON Feature dict with Polygon geometry in WGS84.
    """
    if len(polygon_pixel) < 3:
        raise ValueError(f"Polygon requires at least 3 vertices, got {len(polygon_pixel)}")

    # Pixel -> WGS84; pixel_to_wgs84 returns (lon, lat) tuples
    coords_wgs84 = pixel_to_wgs84(polygon_pixel, georef)

    # Close the ring (GeoJSON requires first == last)
    coords_wgs84 = _close_ring(coords_wgs84)

    # GeoJSON format: [lon, lat] (list, not tuple)
    coordinates = [[lon, lat] for lon, lat in coords_wgs84]

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        },
        "properties": {
            "class": class_name,
            "confidence": round(float(confidence), 4),
        },
    }


def build_feature_collection(features: List[dict]) -> dict:
    """Wrap a list of GeoJSON Features into a FeatureCollection.

    Args:
        features: List of GeoJSON Feature dicts.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    return {"type": "FeatureCollection", "features": features}


def validate_geojson(
    feature_collection: dict,
    tile_bounds_wgs84: Optional[List[float]] = None,
    min_area_px: float = 8.0,
    bounds_eps: float = 1e-6,
) -> Dict[str, Any]:
    """Validate a GeoJSON FeatureCollection.

    Returns a dict with keys: valid, parse_ok, schema_ok,
    geometry_valid_rate, empty_output, errors, repaired_features.

    Args:
        feature_collection: GeoJSON FeatureCollection dict.
        tile_bounds_wgs84: Optional [min_lon, min_lat, max_lon, max_lat] for
            coordinate bounds checking.
        min_area_px: Minimum polygon area (pixel^2). Passed through to
            callers; not directly enforced in validation.
        bounds_eps: Tolerance for bounds checking (degrees).
    """
    result: Dict[str, Any] = {
        "valid": False,
        "parse_ok": False,
        "schema_ok": False,
        "geometry_valid_rate": 0.0,
        "empty_output": False,
        "errors": [],
        "repaired_features": 0,
    }

    # ---- 1. Parse check ----
    if not isinstance(feature_collection, dict):
        result["errors"].append("feature_collection is not a dict")
        return result
    result["parse_ok"] = True

    # ---- 2. Schema check ----
    if feature_collection.get("type") != "FeatureCollection":
        result["errors"].append(
            f"top-level type must be 'FeatureCollection', got {feature_collection.get('type')!r}"
        )
        return result

    if "features" not in feature_collection:
        result["errors"].append("missing 'features' key")
        return result

    features = feature_collection["features"]
    if not isinstance(features, list):
        result["errors"].append("'features' is not a list")
        return result

    result["schema_ok"] = True

    # ---- 3. Empty features ----
    if len(features) == 0:
        result["empty_output"] = True
        result["geometry_valid_rate"] = 1.0
        result["valid"] = True
        return result

    # ---- 4. Per-feature validation ----
    min_lon_b, min_lat_b, max_lon_b, max_lat_b = None, None, None, None
    check_bounds = tile_bounds_wgs84 is not None
    if check_bounds:
        if not isinstance(tile_bounds_wgs84, (list, tuple)) or len(tile_bounds_wgs84) != 4:
            result["errors"].append("tile_bounds_wgs84 must be [min_lon, min_lat, max_lon, max_lat]")
        else:
            min_lon_b, min_lat_b, max_lon_b, max_lat_b = (
                tile_bounds_wgs84[0] - bounds_eps,
                tile_bounds_wgs84[1] - bounds_eps,
                tile_bounds_wgs84[2] + bounds_eps,
                tile_bounds_wgs84[3] + bounds_eps,
            )

    valid_count = 0
    for idx, feat in enumerate(features):
        feat_ok = True

        # Must be a dict
        if not isinstance(feat, dict):
            result["errors"].append(f"Feature[{idx}] is not a dict")
            continue

        # type == Feature
        if feat.get("type") != "Feature":
            result["errors"].append(
                f"Feature[{idx}] type must be 'Feature', got {feat.get('type')!r}"
            )
            feat_ok = False

        # geometry must exist and be a dict
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            result["errors"].append(f"Feature[{idx}] missing 'geometry' or not a dict")
            continue

        # geometry.type == Polygon
        if geom.get("type") != "Polygon":
            result["errors"].append(
                f"Feature[{idx}] geometry.type must be 'Polygon', got {geom.get('type')!r}"
            )
            continue

        # coordinates are valid
        coords = geom.get("coordinates")
        if not isinstance(coords, list) or len(coords) == 0:
            result["errors"].append(f"Feature[{idx}] coordinates is empty or not a list")
            feat_ok = False
        elif not isinstance(coords[0], list) or len(coords[0]) < 4:
            result["errors"].append(
                f"Feature[{idx}] Polygon outer ring has fewer than 4 points (need 3 distinct + 1 closing)"
            )
            feat_ok = False
        else:
            # Validate each coordinate point is [lon, lat]
            bad_pts = False
            for pt_idx, pt in enumerate(coords[0]):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    result["errors"].append(
                        f"Feature[{idx}] coordinate[{pt_idx}] is not [lon, lat]: {pt!r}"
                    )
                    bad_pts = True
                elif not all(isinstance(v, (int, float)) for v in pt):
                    result["errors"].append(
                        f"Feature[{idx}] coordinate[{pt_idx}] has non-numeric values: {pt!r}"
                    )
                    bad_pts = True
            if bad_pts:
                feat_ok = False

        if not feat_ok:
            continue

        # ---- Bounds check ----
        if check_bounds and min_lon_b is not None:
            flat_coords = _flatten_geojson_coords(geom)
            out_of_bounds = []
            for lon, lat in flat_coords:
                if not (min_lon_b <= lon <= max_lon_b and min_lat_b <= lat <= max_lat_b):
                    out_of_bounds.append((lon, lat))
            if out_of_bounds:
                result["errors"].append(
                    f"Feature[{idx}] has {len(out_of_bounds)} coordinates outside "
                    f"tile bounds [{min_lon_b + bounds_eps:.6f}, {min_lat_b + bounds_eps:.6f}, "
                    f"{max_lon_b - bounds_eps:.6f}, {max_lat_b - bounds_eps:.6f}]"
                )
                feat_ok = False

        # ---- Shapely validation ----
        if feat_ok and HAS_SHAPELY:
            try:
                shp_geom = shapely_shape(geom)
                if not shp_geom.is_valid:
                    # Attempt buffer(0) repair
                    repaired = shp_geom.buffer(0)
                    if repaired.is_valid and not repaired.is_empty:
                        result["repaired_features"] += 1
                    else:
                        reason = explain_validity(shp_geom)
                        result["errors"].append(
                            f"Feature[{idx}] geometry invalid (unrepairable): {reason}"
                        )
                        feat_ok = False
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(
                    f"Feature[{idx}] shapely geometry build failed: {exc}"
                )
                feat_ok = False

        if feat_ok:
            valid_count += 1

    result["geometry_valid_rate"] = valid_count / len(features) if len(features) > 0 else 1.0
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
    """Convert Mask R-CNN outputs to GeoJSON FeatureCollections.

    One FeatureCollection per image.

    Args:
        outputs: List of Mask R-CNN ``{boxes, labels, scores, masks}`` dicts.
        metas: List of metadata dicts (must contain ``source_crs`` and
            ``model_transform`` for coordinate conversion).
        score_thresh: Minimum confidence score to keep a detection.
        mask_thresh: Binary threshold for logit masks (> threshold = foreground).
        min_area_px: Minimum polygon area in pixels (passed to
            ``filter_small_polygons``).

    Returns:
        List of GeoJSON FeatureCollection dicts, one per image.
    """
    results: List[dict] = []
    for output, meta in zip(outputs, metas):
        features: List[dict] = []
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        masks = output["masks"].cpu().numpy()  # [N, 1, H, W]

        for i in range(len(scores)):
            if float(scores[i]) < score_thresh:
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

        results.append(build_feature_collection(features))
    return results


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    # ------------------------------------------------------------------
    # Mock georef: maps pixel (0,0) -> (119.3, 35.07), (224,224) -> (119.31, 35.06)
    # ------------------------------------------------------------------
    georef = {
        "source_crs": "EPSG:4326",
        "model_transform": [4.464e-05, 0.0, 119.3, 0.0, -4.464e-05, 35.07],
    }

    # ------------------------------------------------------------------
    # 1. polygon_pixel_to_geojson_feature
    # ------------------------------------------------------------------
    poly_pixel = [(40.0, 50.0), (160.0, 50.0), (160.0, 180.0), (40.0, 180.0)]
    feat = polygon_pixel_to_geojson_feature(poly_pixel, georef, confidence=0.95)
    assert feat["type"] == "Feature", f"Expected Feature, got {feat['type']}"
    assert feat["geometry"]["type"] == "Polygon"
    # Outer ring must be closed (first == last), so at least 5 points for a rectangle
    assert len(feat["geometry"]["coordinates"][0]) >= 4
    assert feat["properties"]["confidence"] == 0.95
    assert feat["properties"]["class"] == "海水养殖区"
    # Verify GeoJSON order is [lon, lat]
    first_pt = feat["geometry"]["coordinates"][0][0]
    assert isinstance(first_pt, list) and len(first_pt) == 2
    print("polygon_pixel_to_geojson_feature: OK")

    # ------------------------------------------------------------------
    # 2. build_feature_collection
    # ------------------------------------------------------------------
    fc = build_feature_collection([feat])
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    print("build_feature_collection: OK")

    # ------------------------------------------------------------------
    # 3. validate_geojson — valid data
    # ------------------------------------------------------------------
    result = validate_geojson(fc, tile_bounds_wgs84=[119.29, 35.05, 119.32, 35.08])
    assert result["valid"], f"Validation failed: {result['errors']}"
    assert result["parse_ok"]
    assert result["schema_ok"]
    assert result["geometry_valid_rate"] == 1.0
    assert not result["empty_output"]
    print(f"validate_geojson (valid): parse_ok={result['parse_ok']}, "
          f"schema_ok={result['schema_ok']}, rate={result['geometry_valid_rate']}")

    # ------------------------------------------------------------------
    # 4. validate_geojson — empty FeatureCollection
    # ------------------------------------------------------------------
    empty_fc = build_feature_collection([])
    result = validate_geojson(empty_fc)
    assert result["empty_output"], "empty FeatureCollection should set empty_output=True"
    assert result["valid"], "Empty FeatureCollection is valid GeoJSON"
    print(f"validate_geojson (empty): empty_output={result['empty_output']}, valid={result['valid']}")

    # ------------------------------------------------------------------
    # 5. validate_geojson — bad schema
    # ------------------------------------------------------------------
    result = validate_geojson({"type": "NotAFeatureCollection"})
    assert not result["valid"]
    assert not result["schema_ok"], f"Expected schema_ok=False, got {result['schema_ok']}"
    assert result["parse_ok"], "parse should still be OK for a dict"
    print(f"validate_geojson (bad schema): schema_ok={result['schema_ok']}")

    # ------------------------------------------------------------------
    # 6. validate_geojson — not a dict
    # ------------------------------------------------------------------
    result = validate_geojson(None)
    assert not result["valid"]
    assert not result["parse_ok"]
    print("validate_geojson (not a dict): OK")

    # ------------------------------------------------------------------
    # 7. validate_geojson — out of bounds
    # ------------------------------------------------------------------
    result = validate_geojson(fc, tile_bounds_wgs84=[120.0, 36.0, 121.0, 37.0])
    assert not result["valid"], "Should be invalid: points are far outside tile bounds"
    assert result["geometry_valid_rate"] == 0.0
    print(f"validate_geojson (out of bounds): rate={result['geometry_valid_rate']}")

    # ------------------------------------------------------------------
    # 8. outputs_to_geojson — synthetic Mask R-CNN output
    # ------------------------------------------------------------------
    # Build a simple 224x224 binary mask with one rectangle
    # Mask R-CNN mask format: [N, 1, H, W]
    synth_mask = np.zeros((1, 1, 224, 224), dtype=np.float32)
    synth_mask[0, 0, 50:180, 40:160] = 1.0  # same rectangle as poly_pixel

    # FakeTensor to simulate .cpu().numpy() calls on torch tensors
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    synth_output_t = {
        "boxes": FakeTensor(np.array([[40, 50, 160, 180]], dtype=np.float32)),
        "labels": FakeTensor(np.array([1], dtype=np.int64)),
        "scores": FakeTensor(np.array([0.95], dtype=np.float32)),
        "masks": FakeTensor(synth_mask),
    }

    geojson_results = outputs_to_geojson([synth_output_t], [georef])
    assert len(geojson_results) == 1
    assert geojson_results[0]["type"] == "FeatureCollection"
    assert len(geojson_results[0]["features"]) == 1
    assert geojson_results[0]["features"][0]["properties"]["confidence"] == 0.95
    print("outputs_to_geojson (synthetic): OK")

    # ------------------------------------------------------------------
    # 9. outputs_to_geojson — score filtering
    # ------------------------------------------------------------------
    synth_low_score = {
        "boxes": FakeTensor(np.array([[40, 50, 160, 180]], dtype=np.float32)),
        "labels": FakeTensor(np.array([1], dtype=np.int64)),
        "scores": FakeTensor(np.array([0.3], dtype=np.float32)),
        "masks": FakeTensor(synth_mask.copy()),
    }
    geojson_filtered = outputs_to_geojson([synth_low_score], [georef], score_thresh=0.5)
    assert len(geojson_filtered[0]["features"]) == 0, "Low-score detection should be filtered out"
    print("outputs_to_geojson (score filter): OK")

    # ------------------------------------------------------------------
    # 10. polygon_pixel_to_geojson_feature — error on degenerate polygon
    # ------------------------------------------------------------------
    try:
        polygon_pixel_to_geojson_feature([(0, 0), (10, 10)], georef)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("polygon_pixel_to_geojson_feature (degenerate): OK")

    print("All geojson_builder tests passed.")

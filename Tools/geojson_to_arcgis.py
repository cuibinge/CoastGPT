import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_FENCED_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert model GeoJSON output text into ArcGIS-importable GeoJSON "
            "(FeatureCollection)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to model output text/json file, or '-' to read stdin.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GeoJSON file path.",
    )
    parser.add_argument(
        "--allow-geometry-collection",
        action="store_true",
        help="Allow GeometryCollection output (disabled by default).",
    )
    return parser.parse_args()


def load_text(path_or_stdin: str) -> str:
    if path_or_stdin == "-":
        return sys.stdin.read()
    return Path(path_or_stdin).read_text(encoding="utf-8")


def try_parse_json(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_object(raw_text: str) -> Any:
    direct = try_parse_json(raw_text)
    if direct is not None:
        return direct

    for m in _FENCED_BLOCK.finditer(raw_text):
        candidate = try_parse_json(m.group(1))
        if candidate is not None:
            return candidate

    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw_text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(raw_text[i:].strip())
            return obj
        except Exception:
            continue

    raise ValueError("No valid JSON object found in input text.")


def _to_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        out = float(v)
    elif isinstance(v, str):
        out = float(v.strip())
    else:
        raise ValueError(f"Coordinate value is not numeric: {type(v)}")
    if not math.isfinite(out):
        raise ValueError("Coordinate value is not finite.")
    return out


def _close_ring_if_needed(ring: List[List[float]]) -> List[List[float]]:
    if len(ring) < 3:
        raise ValueError("Polygon ring requires at least 3 points.")
    first = ring[0]
    last = ring[-1]
    if first != last:
        ring = ring + [first]
    if len(ring) < 4:
        raise ValueError("Closed polygon ring requires at least 4 positions.")
    return ring


def _normalize_position(pos: Sequence[Any]) -> List[float]:
    if len(pos) < 2:
        raise ValueError("Each position must contain at least [lon, lat].")
    return [_to_float(pos[0]), _to_float(pos[1])]


def _normalize_geometry_coordinates(geom_type: str, coords: Any) -> Any:
    t = geom_type.lower()
    if t == "point":
        return _normalize_position(coords)
    if t == "linestring":
        return [_normalize_position(p) for p in coords]
    if t == "polygon":
        rings = []
        for ring in coords:
            clean_ring = [_normalize_position(p) for p in ring]
            rings.append(_close_ring_if_needed(clean_ring))
        return rings
    if t == "multilinestring":
        return [[_normalize_position(p) for p in line] for line in coords]
    if t == "multipolygon":
        polys = []
        for poly in coords:
            rings = []
            for ring in poly:
                clean_ring = [_normalize_position(p) for p in ring]
                rings.append(_close_ring_if_needed(clean_ring))
            polys.append(rings)
        return polys
    if t == "multipoint":
        return [_normalize_position(p) for p in coords]
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def _coerce_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be an object.")
    geom_type = str(geometry.get("type", "")).strip()
    if not geom_type:
        raise ValueError("Geometry.type is required.")
    coords = geometry.get("coordinates", None)

    if isinstance(coords, str):
        parsed = try_parse_json(coords)
        if parsed is not None:
            coords = parsed
        else:
            raise ValueError("Unable to parse string coordinates.")

    clean_coords = _normalize_geometry_coordinates(geom_type, coords)
    return {"type": geom_type, "coordinates": clean_coords}


def normalize_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(feature, dict):
        raise ValueError("Feature item must be object.")

    if feature.get("type") == "Feature":
        geometry = feature.get("geometry", None)
        properties = feature.get("properties", {})
    elif "geometry" in feature:
        geometry = feature.get("geometry", None)
        properties = feature.get("properties", {})
    else:
        raise ValueError("Feature has no geometry.")

    if properties is None or not isinstance(properties, dict):
        properties = {}
    clean_geometry = _coerce_geometry(geometry)
    return {"type": "Feature", "geometry": clean_geometry, "properties": properties}


def to_feature_collection(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        features = obj.get("features", [])
        if not isinstance(features, list):
            raise ValueError("FeatureCollection.features must be a list.")
        clean = [normalize_feature(f) for f in features]
        return {"type": "FeatureCollection", "features": clean}

    if isinstance(obj, dict) and obj.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [normalize_feature(obj)]}

    if isinstance(obj, dict) and "geometry" in obj:
        return {"type": "FeatureCollection", "features": [normalize_feature(obj)]}

    if isinstance(obj, list):
        clean = [normalize_feature(f) for f in obj]
        return {"type": "FeatureCollection", "features": clean}

    raise ValueError("Input JSON cannot be converted to FeatureCollection.")


def validate_for_arcgis(fc: Dict[str, Any], allow_geometry_collection: bool = False) -> None:
    if fc.get("type") != "FeatureCollection":
        raise ValueError("Output root must be FeatureCollection.")
    features = fc.get("features", [])
    if not isinstance(features, list):
        raise ValueError("features must be list.")
    if len(features) == 0:
        raise ValueError("features is empty.")

    for i, feat in enumerate(features):
        if feat.get("type") != "Feature":
            raise ValueError(f"Feature[{i}] type must be Feature.")
        geom = feat.get("geometry", None)
        if not isinstance(geom, dict):
            raise ValueError(f"Feature[{i}] geometry must be object.")
        gtype = str(geom.get("type", ""))
        if not gtype:
            raise ValueError(f"Feature[{i}] geometry.type is missing.")
        if gtype == "GeometryCollection" and not allow_geometry_collection:
            raise ValueError(
                "GeometryCollection is disabled by default for ArcGIS editing compatibility."
            )
        if not isinstance(feat.get("properties", {}), dict):
            raise ValueError(f"Feature[{i}] properties must be object.")


def main() -> None:
    args = parse_args()
    raw_text = load_text(args.input)
    obj = extract_json_object(raw_text)
    fc = to_feature_collection(obj)
    validate_for_arcgis(fc, allow_geometry_collection=bool(args.allow_geometry_collection))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] ArcGIS-ready GeoJSON saved to: {out_path}")
    print(f"[OK] features={len(fc['features'])}")


if __name__ == "__main__":
    main()

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
except Exception:
    mapping = None
    shape = None
    unary_union = None

# Make ``Models`` importable when this script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.geojson_coordinate_utils import (  # noqa: E402
    encode_feature_collection,
    repair_mojibake_in_obj,
    tile_transform_from_feature_collection,
    tile_transform_from_raster,
)


IMAGE_VARIANT_MAP = {
    "Image_FalseColor": "False",
    "Image_TrueColor": "True",
    "Image_Orig": "Orig",
}
DEFAULT_KEEP_PROPERTIES = ("DLMC", "label", "class_name", "category", "target")

DEFAULT_PROMPTS = (
    "<image>\n[DET] Extract the target features from this GF-2 remote sensing image and output a valid GeoJSON FeatureCollection. Return JSON only.",
    "<image>\n[DET] Generate an editable GeoJSON FeatureCollection for ArcGIS from this GF-2 image patch. Return JSON only.",
    "<image>\n[DET] Output the extracted feature information for this image as a GeoJSON FeatureCollection with geometry and properties. Return JSON only.",
)

CONTINUE_MARKER = " <CONTINUE>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stage-3 instruction dataset from GF2 tiled imagery and "
            "GeoJSON labels for end-to-end GeoJSON generation."
        )
    )
    parser.add_argument(
        "--gf2-root",
        type=Path,
        default=Path("GF2"),
        help=(
            "Dataset root. It can be a single sensor root that contains Size_*/Image_* and Label_GeoJSON, "
            "or a parent directory that contains multiple sensor roots such as GF1/GF2/GF6."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="512",
        help="Comma-separated GF2 tile sizes to include, or 'all'. Default: 512",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="Image_FalseColor",
        choices=sorted(IMAGE_VARIANT_MAP.keys()),
        help="Image variant directory to pair with GeoJSON labels.",
    )
    parser.add_argument(
        "--label-subdir",
        type=str,
        default="Label_GeoJSON",
        help="GeoJSON label directory name inside each Size_* directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for GF_geojson_train.json and GF_geojson_train_Image.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of linking them into the generated *_Image directory.",
    )
    parser.add_argument(
        "--compact-answer",
        action="store_true",
        help="Write answers in compact JSON form to reduce token length.",
    )
    parser.add_argument(
        "--keep-crs",
        action="store_true",
        help="Preserve top-level CRS in output FeatureCollection. Disabled by default.",
    )
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=5,
        help="Round GeoJSON coordinates to this many decimals. Default: 5",
    )
    parser.add_argument(
        "--keep-properties",
        type=str,
        default=",".join(DEFAULT_KEEP_PROPERTIES),
        help=(
            "Comma-separated feature property keys to keep. "
            "Use 'all' to keep everything, or 'none' to drop all properties. "
            f"Default: {','.join(DEFAULT_KEEP_PROPERTIES)}"
        ),
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0,
        help=(
            "Initial geometry simplify tolerance in coordinate units (degrees). "
            "0 disables fixed simplify and only uses adaptive simplify when max-answer-chars is exceeded."
        ),
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=3500,
        help=(
            "Try to simplify geometries until compact GeoJSON answer length is within this budget. "
            "Set to 0 to disable adaptive size control. Default: 3500"
        ),
    )
    parser.add_argument(
        "--disable-merge-by-properties",
        action="store_true",
        help="Do not merge geometries that share the same retained properties.",
    )
    parser.add_argument(
        "--disable-split-by-answer-budget",
        action="store_true",
        help="Do not split oversized GeoJSON collections into multiple training samples.",
    )
    parser.add_argument(
        "--prompt-variants",
        type=int,
        default=1,
        help=(
            "Number of prompt variants to emit for each sample. The original "
            "build wrote 3 near-duplicate prompts sharing the same answer, "
            "which wastes training compute. Default is 1."
        ),
    )
    parser.add_argument(
        "--normalize-coords",
        action="store_true",
        help=(
            "Replace absolute lon/lat in answers with tile-normalised [0, 1] "
            "coordinates. The per-tile affine transform is stored in each "
            "sample so inference output can be de-normalised back to lon/lat."
        ),
    )
    parser.add_argument(
        "--tile-padding",
        type=float,
        default=0.02,
        help=(
            "Fallback bbox padding ratio when no raster geotransform is "
            "available. Used to prevent boundary features from saturating to 0/1."
        ),
    )
    parser.add_argument(
        "--repair-mojibake",
        action="store_true",
        default=True,
        help="Detect UTF-8-as-GBK mojibake in property strings and repair it.",
    )
    parser.add_argument(
        "--multiturn",
        action="store_true",
        help="Enable multi-turn conversation mode, append <CONTINUE> marker to non-final answers",
    )
    parser.add_argument(
        "--max-chars-per-turn",
        type=int,
        default=3000,
        help="Maximum characters per turn in multi-turn mode",
    )
    return parser.parse_args()


def parse_size_filter(value: str) -> Optional[Sequence[str]]:
    text = str(value or "").strip()
    if not text or text.lower() == "all":
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts or None


def infer_output_dir(gf2_root: Path, size_filter: Optional[Sequence[str]], image_subdir: str) -> Path:
    sensor_roots = resolve_sensor_roots(gf2_root)
    if size_filter is None:
        size_tag = "all"
    else:
        size_tag = "-".join(str(part) for part in size_filter)
    image_tag = image_subdir.replace("Image_", "").lower()
    if len(sensor_roots) == 1 and sensor_roots[0][1].resolve() == gf2_root.resolve():
        return gf2_root / f"stage3_geojson_{size_tag}_{image_tag}"
    sensor_tag = "-".join(sensor_name.lower() for sensor_name, _ in sensor_roots)
    return gf2_root / f"stage3_geojson_{sensor_tag}_{size_tag}_{image_tag}"


def parse_keep_properties(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    text = str(value or "").strip()
    if not text or text.lower() == "all":
        return None
    if text.lower() == "none":
        return tuple()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return tuple(parts)


def resolve_size_dirs(gf2_root: Path, size_filter: Optional[Sequence[str]]) -> List[Path]:
    candidates = sorted(
        path for path in gf2_root.iterdir()
        if path.is_dir() and path.name.startswith("Size_")
    )
    if size_filter is None:
        return candidates

    wanted = {f"Size_{str(part)}" for part in size_filter}
    selected = [path for path in candidates if path.name in wanted]
    missing = sorted(wanted.difference({path.name for path in selected}))
    if missing:
        raise FileNotFoundError(f"GF2 size directories not found: {missing}")
    return selected


def is_size_root(root: Path) -> bool:
    try:
        return root.exists() and root.is_dir() and any(
            child.is_dir() and child.name.startswith("Size_")
            for child in root.iterdir()
        )
    except Exception:
        return False


def resolve_sensor_roots(dataset_root: Path) -> List[Tuple[str, Path]]:
    dataset_root = Path(dataset_root)
    if is_size_root(dataset_root):
        return [(dataset_root.name, dataset_root)]

    sensor_roots = []
    if dataset_root.exists() and dataset_root.is_dir():
        for child in sorted(dataset_root.iterdir()):
            if child.is_dir() and is_size_root(child):
                sensor_roots.append((child.name, child))

    if sensor_roots:
        return sensor_roots

    raise FileNotFoundError(
        f"No GF sensor roots found under {dataset_root}. Expected either Size_* directories "
        f"or child directories such as GF1/GF2/GF6 that contain Size_*."
    )


def label_stem_to_image_stem(label_stem: str, image_subdir: str) -> str:
    image_variant = IMAGE_VARIANT_MAP[image_subdir]
    marker = "_Label_"
    if marker not in label_stem:
        return label_stem
    prefix, suffix = label_stem.rsplit(marker, 1)
    return f"{prefix}_{image_variant}_{suffix}"


def find_image_for_label(label_path: Path, image_dir: Path, image_subdir: str) -> Optional[Path]:
    image_stem = label_stem_to_image_stem(label_path.stem, image_subdir)
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = image_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def dumps_feature_collection(feature_collection: Dict, compact: bool = True) -> str:
    if compact:
        return json.dumps(feature_collection, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(feature_collection, ensure_ascii=False, indent=2)


def make_feature_collection(
    features: Sequence[Dict],
    collection_name: str,
    crs: Optional[Dict] = None,
) -> Dict:
    feature_collection = {
        "type": "FeatureCollection",
        "features": list(features),
    }
    if collection_name:
        feature_collection["name"] = collection_name
    if isinstance(crs, dict):
        feature_collection["crs"] = crs
    return feature_collection


def _round_float(value: float, decimals: Optional[int]) -> float:
    if decimals is None or decimals < 0:
        return float(value)
    return round(float(value), int(decimals))


def _round_coordinates(obj, decimals: Optional[int]):
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in obj):
            return [_round_float(item, decimals) for item in obj]
        return [_round_coordinates(item, decimals) for item in obj]
    return obj


def round_geometry_coordinates(geometry: Dict, decimals: Optional[int]) -> Dict:
    if decimals is None or decimals < 0:
        return geometry
    geometry = copy.deepcopy(geometry)
    if "coordinates" in geometry:
        geometry["coordinates"] = _round_coordinates(geometry["coordinates"], decimals)
    geometries = geometry.get("geometries")
    if isinstance(geometries, list):
        geometry["geometries"] = [round_geometry_coordinates(item, decimals) for item in geometries if isinstance(item, dict)]
    return geometry


def filter_feature_properties(properties: Dict, keep_properties: Optional[Sequence[str]]) -> Dict:
    if keep_properties is None:
        return dict(properties)
    if not keep_properties:
        return {}
    return {key: properties[key] for key in keep_properties if key in properties}


def _repair_shapely_geometry(geom):
    if geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    try:
        repaired = geom.buffer(0)
        if not repaired.is_empty:
            return repaired
    except Exception:
        pass
    return geom


def merge_feature_geometries(
    features: Sequence[Dict],
    coordinate_decimals: Optional[int],
    simplify_tolerance: float = 0.0,
) -> List[Dict]:
    if shape is None or mapping is None or unary_union is None:
        return list(features)

    grouped: Dict[str, Dict] = {}
    for feature in features:
        geometry = feature.get("geometry")
        properties = feature.get("properties", {})
        if not isinstance(geometry, dict) or not geometry.get("type"):
            continue
        if not isinstance(properties, dict):
            properties = {}
        try:
            geom = _repair_shapely_geometry(shape(geometry))
        except Exception:
            continue
        if geom.is_empty:
            continue
        group_key = json.dumps(properties, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        bucket = grouped.setdefault(group_key, {"properties": properties, "geometries": []})
        bucket["geometries"].append(geom)

    merged_features: List[Dict] = []
    for bucket in grouped.values():
        try:
            merged_geom = _repair_shapely_geometry(unary_union(bucket["geometries"]))
        except Exception:
            merged_geom = None
        if merged_geom is None or merged_geom.is_empty:
            continue
        if simplify_tolerance > 0:
            try:
                merged_geom = _repair_shapely_geometry(
                    merged_geom.simplify(float(simplify_tolerance), preserve_topology=True)
                )
            except Exception:
                pass
        if merged_geom.is_empty:
            continue
        merged_features.append(
            {
                "type": "Feature",
                "geometry": round_geometry_coordinates(mapping(merged_geom), coordinate_decimals),
                "properties": dict(bucket["properties"]),
            }
        )
    return merged_features


def simplify_geometry(geometry: Dict, tolerance: float, coordinate_decimals: Optional[int]) -> Optional[Dict]:
    if tolerance <= 0:
        return round_geometry_coordinates(geometry, coordinate_decimals)
    if shape is None or mapping is None:
        raise RuntimeError("shapely is required for geometry simplification but is not available.")
    geom = _repair_shapely_geometry(shape(geometry))
    simplified = _repair_shapely_geometry(geom.simplify(float(tolerance), preserve_topology=True))
    if simplified.is_empty:
        return None
    return round_geometry_coordinates(mapping(simplified), coordinate_decimals)


def explode_feature(feature: Dict) -> List[Dict]:
    geometry = feature.get("geometry")
    properties = feature.get("properties", {})
    if not isinstance(geometry, dict) or not geometry.get("type"):
        return []
    gtype = geometry.get("type")
    if gtype in {"MultiPolygon", "MultiLineString", "MultiPoint"}:
        child_type = {
            "MultiPolygon": "Polygon",
            "MultiLineString": "LineString",
            "MultiPoint": "Point",
        }[gtype]
        coords = geometry.get("coordinates", [])
        return [
            {
                "type": "Feature",
                "geometry": {"type": child_type, "coordinates": coord},
                "properties": dict(properties),
            }
            for coord in coords
        ]
    if gtype == "GeometryCollection":
        geometries = geometry.get("geometries", [])
        return [
            {
                "type": "Feature",
                "geometry": child,
                "properties": dict(properties),
            }
            for child in geometries
            if isinstance(child, dict) and child.get("type")
        ]
    return [feature]


def split_feature_collection_by_chars(
    feature_collection: Dict,
    max_answer_chars: int,
    explode_multi_geometries: bool = True,
) -> List[Dict]:
    if max_answer_chars <= 0:
        return [feature_collection]

    source_features = feature_collection.get("features", [])
    if not isinstance(source_features, list) or not source_features:
        return [feature_collection]

    parts: List[Dict] = []
    for feature in source_features:
        if explode_multi_geometries:
            exploded = explode_feature(feature)
            parts.extend(exploded if exploded else [feature])
        else:
            parts.append(feature)

    if not parts:
        return [feature_collection]

    collection_name = str(feature_collection.get("name", ""))
    crs = feature_collection.get("crs") if isinstance(feature_collection.get("crs"), dict) else None
    chunks: List[List[Dict]] = []
    current: List[Dict] = []

    for part in parts:
        candidate = current + [part]
        candidate_fc = make_feature_collection(candidate, collection_name=collection_name, crs=crs)
        candidate_chars = len(dumps_feature_collection(candidate_fc, compact=True))
        if current and candidate_chars > max_answer_chars:
            chunks.append(current)
            current = [part]
        else:
            current = candidate
    if current:
        chunks.append(current)

    return [
        make_feature_collection(chunk, collection_name=collection_name, crs=crs)
        for chunk in chunks
        if chunk
    ]


def compress_feature_collection(
    feature_collection: Dict,
    coordinate_decimals: Optional[int],
    keep_properties: Optional[Sequence[str]],
    simplify_tolerance: float,
    max_answer_chars: int,
    collection_name: str,
    merge_by_properties: bool = True,
) -> Tuple[Dict, Dict]:
    original_chars = len(dumps_feature_collection(feature_collection, compact=True))
    compressed = {
        "type": "FeatureCollection",
        "features": [],
    }
    if collection_name:
        compressed["name"] = collection_name
    if isinstance(feature_collection.get("crs"), dict):
        compressed["crs"] = feature_collection["crs"]

    for feature in feature_collection.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type"):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        try:
            geometry = simplify_geometry(
                geometry=geometry,
                tolerance=max(0.0, float(simplify_tolerance)),
                coordinate_decimals=coordinate_decimals,
            )
        except Exception:
            geometry = round_geometry_coordinates(geometry, coordinate_decimals)
        if geometry is None:
            continue
        compressed["features"].append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": filter_feature_properties(properties, keep_properties),
            }
        )

    if merge_by_properties and compressed["features"]:
        merged_features = merge_feature_geometries(
            features=compressed["features"],
            coordinate_decimals=coordinate_decimals,
            simplify_tolerance=max(0.0, float(simplify_tolerance)),
        )
        if merged_features:
            compressed["features"] = merged_features

    final_tolerance = max(0.0, float(simplify_tolerance))
    answer_chars = len(dumps_feature_collection(compressed, compact=True))
    if max_answer_chars > 0 and answer_chars > max_answer_chars and compressed["features"]:
        if shape is None or mapping is None:
            print(
                f"[WARN] shapely unavailable; cannot adaptively simplify {collection_name}. "
                f"answer_chars={answer_chars} > max_answer_chars={max_answer_chars}"
            )
        else:
            best = compressed
            best_chars = answer_chars
            base_tol = final_tolerance if final_tolerance > 0 else 10 ** (-max(int(coordinate_decimals or 5), 1))
            cur_tol = base_tol
            for _ in range(12):
                candidate = {
                    "type": compressed["type"],
                    "features": [],
                }
                if "name" in compressed:
                    candidate["name"] = compressed["name"]
                if "crs" in compressed:
                    candidate["crs"] = compressed["crs"]
                for feature in feature_collection.get("features", []):
                    if not isinstance(feature, dict):
                        continue
                    geometry = feature.get("geometry")
                    if not isinstance(geometry, dict) or not geometry.get("type"):
                        continue
                    properties = feature.get("properties", {})
                    if not isinstance(properties, dict):
                        properties = {}
                    simplified_geometry = simplify_geometry(
                        geometry=geometry,
                        tolerance=cur_tol,
                        coordinate_decimals=coordinate_decimals,
                    )
                    if simplified_geometry is None:
                        continue
                    candidate["features"].append(
                        {
                            "type": "Feature",
                            "geometry": simplified_geometry,
                            "properties": filter_feature_properties(properties, keep_properties),
                        }
                    )
                if merge_by_properties and candidate["features"]:
                    merged_candidate_features = merge_feature_geometries(
                        features=candidate["features"],
                        coordinate_decimals=coordinate_decimals,
                        simplify_tolerance=cur_tol,
                    )
                    if merged_candidate_features:
                        candidate["features"] = merged_candidate_features
                if not candidate["features"]:
                    break
                candidate_chars = len(dumps_feature_collection(candidate, compact=True))
                if candidate_chars < best_chars:
                    best = candidate
                    best_chars = candidate_chars
                    final_tolerance = cur_tol
                if candidate_chars <= max_answer_chars:
                    break
                cur_tol *= 2.0
            compressed = best
            answer_chars = best_chars

    stats = {
        "original_answer_chars": original_chars,
        "compressed_answer_chars": answer_chars,
        "applied_simplify_tolerance": final_tolerance,
        "feature_count": len(compressed["features"]),
    }
    return compressed, stats


def normalize_feature_collection(
    raw_obj: Dict,
    collection_name: str,
    keep_crs: bool = False,
) -> Dict:
    features = []
    raw_type = str(raw_obj.get("type", "")).strip()

    if raw_type == "FeatureCollection" and isinstance(raw_obj.get("features"), list):
        features = raw_obj["features"]
    elif raw_type == "Feature":
        features = [raw_obj]
    elif isinstance(raw_obj.get("features"), list):
        features = raw_obj["features"]
    elif "geometry" in raw_obj:
        features = [{
            "type": "Feature",
            "geometry": raw_obj.get("geometry"),
            "properties": raw_obj.get("properties", {}),
        }]

    clean_features = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type"):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        clean_features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        })

    collection = {
        "type": "FeatureCollection",
        "features": clean_features,
    }
    if collection_name:
        collection["name"] = collection_name
    if keep_crs and isinstance(raw_obj.get("crs"), dict):
        collection["crs"] = raw_obj["crs"]
    return collection


def infer_target_name(feature_collection: Dict) -> str:
    features = feature_collection.get("features", [])
    if not isinstance(features, list) or not features:
        return "target region"
    properties = features[0].get("properties", {})
    if not isinstance(properties, dict):
        return "target region"
    for key in ("DLMC", "label", "class_name", "category", "target"):
        value = properties.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return "target region"


def build_conversations(
    feature_collection: Dict,
    answer: Optional[str] = None,
    prompt_variants: int = 1,
) -> List[Dict[str, str]]:
    answer = answer if answer is not None else dumps_feature_collection(feature_collection, compact=True)
    target_name = infer_target_name(feature_collection)
    prompts = [
        DEFAULT_PROMPTS[0].replace("target features", f"{target_name} features"),
        DEFAULT_PROMPTS[1],
        DEFAULT_PROMPTS[2],
    ]
    n = max(1, min(int(prompt_variants), len(prompts)))
    return [{"Question": prompts[i], "Answer": answer} for i in range(n)]


def build_multiturn_conversations(
    sub_collections: List[Dict],
    first_prompt: str,
    compact_answer: bool = True,
) -> List[Dict[str, str]]:
    """Build multi-turn conversations from split sub FeatureCollections.

    First turn: full task prompt + first set of features
    Subsequent turns: short continuation prompt + corresponding features
    Non-final turns append CONTINUE_MARKER to the answer.
    """
    convs = []
    for i, sub in enumerate(sub_collections):
        answer_text = dumps_feature_collection(sub, compact=compact_answer)
        if i == 0:
            question = first_prompt
        elif i == 1:
            question = "Continue generating the remaining features."
        else:
            question = "Continue."

        is_last = i == len(sub_collections) - 1
        if not is_last:
            answer_text += CONTINUE_MARKER

        convs.append({"Question": question, "Answer": answer_text})
    return convs


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    try:
        os.symlink(src.resolve(), dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def build_dataset(
    gf2_root: Path,
    output_dir: Path,
    size_filter: Optional[Sequence[str]],
    image_subdir: str,
    label_subdir: str,
    copy_images: bool = False,
    compact_answer: bool = False,
    keep_crs: bool = False,
    coordinate_decimals: Optional[int] = 5,
    keep_properties: Optional[Sequence[str]] = DEFAULT_KEEP_PROPERTIES,
    simplify_tolerance: float = 0.0,
    max_answer_chars: int = 3500,
    merge_by_properties: bool = True,
    split_by_answer_budget: bool = True,
    prompt_variants: int = 1,
    multiturn: bool = False,
    max_chars_per_turn: int = 3000,
    normalize_coords: bool = False,
    tile_padding: float = 0.02,
    repair_mojibake: bool = True,
) -> Path:
    gf2_root = gf2_root.resolve()
    output_dir = output_dir.resolve()
    image_output_dir = output_dir / "GF_geojson_train_Image"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    sensor_roots = resolve_sensor_roots(gf2_root)

    samples: List[Dict] = []
    missing_images = 0
    empty_labels = 0
    over_budget = 0
    original_char_lengths: List[int] = []
    compressed_char_lengths: List[int] = []

    for sensor_name, sensor_root in sensor_roots:
        try:
            size_dirs = resolve_size_dirs(sensor_root, size_filter)
        except FileNotFoundError:
            print(f"[WARN] {sensor_name} at {sensor_root} has no matching sizes; skip.")
            continue
        for size_dir in size_dirs:
            image_dir = size_dir / image_subdir
            label_dir = size_dir / label_subdir
            if not image_dir.exists() or not label_dir.exists():
                print(f"[WARN] Skip {size_dir}: missing {image_subdir} or {label_subdir}")
                continue

            for label_path in sorted(label_dir.glob("*.geojson")):
                image_path = find_image_for_label(label_path, image_dir, image_subdir)
                if image_path is None:
                    missing_images += 1
                    print(f"[WARN] Missing image for label: {label_path}")
                    continue

                with label_path.open("r", encoding="utf-8") as f:
                    raw_obj = json.load(f)

                if repair_mojibake:
                    raw_obj = repair_mojibake_in_obj(raw_obj)

                feature_collection = normalize_feature_collection(
                    raw_obj=raw_obj,
                    collection_name=label_path.stem,
                    keep_crs=keep_crs,
                )
                if not feature_collection["features"]:
                    empty_labels += 1
                    print(f"[WARN] Empty feature collection: {label_path}")
                    continue

                feature_collection, stats = compress_feature_collection(
                    feature_collection=feature_collection,
                    coordinate_decimals=coordinate_decimals,
                    keep_properties=keep_properties,
                    simplify_tolerance=simplify_tolerance,
                    max_answer_chars=max_answer_chars,
                    collection_name=label_path.stem,
                    merge_by_properties=merge_by_properties,
                )
                if not feature_collection["features"]:
                    empty_labels += 1
                    print(f"[WARN] Feature collection became empty after compression: {label_path}")
                    continue

                sub_collections = [feature_collection]
                if split_by_answer_budget and max_answer_chars > 0:
                    budget = max_chars_per_turn if multiturn else max_answer_chars
                    sub_collections = split_feature_collection_by_chars(
                        feature_collection=feature_collection,
                        max_answer_chars=budget,
                        explode_multi_geometries=True,
                    )

                dst_image_rel = Path(sensor_name) / image_path.name
                dst_image = image_output_dir / dst_image_rel
                link_or_copy(image_path, dst_image, copy_images=copy_images)
                original_char_lengths.append(int(stats["original_answer_chars"]))

                # Resolve the tile-relative geo transform once per source tile.
                tile_transform = None
                tile_transform_source = "none"
                if normalize_coords:
                    tile_transform = tile_transform_from_raster(image_path)
                    if tile_transform is not None:
                        tile_transform_source = "raster"
                    else:
                        tile_transform = tile_transform_from_feature_collection(
                            feature_collection, pad_ratio=float(tile_padding)
                        )
                        if tile_transform is not None:
                            tile_transform_source = "feature_bbox"
                    if tile_transform is None:
                        print(
                            f"[WARN] Cannot derive tile transform for {label_path.name}; "
                            f"emitting absolute lon/lat instead."
                        )

                base_sample_id = f"{sensor_name}_{image_path.stem}"
                if multiturn and len(sub_collections) > 1:
                    # Encode coordinates for multi-turn output
                    encoded_subs = []
                    for sc in sub_collections:
                        if tile_transform is not None:
                            encoded_sub = encode_feature_collection(
                                sc,
                                transform=tile_transform,
                            )
                        else:
                            encoded_sub = sc
                        encoded_subs.append(encoded_sub)

                    target_name = infer_target_name(feature_collection)
                    convs = build_multiturn_conversations(
                        sub_collections=encoded_subs,
                        first_prompt=DEFAULT_PROMPTS[0].replace(
                            "target features", f"{target_name} features"
                        ),
                        compact_answer=compact_answer,
                    )
                    answer_chars = max(
                        len(dumps_feature_collection(sc, compact=True)) for sc in encoded_subs
                    )
                    coord_encoding = "normalized" if tile_transform is not None else "absolute"
                    sample_record = {
                        "name": dst_image_rel.as_posix(),
                        "sample_id": base_sample_id,
                        "conv": convs,
                        "source_geojson": str(label_path.relative_to(gf2_root)),
                        "sensor": sensor_name,
                        "tile_size": size_dir.name.replace("Size_", ""),
                        "image_variant": image_subdir,
                        "answer_chars": answer_chars,
                        "original_answer_chars": stats["original_answer_chars"],
                        "applied_simplify_tolerance": stats["applied_simplify_tolerance"],
                        "feature_count": len(feature_collection.get("features", [])),
                        "part_count": len(sub_collections),
                        "coord_encoding": coord_encoding,
                    }
                    if tile_transform is not None:
                        sample_record["tile_transform"] = tile_transform
                        sample_record["tile_transform_source"] = tile_transform_source
                    samples.append(sample_record)
                    compressed_char_lengths.append(answer_chars)
                else:
                    # Single-turn: keep original logic, each sub_collection is an independent sample
                    for part_idx, sub_collection in enumerate(sub_collections, start=1):
                        answer_collection = sub_collection
                        if tile_transform is not None:
                            answer_collection = encode_feature_collection(
                                sub_collection,
                                transform=tile_transform,
                            )

                        compact_answer_text = dumps_feature_collection(answer_collection, compact=True)
                        answer_chars = len(compact_answer_text)
                        compressed_char_lengths.append(int(answer_chars))
                        if max_answer_chars > 0 and answer_chars > max_answer_chars:
                            over_budget += 1
                            print(
                                f"[WARN] Answer still exceeds char budget after compression: "
                                f"{label_path.name} chars={answer_chars} budget={max_answer_chars} "
                                f"tol={stats['applied_simplify_tolerance']} part={part_idx}/{len(sub_collections)}"
                            )

                        sample_name = f"{sensor_name}_{image_path.stem}"
                        base_sample_id = sample_name if len(sub_collections) == 1 else f"{sample_name}__part{part_idx}"
                        convs = build_conversations(
                            answer_collection,
                            answer=compact_answer_text,
                            prompt_variants=int(prompt_variants),
                        )
                        if not compact_answer:
                            non_compact_text = dumps_feature_collection(answer_collection, compact=False)
                            for conv in convs:
                                conv["Answer"] = non_compact_text

                        coord_encoding = "normalized" if tile_transform is not None else "absolute"

                        for prompt_idx, conv in enumerate(convs, start=1):
                            prompt_sample_id = (
                                base_sample_id if len(convs) == 1 else f"{base_sample_id}__p{prompt_idx}"
                            )
                            sample_record = {
                                "name": dst_image_rel.as_posix(),
                                "sample_id": prompt_sample_id,
                                "conv": [conv],
                                "source_geojson": str(label_path.relative_to(gf2_root)),
                                "sensor": sensor_name,
                                "tile_size": size_dir.name.replace("Size_", ""),
                                "image_variant": image_subdir,
                                "answer_chars": answer_chars,
                                "original_answer_chars": stats["original_answer_chars"],
                                "applied_simplify_tolerance": stats["applied_simplify_tolerance"],
                                "feature_count": len(sub_collection.get("features", [])),
                                "part_index": part_idx,
                                "part_total": len(sub_collections),
                                "prompt_variant_index": prompt_idx,
                                "prompt_variant_total": len(convs),
                                "coord_encoding": coord_encoding,
                            }
                            if tile_transform is not None:
                                sample_record["tile_transform"] = tile_transform
                                sample_record["tile_transform_source"] = tile_transform_source
                            samples.append(sample_record)

    # Save coordinate transform parameters for inference coordinate de-normalization
    coord_transforms = {}
    for sample in samples:
        if "tile_transform" in sample:
            img_name = sample.get("name", "")
            if img_name and img_name not in coord_transforms:
                coord_transforms[img_name] = sample["tile_transform"]

    if coord_transforms:
        coord_transform_path = output_dir / "coord_transform_train.json"
        with coord_transform_path.open("w", encoding="utf-8") as f:
            json.dump(coord_transforms, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[OK] coord transforms -> {coord_transform_path} ({len(coord_transforms)} entries)")

    out_json = output_dir / "GF_geojson_train.json"
    manifest = output_dir / "GF_geojson_manifest.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"data": samples}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with manifest.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gf2_root": str(gf2_root),
                "sensor_roots": {sensor_name: str(sensor_root) for sensor_name, sensor_root in sensor_roots},
                "sensors": [sensor_name for sensor_name, _ in sensor_roots],
                "output_dir": str(output_dir),
                "sizes": list(size_filter) if size_filter is not None else "all",
                "image_subdir": image_subdir,
                "label_subdir": label_subdir,
                "sample_count": len(samples),
                "missing_images": missing_images,
                "empty_labels": empty_labels,
                "over_budget": over_budget,
                "compression": {
                    "coordinate_decimals": coordinate_decimals,
                    "keep_properties": "all" if keep_properties is None else list(keep_properties),
                    "simplify_tolerance": simplify_tolerance,
                    "max_answer_chars": max_answer_chars,
                    "merge_by_properties": bool(merge_by_properties),
                    "split_by_answer_budget": bool(split_by_answer_budget),
                },
                "coord_encoding": {
                    "normalize_coords": bool(normalize_coords),
                    "tile_padding": float(tile_padding),
                    "prompt_variants": int(prompt_variants),
                    "repair_mojibake": bool(repair_mojibake),
                },
                "answer_stats": {
                    "original_max_chars": max(original_char_lengths) if original_char_lengths else 0,
                    "original_avg_chars": round(sum(original_char_lengths) / len(original_char_lengths), 2) if original_char_lengths else 0,
                    "compressed_max_chars": max(compressed_char_lengths) if compressed_char_lengths else 0,
                    "compressed_avg_chars": round(sum(compressed_char_lengths) / len(compressed_char_lengths), 2) if compressed_char_lengths else 0,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(f"[OK] samples={len(samples)} -> {out_json}")
    print(f"[OK] images -> {image_output_dir}")
    if missing_images:
        print(f"[WARN] missing_images={missing_images}")
    if empty_labels:
        print(f"[WARN] empty_labels={empty_labels}")
    return out_json


def main() -> None:
    args = parse_args()
    size_filter = parse_size_filter(args.sizes)
    output_dir = args.output_dir or infer_output_dir(args.gf2_root, size_filter, args.image_subdir)

    if not args.gf2_root.exists():
        raise FileNotFoundError(f"GF2 root not found: {args.gf2_root}")

    build_dataset(
        gf2_root=args.gf2_root,
        output_dir=output_dir,
        size_filter=size_filter,
        image_subdir=args.image_subdir,
        label_subdir=args.label_subdir,
        copy_images=bool(args.copy_images),
        compact_answer=bool(args.compact_answer),
        keep_crs=bool(args.keep_crs),
        coordinate_decimals=args.coordinate_decimals,
        keep_properties=parse_keep_properties(args.keep_properties),
        simplify_tolerance=float(args.simplify_tolerance),
        max_answer_chars=int(args.max_answer_chars),
        merge_by_properties=not bool(args.disable_merge_by_properties),
        split_by_answer_budget=not bool(args.disable_split_by_answer_budget),
        prompt_variants=int(args.prompt_variants),
        normalize_coords=bool(args.normalize_coords),
        tile_padding=float(args.tile_padding),
        repair_mojibake=bool(args.repair_mojibake),
        multiturn=bool(args.multiturn),
        max_chars_per_turn=int(args.max_chars_per_turn),
    )


if __name__ == "__main__":
    main()

"""Prepare usable CoastGPT-Bench subsets for instruction training.

The builder keeps dataset conversion outside the core model path. Feature
requests remain free-text training data, while vector answers are represented
as generic GeoJSON FeatureCollections.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ID = "cuibinge/CoastGPT-Bench"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_IMAGE_DIR = "CoastBench_Image"
DEFAULT_JSON_NAME = "CoastBench.json"
OPEN_VOCAB_VECTOR_RULE = (
    "Treat requested features as open vocabulary queries. "
    "Return only matching objects. "
    "Do not assign background, non-requested, or ambiguous regions to any known category."
)


@dataclass
class BuildStats:
    vg_records: int = 0
    vqa_records: int = 0
    caption_records: int = 0
    geojson_records: int = 0
    classification_records: int = 0
    skipped_without_image: int = 0

    @property
    def total_records(self) -> int:
        return (
            self.vg_records
            + self.vqa_records
            + self.caption_records
            + self.geojson_records
            + self.classification_records
        )


def _hf_url(path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{quoted}"


def _api_siblings() -> List[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO_ID}", timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["rfilename"] for item in payload.get("siblings", []) if "rfilename" in item]


def _download_file(path: str, target: Path, overwrite: bool = False, retries: int = 5) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return target
    tmp_target = target.with_suffix(target.suffix + ".part")
    if tmp_target.exists():
        tmp_target.unlink()
    last_error = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            urllib.request.urlretrieve(_hf_url(path), tmp_target)
            tmp_target.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if tmp_target.exists():
                tmp_target.unlink()
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed to download {path} after {retries} attempts: {last_error}") from last_error
    return target


def _download_many(paths: Sequence[str], source_root: Path, overwrite: bool = False, workers: int = 16) -> None:
    if not paths:
        return
    worker_count = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_download_file, rel_path, source_root / rel_path, overwrite): rel_path
            for rel_path in paths
        }
        completed = 0
        for future in as_completed(futures):
            rel_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed downloading {rel_path}: {exc}") from exc
            completed += 1
            if completed % 100 == 0 or completed == len(paths):
                print(f"downloaded {completed}/{len(paths)} files")


def _extract_zip(path: Path, target_dir: Path, overwrite: bool = False) -> Path:
    marker = target_dir / f".{path.stem}.extracted"
    if marker.exists() and not overwrite:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return target_dir


def extract_existing_archives(source_root: Path, overwrite: bool = False) -> None:
    archive_targets = {
        "classification-22.zip": source_root,
        "image_caption_qa_with_time_place.zip": source_root,
        "VQA_10_image.zip": source_root / "VQA_10",
        "big_image_geojson_EN.zip": source_root,
        "big_image_rgb.zip": source_root,
        "big_image_tif.zip": source_root,
        "geojson_EN.zip": source_root,
        "image_rgb.zip": source_root,
        "image_tif.zip": source_root,
    }
    for archive in source_root.rglob("*.zip"):
        target = archive_targets.get(archive.name)
        if target is None and archive.parent.name == "image-caption":
            target = source_root
        if target is None:
            continue
        _extract_zip(archive, target, overwrite=overwrite)


def download_selected_assets(
    source_root: Path,
    include_vg: bool,
    include_vqa: bool,
    include_caption: bool,
    include_geojson: bool,
    include_classification: bool,
    overwrite: bool = False,
) -> None:
    """Download selected CoastGPT-Bench assets into a local raw cache."""
    source_root.mkdir(parents=True, exist_ok=True)
    siblings = _api_siblings()

    if include_vg:
        vg_paths = [p for p in siblings if p.startswith("VG_coastline/")]
        _download_many(vg_paths, source_root=source_root, overwrite=overwrite, workers=16)

    if include_vqa:
        for rel_path in siblings:
            if rel_path.startswith("VQA_10/"):
                local_path = _download_file(rel_path, source_root / rel_path, overwrite=overwrite)
                if local_path.suffix.lower() == ".zip":
                    _extract_zip(local_path, source_root / "VQA_10", overwrite=overwrite)

    if include_caption:
        for rel_path in (
            "image_caption_qa_with_time_place.zip",
            "image-caption/image_rgb.zip",
        ):
            local_path = _download_file(rel_path, source_root / rel_path, overwrite=overwrite)
            _extract_zip(local_path, source_root, overwrite=overwrite)

    if include_geojson:
        for rel_path in (
            "image-caption/geojson_EN.zip",
            "image-caption/big_image_geojson_EN.zip",
            "image-caption/image_rgb.zip",
        ):
            local_path = _download_file(rel_path, source_root / rel_path, overwrite=overwrite)
            _extract_zip(local_path, source_root, overwrite=overwrite)

    if include_classification:
        local_path = _download_file("classification-22.zip", source_root / "classification-22.zip", overwrite=overwrite)
        _extract_zip(local_path, source_root, overwrite=overwrite)


def _load_json(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ascii_id(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isascii() and (char.isalnum() or char in "-_."):
            safe.append(char)
        else:
            safe.append("_")
    value = "".join(safe).strip("._")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "sample"


def _copy_or_link(source: Path, target: Path, copy_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if copy_images:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source.resolve())
    except Exception:
        shutil.copy2(source, target)


def _index_images(source_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.lower(), []).append(path)
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def _find_image(index: Dict[str, List[Path]], name: str) -> Optional[Path]:
    raw = str(name or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if not raw.lower().endswith(IMAGE_SUFFIXES):
        candidates.extend(f"{raw}{suffix}" for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        matches = index.get(Path(candidate).name.lower()) or index.get(Path(candidate).stem.lower())
        if matches:
            return matches[0]
    return None


def _record_name(prefix: str, source_image: Path, serial: int) -> str:
    return f"{prefix}_{serial:08d}_{_ascii_id(source_image.stem)}{source_image.suffix.lower()}"


def _line_geojson(points: Sequence[Sequence[float]], source_id: str) -> Dict:
    coordinates = []
    for point in points:
        if len(point) < 2:
            continue
        coordinates.append([round(float(point[0]), 7), round(float(point[1]), 7)])
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {"source_id": _ascii_id(source_id)},
            }
        ],
    }


def _normalize_coordinates(value):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(v, (int, float, str)) for v in value[:2]):
            try:
                return [round(float(value[0]), 7), round(float(value[1]), 7)]
            except Exception:
                pass
        return [_normalize_coordinates(item) for item in value]
    return value


def _feature_collection_from_features(features: object, source_id: str) -> Optional[Dict]:
    if not isinstance(features, list) or len(features) == 0:
        return None
    normalized = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        geometry = dict(geometry)
        if "coordinates" in geometry:
            geometry["coordinates"] = _normalize_coordinates(geometry["coordinates"])
        normalized.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "source_id": _ascii_id(source_id),
                    "feature_index": idx,
                },
            }
        )
    if not normalized:
        return None
    return {"type": "FeatureCollection", "features": normalized}


def _json_answer(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _append_record(
    records: List[Dict],
    image_dir: Path,
    source_image: Optional[Path],
    target_name: str,
    conv: List[Dict],
    copy_images: bool,
    stats: BuildStats,
) -> bool:
    if source_image is None or not source_image.exists():
        stats.skipped_without_image += 1
        return False
    _copy_or_link(source_image, image_dir / target_name, copy_images=copy_images)
    records.append({"name": target_name, "conv": conv})
    return True


def _build_vg_records(
    source_root: Path,
    image_index: Dict[str, List[Path]],
    records: List[Dict],
    image_dir: Path,
    copy_images: bool,
    stats: BuildStats,
) -> None:
    answer_files = sorted((source_root / "VG_coastline").glob("*_coast_answer.json"))
    for answer_file in answer_files:
        payload = _load_json(answer_file)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        prefix = _ascii_id(answer_file.stem.replace("_coast_answer", "vg"))
        for item in items:
            if not isinstance(item, dict):
                continue
            image_name = str(item.get("img", ""))
            source_image = _find_image(image_index, image_name)
            try:
                points = ast.literal_eval(str(item.get("answer", "")))
            except Exception:
                continue
            target = _record_name(prefix, source_image, stats.vg_records) if source_image else f"{prefix}_{stats.vg_records:08d}.tif"
            answer = _json_answer(_line_geojson(points, source_id=image_name))
            conv = [
                {
                    "Question": (
                        "<image>[VG] Extract the requested vector object from this image "
                        f"and return one valid GeoJSON FeatureCollection. {OPEN_VOCAB_VECTOR_RULE} Return JSON only."
                    ),
                    "Answer": answer,
                }
            ]
            if _append_record(records, image_dir, source_image, target, conv, copy_images, stats):
                stats.vg_records += 1


def _build_vqa_records(
    source_root: Path,
    image_index: Dict[str, List[Path]],
    records: List[Dict],
    image_dir: Path,
    copy_images: bool,
    stats: BuildStats,
) -> None:
    json_dir = source_root / "VQA_10" / "VQA_10_JSON"
    for json_path in sorted(json_dir.glob("VQA*.json")):
        payload = _load_json(json_path)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            source_image = _find_image(image_index, str(item.get("name", "")))
            target = _record_name("vqa", source_image, stats.vqa_records) if source_image else f"vqa_{stats.vqa_records:08d}.png"
            conv = []
            for qa in item.get("conv", []):
                if not isinstance(qa, dict):
                    continue
                question = str(qa.get("Question", "")).replace("[VQA]", "").strip()
                answer = str(qa.get("Answer", "")).strip()
                if question and answer:
                    conv.append({"Question": question, "Answer": answer})
            if conv and _append_record(records, image_dir, source_image, target, conv, copy_images, stats):
                stats.vqa_records += 1


def _build_caption_records(
    source_root: Path,
    image_index: Dict[str, List[Path]],
    records: List[Dict],
    image_dir: Path,
    copy_images: bool,
    stats: BuildStats,
) -> None:
    json_root = source_root / "image_caption_qa_with_time_place"
    for json_path in sorted(json_root.glob("*_caption_qa_with_time_place.json")):
        payload = _load_json(json_path)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            source_image = _find_image(image_index, str(item.get("name", "")))
            target = _record_name("caption", source_image, stats.caption_records) if source_image else f"caption_{stats.caption_records:08d}.png"
            conv = []
            for qa in item.get("conv", []):
                if not isinstance(qa, dict):
                    continue
                question = str(qa.get("Question", "")).strip()
                answer = str(qa.get("Answer", "")).strip()
                if question.startswith("When and where"):
                    continue
                question = question.replace("[CAP]", "").strip()
                if question and answer:
                    conv.append({"Question": question, "Answer": answer})
            if conv and _append_record(records, image_dir, source_image, target, conv, copy_images, stats):
                stats.caption_records += 1


def _build_geojson_records(
    source_root: Path,
    image_index: Dict[str, List[Path]],
    records: List[Dict],
    image_dir: Path,
    copy_images: bool,
    stats: BuildStats,
) -> None:
    geojson_root = source_root / "geojson_EN"
    for geojson_path in sorted(geojson_root.glob("*_captions.geojson")):
        payload = _load_json(geojson_path)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("name", ""))
            source_image = _find_image(image_index, source_id)
            target = _record_name("geojson", source_image, stats.geojson_records) if source_image else f"geojson_{stats.geojson_records:08d}.png"
            fc = _feature_collection_from_features(item.get("features"), source_id=source_id)
            if fc is None:
                continue
            conv = [
                {
                    "Question": (
                        "<image>[DET] Return the georeferenced spatial footprint for this image "
                        "as one valid GeoJSON FeatureCollection. Return JSON only."
                    ),
                    "Answer": _json_answer(fc),
                }
            ]
            if _append_record(records, image_dir, source_image, target, conv, copy_images, stats):
                stats.geojson_records += 1


def _classification_roots(source_root: Path) -> Iterable[Path]:
    candidates = [
        source_root / "classification-22",
        source_root / "classification_22",
        source_root / "classification",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            yield candidate
    for candidate in source_root.iterdir() if source_root.exists() else []:
        if candidate.is_dir() and "classification" in candidate.name.lower():
            yield candidate


def _is_ascii_label(text: str) -> bool:
    return bool(text) and text.isascii() and any(char.isalpha() for char in text)


def _build_classification_records(
    source_root: Path,
    records: List[Dict],
    image_dir: Path,
    copy_images: bool,
    stats: BuildStats,
) -> None:
    seen = set()
    for class_root in _classification_roots(source_root):
        for image_path in class_root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = image_path.parent.name.strip()
            if not _is_ascii_label(label):
                continue
            key = str(image_path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            target = _record_name("classification", image_path, stats.classification_records)
            conv = [
                {
                    "Question": "<image>Classify the remote sensing scene shown in this image.",
                    "Answer": label,
                }
            ]
            if _append_record(records, image_dir, image_path, target, conv, copy_images, stats):
                stats.classification_records += 1


def build_coastgpt_bench(
    source_root: Path,
    output_root: Path,
    include_vg: bool = True,
    include_vqa: bool = True,
    include_caption: bool = True,
    include_geojson: bool = True,
    include_classification: bool = True,
    copy_images: bool = False,
    extract_archives: bool = True,
) -> Tuple[Path, BuildStats]:
    if extract_archives:
        extract_existing_archives(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = output_root / DEFAULT_IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    image_index = _index_images(source_root)
    records: List[Dict] = []
    stats = BuildStats()

    if include_vg:
        _build_vg_records(source_root, image_index, records, image_dir, copy_images, stats)
    if include_vqa:
        _build_vqa_records(source_root, image_index, records, image_dir, copy_images, stats)
    if include_caption:
        _build_caption_records(source_root, image_index, records, image_dir, copy_images, stats)
    if include_geojson:
        _build_geojson_records(source_root, image_index, records, image_dir, copy_images, stats)
    if include_classification:
        _build_classification_records(source_root, records, image_dir, copy_images, stats)

    output_json = output_root / DEFAULT_JSON_NAME
    output_json.write_text(
        json.dumps({"data": records}, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    summary_path = output_root / "CoastBench_summary.json"
    summary_path.write_text(json.dumps(stats.__dict__, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_json, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoastGPT-Bench for CoastGPT training")
    parser.add_argument("--source-root", type=Path, required=True, help="Local raw CoastGPT-Bench cache")
    parser.add_argument("--output-root", type=Path, required=True, help="Prepared training dataset root")
    parser.add_argument("--download", action="store_true", help="Download selected assets before building")
    parser.add_argument("--download-vg", action="store_true", help="Download visual grounding images and labels")
    parser.add_argument("--download-vqa", action="store_true", help="Download VQA images and labels")
    parser.add_argument("--download-caption", action="store_true", help="Download caption images and labels")
    parser.add_argument("--download-geojson", action="store_true", help="Download GeoJSON image and label assets")
    parser.add_argument("--download-classification", action="store_true", help="Download scene classification assets")
    parser.add_argument("--overwrite-downloads", action="store_true", help="Overwrite downloaded files")
    parser.add_argument("--no-extract-existing", action="store_true", help="Do not extract existing local zip files")
    parser.add_argument("--no-vg", action="store_true", help="Skip visual grounding records")
    parser.add_argument("--no-vqa", action="store_true", help="Skip VQA records")
    parser.add_argument("--no-caption", action="store_true", help="Skip caption records")
    parser.add_argument("--no-geojson", action="store_true", help="Skip GeoJSON vector records")
    parser.add_argument("--no-classification", action="store_true", help="Skip scene classification records")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_vg = not args.no_vg
    include_vqa = not args.no_vqa
    include_caption = not args.no_caption
    include_geojson = not args.no_geojson
    include_classification = not args.no_classification

    if args.download:
        download_selected_assets(
            source_root=args.source_root,
            include_vg=args.download_vg or include_vg,
            include_vqa=args.download_vqa or include_vqa,
            include_caption=args.download_caption or include_caption,
            include_geojson=args.download_geojson or include_geojson,
            include_classification=args.download_classification or include_classification,
            overwrite=args.overwrite_downloads,
        )

    output_json, stats = build_coastgpt_bench(
        source_root=args.source_root,
        output_root=args.output_root,
        include_vg=include_vg,
        include_vqa=include_vqa,
        include_caption=include_caption,
        include_geojson=include_geojson,
        include_classification=include_classification,
        copy_images=args.copy_images,
        extract_archives=not args.no_extract_existing,
    )
    print(f"wrote={output_json}")
    print(f"records={stats.total_records}")
    print(json.dumps(stats.__dict__, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

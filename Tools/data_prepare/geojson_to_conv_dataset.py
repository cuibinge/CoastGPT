import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_QUESTION = (
    "<image>\nOutput the GeoJSON FeatureCollection for this image patch. "
    "Return JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GeoJSON annotations into CoastGPT instruction format "
            "by adding conv=[{Question, Answer}] to each record."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains *.geojson files (default: data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root directory for generated *.json files. "
            "If omitted, files are written next to source GeoJSON files."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.geojson",
        help="Glob pattern under input-root to match source files.",
    )
    parser.add_argument(
        "--question-template",
        type=str,
        default=DEFAULT_QUESTION,
        help="Instruction template. Use {name} placeholder if needed.",
    )
    parser.add_argument(
        "--append-conv",
        action="store_true",
        help="Append generated QA to existing conv list instead of replacing it.",
    )
    parser.add_argument(
        "--skip-existing-json",
        action="store_true",
        help="Skip writing if target *.json already exists.",
    )
    parser.add_argument(
        "--compact-answer",
        action="store_true",
        help="Write GeoJSON answer in compact JSON form to reduce token length.",
    )
    return parser.parse_args()


def normalize_records(raw: Any) -> Tuple[List[Any], str]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"], "dict_data"
    if isinstance(raw, list):
        return raw, "list"
    if isinstance(raw, dict):
        return [raw], "single_dict"
    raise ValueError(f"Unsupported source JSON type: {type(raw)}")


def rebuild_container(original: Any, record_style: str, records: List[Any]) -> Any:
    if record_style == "dict_data":
        out = copy.deepcopy(original)
        out["data"] = records
        return out
    if record_style == "list":
        return records
    if record_style == "single_dict":
        return records[0] if records else {}
    raise ValueError(f"Unknown record style: {record_style}")


def build_feature_collection(item: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(item, dict):
        if item.get("type") == "FeatureCollection" and isinstance(item.get("features"), list):
            return {"type": "FeatureCollection", "features": item["features"]}

        if isinstance(item.get("features"), list):
            return {"type": "FeatureCollection", "features": item["features"]}

        if item.get("type") == "Feature" and "geometry" in item:
            return {"type": "FeatureCollection", "features": [item]}

        if "geometry" in item:
            feature = {
                "type": "Feature",
                "geometry": item.get("geometry"),
                "properties": item.get("properties", {}),
            }
            return {"type": "FeatureCollection", "features": [feature]}

    return {"type": "FeatureCollection", "features": []}


def build_question(template: str, item: Dict[str, Any]) -> str:
    name = ""
    if isinstance(item, dict):
        raw_name = item.get("name", item.get("filename", ""))
        if isinstance(raw_name, list):
            raw_name = raw_name[0] if raw_name else ""
        name = str(raw_name)
    return template.replace("{name}", name)


def build_answer_text(feature_collection: Dict[str, Any], compact: bool) -> str:
    if compact:
        return json.dumps(feature_collection, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(feature_collection, ensure_ascii=False, indent=2)


def target_path_for(src_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = src_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".json")


def convert_file(
    src_path: Path,
    dst_path: Path,
    question_template: str,
    append_conv: bool,
    compact_answer: bool,
) -> Tuple[int, int]:
    with src_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records, record_style = normalize_records(raw)
    new_records: List[Any] = []
    converted = 0

    for item in records:
        if not isinstance(item, dict):
            new_records.append(item)
            continue

        new_item = copy.deepcopy(item)
        feature_collection = build_feature_collection(new_item)
        qa = {
            "Question": build_question(question_template, new_item),
            "Answer": build_answer_text(feature_collection, compact_answer),
        }

        if append_conv and isinstance(new_item.get("conv"), list):
            new_item["conv"] = copy.deepcopy(new_item["conv"]) + [qa]
        else:
            new_item["conv"] = [qa]

        new_records.append(new_item)
        converted += 1

    out_data = rebuild_container(raw, record_style, new_records)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return converted, len(records)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or args.input_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    src_files = sorted(input_root.glob(args.pattern))
    src_files = [p for p in src_files if p.is_file() and p.suffix.lower() == ".geojson"]

    if not src_files:
        print(f"[WARN] No source files matched: {input_root / args.pattern}")
        return

    total_files = 0
    total_records = 0
    total_converted = 0

    for src in src_files:
        dst = target_path_for(src, input_root, output_root)
        if args.skip_existing_json and dst.exists():
            print(f"[SKIP] {dst}")
            continue

        converted, total = convert_file(
            src_path=src,
            dst_path=dst,
            question_template=args.question_template,
            append_conv=args.append_conv,
            compact_answer=args.compact_answer,
        )

        total_files += 1
        total_records += total
        total_converted += converted
        print(f"[OK] {src} -> {dst} ({converted}/{total} records)")

    print(
        "[DONE] files=%d, records=%d, converted=%d, output_root=%s"
        % (total_files, total_records, total_converted, output_root)
    )


if __name__ == "__main__":
    main()

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import ml_collections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

from Tools.run_geojson_batch_eval import (  # noqa: E402
    ConfigArgumentParser,
    _generate_single_prediction,
    _load_inference_bundle,
    str2bool,
)


def _load_stage2_items(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        items: List[Dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                items.append(value)
        return items
    return []


def _resolve_default_image_root(dataset_path: Path) -> Path:
    return dataset_path.parent / f"{dataset_path.stem}_Image"


def _candidate_image_paths(raw: Path, dataset_path: Path, image_root: Optional[Path]) -> List[Path]:
    candidates = [raw] if raw.is_absolute() else []
    roots = []
    if image_root is not None:
        roots.append(image_root)
    roots.append(dataset_path.parent)

    names = [raw]
    if raw.suffix.lower() not in _IMAGE_SUFFIXES:
        stems = [raw.name]
        if raw.name.isdigit():
            stems.extend([raw.name.zfill(5), raw.name.zfill(6)])
        for stem in stems:
            for suffix in _IMAGE_SUFFIXES:
                names.append(raw.with_name(stem + suffix))
        for child_name in ("naip.png", "rgb.jpg", "rgb.png"):
            names.append(raw / child_name)

    for root in roots:
        for name in names:
            if not name.is_absolute():
                candidates.append(root / name)
    return candidates


def _resolve_image_path(item: Dict[str, Any], dataset_path: Path, image_root: Optional[Path]) -> Path:
    for key in ("image", "image_file", "name", "filename", "img"):
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        raw = Path(text)
        candidates = _candidate_image_paths(raw, dataset_path, image_root)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Unable to resolve image path for item keys={list(item.keys())}")


def _iter_turns(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    conv = item.get("conv")
    if isinstance(conv, list):
        return [turn for turn in conv if isinstance(turn, dict)]
    if isinstance(conv, dict):
        return [conv]
    if "question" in item or "Question" in item:
        return [item]
    return []


def _get_text(turn: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = turn.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _build_physical_prompt_text(item: Dict[str, Any], dataset_path: Path, image_path: Path) -> str:
    dataset_name = dataset_path.stem
    parts = [f"[Dataset: {dataset_name}]"] if dataset_name else []
    path_lower = str(image_path).lower()
    for needle, value in (
        ("sentinel", "Sentinel"),
        ("landsat", "Landsat"),
        ("gaofen", "GF"),
        ("worldview", "WorldView"),
        ("planet", "Planet"),
        ("jl1", "JL-1"),
        ("jilin", "JL-1"),
    ):
        if needle in path_lower:
            parts.append(f"[Sensor: {value}]")
            break
    return " ".join(parts)


def flatten_stage2_entries(
    *,
    dataset_path: Path,
    image_root: Optional[Path] = None,
    max_questions_per_image: int = 0,
) -> List[Dict[str, Any]]:
    items = _load_stage2_items(dataset_path)
    root = image_root if image_root is not None else _resolve_default_image_root(dataset_path)
    entries: List[Dict[str, Any]] = []

    for item_idx, item in enumerate(items):
        image_path = _resolve_image_path(item, dataset_path, root)
        image_name = str(item.get("name") or item.get("filename") or item.get("img") or image_path.name)
        physical_prompt_text = _build_physical_prompt_text(item, dataset_path, image_path)
        turns = _iter_turns(item)
        if max_questions_per_image > 0:
            turns = turns[:max_questions_per_image]
        for turn_idx, turn in enumerate(turns):
            question = _get_text(turn, "Question", "question")
            answer = _get_text(turn, "Answer", "answer")
            if not question or not answer:
                continue
            entries.append(
                {
                    "sample_id": f"{image_name}#{item_idx}:{turn_idx}",
                    "image_name": image_name,
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "task_type": infer_task_type(question, answer),
                    "physical_prompt_text": physical_prompt_text,
                    "item_index": item_idx,
                    "turn_index": turn_idx,
                }
            )
    return entries


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_bbox(text: str) -> Optional[List[float]]:
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(text or ""))
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in values[:4]]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def infer_task_type(question: str, answer: str) -> str:
    question_text = str(question or "").strip().lower()
    answer_text = normalize_answer(answer)
    if question_text.startswith("[vg]") or _parse_bbox(answer) is not None:
        return "vg"
    if question_text.startswith("[cap]"):
        return "cap"
    if question_text.startswith("[cls]"):
        return "cls"
    if "rural" in question_text and "urban" in question_text:
        return "cls"
    if len(answer_text.split()) <= 4 and any(
        token in question_text for token in ("class", "category", "classification")
    ):
        return "cls"
    return "vqa"


def _extract_label_prediction(prediction: str, label_choices: Optional[Sequence[str]] = None) -> str:
    pred = normalize_answer(prediction)
    if not pred:
        return ""

    choices = []
    for choice in label_choices or []:
        label = normalize_answer(choice)
        if label and label not in choices:
            choices.append(label)

    matches = []
    for label in sorted(choices, key=lambda item: (-len(item.split()), -len(item))):
        match = re.search(rf"(?<!\w){re.escape(label)}(?!\w)", pred)
        if match:
            matches.append((match.start(), label))
    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    tokens = pred.split()
    return tokens[0] if tokens else ""


def collect_label_choices(entries: Sequence[Dict[str, Any]]) -> List[str]:
    choices = []
    for entry in entries:
        if str(entry.get("task_type", "")) in {"vg", "cap"}:
            continue
        answer = normalize_answer(str(entry.get("answer", "")))
        if not answer:
            continue
        if len(answer) > 64 or len(answer.split()) > 4:
            continue
        if answer not in choices:
            choices.append(answer)
    return sorted(choices)


def _bbox_iou(prediction: str, answer: str) -> Optional[float]:
    pred_box = _parse_bbox(prediction)
    answer_box = _parse_bbox(answer)
    if pred_box is None or answer_box is None:
        return None
    px1, py1, px2, py2 = pred_box
    ax1, ay1, ax2, ay2 = answer_box
    ix1, iy1 = max(px1, ax1), max(py1, ay1)
    ix2, iy2 = min(px2, ax2), min(py2, ay2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    answer_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union = pred_area + answer_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _metric_tokens(text: str) -> List[str]:
    return [token for token in normalize_answer(text).split() if token not in {"a", "an", "the"}]


def _caption_token_f1(prediction: str, answer: str) -> float:
    pred_tokens = _metric_tokens(prediction)
    answer_tokens = _metric_tokens(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in answer_tokens:
        count = pred_counts.get(token, 0)
        if count > 0:
            overlap += 1
            pred_counts[token] = count - 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        cur = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                cur.append(prev[idx - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[idx]))
        prev = cur
    return prev[-1]


def _caption_rouge_l(prediction: str, answer: str) -> float:
    pred_tokens = _metric_tokens(prediction)
    answer_tokens = _metric_tokens(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    return _lcs_length(pred_tokens, answer_tokens) / len(answer_tokens)


def score_answer(
    prediction: str,
    answer: str,
    label_choices: Optional[Sequence[str]] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    pred = normalize_answer(prediction)
    gt = normalize_answer(answer)
    task = task_type or infer_task_type("", answer)
    choices = list(label_choices or [])
    if gt and gt not in choices:
        choices.append(gt)
    label_prediction = _extract_label_prediction(prediction, choices)
    scores: Dict[str, Any] = {
        "task_type": task,
        "exact": bool(gt) and pred == gt,
        "contains": bool(gt) and (pred == gt or gt in pred),
    }
    if task not in {"vg", "cap"}:
        scores.update(
            {
                "label_answer": gt,
                "label_prediction": label_prediction,
                "label_exact": bool(gt) and label_prediction == gt,
            }
        )
    if task == "vg":
        iou = _bbox_iou(prediction, answer)
        scores["vg_iou"] = float(iou) if iou is not None else 0.0
        scores["vg_acc_at_025"] = bool(iou is not None and iou >= 0.25)
        scores["vg_acc_at_05"] = bool(iou is not None and iou >= 0.5)
        scores["vg_ap_at_05"] = 1.0 if scores["vg_acc_at_05"] else 0.0
    elif task == "cap":
        scores["cap_token_f1"] = _caption_token_f1(prediction, answer)
        scores["cap_rouge_l"] = _caption_rouge_l(prediction, answer)
    return scores


_SCORE_FIELD_KEYS = {
    "task_type",
    "exact",
    "contains",
    "label_answer",
    "label_prediction",
    "label_exact",
    "vg_iou",
    "vg_acc_at_025",
    "vg_acc_at_05",
    "vg_ap_at_05",
    "cap_token_f1",
    "cap_rouge_l",
}


def rescore_existing_record(
    record: Dict[str, Any],
    answer: str,
    label_choices: Optional[Sequence[str]] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    rescored = {key: value for key, value in record.items() if key not in _SCORE_FIELD_KEYS}
    rescored.update(
        score_answer(
            str(record.get("prediction", "")),
            answer,
            label_choices=label_choices,
            task_type=task_type,
        )
    )
    return rescored


def _summarize_task(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    result: Dict[str, Any] = {"total": total}
    if total == 0:
        return result
    result["exact_rate"] = sum(1 for r in records if r.get("exact")) / total
    result["contains_rate"] = sum(1 for r in records if r.get("contains")) / total
    if any("label_exact" in r for r in records):
        result["label_exact_rate"] = sum(1 for r in records if r.get("label_exact")) / total
    vg_records = [r for r in records if "vg_iou" in r]
    if vg_records:
        result["vg_mean_iou"] = sum(float(r.get("vg_iou", 0.0)) for r in vg_records) / len(vg_records)
        result["vg_acc_at_025_rate"] = sum(
            1
            for r in vg_records
            if r.get("vg_acc_at_025", float(r.get("vg_iou", 0.0)) >= 0.25)
        ) / len(vg_records)
        result["vg_acc_at_05_rate"] = sum(
            1
            for r in vg_records
            if r.get("vg_acc_at_05", float(r.get("vg_iou", 0.0)) >= 0.5)
        ) / len(vg_records)
        result["vg_ap_at_05"] = sum(
            float(r.get("vg_ap_at_05", 1.0 if r.get("vg_acc_at_05", float(r.get("vg_iou", 0.0)) >= 0.5) else 0.0))
            for r in vg_records
        ) / len(vg_records)
    cap_records = [r for r in records if "cap_token_f1" in r or "cap_rouge_l" in r]
    if cap_records:
        result["cap_token_f1"] = sum(float(r.get("cap_token_f1", 0.0)) for r in cap_records) / len(cap_records)
        result["cap_rouge_l"] = sum(float(r.get("cap_rouge_l", 0.0)) for r in cap_records) / len(cap_records)
    return result


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}
    exact = sum(1 for r in records if r.get("exact"))
    contains = sum(1 for r in records if r.get("contains"))
    label_exact = sum(1 for r in records if r.get("label_exact"))
    all_unk = sum(1 for r in records if r.get("all_unk"))
    avg_unk_ratio = sum(float(r.get("unk_ratio", 0.0)) for r in records) / total
    label_confusion: Dict[str, Dict[str, int]] = {}
    for record in records:
        answer = str(record.get("label_answer", "")).strip()
        prediction = str(record.get("label_prediction", "")).strip()
        if not answer:
            continue
        label_confusion.setdefault(answer, {})
        label_confusion[answer][prediction] = label_confusion[answer].get(prediction, 0) + 1
    label_recall_by_answer = {}
    for answer, pred_counts in label_confusion.items():
        answer_total = sum(pred_counts.values())
        label_recall_by_answer[answer] = (
            pred_counts.get(answer, 0) / answer_total if answer_total > 0 else 0.0
        )
    label_balanced_accuracy = (
        sum(label_recall_by_answer.values()) / len(label_recall_by_answer)
        if label_recall_by_answer
        else 0.0
    )
    by_task: Dict[str, Dict[str, Any]] = {}
    task_names = sorted({str(record.get("task_type", "unknown") or "unknown") for record in records})
    for task_name in task_names:
        by_task[task_name] = _summarize_task(
            [
                record
                for record in records
                if str(record.get("task_type", "unknown") or "unknown") == task_name
            ]
        )
    return {
        "total": total,
        "exact_count": exact,
        "exact_rate": exact / total,
        "contains_count": contains,
        "contains_rate": contains / total,
        "label_exact_count": label_exact,
        "label_exact_rate": label_exact / total,
        "label_balanced_accuracy": label_balanced_accuracy,
        "label_recall_by_answer": label_recall_by_answer,
        "label_confusion": label_confusion,
        "by_task": by_task,
        "all_unk_count": all_unk,
        "all_unk_rate": all_unk / total,
        "avg_unk_ratio": avg_unk_ratio,
    }


def _collect_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return existing
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("sample_id"):
            existing[str(record["sample_id"])] = record
    return existing


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _apply_wavelet_overrides(config: ml_collections.ConfigDict) -> None:
    if not hasattr(config, "wavelet_adapter_enabled"):
        return
    wavelet_cfg = getattr(config, "wavelet_adapter", ml_collections.ConfigDict())
    wavelet_cfg.enabled = bool(config.wavelet_adapter_enabled)
    wavelet_cfg.mode = str(getattr(config, "wavelet_adapter_mode", "learnable_direct"))
    wavelet_cfg.in_channels = int(getattr(config, "wavelet_adapter_in_channels", 4))
    wavelet_cfg.multiband_channels = int(getattr(config, "wavelet_adapter_multiband_channels", 4))
    wavelet_cfg.target_channels = int(getattr(config, "wavelet_adapter_target_channels", 3))
    wavelet_cfg.init = str(getattr(config, "wavelet_adapter_init", "rgb_from_bgrn"))
    wavelet_cfg.normalize_output = bool(getattr(config, "wavelet_adapter_normalize_output", True))
    wavelet_cfg.output_mean = list(getattr(config, "wavelet_adapter_output_mean", [0.485, 0.456, 0.406]))
    wavelet_cfg.output_std = list(getattr(config, "wavelet_adapter_output_std", [0.229, 0.224, 0.225]))
    config.wavelet_adapter = wavelet_cfg


def parse_args() -> ml_collections.ConfigDict:
    parser = ConfigArgumentParser()
    parser.add_argument("--dataset-json", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--predictions-jsonl", type=str, default=None)
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-questions-per-image", type=int, default=1)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--skip-inference", type=str2bool, default=False)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "bf16", "float32", "fp32"], default=None)
    parser.add_argument("--fp16", type=str2bool, default=None)
    parser.add_argument("--bf16", type=str2bool, default=None)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--skip-text-lora", type=str2bool, default=False)
    parser.add_argument("--diag-on-unk", type=str2bool, default=False)
    parser.add_argument("--diag-topk", type=int, default=8)
    parser.add_argument("--accelerator", default="npu", type=str, choices=["cpu", "gpu", "npu", "mps"])
    parser.add_argument("--use-checkpoint", default=False, type=str2bool)
    parser.add_argument("--multiband-inference-mode", choices=["auto", "off", "zero-extra"], default="off")
    parser.add_argument("--multiband-max-channels", type=int, default=4)

    parser.add_argument("--wavelet-adapter-enabled", type=str2bool, default=None)
    parser.add_argument("--wavelet-adapter-mode", type=str, default="learnable_direct")
    parser.add_argument("--wavelet-adapter-in-channels", type=int, default=4)
    parser.add_argument("--wavelet-adapter-multiband-channels", type=int, default=4)
    parser.add_argument("--wavelet-adapter-target-channels", type=int, default=3)
    parser.add_argument("--wavelet-adapter-init", type=str, default="rgb_from_bgrn")
    parser.add_argument("--wavelet-adapter-normalize-output", type=str2bool, default=True)

    config = ml_collections.ConfigDict(parser.parse_args())
    _apply_wavelet_overrides(config)
    return config


def main() -> None:
    config = parse_args()
    dataset_path = Path(config.dataset_json)
    image_root = Path(config.image_root) if getattr(config, "image_root", None) else None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_jsonl = Path(getattr(config, "predictions_jsonl", None) or output_dir / "predictions.jsonl")
    eval_json = Path(getattr(config, "eval_json", None) or output_dir / "eval_summary.json")

    entries = flatten_stage2_entries(
        dataset_path=dataset_path,
        image_root=image_root,
        max_questions_per_image=int(config.max_questions_per_image),
    )
    if int(config.limit) > 0:
        entries = entries[: int(config.limit)]
    label_choices = collect_label_choices(entries)

    existing = _collect_existing(predictions_jsonl) if bool(config.resume) else {}
    if predictions_jsonl.exists() and not bool(config.resume):
        predictions_jsonl.unlink()
        existing = {}

    if not bool(config.skip_inference):
        model, tokenizer, vision_processor, device, dtype = _load_inference_bundle(config)
        for idx, entry in enumerate(entries, start=1):
            sample_id = str(entry["sample_id"])
            if sample_id in existing:
                if not bool(config.quiet):
                    print(f"[{idx}/{len(entries)}] skip existing sample_id={sample_id}")
                continue
            pred_result = _generate_single_prediction(
                config=config,
                model=model,
                tokenizer=tokenizer,
                vision_processor=vision_processor,
                device=device,
                dtype=dtype,
                image_path=entry["image_path"],
                prompt_text=entry["question"],
                physical_prompt_text=str(entry.get("physical_prompt_text", "")),
            )
            scores = score_answer(
                str(pred_result["prediction"]),
                str(entry["answer"]),
                label_choices=label_choices,
                task_type=str(entry.get("task_type", "")),
            )
            record = {
                "sample_id": sample_id,
                "image": str(entry["image_path"]),
                "question": entry["question"],
                "answer": entry["answer"],
                "task_type": entry.get("task_type", ""),
                "prediction": pred_result["prediction"],
                "raw_output": pred_result["raw_output"],
                "unk_ratio": pred_result["unk_ratio"],
                "all_unk": pred_result["all_unk"],
                **scores,
            }
            _append_jsonl(predictions_jsonl, record)
            existing[sample_id] = record
            if not bool(config.quiet):
                print(f"[{idx}/{len(entries)}] sample_id={sample_id} exact={scores['exact']} contains={scores['contains']}")

    existing_records = _collect_existing(predictions_jsonl)
    records: List[Dict[str, Any]] = []
    for entry in entries:
        sample_id = str(entry["sample_id"])
        record = existing_records.get(sample_id)
        if record is None:
            continue
        rescored = rescore_existing_record(
            record,
            answer=str(entry["answer"]),
            label_choices=label_choices,
            task_type=str(entry.get("task_type", "")),
        )
        records.append(rescored)
    payload = {
        "dataset_json": str(dataset_path),
        "image_root": str(image_root or _resolve_default_image_root(dataset_path)),
        "predictions": str(predictions_jsonl),
        "summary": summarize(records),
        "details": records,
    }
    eval_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"[DONE] predictions={predictions_jsonl}")
    print(f"[DONE] eval={eval_json}")


if __name__ == "__main__":
    main()

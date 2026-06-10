import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import ml_collections
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.eval_geojson import (
    _candidate_ids_from_dataset_entry,
    evaluate_entry,
    load_dataset_entries,
    load_prediction_map,
    parse_property_keys,
    summarize_results,
)
from Tools.geojson_to_arcgis import (
    extract_json_object,
    to_feature_collection,
    validate_for_arcgis,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    text = str(v).lower()
    if text in ("yes", "true", "t", "y", "1"):
        return True
    if text in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class ConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser(add_help=False)
        self.config_parser.add_argument(
            "-c",
            "--config",
            default="Configs/inference.yaml",
            metavar="FILE",
            help="where to load YAML configuration",
        )
        super().__init__(
            *args,
            parents=[self.config_parser],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **kwargs,
        )

    def parse_args(self, args=None):
        res, remaining_argv = self.config_parser.parse_known_args(args)
        config_vars: Dict[str, Any] = {}
        if res.config is not None:
            with open(res.config, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a dict: {res.config}")
            config_vars.update(loaded)

        namespace = vars(super().parse_args(remaining_argv))
        config_vars.update({k: v for k, v in namespace.items() if v is not None})
        return config_vars


def parse_args() -> ml_collections.ConfigDict:
    parser = ConfigArgumentParser()
    parser.add_argument("--opts", default=None, nargs="+")

    parser.add_argument("--dataset-json", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--predictions-jsonl", type=str, default=None)
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--normalized-geojson-dir", type=str, default=None)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--skip-inference", type=str2bool, default=False)
    parser.add_argument("--skip-eval", type=str2bool, default=False)
    parser.add_argument("--save-normalized-geojson", type=str2bool, default=True)

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--skip-text-lora", type=str2bool, default=False)
    parser.add_argument("--diag-on-unk", type=str2bool, default=True)
    parser.add_argument("--diag-topk", type=int, default=8)

    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "npu", "mps"],
    )
    parser.add_argument("--use-checkpoint", default=False, type=str2bool)

    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--property-keys", type=str, default="DLMC,label,class_name,category,target")
    parser.add_argument("--allow-geometry-collection", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    config = parser.parse_args()
    return ml_collections.config_dict.ConfigDict(config)


def _resolve_output_paths(config: ml_collections.ConfigDict) -> Tuple[Path, Path, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl_value = getattr(config, "predictions_jsonl", None)
    eval_json_value = getattr(config, "eval_json", None)
    normalized_geojson_dir_value = getattr(config, "normalized_geojson_dir", None)

    predictions_jsonl = (
        Path(predictions_jsonl_value)
        if predictions_jsonl_value
        else output_dir / "predictions.jsonl"
    )
    eval_json = Path(eval_json_value) if eval_json_value else output_dir / "eval_summary.json"
    normalized_geojson_dir = (
        Path(normalized_geojson_dir_value)
        if normalized_geojson_dir_value
        else output_dir / "normalized_geojson"
    )

    predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    if bool(config.save_normalized_geojson):
        normalized_geojson_dir.mkdir(parents=True, exist_ok=True)

    return predictions_jsonl, eval_json, normalized_geojson_dir


def _resolve_image_root(dataset_path: Path, config: ml_collections.ConfigDict) -> Optional[Path]:
    image_root_value = getattr(config, "image_root", None)
    if image_root_value:
        return Path(image_root_value)

    candidate = dataset_path.parent / f"{dataset_path.stem}_Image"
    if candidate.exists():
        return candidate
    return None


def _collect_existing_sample_ids(predictions_jsonl: Path) -> Set[str]:
    sample_ids: Set[str] = set()
    if not predictions_jsonl.exists():
        return sample_ids

    for raw_line in predictions_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            sample_ids.add(sample_id)
    return sample_ids


def _sanitize_filename(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "sample"
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r'[<>:"|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:200] or "sample"


def _extract_sample_id(entry: Dict[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "name", "filename"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"sample_{idx:06d}"


def _resolve_image_path(
    entry: Dict[str, Any],
    dataset_path: Path,
    image_root: Optional[Path],
) -> Path:
    candidates: List[Path] = []
    raw_candidates: List[str] = []
    for key in ("image", "image_file", "name", "filename"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        raw_candidates.append(text)
        candidate_path = Path(text)
        if candidate_path.is_absolute():
            candidates.append(candidate_path)
        else:
            if image_root is not None:
                candidates.append(image_root / candidate_path)
            candidates.append(dataset_path.parent / candidate_path)

    for path in candidates:
        if path.exists():
            return path

    hint = ", ".join(raw_candidates[:4]) if raw_candidates else "<none>"
    raise FileNotFoundError(
        f"Unable to resolve image path for sample. dataset={dataset_path} candidates={hint}"
    )


def _extract_question(entry: Dict[str, Any], override_prompt: Optional[str]) -> str:
    if override_prompt:
        return str(override_prompt)
    conv = entry.get("conv", None)
    if not isinstance(conv, list) or not conv:
        raise ValueError("Dataset entry has no conv list.")
    question = conv[0].get("Question", None)
    if question is None:
        raise ValueError("Dataset entry conv[0] has no Question.")
    return str(question)


def _load_inference_bundle(config: ml_collections.ConfigDict):
    from Dataset.build_transform import build_vlp_transform
    from Inference import (
        _fix_tokenizer_ids,
        _load_checkpoint,
        _normalize_inference_runtime,
        _normalize_npu_config,
        _resolve_device,
        _to_dtype_name,
    )
    from Models.coastgpt import CoastGPT

    _normalize_npu_config(config)
    _normalize_inference_runtime(config)

    torch.manual_seed(int(getattr(config, "seed", 322)))
    device = _resolve_device(config)
    dtype = _to_dtype_name(getattr(config, "dtype", "float16"))

    model = CoastGPT(config)
    vision_processor = (
        model.get_image_processor()
        if bool(getattr(config, "hf_model", False))
        else build_vlp_transform(config, is_train=False)
    )
    model.to(dtype)

    model_path = getattr(config, "model_path", None)
    if not model_path:
        raise ValueError("--model-path is required unless --skip-inference True is used.")
    msg = _load_checkpoint(
        model,
        model_path,
        skip_text_lora=bool(getattr(config, "skip_text_lora", False)),
    )
    if msg is not None:
        print(msg)

    tokenizer = model.language.tokenizer
    _fix_tokenizer_ids(tokenizer, model)

    model.to(device)
    model.eval()
    return model, tokenizer, vision_processor, device, dtype


def _generate_single_prediction(
    *,
    config: ml_collections.ConfigDict,
    model: Any,
    tokenizer,
    vision_processor,
    device: torch.device,
    dtype: torch.dtype,
    image_path: Path,
    prompt_text: str,
) -> Dict[str, Any]:
    from Dataset.conversation import SeparatorStyle, default_conversation
    from Inference import (
        _build_generation_kwargs,
        _build_image_tensor,
        _calc_unk_stats,
        _decode_new_tokens,
        _normalize_user_instruction,
        _run_logits_probe,
    )
    from Models import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        tokenizer_image_token,
    )
    from Models.utils import KeywordsStoppingCriteria

    image_tensor = _build_image_tensor(config, vision_processor, str(image_path), device, dtype)

    user_prompt = _normalize_user_instruction(prompt_text)
    if image_tensor is not None:
        if bool(getattr(config, "tune_im_start", False)):
            user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_prompt
        else:
            user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    gen_kwargs = _build_generation_kwargs(config, tokenizer, stopping_criteria)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            **gen_kwargs,
        )

    outputs, raw_outputs, new_tokens = _decode_new_tokens(
        tokenizer, output_ids, int(input_ids.shape[1]), stop_str
    )
    all_unk, unk_ratio = _calc_unk_stats(new_tokens, tokenizer)

    if outputs == "":
        if bool(getattr(config, "diag_on_unk", True)):
            _run_logits_probe(
                model=model,
                input_ids=input_ids,
                image_tensor=image_tensor,
                base_gen_kwargs=gen_kwargs,
                tokenizer=tokenizer,
                topk=int(getattr(config, "diag_topk", 8)),
            )
        visible_text = raw_outputs.strip()
    else:
        visible_text = outputs
        if bool(getattr(config, "diag_on_unk", True)) and all_unk:
            _run_logits_probe(
                model=model,
                input_ids=input_ids,
                image_tensor=image_tensor,
                base_gen_kwargs=gen_kwargs,
                tokenizer=tokenizer,
                topk=int(getattr(config, "diag_topk", 8)),
            )

    result = {
        "prediction": visible_text,
        "raw_output": raw_outputs,
        "all_unk": bool(all_unk),
        "unk_ratio": float(unk_ratio),
        "prompt_length": int(input_ids.shape[1]),
        "generated_tokens": int(new_tokens.shape[0]),
    }

    del input_ids, output_ids, new_tokens, image_tensor
    return result


def _normalize_prediction_text(
    prediction_text: str,
    allow_geometry_collection: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw_obj = extract_json_object(prediction_text)
        fc = to_feature_collection(raw_obj)
        validate_for_arcgis(fc, allow_geometry_collection=allow_geometry_collection)
        return fc, None
    except Exception as exc:
        return None, str(exc)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_evaluation(
    *,
    dataset_path: Path,
    entries: Sequence[Dict[str, Any]],
    predictions_jsonl: Path,
    eval_json: Path,
    config: ml_collections.ConfigDict,
) -> Dict[str, Any]:
    pred_map = load_prediction_map(predictions_jsonl)
    property_keys = parse_property_keys(config.property_keys)

    results: List[Dict[str, Any]] = []
    for entry in entries:
        pred_text = None
        for candidate_id in _candidate_ids_from_dataset_entry(entry):
            if candidate_id in pred_map:
                pred_text = pred_map[candidate_id]
                break
        results.append(
            evaluate_entry(
                entry=entry,
                pred_text=pred_text,
                iou_threshold=float(config.iou_threshold),
                property_keys=property_keys,
                allow_geometry_collection=bool(config.allow_geometry_collection),
            )
        )

    summary = summarize_results(results)
    payload = {
        "dataset_json": str(dataset_path),
        "predictions": str(predictions_jsonl),
        "iou_threshold": float(config.iou_threshold),
        "property_keys": "all" if property_keys is None else list(property_keys),
        "summary": summary,
        "details": results,
    }
    eval_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("sample_complete_match_rate") is not None:
        print(
            "[OK] "
            f"ArcGIS-ready={summary.get('arcgis_ready_rate', 0.0):.4f}, "
            f"F1@IoU={summary.get('f1_iou', 0.0):.4f}, "
            f"sample_complete_match={summary.get('sample_complete_match_rate', 0.0):.4f}"
        )


def main() -> None:
    config = parse_args()
    config.adjust_norm = False

    dataset_path = Path(config.dataset_json)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset-json not found: {dataset_path}")

    entries = load_dataset_entries(dataset_path)
    if int(config.limit) > 0:
        entries = entries[: int(config.limit)]

    predictions_jsonl, eval_json, normalized_geojson_dir = _resolve_output_paths(config)
    image_root = _resolve_image_root(dataset_path, config)

    if bool(config.skip_inference):
        if not predictions_jsonl.exists():
            raise FileNotFoundError(
                f"--skip-inference True was set, but predictions file does not exist: {predictions_jsonl}"
            )
    else:
        existing_ids = _collect_existing_sample_ids(predictions_jsonl) if bool(config.resume) else set()
        if predictions_jsonl.exists() and not bool(config.resume):
            predictions_jsonl.unlink()
            existing_ids = set()

        model, tokenizer, vision_processor, device, dtype = _load_inference_bundle(config)

        total = len(entries)
        for idx, entry in enumerate(entries, start=1):
            sample_id = _extract_sample_id(entry, idx - 1)
            if sample_id in existing_ids:
                if not bool(config.quiet):
                    print(f"[{idx}/{total}] skip existing sample_id={sample_id}")
                continue

            image_path = _resolve_image_path(entry, dataset_path, image_root)
            question = _extract_question(entry, getattr(config, "prompt", None))
            pred_result = _generate_single_prediction(
                config=config,
                model=model,
                tokenizer=tokenizer,
                vision_processor=vision_processor,
                device=device,
                dtype=dtype,
                image_path=image_path,
                prompt_text=question,
            )

            normalized_fc = None
            normalized_error = None
            normalized_geojson_path = None
            prediction_text = str(pred_result["prediction"] or "").strip()
            if prediction_text:
                normalized_fc, normalized_error = _normalize_prediction_text(
                    prediction_text,
                    allow_geometry_collection=bool(config.allow_geometry_collection),
                )
                if normalized_fc is not None and bool(config.save_normalized_geojson):
                    normalized_geojson_path = normalized_geojson_dir / f"{_sanitize_filename(sample_id)}.geojson"
                    normalized_geojson_path.write_text(
                        json.dumps(normalized_fc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

            record = {
                "sample_id": sample_id,
                "name": entry.get("name", ""),
                "image": str(image_path),
                "question": question,
                "prediction": pred_result["prediction"],
                "raw_output": pred_result["raw_output"],
                "prompt_length": pred_result["prompt_length"],
                "generated_tokens": pred_result["generated_tokens"],
                "unk_ratio": pred_result["unk_ratio"],
                "all_unk": pred_result["all_unk"],
                "normalized_geojson_path": str(normalized_geojson_path) if normalized_geojson_path else None,
                "normalized_geojson_error": normalized_error,
            }
            _append_jsonl(predictions_jsonl, record)
            existing_ids.add(sample_id)

            if not bool(config.quiet):
                status = "ok" if normalized_fc is not None else f"normalize_failed={normalized_error}"
                print(f"[{idx}/{total}] sample_id={sample_id} {status}")

    if bool(config.skip_eval):
        print(f"[DONE] predictions written to {predictions_jsonl}")
        return

    payload = _run_evaluation(
        dataset_path=dataset_path,
        entries=entries,
        predictions_jsonl=predictions_jsonl,
        eval_json=eval_json,
        config=config,
    )
    _print_summary(payload["summary"])
    print(f"[DONE] predictions={predictions_jsonl}")
    print(f"[DONE] eval={eval_json}")


if __name__ == "__main__":
    main()

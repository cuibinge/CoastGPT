import argparse
import importlib.util
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
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "bf16", "float32", "fp32"], default=None)
    parser.add_argument("--fp16", type=str2bool, default=None)
    parser.add_argument("--bf16", type=str2bool, default=None)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--skip-text-lora", type=str2bool, default=False)
    parser.add_argument("--diag-on-unk", type=str2bool, default=True)
    parser.add_argument("--diag-topk", type=int, default=8)
    parser.add_argument(
        "--multiband-inference-mode",
        type=str,
        choices=["auto", "off", "zero-extra"],
        default="auto",
        help="auto loads TIFF multi-band input, off uses RGB only, zero-extra zeros bands after RGB.",
    )
    parser.add_argument("--multiband-max-channels", type=int, default=4)

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


def _build_multiband_tensor_for_generation(
    *,
    config: ml_collections.ConfigDict,
    image_path: Path,
    image_tensor: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mode = str(getattr(config, "multiband_inference_mode", "auto")).strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return None, None

    module_path = PROJECT_ROOT / "Dataset" / "multiband_source.py"
    spec = importlib.util.spec_from_file_location("multiband_source", module_path)
    multiband_source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multiband_source)

    max_channels = int(getattr(config, "multiband_max_channels", 4))
    multiband = multiband_source.load_multiband_tensor(
        image_path,
        output_size=None,
        max_channels=max_channels,
    )
    if multiband is None:
        return None, None

    wavelet_cfg = getattr(config, "wavelet_adapter", {})
    expected_channels = int(
        wavelet_cfg.get("in_channels", wavelet_cfg.get("multiband_channels", max_channels))
        if hasattr(wavelet_cfg, "get")
        else max_channels
    )
    if multiband.shape[0] < expected_channels:
        return None, None

    if mode == "zero-extra" and multiband.shape[0] > 3:
        multiband = multiband.clone()
        multiband[3:, :, :] = 0.0

    multiband = multiband.unsqueeze(0).to(device=device, dtype=torch.float32)
    valid_multiband = torch.ones((1,), device=device, dtype=torch.bool)
    return multiband, valid_multiband


def _ensure_image_token(
    prompt_text: str,
    *,
    tune_im_start: bool,
    default_image_token: str,
    default_im_start_token: str,
    default_im_end_token: str,
) -> str:
    prompt_text = str(prompt_text or "").strip()
    image_token = default_image_token
    if tune_im_start:
        image_token = default_im_start_token + default_image_token + default_im_end_token
    if default_image_token in prompt_text:
        text = (
            prompt_text.replace(default_im_start_token, "")
            .replace(default_im_end_token, "")
            .replace(default_image_token, "")
            .strip()
        )
        return (image_token + ("\n" + text if text else "")).strip()
    return (image_token + ("\n" + prompt_text if prompt_text else "")).strip()


def _extract_free_element_text(text: str) -> str:
    text_lower = re.sub(r"\[[a-z0-9_]+\]", " ", str(text or "").lower())
    patterns = [
        r"(?:find|locate|detect|identify|segment|extract)\s+(?:a|an|the)?\s*([a-z][a-z0-9 -]{2,64})",
        r"(?:about|of|for)\s+(?:the|a|an)?\s*([a-z][a-z0-9 -]{2,64})",
    ]
    stop_terms = {
        "image",
        "scene",
        "picture",
        "photo",
        "target",
        "object",
        "area",
        "region",
        "class",
        "category",
        "dimensions",
    }
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if not match:
            continue
        candidate = match.group(1).strip(" .,;:!?\"'()[]{}")
        candidate = re.split(
            r",|\.|;|\?|!|\band\b|\bwith\b|\bthat\b|\bwhich\b|\bin\b|\bon\b",
            candidate,
        )[0].strip()
        words = [word for word in candidate.split() if word]
        if not words:
            continue
        candidate = " ".join(words[:4])
        if candidate in stop_terms or len(candidate) < 3:
            continue
        return candidate
    return ""


def _infer_semantic_route_texts(prompt_text: str) -> Tuple[str, str]:
    text_lower = str(prompt_text or "").lower()
    if "[cls]" in text_lower or ("rural" in text_lower and "urban" in text_lower):
        return "场景分类", "土地覆盖"
    if "[vg]" in text_lower or "[loc]" in text_lower or "bbox" in text_lower or "bounding box" in text_lower:
        return "视觉定位", _extract_free_element_text(prompt_text) or "无"
    if "[cap]" in text_lower or "[caption]" in text_lower or "caption" in text_lower:
        return "描述", "无"
    if any(token in text_lower for token in ("question", "answer", "what", "where", "when", "why", "how")):
        return "视觉问答", "无"
    return "描述", "无"


def _tokenize_route_text(
    tokenizer,
    text: str,
    device: torch.device,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    tokenize_kwargs = dict(
        return_tensors="pt",
        padding="max_length" if max_length is not None else True,
        truncation=True,
    )
    if max_length is not None:
        tokenize_kwargs["max_length"] = int(max_length)
    tokens = tokenizer([text], **tokenize_kwargs)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _build_semantic_route_inputs(
    tokenizer,
    prompt_text: str,
    device: torch.device,
    physical_prompt_text: str = "",
    physical_prompt_max_len: int = 64,
    task_text_max_len: int = 16,
    element_text_max_len: int = 16,
) -> Dict[str, Optional[torch.Tensor]]:
    task_text, element_text = _infer_semantic_route_texts(prompt_text)
    task_ids, task_mask = _tokenize_route_text(
        tokenizer,
        task_text,
        device,
        max_length=task_text_max_len,
    )
    element_ids, element_mask = _tokenize_route_text(
        tokenizer,
        element_text,
        device,
        max_length=element_text_max_len,
    )
    route_inputs = {
        "task_text_ids": task_ids,
        "task_text_attention_mask": task_mask,
        "element_text_ids": element_ids,
        "element_text_attention_mask": element_mask,
    }
    if str(physical_prompt_text or "").strip():
        physical_ids, physical_mask = _tokenize_route_text(
            tokenizer,
            physical_prompt_text,
            device,
            max_length=physical_prompt_max_len,
        )
        route_inputs["physical_prompt_ids"] = physical_ids
        route_inputs["physical_prompt_attention_mask"] = physical_mask
    return route_inputs


def _infer_decode_prompt_len(output_ids: torch.Tensor, input_ids: torch.Tensor) -> int:
    if not torch.is_tensor(output_ids) or not torch.is_tensor(input_ids):
        return 0
    if output_ids.ndim != 2 or input_ids.ndim != 2:
        return 0
    prompt_len = int(input_ids.shape[1])
    if prompt_len <= 0 or int(output_ids.shape[1]) < prompt_len:
        return 0
    output_prefix = output_ids[0, :prompt_len].detach().cpu()
    prompt_ids = input_ids[0, :prompt_len].detach().cpu()
    if torch.equal(output_prefix, prompt_ids):
        return prompt_len
    return 0


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
    physical_prompt_text: str = "",
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
    multiband_tensor, valid_multiband = _build_multiband_tensor_for_generation(
        config=config,
        image_path=image_path,
        image_tensor=image_tensor,
        device=device,
        dtype=dtype,
    )

    user_prompt = _normalize_user_instruction(prompt_text)
    if image_tensor is not None:
        user_prompt = _ensure_image_token(
            user_prompt,
            tune_im_start=bool(getattr(config, "tune_im_start", False)),
            default_image_token=DEFAULT_IMAGE_TOKEN,
            default_im_start_token=DEFAULT_IM_START_TOKEN,
            default_im_end_token=DEFAULT_IM_END_TOKEN,
        )

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
        semantic_route_inputs = _build_semantic_route_inputs(
            tokenizer,
            prompt_text,
            device,
            physical_prompt_text=physical_prompt_text,
            physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
            task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
            element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
        )
        generate_inputs = {
            "input_ids": input_ids,
            "images": image_tensor,
            **gen_kwargs,
            **semantic_route_inputs,
        }
        if multiband_tensor is not None:
            generate_inputs["multiband"] = multiband_tensor
            generate_inputs["valid_multiband"] = valid_multiband
        output_ids = model.generate(**generate_inputs)

    decode_prompt_len = _infer_decode_prompt_len(output_ids, input_ids)
    outputs, raw_outputs, new_tokens = _decode_new_tokens(
        tokenizer, output_ids, decode_prompt_len, stop_str
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
        "decode_prompt_length": int(decode_prompt_len),
        "generated_tokens": int(new_tokens.shape[0]),
    }

    del input_ids, output_ids, new_tokens, image_tensor, multiband_tensor, valid_multiband
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

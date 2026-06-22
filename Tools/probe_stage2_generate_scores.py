import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import ml_collections
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.probe_stage2_teacher_forced_vs_generate import (  # noqa: E402
    LABEL_CHOICES,
    _build_dataset_entries,
    _label_token_ids,
    normalize_label,
    resolve_dataset_stage,
    summarize_label_pairs,
)


def pick_label_from_logits(logits: torch.Tensor, label_token_ids: Dict[str, int]) -> Dict[str, Any]:
    if logits.ndim == 2:
        row = logits[0]
    elif logits.ndim == 1:
        row = logits
    else:
        raise ValueError(f"expected 1D/2D logits, got shape={tuple(logits.shape)}")
    row = row.detach().float().cpu()
    scores = {label: float(row[int(token_id)].item()) for label, token_id in label_token_ids.items()}
    pred_label = max(scores.items(), key=lambda item: item[1])[0] if scores else ""
    return {"pred_label": pred_label, "scores": scores}


def compare_label_logits(
    manual_logits: torch.Tensor,
    generate_logits: torch.Tensor,
    label_token_ids: Dict[str, int],
) -> Dict[str, Any]:
    manual = pick_label_from_logits(manual_logits, label_token_ids)
    generated = pick_label_from_logits(generate_logits, label_token_ids)
    diffs = {
        label: round(abs(manual["scores"][label] - generated["scores"][label]), 6)
        for label in label_token_ids
    }
    return {
        "manual_pred_label": manual["pred_label"],
        "generate_score_pred_label": generated["pred_label"],
        "pred_match": manual["pred_label"] == generated["pred_label"],
        "manual_scores": manual["scores"],
        "generate_scores": generated["scores"],
        "label_score_abs_diff": diffs,
        "max_label_score_abs_diff": max(diffs.values()) if diffs else 0.0,
    }


def _move_image_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = tensor.to(device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor


def _prepare_prompt(tokenizer, prompt_text: str, config: ml_collections.ConfigDict, device: torch.device):
    from Dataset.conversation import SeparatorStyle, default_conversation
    from Inference import _normalize_user_instruction
    from Models import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        tokenizer_image_token,
    )
    from Models.utils import KeywordsStoppingCriteria
    from Tools.run_geojson_batch_eval import _ensure_image_token

    user_prompt = _normalize_user_instruction(prompt_text)
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
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    return prompt, input_ids, stopping_criteria


def _embed_routes(model, route_inputs: Dict[str, Any]) -> Dict[str, Any]:
    physical_prompt_embs = model._embed_semantic_text_ids(
        route_inputs.get("physical_prompt_ids"),
        attention_mask=route_inputs.get("physical_prompt_attention_mask"),
    )
    task_text_embs = model._embed_semantic_text_ids(
        route_inputs.get("task_text_ids"),
        attention_mask=route_inputs.get("task_text_attention_mask"),
    )
    element_text_embs = model._embed_semantic_text_ids(
        route_inputs.get("element_text_ids"),
        attention_mask=route_inputs.get("element_text_attention_mask"),
    )
    return {
        "physical_prompt_embs": physical_prompt_embs,
        "task_text_embs": task_text_embs,
        "element_text_embs": element_text_embs,
    }


def _manual_generation_prompt_logits(
    *,
    model,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    route_inputs: Dict[str, Any],
    route_embs: Dict[str, Any],
    multiband_tensor,
    valid_multiband,
):
    images = image_tensor
    if multiband_tensor is not None:
        wavelet_data = {"rgb": image_tensor, "multiband": multiband_tensor}
        if valid_multiband is not None:
            wavelet_data["valid_multiband"] = valid_multiband
        model._maybe_apply_wavelet_adapter(wavelet_data, {})
        images = wavelet_data.get("rgb", image_tensor)

    image_embedding = model.encode_image(
        images,
        pool=False,
        physical_prompt_embs=route_embs.get("physical_prompt_embs"),
        task_text_embs=route_embs.get("task_text_embs"),
        element_text_embs=route_embs.get("element_text_embs"),
        physical_prompt_attention_mask=route_inputs.get("physical_prompt_attention_mask"),
        task_text_attention_mask=route_inputs.get("task_text_attention_mask"),
        element_text_attention_mask=route_inputs.get("element_text_attention_mask"),
    )
    _, attention_mask, past_key_values, inputs_embeds, _ = model.language.prepare_inputs_for_multimodal(
        input_ids=input_ids,
        attention_mask=None,
        labels=None,
        image_embedding=image_embedding,
        past_key_values=None,
    )
    outputs = model.language.get_text_encoder()(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=False,
        return_dict=True,
    )
    return outputs.logits[:, -1, :]


def _score_one(
    *,
    config: ml_collections.ConfigDict,
    model,
    tokenizer,
    vision_processor,
    device: torch.device,
    dtype: torch.dtype,
    entry: Dict[str, Any],
    label_token_ids: Dict[str, int],
) -> Dict[str, Any]:
    from Inference import _build_generation_kwargs, _build_image_tensor
    from Tools.run_geojson_batch_eval import (
        _build_multiband_tensor_for_generation,
        _build_semantic_route_inputs,
    )

    image_path = Path(entry["image_path"])
    image_tensor = _build_image_tensor(config, vision_processor, str(image_path), device, dtype)
    image_tensor = _move_image_tensor(image_tensor, device, dtype)
    multiband_tensor, valid_multiband = _build_multiband_tensor_for_generation(
        config=config,
        image_path=image_path,
        image_tensor=image_tensor,
        device=device,
        dtype=dtype,
    )
    prompt, input_ids, stopping_criteria = _prepare_prompt(
        tokenizer,
        str(entry["question"]),
        config,
        device,
    )
    route_inputs = _build_semantic_route_inputs(
        tokenizer,
        str(entry["question"]),
        device,
        physical_prompt_text=str(entry.get("physical_prompt", "")),
    )
    route_embs = _embed_routes(model, route_inputs)
    gen_kwargs = _build_generation_kwargs(config, tokenizer, stopping_criteria)
    gen_kwargs.update(
        {
            "max_new_tokens": 1,
            "min_new_tokens": 0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
    )

    generate_inputs = {
        "input_ids": input_ids,
        "images": image_tensor,
        **gen_kwargs,
        **route_inputs,
    }
    if multiband_tensor is not None:
        generate_inputs["multiband"] = multiband_tensor
        generate_inputs["valid_multiband"] = valid_multiband

    with torch.inference_mode():
        manual_logits = _manual_generation_prompt_logits(
            model=model,
            input_ids=input_ids,
            image_tensor=image_tensor,
            route_inputs=route_inputs,
            route_embs=route_embs,
            multiband_tensor=multiband_tensor,
            valid_multiband=valid_multiband,
        )
        generated = model.generate(**generate_inputs)

    if not getattr(generated, "scores", None):
        raise RuntimeError("model.generate did not return output scores")
    generate_logits = generated.scores[0]
    comparison = compare_label_logits(manual_logits, generate_logits, label_token_ids)
    generated_token_id = int(generated.sequences[0, -1].detach().cpu().item())
    comparison.update(
        {
            "idx": int(entry["idx"]),
            "image_path": str(image_path),
            "question": str(entry["question"]),
            "answer": str(entry["answer"]),
            "answer_label": str(entry["answer_label"]),
            "physical_prompt": str(entry.get("physical_prompt", "")),
            "prompt_tail": prompt[-160:],
            "generated_token_id": generated_token_id,
            "generated_token_text": tokenizer.decode([generated_token_id]),
        }
    )
    return comparison


def _run_probe(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    from Dataset.build_transform import build_vlp_transform
    from Dataset.cap_dataset import InstructDatasetWithTaskId
    from Dataset.conversation import conv_templates
    from Tools.run_geojson_batch_eval import _load_inference_bundle

    dataset_stage = resolve_dataset_stage(config)
    model, tokenizer, eval_transform, device, dtype = _load_inference_bundle(config)
    dataset = InstructDatasetWithTaskId(
        root=config.data_path,
        transform=build_vlp_transform(config, is_train=True),
        tokenizer=tokenizer,
        crop_size=int(getattr(config, "crop_size", 224)),
        stage=dataset_stage,
        prompt_type=str(getattr(config, "prompt_template", "llava_llama_2")),
        tune_im_start=bool(getattr(config, "tune_im_start", False)),
    )
    entries = _build_dataset_entries(dataset, int(getattr(config, "limit", 0)))
    label_token_ids = _label_token_ids(tokenizer, LABEL_CHOICES)

    from Dataset import conversation as conversation_lib

    conversation_lib.default_conversation = conv_templates[str(getattr(config, "prompt_template", "llava_llama_2"))]

    manual_pairs: List[Tuple[str, str]] = []
    generate_score_pairs: List[Tuple[str, str]] = []
    match_count = 0
    max_diffs = []
    samples = []
    errors = []
    sample_dump = int(getattr(config, "sample_dump", 12))
    for entry in entries:
        try:
            result = _score_one(
                config=config,
                model=model,
                tokenizer=tokenizer,
                vision_processor=eval_transform,
                device=device,
                dtype=dtype,
                entry=entry,
                label_token_ids=label_token_ids,
            )
        except Exception as exc:
            errors.append({"idx": int(entry["idx"]), "error": repr(exc)})
            continue
        answer_label = normalize_label(entry["answer"])
        manual_pairs.append((answer_label, str(result["manual_pred_label"])))
        generate_score_pairs.append((answer_label, str(result["generate_score_pred_label"])))
        if bool(result["pred_match"]):
            match_count += 1
        max_diffs.append(float(result["max_label_score_abs_diff"]))
        if len(samples) < sample_dump:
            samples.append(result)
        if str(device).startswith("npu"):
            try:
                import torch_npu  # noqa: F401

                torch.npu.empty_cache()
            except Exception:
                pass

    total_compared = len(manual_pairs)
    return {
        "data_path": str(config.data_path),
        "model_path": str(config.model_path),
        "limit": len(entries),
        "label_token_ids": label_token_ids,
        "manual_generation_prompt_first_token": summarize_label_pairs(manual_pairs),
        "hf_generate_score_first_token": summarize_label_pairs(generate_score_pairs),
        "manual_vs_generate_pred_match_rate_pct": (match_count / total_compared * 100.0) if total_compared else 0.0,
        "max_label_score_abs_diff_max": max(max_diffs) if max_diffs else 0.0,
        "max_label_score_abs_diff_mean": (sum(max_diffs) / len(max_diffs)) if max_diffs else 0.0,
        "errors": errors,
        "samples": samples,
    }


def _parse_args() -> ml_collections.ConfigDict:
    from Trainer.utils import ConfigArgumentParser, str2bool

    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-dump", type=int, default=12)
    parser.add_argument("--accelerator", type=str, default=None, choices=["cpu", "npu", "gpu", "mps"])
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--bf16", type=str2bool, default=None)
    parser.add_argument("--fp16", type=str2bool, default=None)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--diag-on-unk", type=str2bool, default=False)
    raw = parser.parse_args(wandb=True)
    config = ml_collections.ConfigDict(raw)
    if not getattr(config, "data_path", None):
        raise ValueError("--data-path is required")
    if not getattr(config, "model_path", None):
        raise ValueError("--model-path is required")
    if not getattr(config, "output", None):
        raise ValueError("--output is required")
    return config


def main() -> None:
    config = _parse_args()
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run_probe(config)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "limit",
                    "manual_generation_prompt_first_token",
                    "hf_generate_score_first_token",
                    "manual_vs_generate_pred_match_rate_pct",
                    "max_label_score_abs_diff_max",
                    "max_label_score_abs_diff_mean",
                    "errors",
                )
            },
            ensure_ascii=False,
        )
    )
    print(f"[probe] wrote {output}")


if __name__ == "__main__":
    main()

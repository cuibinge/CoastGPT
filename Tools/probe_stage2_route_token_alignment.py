#!/usr/bin/env python
"""Compare Stage2 first-token logits under eval-style vs train-style route tokens.

This is a read-only diagnostic tool for the Stage2 LR overfit failure mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import ml_collections
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.probe_stage2_generate_scores import (  # noqa: E402
    LABEL_CHOICES,
    _build_dataset_entries,
    _embed_routes,
    _label_token_ids,
    _manual_generation_prompt_logits,
    _move_image_tensor,
    _prepare_prompt,
    normalize_label,
    pick_label_from_logits,
    resolve_dataset_stage,
    summarize_label_pairs,
)


def _tokenize_train_style(
    tokenizer,
    text: str,
    *,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [str(text or "")],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return tokens.input_ids.to(device), tokens.attention_mask.to(device)


def _build_train_style_route_inputs(
    tokenizer,
    entry: Dict[str, Any],
    config: ml_collections.ConfigDict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    physical_ids, physical_mask = _tokenize_train_style(
        tokenizer,
        str(entry.get("physical_prompt", "")),
        max_len=int(getattr(config, "physical_prompt_max_len", 64)),
        device=device,
    )
    task_ids, task_mask = _tokenize_train_style(
        tokenizer,
        str(entry.get("task_text", "描述")),
        max_len=int(getattr(config, "task_text_max_len", 16)),
        device=device,
    )
    element_ids, element_mask = _tokenize_train_style(
        tokenizer,
        str(entry.get("element_text", "无")),
        max_len=int(getattr(config, "element_text_max_len", 16)),
        device=device,
    )
    return {
        "physical_prompt_ids": physical_ids,
        "physical_prompt_attention_mask": physical_mask,
        "task_text_ids": task_ids,
        "task_text_attention_mask": task_mask,
        "element_text_ids": element_ids,
        "element_text_attention_mask": element_mask,
    }


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
    from Inference import _build_image_tensor
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
    _, input_ids, _ = _prepare_prompt(
        tokenizer,
        str(entry["question"]),
        config,
        device,
    )
    eval_route_inputs = _build_semantic_route_inputs(
        tokenizer,
        str(entry["question"]),
        device,
        physical_prompt_text=str(entry.get("physical_prompt", "")),
    )
    train_route_inputs = _build_train_style_route_inputs(tokenizer, entry, config, device)

    with torch.inference_mode():
        eval_route_logits = _manual_generation_prompt_logits(
            model=model,
            input_ids=input_ids,
            image_tensor=image_tensor,
            route_inputs=eval_route_inputs,
            route_embs=_embed_routes(model, eval_route_inputs),
            multiband_tensor=multiband_tensor,
            valid_multiband=valid_multiband,
        )
        train_route_logits = _manual_generation_prompt_logits(
            model=model,
            input_ids=input_ids,
            image_tensor=image_tensor,
            route_inputs=train_route_inputs,
            route_embs=_embed_routes(model, train_route_inputs),
            multiband_tensor=multiband_tensor,
            valid_multiband=valid_multiband,
        )

    eval_pick = pick_label_from_logits(eval_route_logits, label_token_ids)
    train_pick = pick_label_from_logits(train_route_logits, label_token_ids)
    return {
        "idx": int(entry["idx"]),
        "image_path": str(image_path),
        "answer": str(entry["answer"]),
        "answer_label": normalize_label(str(entry["answer"])),
        "question": str(entry["question"]),
        "physical_prompt": str(entry.get("physical_prompt", "")),
        "task_text": str(entry.get("task_text", "")),
        "element_text": str(entry.get("element_text", "")),
        "eval_route_pred_label": eval_pick["pred_label"],
        "train_route_pred_label": train_pick["pred_label"],
        "eval_route_scores": eval_pick["scores"],
        "train_route_scores": train_pick["scores"],
        "pred_changed": eval_pick["pred_label"] != train_pick["pred_label"],
    }


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
    indices = getattr(config, "indices", None)
    if indices:
        wanted = {int(x) for x in str(indices).replace(";", ",").split(",") if x.strip()}
        entries = [entry for entry in entries if int(entry["idx"]) in wanted]
    label_token_ids = _label_token_ids(tokenizer, LABEL_CHOICES)

    from Dataset import conversation as conversation_lib

    conversation_lib.default_conversation = conv_templates[str(getattr(config, "prompt_template", "llava_llama_2"))]

    eval_pairs: List[Tuple[str, str]] = []
    train_pairs: List[Tuple[str, str]] = []
    samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    changed: List[Dict[str, Any]] = []
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
        answer_label = str(result["answer_label"])
        eval_pairs.append((answer_label, str(result["eval_route_pred_label"])))
        train_pairs.append((answer_label, str(result["train_route_pred_label"])))
        keep_sample = len(samples) < int(getattr(config, "sample_dump", 16))
        if bool(result["pred_changed"]):
            changed.append(result)
            keep_sample = True
        if keep_sample:
            samples.append(result)
        if str(device).startswith("npu"):
            try:
                import torch_npu  # noqa: F401

                torch.npu.empty_cache()
            except Exception:
                pass

    return {
        "data_path": str(config.data_path),
        "model_path": str(config.model_path),
        "limit": len(entries),
        "label_token_ids": label_token_ids,
        "eval_style_route": summarize_label_pairs(eval_pairs),
        "train_style_route": summarize_label_pairs(train_pairs),
        "changed_count": len(changed),
        "errors": errors,
        "changed": changed,
        "samples": samples,
    }


def _parse_args() -> ml_collections.ConfigDict:
    from Trainer.utils import ConfigArgumentParser, str2bool

    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--sample-dump", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default=None, choices=["cpu", "npu", "gpu", "mps"])
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--bf16", type=str2bool, default=None)
    parser.add_argument("--fp16", type=str2bool, default=None)
    parser.add_argument("--force-safe-npu", type=str2bool, default=True)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--min-new-tokens", type=int, default=0)
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
    print(json.dumps({
        "limit": result["limit"],
        "eval_style_route": result["eval_style_route"],
        "train_style_route": result["train_style_route"],
        "changed_count": result["changed_count"],
        "errors": result["errors"],
    }, ensure_ascii=False))
    print(f"[route-token-probe] wrote {output}")


if __name__ == "__main__":
    main()

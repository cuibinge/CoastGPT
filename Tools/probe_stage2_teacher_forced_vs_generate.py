#!/usr/bin/env python
"""Compare Stage2 generation collapse against teacher-forced label logits.

This is a diagnostic tool. It intentionally does not change model behavior.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ml_collections
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IGNORE_INDEX = -100
LABEL_CHOICES = ("rural", "urban")


def first_label_index(labels: torch.Tensor) -> Optional[int]:
    """Return the first supervised token index in a single-sample label tensor."""
    if labels.ndim == 2:
        if labels.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got labels shape {tuple(labels.shape)}")
        labels = labels[0]
    valid = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False)
    if valid.numel() == 0:
        return None
    return int(valid[0].item())


def _pct(value: float) -> float:
    return round(float(value) * 100.0, 6)


def summarize_label_pairs(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """Summarize (answer_label, predicted_label) pairs."""
    total = len(pairs)
    correct = sum(1 for answer, pred in pairs if answer == pred)
    by_answer: Dict[str, Counter] = defaultdict(Counter)
    for answer, pred in pairs:
        by_answer[str(answer)][str(pred)] += 1

    recalls: Dict[str, float] = {}
    for answer in sorted(by_answer):
        denom = sum(by_answer[answer].values())
        recalls[answer] = _pct(by_answer[answer].get(answer, 0) / denom) if denom else 0.0

    balanced = sum(recalls.values()) / len(recalls) if recalls else 0.0
    return {
        "total": total,
        "label_exact_rate_pct": _pct(correct / total) if total else 0.0,
        "label_balanced_accuracy_pct": round(float(balanced), 6),
        "label_recall_by_answer": recalls,
        "label_confusion": {
            answer: dict(sorted(pred_counts.items()))
            for answer, pred_counts in sorted(by_answer.items())
        },
    }


def normalize_label(text: str) -> str:
    text = str(text or "").lower().strip()
    if "urban" in text:
        return "urban"
    if "rural" in text:
        return "rural"
    return text.split()[0] if text.split() else ""


def resolve_dataset_stage(config: ml_collections.ConfigDict) -> int:
    """Capture the training dataset stage before inference load mutates config."""
    try:
        return int(getattr(config, "stage", 2))
    except Exception:
        return 2


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def _embed_text_field(model, batch: Dict[str, Any], field: str) -> Optional[torch.Tensor]:
    ids_key = f"{field}_ids"
    mask_key = f"{field}_attention_mask"
    out_key = f"{field}_embs"
    ids = batch.get(ids_key)
    if ids is None or not torch.is_tensor(ids):
        return None
    emb_layer = model.language.get_text_encoder().get_input_embeddings()
    embs = emb_layer(ids)
    mask = batch.get(mask_key)
    if mask is not None and torch.is_tensor(mask):
        embs = embs * mask.to(embs.device).unsqueeze(-1).to(embs.dtype)
    batch[out_key] = embs
    return embs


def _build_dataset_entries(dataset, limit: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    n = len(dataset) if limit <= 0 else min(limit, len(dataset))
    for idx in range(n):
        conv = dataset.cap_list[idx]
        if isinstance(conv, list):
            turn = conv[0]
        else:
            turn = conv
        question = str(turn.get("Question", turn.get("question", ""))).strip()
        answer = str(turn.get("Answer", turn.get("answer", ""))).strip()
        entries.append(
            {
                "idx": idx,
                "image_path": str(dataset.img_list[idx]),
                "question": question,
                "answer": answer,
                "answer_label": normalize_label(answer),
                "task_text": dataset.task_texts[idx] if idx < len(dataset.task_texts) else "",
                "element_text": dataset.element_texts[idx] if idx < len(dataset.element_texts) else "",
                "physical_prompt": dataset._build_physical_prompt(
                    dataset.sample_phys_meta[idx] if idx < len(dataset.sample_phys_meta) else None
                ),
            }
        )
    return entries


def _label_token_ids(tokenizer, labels: Sequence[str]) -> Dict[str, int]:
    token_ids: Dict[str, int] = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"label {label!r} produced no token ids")
        token_ids[label] = int(ids[0])
    return token_ids


def _teacher_forced_one(
    *,
    model,
    tokenizer,
    dataset,
    collator,
    idx: int,
    device: torch.device,
    dtype: torch.dtype,
    label_token_ids: Dict[str, int],
    projection_mode: str = "train",
) -> Dict[str, Any]:
    if projection_mode not in {"train", "generate"}:
        raise ValueError(f"unsupported projection_mode={projection_mode!r}")
    instance = dataset[idx]
    batch = collator([instance])
    batch = _move_batch_to_device(batch, device, dtype)

    with torch.inference_mode():
        physical_prompt_embs = _embed_text_field(model, batch, "physical_prompt")
        _embed_text_field(model, batch, "task_text")
        _embed_text_field(model, batch, "element_text")

        model._maybe_apply_wavelet_adapter(batch, {})
        if model.vision.__class__.__name__ == "DualVisionEncoder":
            image_seq, _, _ = model.vision.encode_with_spatial(
                batch["rgb"],
                physical_prompt_embs=physical_prompt_embs,
            )
        else:
            image_seq = model.vision(batch)

        if projection_mode == "train":
            multimodal_embedding = model.multimodal(batch, image_embedding=image_seq)
        else:
            multimodal_embedding = model.multimodal.encode_test(
                image_seq,
                physical_prompts=physical_prompt_embs,
                task_text_embs=batch.get("task_text_embs"),
                element_text_embs=batch.get("element_text_embs"),
                physical_prompt_mask=batch.get("physical_prompt_attention_mask"),
                task_text_mask=batch.get("task_text_attention_mask"),
                element_text_mask=batch.get("element_text_attention_mask"),
            )
        _, attention_mask, past_key_values, inputs_embeds, labels = model.language.prepare_inputs_for_multimodal(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
            image_embedding=multimodal_embedding,
            past_key_values=None,
        )
        if inputs_embeds is None or labels is None:
            raise RuntimeError("prepare_inputs_for_multimodal did not return inputs_embeds/labels")
        target_pos = first_label_index(labels)
        if target_pos is None:
            return {"idx": idx, "error": "all_labels_ignored"}
        if target_pos <= 0:
            return {"idx": idx, "error": f"first label position is {target_pos}"}

        outputs = model.language.get_text_encoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[0, target_pos - 1].float()
        scores = {label: float(logits[token_id].detach().cpu().item()) for label, token_id in label_token_ids.items()}
        pred_label = max(scores.items(), key=lambda item: item[1])[0]
        top_token_id = int(torch.argmax(logits).detach().cpu().item())
        target_token_id = int(labels[0, target_pos].detach().cpu().item())

    return {
        "idx": idx,
        "teacher_pred_label": pred_label,
        "teacher_label_scores": scores,
        "target_pos": int(target_pos),
        "target_token_id": target_token_id,
        "target_token_text": tokenizer.decode([target_token_id]),
        "top_token_id": top_token_id,
        "top_token_text": tokenizer.decode([top_token_id]),
    }


def _run_probe(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    from Dataset.build_transform import build_vlp_transform
    from Dataset.cap_dataset import DataCollatorForSupervisedDataset, InstructDatasetWithTaskId
    from Dataset.conversation import conv_templates
    from Tools.run_geojson_batch_eval import _generate_single_prediction, _load_inference_bundle
    from Tools.run_stage2_batch_eval import _extract_label_prediction

    dataset_stage = resolve_dataset_stage(config)
    model, tokenizer, eval_transform, device, dtype = _load_inference_bundle(config)

    train_transform = build_vlp_transform(config, is_train=True)
    dataset = InstructDatasetWithTaskId(
        root=config.data_path,
        transform=train_transform,
        tokenizer=tokenizer,
        crop_size=int(getattr(config, "crop_size", 224)),
        stage=dataset_stage,
        prompt_type=str(getattr(config, "prompt_template", "llava_llama_2")),
        tune_im_start=bool(getattr(config, "tune_im_start", False)),
    )
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
        task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
        element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
    )
    entries = _build_dataset_entries(dataset, int(getattr(config, "limit", 0)))
    label_token_ids = _label_token_ids(tokenizer, LABEL_CHOICES)

    generation_pairs: List[Tuple[str, str]] = []
    teacher_pairs: List[Tuple[str, str]] = []
    teacher_generate_embedding_pairs: List[Tuple[str, str]] = []
    samples: List[Dict[str, Any]] = []
    teacher_errors: List[Dict[str, Any]] = []
    teacher_generate_embedding_errors: List[Dict[str, Any]] = []

    # Ensure generation uses the configured prompt template after dataset construction.
    from Dataset import conversation as conversation_lib

    conversation_lib.default_conversation = conv_templates[str(getattr(config, "prompt_template", "llava_llama_2"))]

    for entry in entries:
        idx = int(entry["idx"])
        gen = _generate_single_prediction(
            config=config,
            model=model,
            tokenizer=tokenizer,
            vision_processor=eval_transform,
            device=device,
            dtype=dtype,
            image_path=Path(entry["image_path"]),
            prompt_text=entry["question"],
            physical_prompt_text=entry.get("physical_prompt", ""),
        )
        gen_label = _extract_label_prediction(gen.get("prediction", ""), LABEL_CHOICES)
        answer_label = entry["answer_label"]
        generation_pairs.append((answer_label, gen_label))

        teacher = _teacher_forced_one(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            collator=collator,
            idx=idx,
            device=device,
            dtype=dtype,
            label_token_ids=label_token_ids,
            projection_mode="train",
        )
        if "error" in teacher:
            teacher_errors.append(teacher)
            teacher_label = ""
        else:
            teacher_label = str(teacher["teacher_pred_label"])
            teacher_pairs.append((answer_label, teacher_label))

        teacher_generate_embedding = _teacher_forced_one(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            collator=collator,
            idx=idx,
            device=device,
            dtype=dtype,
            label_token_ids=label_token_ids,
            projection_mode="generate",
        )
        if "error" in teacher_generate_embedding:
            teacher_generate_embedding_errors.append(teacher_generate_embedding)
            teacher_generate_embedding_label = ""
        else:
            teacher_generate_embedding_label = str(teacher_generate_embedding["teacher_pred_label"])
            teacher_generate_embedding_pairs.append((answer_label, teacher_generate_embedding_label))

        if len(samples) < int(getattr(config, "sample_dump", 12)):
            samples.append(
                {
                    **entry,
                    "generation_prediction": gen.get("prediction", ""),
                    "generation_label": gen_label,
                    "teacher_label": teacher_label,
                    "teacher": teacher,
                    "teacher_generate_embedding_label": teacher_generate_embedding_label,
                    "teacher_generate_embedding": teacher_generate_embedding,
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        "generation": summarize_label_pairs(generation_pairs),
        "teacher_forced_first_token": summarize_label_pairs(teacher_pairs),
        "teacher_forced_generate_embedding_first_token": summarize_label_pairs(
            teacher_generate_embedding_pairs
        ),
        "teacher_errors": teacher_errors,
        "teacher_generate_embedding_errors": teacher_generate_embedding_errors,
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
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
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
                    "generation",
                    "teacher_forced_first_token",
                    "teacher_forced_generate_embedding_first_token",
                )
            },
            ensure_ascii=False,
        )
    )
    print(f"[probe] wrote {output}")


if __name__ == "__main__":
    main()

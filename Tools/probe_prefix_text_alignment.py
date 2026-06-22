#!/usr/bin/env python
"""Compare Stage2 training-answer prefix with inference-generation prompt.

This is a read-only diagnostic script. It does not load model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IGNORE_INDEX = -100


def _tail(values: List[int], n: int = 16) -> List[int]:
    return values[-n:] if len(values) > n else values


def _decode_ids(tokenizer, ids: List[int]) -> List[str]:
    out = []
    for token_id in ids:
        if int(token_id) < 0:
            out.append(f"<image:{token_id}>")
        else:
            out.append(tokenizer.decode([int(token_id)]))
    return out


def _route_token_compare(tokenizer, text: str, max_len: int) -> Dict[str, Any]:
    train_tokens = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )
    eval_tokens = tokenizer(
        [text],
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    train_ids = list(train_tokens["input_ids"][0])
    eval_ids = list(eval_tokens["input_ids"][0])
    return {
        "text": text,
        "max_len": max_len,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "train_attention_mask": list(train_tokens["attention_mask"][0]),
        "eval_attention_mask": list(eval_tokens["attention_mask"][0]),
        "train_tokens": _decode_ids(tokenizer, train_ids),
        "eval_tokens": _decode_ids(tokenizer, eval_ids),
        "ids_equal": train_ids == eval_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--indices", default="0,1,17,41,44")
    parser.add_argument("--prompt-template", default="llava_llama_2")
    parser.add_argument("--tokenizer-path", default="meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()

    from Dataset import conversation as conversation_lib
    from Dataset.cap_dataset import preprocess, preprocess_multimodal
    from Dataset.conversation import conv_templates
    from Models import IMAGE_TOKEN_INDEX, tokenizer_image_token
    from Tools.probe_stage2_teacher_forced_vs_generate import first_label_index
    from Tools.run_geojson_batch_eval import _ensure_image_token
    from Inference import _normalize_user_instruction
    from Models import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    conversation_lib.default_conversation = conv_templates[args.prompt_template]

    data_path = Path(args.data_path)
    json_path = data_path / "LR.json"
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    if args.limit > 0:
        items = items[: args.limit]

    requested = [int(x) for x in str(args.indices).replace(";", ",").split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []

    for idx in requested:
        item = items[idx]
        conv_data = item.get("conv", item)
        if isinstance(conv_data, list):
            turn = conv_data[0]
        else:
            turn = conv_data
        question = str(turn.get("Question", turn.get("question", ""))).strip()
        if DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + question
        answer = str(turn.get("Answer", turn.get("answer", ""))).strip()

        training_source = [{"Question": question, "Answer": answer}]
        training_source = preprocess_multimodal(training_source, tune_im_start=False)
        train_encoded = preprocess(training_source, tokenizer, has_image=True)
        train_input_ids = train_encoded["input_ids"][0].tolist()
        train_labels = train_encoded["labels"][0]
        first_idx = first_label_index(train_labels)
        if first_idx is None:
            raise RuntimeError(f"idx={idx} has no supervised label")
        train_prefix_ids = train_input_ids[:first_idx]
        target_id = int(train_labels[first_idx].item())

        user_prompt = _normalize_user_instruction(question)
        user_prompt = _ensure_image_token(
            user_prompt,
            tune_im_start=False,
            default_image_token=DEFAULT_IMAGE_TOKEN,
            default_im_start_token=DEFAULT_IM_START_TOKEN,
            default_im_end_token=DEFAULT_IM_END_TOKEN,
        )
        gen_conv = conversation_lib.default_conversation.copy()
        gen_conv.append_message(gen_conv.roles[0], user_prompt)
        gen_conv.append_message(gen_conv.roles[1], None)
        gen_prompt = gen_conv.get_prompt()
        gen_ids = tokenizer_image_token(
            gen_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors=None,
        )
        gen_space_ids = tokenizer_image_token(
            gen_prompt + " ",
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors=None,
        )

        rows.append(
            {
                "idx": idx,
                "image": item.get("name") or item.get("filename") or item.get("image"),
                "question": question,
                "answer": answer,
                "target_id": target_id,
                "target_text": tokenizer.decode([target_id]),
                "train_first_label_raw_idx": int(first_idx),
                "train_prefix_len": len(train_prefix_ids),
                "gen_prompt_len": len(gen_ids),
                "gen_prompt_plus_space_len": len(gen_space_ids),
                "train_equals_gen": train_prefix_ids == gen_ids,
                "train_equals_gen_plus_space": train_prefix_ids == gen_space_ids,
                "gen_plus_space_extra_ids": gen_space_ids[len(gen_ids):],
                "gen_plus_space_extra_tokens": _decode_ids(tokenizer, gen_space_ids[len(gen_ids):]),
                "train_prefix_tail_ids": _tail(train_prefix_ids),
                "gen_prompt_tail_ids": _tail(gen_ids),
                "gen_prompt_plus_space_tail_ids": _tail(gen_space_ids),
                "train_prefix_tail_tokens": _decode_ids(tokenizer, _tail(train_prefix_ids)),
                "gen_prompt_tail_tokens": _decode_ids(tokenizer, _tail(gen_ids)),
                "gen_prompt_plus_space_tail_tokens": _decode_ids(tokenizer, _tail(gen_space_ids)),
                "gen_prompt_tail_repr": repr(gen_prompt[-160:]),
                "gen_prompt_plus_space_tail_repr": repr((gen_prompt + " ")[-160:]),
            }
        )

    summary = {
        "data_path": str(data_path),
        "prompt_template": args.prompt_template,
        "indices": requested,
        "all_train_equals_gen": all(r["train_equals_gen"] for r in rows),
        "all_train_equals_gen_plus_space": all(r["train_equals_gen_plus_space"] for r in rows),
        "route_tokenization": {
            "physical_prompt": _route_token_compare(tokenizer, "[Dataset: LR]", 64),
            "task_text": _route_token_compare(tokenizer, "场景分类", 16),
            "element_text": _route_token_compare(tokenizer, "土地覆盖", 16),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("all_train_equals_gen", "all_train_equals_gen_plus_space", "indices")}, ensure_ascii=False))
    print(f"[prefix-text-probe] wrote {output}")


if __name__ == "__main__":
    main()

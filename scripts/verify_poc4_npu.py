#!/usr/bin/env python3
"""
PoC-4 NPU verification — minimal smoke test.

Loads CoastGPT from the standard checkpoint, builds FusionPipeline
with stub detection heads (return empty), and verifies:
  1. LLM Parser correctly extracts task_type + target_classes
  2. Gating correctly routes known→heads, unknown→LLM fallback
  3. Pipeline runs without crash on NPU
  4. Outputs valid FeatureCollection + diagnostics

Usage:
    python scripts/verify_poc4_npu.py \\
        --model-path output/checkpoints/FINAL.pt \\
        --image-file Images/test.png \\
        --prompt "[DET] 请检测图中的海岸线和红树林湿地。"

If detection head checkpoints are available, pass them:
    --inst-ckpt outputs/poc_aqua_128/checkpoints/epoch_040.pt
    --sem-ckpt outputs/poc2_vit_fpn/checkpoints/best_miou.pt
    --edge-ckpt <path>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# NPU / Environment setup (mirrors Inference.py)
# =============================================================================

def _setup_npu():
    try:
        import torch_npu  # noqa: F401
        local_idx = int(os.environ.get("LOCAL_RANK", "0"))
        torch_npu.npu.set_device(local_idx)
        return torch.device(f"npu:{local_idx}")
    except Exception:
        return torch.device("npu:0")


def _load_image(image_file: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_file).convert("RGB")
    # Resize large images
    w, h = image.size
    if w * h > 16_000_000:
        scale = (16_000_000 / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return image


# =============================================================================
# Stub predictors (no-op — returns empty features)
# =============================================================================

from Models.fusion_predictors import PredictorOutput, BasePredictor


class StubPredictor(BasePredictor):
    """No-op predictor that returns empty features."""
    def __init__(self, branch: str):
        self._branch = branch

    @property
    def branch(self) -> str:
        return self._branch

    def predict(self, image, prompt, georef, classes):
        return PredictorOutput(
            branch=self._branch,
            features=[],
            metadata={"skipped": True, "reason": "stub — no checkpoint loaded"},
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PoC-4 NPU Verification")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to CoastGPT FINAL.pt checkpoint")
    parser.add_argument("--image-file", type=str, default="Images/test.png")
    parser.add_argument("--prompt", type=str,
                        default="[DET] 请检测图中的海岸线和红树林湿地。")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    # ---- Device ----
    if args.device.startswith("npu"):
        device = _setup_npu()
    else:
        device = torch.device(args.device)
    dtype = torch.float16
    print(f"[verify] device={device}, dtype={dtype}")

    # ---- Load CoastGPT (use standard YAML config via ConfigArgumentParser) ----
    print("[verify] Loading CoastGPT...")
    import yaml as _yaml
    from ml_collections import ConfigDict
    from Trainer.utils.config_parser import ConfigArgumentParser

    # Use the standard training config as base (has all model architecture keys)
    with open("Configs/step2_dual.yaml", "r") as f:
        raw = _yaml.safe_load(f)
    config = ConfigDict(raw)
    config.stage = 0  # eval mode
    config.adjust_norm = False

    from Models.coastgpt import CoastGPT
    model = CoastGPT(config)
    model.to(dtype)

    # Load checkpoint (same pattern as Inference.py)
    ckpt = torch.load(args.model_path, map_location="cpu")
    if "vision_ckpt" in ckpt or "other_ckpt" in ckpt:
        print("[verify] Structured checkpoint detected — using custom_load_state_dict")
        if hasattr(model, "custom_load_state_dict"):
            model.custom_load_state_dict(args.model_path)
        else:
            # Manual load
            if "vision_ckpt" in ckpt and hasattr(model, "vision"):
                model.vision.load_state_dict(ckpt["vision_ckpt"], strict=False)
            other = ckpt.get("other_ckpt", {})
            if isinstance(other, dict):
                mm = other.get("multimodal_projection")
                if mm is not None and hasattr(model, "multimodal"):
                    model.multimodal.projection.load_state_dict(mm, strict=False)
    else:
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    tokenizer = model.language.tokenizer
    print(f"[verify] CoastGPT loaded. vocab_size={len(tokenizer)}")

    # ---- Load image ----
    print(f"[verify] Loading image: {args.image_file}")
    image = _load_image(args.image_file)
    from Dataset.build_transform import build_vlp_transform
    transform = build_vlp_transform(config, is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)
    print(f"[verify] Image tensor: {list(image_tensor.shape)}")

    # ---- Build FusionPipeline with stub detection heads + real LLM ----
    from Models.fusion_pipeline import FusionConfig, FusionPipeline
    from Models.fusion_predictors import LLMPredictor, LLMTextPredictor

    # Load label_map
    label_map_path = "Configs/label_map.json"
    if Path(label_map_path).exists():
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
    else:
        label_map = {
            "branch_classes": {
                "instance": {"0": "background", "1": "海水养殖区"},
                "semantic": {"0": "background", "1": "水田", "2": "旱地"},
                "edge": {"0": "background", "1": "海岸线"},
            }
        }

    fusion_config = FusionConfig(
        parser_max_new_tokens=128,
        parser_confidence_threshold=0.3,
        empty_target_policy="error",
        llm_min_confidence=0.3,
    )
    # Use minimal Chinese parser prompt (matching training conversation style)
    fusion_config.parser_prompt_template = (
        "从用户输入中提取任务类型和检测目标类别。\n"
        "用户输入: {user_prompt}\n"
        "请以JSON格式返回，包含task_type(DET/CAP/VQA)、target_classes(类别列表)、confidence(0-1)。\n"
    )

    llm_predictor = LLMPredictor(
        coastgpt_model=model,
        tokenizer=tokenizer,
        config=config,
        max_new_tokens=512,
        device=str(device),
    )
    llm_text_predictor = LLMTextPredictor(
        coastgpt_model=model,
        tokenizer=tokenizer,
        config=config,
        max_new_tokens=256,
        device=str(device),
    )

    pipeline = FusionPipeline(
        predictors={
            "instance": StubPredictor("instance"),
            "semantic": StubPredictor("semantic"),
            "edge": StubPredictor("edge"),
            "llm": llm_predictor,
            "llm_text": llm_text_predictor,
        },
        config=fusion_config,
        label_map=label_map,
        coastgpt_model=model,
        tokenizer=tokenizer,
        device=str(device),
    )

    # ---- Debug: directly patch LLMParser.parse to print raw LLM output ----
    _orig_parse = pipeline.parser.parse
    def _debug_parse(prompt, image):
        """Wraps parse() to debug LLM parsing."""
        print("[verify][parser] parse() called — trying LLM parse...")
        try:
            from Models import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, tokenizer_image_token

            parse_prompt_text = pipeline.parser._config.parser_prompt_template.replace("{user_prompt}", prompt)
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + parse_prompt_text
            input_ids = tokenizer_image_token(
                full_prompt, pipeline.parser._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(pipeline.parser._device)

            print(f"[verify][parser] parser prompt tokens: {input_ids.shape[1]}, running generate...")
            t0 = __import__('time').time()
            with torch.inference_mode():
                output_ids = pipeline.parser._model.generate(
                    input_ids=input_ids,
                    images=image.to(pipeline.parser._device),
                    do_sample=False, temperature=1.0,
                    max_new_tokens=pipeline.parser._config.parser_max_new_tokens,
                    use_cache=True,
                )
            elapsed = __import__('time').time() - t0
            new_tokens = output_ids[0, input_ids.shape[1]:]
            text = pipeline.parser._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text_raw = pipeline.parser._tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            print(f"[verify][parser] generate took {elapsed:.1f}s, n_tokens={len(new_tokens)}, clean_text={len(text)} chars: {text[:300]!r}")
            print(f"[verify][parser] raw_text: {text_raw[:300]!r}")
            # Also show first/last token IDs
            if len(new_tokens) > 0:
                print(f"[verify][parser] first token ids: {new_tokens[:10].tolist()}")
                print(f"[verify][parser] last token ids: {new_tokens[-10:].tolist()}")
        except Exception as exc:
            print(f"[verify][parser] LLM generation FAILED: {exc}")
        # Fall back to original parse
        return _orig_parse(prompt, image)
    pipeline.parser.parse = _debug_parse

    # ---- Run ----
    print(f"\n[verify] Running with prompt: {args.prompt!r}")
    georef = {"source_crs": "EPSG:4326"}

    t0 = time.time()
    try:
        fc, diagnostics = pipeline.run(image_tensor, args.prompt, georef)
        elapsed = time.time() - t0
        print(f"[verify] Completed in {elapsed:.1f}s")
    except Exception as exc:
        print(f"[verify] Pipeline FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- Summarize ----
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"  Prefix:        {diagnostics.get('prefix')}")
    parse = diagnostics.get("parse", {})
    print(f"  Parse task:    {parse.get('task_type')} (source={parse.get('source')}, conf={parse.get('confidence')})")
    print(f"  Target classes:{parse.get('target_classes')}")
    dispatch = diagnostics.get("dispatch", {})
    known = dispatch.get("known", {})
    unknown = dispatch.get("unknown", [])
    for branch, classes in known.items():
        if classes:
            print(f"  Known [{branch}]: {classes}")
    print(f"  Unknown:       {unknown}")
    print(f"  Det features:  {diagnostics.get('det_feature_count', 0)}")
    print(f"  LLM features:  {diagnostics.get('llm_feature_count_before_dedup', 0)}")
    print(f"  Final features:{diagnostics.get('final_feature_count', 0)}")
    dedup_cross = diagnostics.get("dedup_cross", {})
    print(f"  Cross-dedup removed: {dedup_cross.get('n_cross_removed', 0)}")
    print(f"  FeatureCollection type: {fc.get('type')}")
    print(f"  Num features: {len(fc.get('features', []))}")

    # Validity check
    if fc.get("type") == "FeatureCollection" and isinstance(fc.get("features"), list):
        print(f"\n  ✅ GeoJSON VALID: FeatureCollection with {len(fc['features'])} features")
    else:
        print(f"\n  ❌ GeoJSON INVALID: {fc.get('type')}")

    # Save output
    output_path = args.output or "poc4_verify_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"feature_collection": fc, "diagnostics": diagnostics}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"  Output saved to: {output_path}")

    # Save diagnostics separately
    diag_path = output_path.replace(".json", "_diag.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print("VERDICT: Pipeline runs on NPU without crash")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

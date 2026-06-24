#!/usr/bin/env python3
"""
PoC-4 NPU verification — minimal smoke test.

Loads CoastGPT from checkpoint, builds FusionPipeline with real LLM predictors
and stub detection heads, then verifies pipeline orchestration on NPU.

Usage:
    python scripts/verify_poc4_npu.py \\
        --model-path output/checkpoints/FINAL.pt \\
        --image-file Images/test.png \\
        --prompt "[DET] 请检测图中的海岸线和红树林湿地。"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _setup_npu():
    import torch_npu
    local_idx = int(os.environ.get("LOCAL_RANK", "0"))
    torch_npu.npu.set_device(local_idx)
    return torch.device(f"npu:{local_idx}")


def _load_image(image_file: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_file).convert("RGB")
    w, h = image.size
    if w * h > 16_000_000:
        scale = (16_000_000 / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return image


from Models.fusion_predictors import PredictorOutput, BasePredictor


class StubPredictor(BasePredictor):
    def __init__(self, branch: str):
        self._branch = branch
    @property
    def branch(self) -> str:
        return self._branch
    def predict(self, image, prompt, georef, classes):
        return PredictorOutput(branch=self._branch, features=[], metadata={"skipped": True, "reason": "stub"})


def main():
    parser = argparse.ArgumentParser(description="PoC-4 NPU Verification")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, default="Images/test.png")
    parser.add_argument("--prompt", type=str, default="[DET] 请检测图中的海岸线和红树林湿地。")
    parser.add_argument("--output", type=str, default="poc4_verify_output.json")
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device = _setup_npu() if args.device.startswith("npu") else torch.device(args.device)
    dtype = torch.float16
    print(f"[verify] device={device}, dtype={dtype}")

    # ---- Load config ----
    import yaml as _yaml
    from ml_collections import ConfigDict
    with open("Configs/step2_dual.yaml", "r") as f:
        raw = _yaml.safe_load(f)
    config = ConfigDict(raw)
    config.stage = 0
    config.adjust_norm = False
    config.lora.enable = False   # 推理不需要 PeftModel 包装
    # NPU safety
    bits = int(getattr(config, "bits", 16))
    if bits in (4, 8):
        print(f"[verify] NPU safe: bits={bits} -> 16")
        config.bits = 16
    if not bool(getattr(config, "fp16", False)):
        config.fp16 = True; config.bf16 = False

    # ---- Load CoastGPT ----
    # Priority: FINAL_merged.pt > FINAL.pt + manual LoRA merge
    merged_path = args.model_path.replace(".pt", "_merged.pt")
    print(f"[verify] Loading CoastGPT (lora.enable=False)...")
    torch.manual_seed(322)
    from Models.coastgpt import CoastGPT
    model = CoastGPT(config)     # 干净的 CustomLlamaForCausalLM，无 PeftModel 壳
    model.to(dtype)

    if Path(merged_path).exists():
        print(f"[verify] Loading merged checkpoint: {merged_path}")
        ckpt = torch.load(merged_path, map_location="cpu")
        model.load_state_dict(ckpt["vision_ckpt"], strict=False)
        model.load_state_dict(ckpt["other_ckpt"], strict=False)
    else:
        print(f"[verify] Loading FINAL.pt + manual LoRA merge...")
        ckpt = torch.load(args.model_path, map_location="cpu")
        model.vision.load_state_dict(ckpt["vision_ckpt"], strict=False)
        other = ckpt["other_ckpt"]
        model.multimodal.projection.load_state_dict(other["multimodal_projection"], strict=False)
        te = model.language.get_text_encoder()
        ew = other["embed_tokens"]["weight"]
        te.get_input_embeddings().weight.data.copy_(ew[:te.get_input_embeddings().weight.shape[0], :])
        # Manual LoRA merge
        lora_file = _REPO_ROOT / "TextLoRA" / "adapter_model.safetensors"
        if lora_file.exists():
            from safetensors.torch import load_file
            import re, json
            with open(_REPO_ROOT / "TextLoRA" / "adapter_config.json") as f:
                lora_cfg = json.load(f)
            lora_state = load_file(str(lora_file))
            scale = lora_cfg["lora_alpha"] / lora_cfg["r"]
            base_sd = model.state_dict()
            merged = 0
            for key in list(lora_state.keys()):
                m = re.match(r'base_model\.model\.(.+)\.lora_A\.weight$', key)
                if not m: continue
                bp = m.group(1)
                bk = key.replace('.lora_A.weight', '.lora_B.weight')
                if bk not in lora_state: continue
                delta = (lora_state[bk].float() @ lora_state[key].float()) * scale
                target = f'language.text_encoder.{bp}.weight'
                if target in base_sd:
                    base_sd[target] = base_sd[target].float() + delta
                    merged += 1
            model.load_state_dict(base_sd, strict=True)
            print(f"[verify]  LoRA merged: {merged}/{len(lora_state)//2} adapters")

    model.to(device); model.eval()
    tokenizer = model.language.tokenizer
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if pad_id is None and eos_id is not None:
        tokenizer.pad_token_id = eos_id
    if pad_id is not None and unk_id is not None and eos_id is not None and pad_id == unk_id:
        tokenizer.pad_token_id = eos_id
    model.language.tokenizer.pad_token_id = tokenizer.pad_token_id
    model.language.get_text_encoder().config.pad_token_id = tokenizer.pad_token_id
    if eos_id is not None:
        model.language.get_text_encoder().config.eos_token_id = eos_id
    print(f"[verify] Model ready. vocab={len(tokenizer)}, pad={tokenizer.pad_token_id}")

    # ---- Load image ----
    print(f"[verify] Loading image: {args.image_file}")
    image = _load_image(args.image_file)
    from Dataset.build_transform import build_vlp_transform
    transform = build_vlp_transform(config, is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)
    print(f"[verify] Image: {list(image_tensor.shape)}")

    # ---- Load label_map ----
    label_map_path = "Configs/label_map.json"
    if Path(label_map_path).exists():
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
    else:
        label_map = {"branch_classes": {
            "instance": {"0": "background", "1": "海水养殖区"},
            "semantic": {"0": "background", "1": "水田", "2": "旱地"},
            "edge": {"0": "background", "1": "海岸线"},
        }}

    # ---- Build FusionPipeline ----
    from Models.fusion_pipeline import FusionConfig, FusionPipeline
    from Models.fusion_predictors import LLMPredictor, LLMTextPredictor

    fusion_config = FusionConfig(
        parser_max_new_tokens=128,
        parser_confidence_threshold=0.3,
        empty_target_policy="error",
        llm_min_confidence=0.3,
    )
    # Simple [DET]-style parser prompt the model understands
    fusion_config.parser_prompt_template = (
        "[DET] 从以下用户请求中提取任务类型和目标类别，以JSON格式输出，"
        "包含task_type(DET/CAP/VQA)、target_classes(类别列表)、confidence(0-1)。\n"
        "用户请求: {user_prompt}"
    )

    llm_p = LLMPredictor(model, tokenizer, config, max_new_tokens=512, device=str(device))
    llm_t = LLMTextPredictor(model, tokenizer, config, max_new_tokens=256, device=str(device))

    pipeline = FusionPipeline(
        predictors={
            "instance": StubPredictor("instance"),
            "semantic": StubPredictor("semantic"),
            "edge": StubPredictor("edge"),
            "llm": llm_p,
            "llm_text": llm_t,
        },
        config=fusion_config, label_map=label_map,
        coastgpt_model=model, tokenizer=tokenizer, device=str(device),
    )

    # ---- Run ----
    print(f"\n[verify] Prompt: {args.prompt!r}")
    georef = {"source_crs": "EPSG:4326"}

    t0 = time.time()
    try:
        fc, diagnostics = pipeline.run(image_tensor, args.prompt, georef)
        elapsed = time.time() - t0
        print(f"[verify] Pipeline completed in {elapsed:.1f}s")
    except Exception as exc:
        print(f"[verify] Pipeline FAILED: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    parse = diagnostics.get("parse", {})
    dispatch = diagnostics.get("dispatch", {})
    known = dispatch.get("known", {})
    unknown = dispatch.get("unknown", [])
    print(f"  Prefix:        {diagnostics.get('prefix')}")
    print(f"  Parse:         task={parse.get('task_type')}, source={parse.get('source')}, conf={parse.get('confidence')}")
    print(f"  Target classes:{parse.get('target_classes')}")
    for branch, classes in known.items():
        if classes:
            print(f"  Known [{branch}]: {classes}")
    print(f"  Unknown:       {unknown}")
    print(f"  Det features:  {diagnostics.get('det_feature_count', 0)}")
    llm_before = diagnostics.get('llm_feature_count_before_dedup', 0)
    print(f"  LLM features:  {llm_before}")
    print(f"  Final features:{diagnostics.get('final_feature_count', 0)}")
    print(f"  FeatureCollection: {fc.get('type')}, {len(fc.get('features', []))} features")

    # Save
    out = {"feature_collection": fc, "diagnostics": diagnostics}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Output: {args.output}")

    # Diag
    diag_path = args.output.replace(".json", "_diag.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    valid = fc.get("type") == "FeatureCollection" and isinstance(fc.get("features"), list)
    print(f"VERDICT: {'✅ Pipeline runs on NPU, GeoJSON ' + ('valid' if valid else 'INVALID')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

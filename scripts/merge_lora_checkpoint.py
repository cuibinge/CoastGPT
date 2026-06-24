#!/usr/bin/env python3
"""
Merge Stage 2 LoRA → single checkpoint compatible with Inference.py.

Bypasses PEFT entirely — manually computes LoRA delta and adds to base weights.
This preserves the original state_dict key structure, so the merged checkpoint
loads cleanly with load_state_dict(strict=False) or _load_checkpoint.

Usage:
    python scripts/merge_lora_checkpoint.py

Requires: safetensors (pip install safetensors)
Produces: output/checkpoints/FINAL_merged.pt (structured, same format as FINAL.pt)
"""
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_lora_weights(lora_dir: str) -> dict:
    """Load LoRA adapter weights from safetensors, return {param_name: tensor}."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")

    adapter_file = Path(lora_dir) / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_file}")
    return load_file(str(adapter_file))


def merge_lora_into_weights(base_state: dict, lora_state: dict, lora_alpha: int = 256, lora_r: int = 128) -> dict:
    """
    Compute LoRA delta = lora_B @ lora_A * (alpha / r) and add to base weights.

    LoRA keys look like:
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    Base keys look like:
      model.layers.0.self_attn.q_proj.weight

    Returns modified base_state dict.
    """
    import re

    scale = lora_alpha / lora_r
    merged_count = 0

    # Group lora_A and lora_B pairs
    # Key format: base_model.model.model.layers.X.mlp.down_proj.lora_A.weight
    lora_pairs = {}
    for key, tensor in lora_state.items():
        m = re.match(r'base_model\.model\.(.+)\.lora_([AB])\.weight$', key)
        if not m:
            continue
        base_path = m.group(1)  # e.g. "model.layers.0.self_attn.q_proj"
        lora_type = m.group(2)  # "A" or "B"
        lora_pairs.setdefault(base_path, {})[lora_type] = tensor

    # Diagnostic: check what keys actually exist for model layers
    layer_keys = [k for k in base_state if 'layer' in k.lower()][:5]
    mlp_keys = [k for k in base_state if 'down_proj' in k][:5]
    print(f"  DEBUG: sample 'layer' keys: {layer_keys}")
    print(f"  DEBUG: sample 'down_proj' keys: {mlp_keys}")
    first_base = list(lora_pairs.keys())[0] if lora_pairs else None
    if first_base:
        test_key = f"language.text_encoder.{first_base}.weight"
        print(f"  DEBUG: LoRA base_path={first_base}, looking for {test_key}: {test_key in base_state}")

    for base_path, pair in lora_pairs.items():
        if 'A' not in pair or 'B' not in pair:
            continue
        lora_A = pair['A']
        lora_B = pair['B']
        # LoRA delta: B @ A * scale
        # lora_A: [r, in_features], lora_B: [out_features, r]
        delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scale

        # LoRA base_path: "model.layers.0.self_attn.q_proj"
        # Base key in state_dict: "language.text_encoder.model.layers.0.self_attn.q_proj.weight"
        base_key = f"language.text_encoder.{base_path}.weight"
        if base_key in base_state:
            base_state[base_key] = base_state[base_key].to(torch.float32) + delta.to(base_state[base_key].dtype)
            merged_count += 1

    print(f"  Merged {merged_count} LoRA adapters into base weights (alpha={lora_alpha}, r={lora_r})")
    return base_state


def main():
    import yaml
    from ml_collections import ConfigDict

    # ---- Load config ----
    with open("Configs/step2_dual.yaml", "r") as f:
        raw = yaml.safe_load(f)
    config = ConfigDict(raw)
    config.stage = 0  # fresh model
    config.adjust_norm = False
    config.accelerator = "cpu"; config.bits = 16; config.fp16 = False; config.bf16 = False

    # ---- Load LoRA adapter weights ----
    lora_path = str(_REPO_ROOT / "TextLoRA")
    # Read lora config for alpha/r
    import json
    with open(Path(lora_path) / "adapter_config.json") as f:
        lora_config = json.load(f)
    lora_alpha = lora_config.get("lora_alpha", 256)
    lora_r = lora_config.get("r", 128)

    print(f"[1/4] Loading LoRA weights from {lora_path}...")
    lora_state = load_lora_weights(lora_path)
    print(f"  Loaded {len(lora_state)} LoRA parameters")

    # ---- Load base model + FINAL.pt weights ----
    print("[2/4] Loading base model + FINAL.pt...")
    from Models.coastgpt import CoastGPT
    model = CoastGPT(config)
    model.to(torch.float32)

    ckpt = torch.load("output/checkpoints/FINAL.pt", map_location="cpu")
    model.vision.load_state_dict(ckpt["vision_ckpt"], strict=False)
    other = ckpt["other_ckpt"]
    model.multimodal.projection.load_state_dict(other["multimodal_projection"], strict=False)

    # Load embed_tokens (truncate vocab if needed)
    te = model.language.get_text_encoder()
    ew = other["embed_tokens"]["weight"]
    mv = te.get_input_embeddings().weight.shape[0]
    te.get_input_embeddings().weight.data.copy_(ew[:mv, :])
    if "lm_head" in other and len(other["lm_head"]) > 0:
        lm_w = list(other["lm_head"].values())[0]
        if te.get_output_embeddings() is not None:
            te.get_output_embeddings().weight.data.copy_(lm_w[:mv, :])

    print("  Base weights loaded")

    # ---- Merge LoRA into base weights (manual, no PEFT) ----
    print("[3/4] Merging LoRA into base weights (manual computation)...")
    base_state = model.state_dict()
    base_state = merge_lora_into_weights(base_state, lora_state, lora_alpha, lora_r)
    model.load_state_dict(base_state, strict=True)  # strict=True — must match perfectly
    print("  Merge complete, state_dict intact")
    model.eval()

    # ---- Save as structured checkpoint ----
    output_path = "output/checkpoints/FINAL_merged.pt"
    print(f"[4/4] Saving merged checkpoint to {output_path}...")

    # Build other_ckpt: start with original FINAL.pt's other_ckpt,
    # then overlay merged text_encoder weights on top
    full_state = model.state_dict()
    vision_state = {k[len("vision."):]: v for k, v in full_state.items() if k.startswith("vision.")}

    # Start from original other_ckpt (preserves multimodal_projection, embed_tokens, lm_head structure)
    other_state = dict(ckpt["other_ckpt"])
    # Overlay merged text_encoder weights (these are the LoRA-merged attention layers)
    for k, v in full_state.items():
        if k.startswith("language.text_encoder."):
            # Store under language.text_encoder.* so _load_checkpoint can find them
            other_state[k] = v

    merged_ckpt = {"vision_ckpt": vision_state, "other_ckpt": other_state}
    torch.save(merged_ckpt, output_path)
    size_gb = Path(output_path).stat().st_size / 1024**3
    print(f"  Saved: {size_gb:.1f} GB")

    # ---- Verify: reload into fresh model (same path as Inference.py) ----
    print("[verify] Reloading merged checkpoint...")
    config2 = ConfigDict(raw)
    config2.stage = 0; config2.adjust_norm = False
    config2.accelerator = "cpu"; config2.bits = 16; config2.fp16 = False; config2.bf16 = False

    m2 = CoastGPT(config2)
    ckpt2 = torch.load(output_path, map_location="cpu")

    # Same loading as Inference.py _load_checkpoint + custom_load_state_dict
    m2.vision.load_state_dict(ckpt2["vision_ckpt"], strict=False)
    o2 = ckpt2["other_ckpt"]
    m2.multimodal.projection.load_state_dict(o2["multimodal_projection"], strict=False)
    te2 = m2.language.get_text_encoder()
    e2w = o2["embed_tokens"]["weight"]
    v2 = te2.get_input_embeddings().weight.shape[0]
    te2.get_input_embeddings().weight.data.copy_(e2w[:v2, :])
    if "lm_head" in o2 and len(o2["lm_head"]) > 0:
        lm2w = list(o2["lm_head"].values())[0]
        if te2.get_output_embeddings() is not None:
            te2.get_output_embeddings().weight.data.copy_(lm2w[:v2, :])

    # Load merged text_encoder weights (language.text_encoder.*)
    text_keys = {k: v for k, v in o2.items() if k.startswith("language.text_encoder.")}
    msg2 = m2.load_state_dict(text_keys, strict=False)
    print(f"  text_encoder loaded: missing={len(msg2.missing_keys)}, unexpected={len(msg2.unexpected_keys)}")

    if len(msg2.unexpected_keys) == 0:
        print("[verify] PERFECT — zero unexpected keys.")
    else:
        print(f"[verify] {len(msg2.unexpected_keys)} unexpected keys.")

    print("\nDone! Merged checkpoint: output/checkpoints/FINAL_merged.pt")
    print("Inference: python Inference.py --model-path output/checkpoints/FINAL_merged.pt --skip-text-lora True")


if __name__ == "__main__":
    main()

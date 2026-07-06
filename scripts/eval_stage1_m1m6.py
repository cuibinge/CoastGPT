#!/usr/bin/env python3
"""
M1-M6 Stage 1 evaluation suite.
Usage: python scripts/eval_stage1_m1m6.py --checkpoint <ckpt.pt> --config <config.json>
Runs: M1 (embedding cos), M2 (zero visual), M3 (shuffle), M4 (yes/no),
      M5 (multi-choice), M6 (cross-image diversity)
"""
import json, os, re, sys, torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Models.coastgpt import CoastGPT
from Models import IMAGE_TOKEN_INDEX
from Models.language_model import tokenizer_image_token
import ml_collections

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TFORM = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

def cos(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))

def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = ml_collections.ConfigDict(json.load(f))
    cfg.stage = 1; cfg.accelerator = "npu"; cfg.lora.enable = False
    model = CoastGPT(cfg).eval().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("vision_ckpt", "rgb_ckpt", "module")):
        import gc

        del ckpt
        gc.collect()
        print("[Eval] Loading project-format checkpoint via CoastGPT.custom_load_state_dict")
        model.custom_load_state_dict(checkpoint_path, strict=False)
        return model
    state = ckpt["model"] if "model" in ckpt else ckpt
    ms = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"[Eval] Loaded flat checkpoint tensors: {len(compatible)}/{len(ms)}")
    return model

def gather_images(max_n=20):
    paths = []
    for d in ["Images", "artifacts"]:
        dp = os.path.join(ROOT, d)
        if not os.path.isdir(dp): continue
        for f in sorted(os.listdir(dp)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(dp, f))
    # Ensure at least 4 distinct scenes
    return paths[:max_n]

def build_prompt_ids(prompt_text, tok, device):
    ids = tokenizer_image_token(
        f"<image>\n{prompt_text}",
        tok,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return ids.unsqueeze(0).to(device)

def clean_generation(text):
    text = str(text or "")
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.replace("<image>", "\n").replace("</image>", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def parse_yes_no(text):
    match = re.search(r"\b(yes|no)\b", clean_generation(text).lower())
    return match.group(1) if match else "unknown"

def parse_choice(text, allowed="ABCDE"):
    cleaned = clean_generation(text).strip().upper()
    match = re.search(rf"\b([{re.escape(allowed)}])\b", cleaned)
    if match:
        return match.group(1)
    for ch in cleaned:
        if ch in allowed:
            return ch
    return "?"

def run_m1(model, imgs_t, device):
    """M1: Embedding cross-image cosine similarity."""
    with torch.no_grad():
        all_gate, all_proj, all_experts = [], [], []
        for img_t in imgs_t:
            iseq, _, _ = model.vision.encode_with_spatial(img_t)
            moe = model.multimodal.projection
            ie = moe._project_image_embs(iseq)
            g = moe._build_visual_gate_feature(ie)
            p_out = model.multimodal({}, image_embedding=iseq)
            e_outs = [moe.experts[e](ie, physical_queries=None) for e in range(moe.num_experts)]
            all_gate.append(g); all_proj.append(p_out); all_experts.append(e_outs)

    n = len(imgs_t)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    gate_cos_vals = [cos(all_gate[i], all_gate[j]) for i, j in pairs]
    proj_cos_vals = [cos(all_proj[i].mean(dim=1), all_proj[j].mean(dim=1)) for i, j in pairs]
    exp_cos_vals = []
    for e in range(model.multimodal.projection.num_experts):
        exp_cos_vals.append([cos(all_experts[i][e].mean(dim=1), all_experts[j][e].mean(dim=1))
                            for i, j in pairs])

    return {
        "gate_feat_cos_mean": float(np.mean(gate_cos_vals)),
        "gate_feat_cos_max": float(np.max(gate_cos_vals)),
        "proj_emb_cos_mean": float(np.mean(proj_cos_vals)),
        "proj_emb_cos_max": float(np.max(proj_cos_vals)),
        "expert_cos_mean": [float(np.mean(ec)) for ec in exp_cos_vals],
        "expert_cos_max": [float(np.max(ec)) for ec in exp_cos_vals],
        "m1_pass": float(np.mean(proj_cos_vals)) < 0.90 and float(np.mean(gate_cos_vals)) < 0.90,
    }

def generate(model, img_t, prompt_text, tok, device, max_tokens=60, do_sample=True, temp=0.8):
    ids = build_prompt_ids(prompt_text, tok, device)
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=do_sample, temperature=temp)
    if do_sample:
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        out = model.generate(input_ids=ids, images=img_t, **gen_kwargs)
    return clean_generation(tok.decode(out[0], skip_special_tokens=True))

def generate_zero_visual(model, img_t, prompt_text, tok, device, max_tokens=60, do_sample=True, temp=0.8):
    """M2 helper: replace visual embeddings with zeros."""
    ids = build_prompt_ids(prompt_text, tok, device)
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=do_sample, temperature=temp)
    if do_sample:
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        iseq, _, _ = model.vision.encode_with_spatial(img_t)
        mm = model.multimodal({}, image_embedding=iseq)
        zero_mm = torch.zeros_like(mm)
        _, attention_mask, past_key_values, inputs_embeds, _ = model.language.prepare_inputs_for_multimodal(
            input_ids=ids,
            attention_mask=None,
            labels=None,
            image_embedding=zero_mm,
            past_key_values=None,
        )
        inputs_embeds = torch.nan_to_num(inputs_embeds)
        llm_out = model.language.text_encoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            **gen_kwargs,
        )
    return clean_generation(tok.decode(llm_out[0], skip_special_tokens=True))

def run_m2_m3(model, imgs_t, tok, device, img_labels):
    """M2: Zero visual, M3: Image sensitivity (same prompt, different images)."""
    prompt = "[CAP] Describe this image concisely."
    results = {"m2": [], "m3": []}

    # M2: normal vs zero visual
    for idx, label in enumerate(img_labels):
        norm = generate(model, imgs_t[idx], prompt, tok, device, max_tokens=80, do_sample=True, temp=0.8)
        zero = generate_zero_visual(model, imgs_t[idx], prompt, tok, device, max_tokens=80, do_sample=True, temp=0.8)
        results["m2"].append({"label": label, "normal": norm[:200], "zero": zero[:200]})

    # M3: different images, same prompt — check if outputs vary
    outputs = [generate(model, imgs_t[idx], prompt, tok, device, max_tokens=80, do_sample=True, temp=0.8)
              for idx in range(len(img_labels))]
    first80 = [o[:80] for o in outputs]
    results["m3"] = {"outputs": [{"label": l, "text": o[:200]} for l, o in zip(img_labels, outputs)],
                     "unique_first80": len(set(first80)),
                     "total": len(first80),
                     "m3_pass": len(set(first80)) == len(first80)}
    return results

def run_m4(model, imgs_t, tok, device, img_labels):
    """M4: Yes/No structured questions."""
    questions = [
        "Is there a coastline or ocean visible in this image? Answer only yes or no.",
        "Is there dense forest or rainforest visible in this image? Answer only yes or no.",
        "Is there urban area or buildings visible in this image? Answer only yes or no.",
        "Is there a water body (lake, river, ocean) visible in this image? Answer only yes or no.",
    ]
    results = []
    for q in questions:
        qr = {"question": q, "answers": []}
        for idx, label in enumerate(img_labels):
            out = generate(model, imgs_t[idx], q, tok, device, max_tokens=20, do_sample=False, temp=1.0)
            ans = parse_yes_no(out)
            qr["answers"].append({"label": label, "answer": ans, "raw": out[:60]})
        results.append(qr)
    return results

def run_m5(model, imgs_t, tok, device, img_labels):
    """M5: Multi-choice classification."""
    mc_prompts = [
        ("What is the MAIN land cover type in this image?\nA. Coastline or beach\nB. Dense forest\nC. Desert\nD. Urban area\nE. Farmland\nAnswer with a single letter A-E.", None),
        ("What is the DOMINANT terrain in this image?\nA. Mountain\nB. Flat plain\nC. Hilly\nD. Water surface\nAnswer with a single letter A-D.", None),
    ]
    results = []
    for prompt, _ in mc_prompts:
        pr = {"prompt": prompt[:100], "answers": []}
        allowed = "ABCDE" if "A-E" in prompt else "ABCD"
        for idx, label in enumerate(img_labels):
            out = generate(model, imgs_t[idx], prompt, tok, device, max_tokens=5, do_sample=False, temp=1.0)
            ans_letter = parse_choice(out, allowed=allowed)
            pr["answers"].append({"label": label, "choice": ans_letter, "raw": out[:60]})
        results.append(pr)
    return results

def run_m6(model, imgs_t, tok, device):
    """M6: Different images → different outputs (greedy decode to check collapse)."""
    prompt = "[CAP] Describe what you see in this image."
    outputs = []
    for img_t in imgs_t:
        out = generate(model, img_t, prompt, tok, device, max_tokens=80, do_sample=False, temp=1.0)
        outputs.append(out[:200])
    unique_first50 = len(set(o[:50] for o in outputs))
    return {"outputs": outputs, "unique_first50": unique_first50, "total": len(outputs),
            "m6_pass": unique_first50 >= len(outputs) * 0.5}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--output", default=None)
    ap.add_argument("--m1-only", action="store_true")
    ap.add_argument("--skip-gen", action="store_true", help="skip generation tests (M2-M6)")
    args = ap.parse_args()

    try: import torch_npu
    except Exception:
        if "npu" in args.device: args.device = "cpu"

    ckpt_name = os.path.basename(args.checkpoint)
    out_dir = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_{ckpt_name.replace('.pt','')}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Eval] Loading model...")
    model = load_model(args.config, args.checkpoint, args.device)
    tok = model.language.tokenizer

    img_paths = gather_images(args.max_images)
    if len(img_paths) < 4:
        print(f"[ERROR] Need >=4 images, found {len(img_paths)}")
        return
    img_labels = [os.path.splitext(os.path.basename(p))[0][:30] for p in img_paths]

    imgs_t = []
    for p in img_paths:
        try:
            imgs_t.append(TFORM(Image.open(p).convert('RGB')).unsqueeze(0).to(args.device))
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")

    print(f"[Eval] {len(imgs_t)} images loaded. Running M1...")

    # M1 always runs
    m1 = run_m1(model, imgs_t, args.device)
    print(f"  M1: gate={m1['gate_feat_cos_mean']:.4f} proj={m1['proj_emb_cos_mean']:.4f}  "
          f"{'✅' if m1['m1_pass'] else '🔴'}")

    result = {"checkpoint": args.checkpoint, "n_images": len(imgs_t), "M1": m1}

    if not args.skip_gen and not args.m1_only:
        print("[Eval] Running M2/M3 (generation tests)...")
        m2_m3 = run_m2_m3(model, imgs_t, tok, args.device, img_labels)
        result["M2"] = m2_m3["m2"]
        result["M3"] = m2_m3["m3"]
        print(f"  M3: {m2_m3['m3']['unique_first80']}/{m2_m3['m3']['total']} unique first-80-char")

        print("[Eval] Running M4 (yes/no)...")
        m4 = run_m4(model, imgs_t, tok, args.device, img_labels)
        result["M4"] = m4

        print("[Eval] Running M5 (multi-choice)...")
        m5 = run_m5(model, imgs_t, tok, args.device, img_labels)
        result["M5"] = m5

        print("[Eval] Running M6 (cross-image diversity)...")
        m6 = run_m6(model, imgs_t, tok, args.device)
        result["M6"] = m6
        print(f"  M6: {m6['unique_first50']}/{m6['total']} unique first-50-char")

    # Summary verdict
    verdict = []
    verdict.append(f"M1 (embedding cos): {'PASS ✅' if m1['m1_pass'] else 'FAIL 🔴'}")
    if "M3" in result:
        verdict.append(f"M3 (image sensitivity): {'PASS ✅' if result['M3']['m3_pass'] else 'FAIL 🔴'}")
    if "M6" in result:
        verdict.append(f"M6 (output diversity): {'PASS ✅' if result['M6']['m6_pass'] else 'FAIL 🔴'}")

    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY — {ckpt_name}")
    for v in verdict:
        print(f"  {v}")

    # Save
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    main()

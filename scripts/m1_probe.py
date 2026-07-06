#!/usr/bin/env python3
"""M1 probe: cross-image embedding cosine similarity on any Stage 1 checkpoint."""
import json, os, sys, torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Models.coastgpt import CoastGPT
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
        print("[M1 Probe] Loading project-format checkpoint via CoastGPT.custom_load_state_dict")
        model.custom_load_state_dict(checkpoint_path, strict=False)
        return model
    state = ckpt["model"] if "model" in ckpt else ckpt
    ms = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"[M1 Probe] Loaded flat checkpoint tensors: {len(compatible)}/{len(ms)}")
    return model

def gather_image_paths(root_dirs):
    paths = []
    for d in root_dirs:
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(d, f))
    return paths

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="consolidated .pt checkpoint")
    ap.add_argument("--config", required=True, help="config.json from run dir")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try: import torch_npu
    except Exception:
        if "npu" in args.device: args.device = "cpu"

    print(f"[M1 Probe] Loading model from {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, args.device)

    # Gather test images
    img_dirs = ["Images", "artifacts", "Docs"]
    img_paths = []
    for d in img_dirs:
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            img_paths.extend([os.path.join(p, f) for f in sorted(os.listdir(p))
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    img_paths = img_paths[:args.max_images]
    if len(img_paths) < 4:
        print("[ERROR] Need at least 4 images for cosine probe. Found:", len(img_paths))
        return
    print(f"[M1 Probe] Using {len(img_paths)} images")

    # Load and preprocess
    imgs_t = []
    for p in img_paths:
        try:
            imgs_t.append(TFORM(Image.open(p).convert('RGB')).unsqueeze(0).to(args.device))
        except Exception:
            pass
    print(f"[M1 Probe] Loaded {len(imgs_t)} images")

    # Extract embeddings
    with torch.no_grad():
        all_iseq, all_gate, all_proj, all_experts = [], [], [], []
        for img_t in imgs_t:
            iseq, _, _ = model.vision.encode_with_spatial(img_t)
            moe = model.multimodal.projection
            ie = moe._project_image_embs(iseq)
            g = moe._build_visual_gate_feature(ie)
            p_out = model.multimodal({}, image_embedding=iseq)
            e_outs = [moe.experts[e](ie, physical_queries=None) for e in range(moe.num_experts)]
            all_iseq.append(iseq); all_gate.append(g); all_proj.append(p_out); all_experts.append(e_outs)

    # Compute pairwise cosines
    n = len(imgs_t)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    src_cos, gate_cos, proj_cos = [], [], []
    exp_cos = [[] for _ in range(model.multimodal.projection.num_experts)]

    for i, j in pairs:
        src_cos.append(cos(all_iseq[i].mean(dim=1), all_iseq[j].mean(dim=1)))
        gate_cos.append(cos(all_gate[i], all_gate[j]))
        proj_cos.append(cos(all_proj[i].mean(dim=1), all_proj[j].mean(dim=1)))
        for e in range(len(exp_cos)):
            exp_cos[e].append(cos(all_experts[i][e].mean(dim=1), all_experts[j][e].mean(dim=1)))

    # Report
    def stat(vals, name, target=0.90):
        m = np.mean(vals); mx = np.max(vals); mn = np.min(vals)
        flag = "✅" if m < target else ("⚠️" if m < target + 0.05 else "🔴")
        print(f"  {name:<20} mean={m:.4f}  min={mn:.4f}  max={mx:.4f}  (M1 target: <{target}) {flag}")
        return m

    print("\n" + "=" * 60)
    print(f"  M1 EMBEDDING PROBE — {os.path.basename(args.checkpoint)}")
    print(f"  {len(pairs)} pairs across {n} images")
    print("=" * 60)
    m_src = stat(src_cos, "image_seq", 0.90)
    m_gate = stat(gate_cos, "gate_feat", 0.90)
    m_proj = stat(proj_cos, "proj_emb", 0.90)
    for e in range(len(exp_cos)):
        stat(exp_cos[e], f"expert_{e}_out", 0.90)

    print(f"\n  M1 VERDICT:", end=" ")
    if m_proj < 0.90 and m_gate < 0.90:
        print("PASS ✅  (both proj_emb and gate_feat < 0.90)")
    elif m_proj < 0.90 or m_gate < 0.90:
        print("PARTIAL ⚠️  (one below 0.90, one above)")
    else:
        print("FAIL 🔴  (embedding collapse — proj and gate both > 0.90)")

    # Save
    result = {
        "checkpoint": args.checkpoint,
        "n_images": n, "n_pairs": len(pairs),
        "image_seq_cos": float(m_src),
        "gate_feat_cos": float(m_gate),
        "proj_emb_cos": float(m_proj),
        "expert_cos": [float(np.mean(ec)) for ec in exp_cos],
    }
    out_path = args.output or os.path.join(os.path.dirname(args.checkpoint), "m1_probe.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()

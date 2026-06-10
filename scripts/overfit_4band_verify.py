#!/usr/bin/env python3
"""
小规模 4 波段 overfit 验证：对比 3 通道 vs 4 通道（RGB+NIR）loss 收敛。

用法:
  python scripts/overfit_4band_verify.py --in-chans 3 --device npu:1
  python scripts/overfit_4band_verify.py --in-chans 4 --device npu:1
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict

warnings.filterwarnings("ignore")
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Dataset (pre-cached in memory)
# ---------------------------------------------------------------------------


def sample_tif_paths(root: str = "/home/ma-user/work/Stage3Data/养殖区", n: int = 100, seed: int = 42) -> List[str]:
    rng = np.random.RandomState(seed)
    all_tifs = []
    for dirpath, _, filenames in os.walk(root):
        if dirpath.endswith("Image_Orig"):
            for fn in filenames:
                if fn.endswith(".tif") and not fn.startswith("."):
                    all_tifs.append(os.path.join(dirpath, fn))
    if len(all_tifs) < n:
        raise RuntimeError(f"Only {len(all_tifs)} TIFs found, need {n}")
    idx = rng.choice(len(all_tifs), size=n, replace=False)
    return [all_tifs[i] for i in idx]


def load_tif_4band(path: str) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(path).astype(np.float32)
    arr = np.clip(arr, 0, None)
    for b in range(arr.shape[2]):
        bmin, bmax = arr[:, :, b].min(), arr[:, :, b].max()
        if bmax > bmin + 1e-6:
            arr[:, :, b] = (arr[:, :, b] - bmin) / (bmax - bmin)
        else:
            arr[:, :, b] = 0.0
    return arr


def precache_dataset(paths: List[str], image_size: int = 224) -> torch.Tensor:
    """将所有 TIF 预加载到显存/内存中，避免训练时反复 I/O."""
    print(f"Pre-caching {len(paths)} TIFs...")
    tensors = []
    for i, p in enumerate(paths):
        arr = load_tif_4band(p)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [4, H, W]
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        H = (image_size // 16) * 16
        W = (image_size // 16) * 16
        t = t[:, :H, :W]
        tensors.append(t)
        if (i + 1) % 20 == 0:
            print(f"  cached {i+1}/{len(paths)}")
    result = torch.stack(tensors, dim=0)  # [N, 4, H, W]
    print(f"Cache shape: {list(result.shape)}, dtype={result.dtype}, memory={result.element_size() * result.numel() / 1e6:.1f}MB")
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SimpleReconHead(nn.Module):
    def __init__(self, in_ch: int = 1024, out_ch: int = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


def expand_conv2d_in_channels(old_conv: nn.Conv2d, new_in_ch: int) -> nn.Conv2d:
    old_w = old_conv.weight.data
    out_ch = old_w.shape[0]
    k = old_w.shape[2:]
    new_conv = nn.Conv2d(new_in_ch, out_ch, kernel_size=k, stride=old_conv.stride,
                         padding=old_conv.padding, bias=old_conv.bias is not None)
    new_conv.weight.data[:, :3] = old_w.clone()
    new_conv.weight.data[:, 3:] = old_w.mean(dim=1, keepdim=True)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data.clone()
    return new_conv


def expand_vision_encoder_channels(encoder, in_chans: int):
    if in_chans == 3:
        return
    ge = encoder.global_encoder
    if hasattr(ge, "patch_embed") and hasattr(ge.patch_embed, "proj"):
        old = ge.patch_embed.proj
        ge.patch_embed.proj = expand_conv2d_in_channels(old, in_chans)
        print(f"[expand] global patch_embed.proj: {old.in_channels}→{in_chans}")

    le = encoder.local_encoder
    for full_name, module in le.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            parts = full_name.split(".")
            parent = le
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            leaf = parts[-1]
            new_mod = expand_conv2d_in_channels(module, in_chans)
            if leaf.isdigit():
                parent[int(leaf)] = new_mod
            else:
                setattr(parent, leaf, new_mod)
            print(f"[expand] local {full_name}: 3→{in_chans}")
            break


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config(in_chans: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.alignment_dim = 1024
    cfg.adjust_norm = False
    cfg.stage = 1
    rgb_cfg = ConfigDict()
    rgb_cfg.arch = "dual"
    rgb_cfg.default_input_size = [224, 224]
    rgb_cfg.global_encoder_name = "dinov3_vitl16"
    rgb_cfg.local_encoder_name = "convnext_base"
    rgb_cfg.local_source = "dino"
    rgb_cfg.local_ckpt_path = str(_REPO_ROOT / "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth")
    rgb_cfg.global_ckpt_path = str(_REPO_ROOT / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    rgb_cfg.freeze_global = False
    rgb_cfg.freeze_local = False
    rgb_cfg.attn_pooler = ConfigDict()
    rgb_cfg.attn_pooler.num_query = 144
    rgb_cfg.attn_pooler.num_attn_heads = 16
    rgb_cfg.attn_pooler.num_layers = 6
    cfg.rgb_vision = rgb_cfg
    cfg.physics = ConfigDict()
    cfg.physics.enabled = False
    cfg.physics.prompt_enabled = False
    cfg.moe_proj = ConfigDict()
    cfg.moe_proj.include_physical_prompt = False
    cfg.use_checkpoint = False
    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_overfit(in_chans: int, device: str, n_samples: int = 100, n_steps: int = 200):
    print(f"\n{'='*60}")
    print(f"Overfit experiment: in_chans={in_chans}")
    print(f"{'='*60}")

    # --- Data: pre-cache all TIFs in memory ---
    paths = sample_tif_paths(n=n_samples)
    cache = precache_dataset(paths, image_size=224)  # [N, 4, H, W]
    N = cache.shape[0]

    # --- Model ---
    from Models.dual_vision_encoder import DualVisionEncoder

    cfg = build_config(in_chans)
    vision = DualVisionEncoder(cfg)
    expand_vision_encoder_channels(vision, in_chans)

    for n, p in vision.named_parameters():
        p.requires_grad = False
    ge = vision.global_encoder
    if hasattr(ge, "patch_embed") and hasattr(ge.patch_embed, "proj"):
        for p in ge.patch_embed.proj.parameters():
            p.requires_grad = True
    le = vision.local_encoder
    for name, module in le.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == in_chans:
            for p in module.parameters():
                p.requires_grad = True
            break
    if hasattr(ge, "blocks") and len(ge.blocks) > 0:
        for p in ge.blocks[0].parameters():
            p.requires_grad = True

    recon = SimpleReconHead(in_ch=1024, out_ch=in_chans)
    vision = vision.to(device)
    recon = recon.to(device)
    cache = cache.to(device)

    trainable = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in recon.parameters())
    print(f"Trainable params: {trainable:,}")

    opt = torch.optim.AdamW(
        list(vision.parameters()) + list(recon.parameters()), lr=1e-3, weight_decay=0.0
    )

    loss_history = []
    vision.train()
    recon.train()
    batch_size = 4
    t_start = time.time()

    for step in range(n_steps):
        # Random batch from cache
        idx = torch.randperm(N, device=cache.device)[:batch_size]
        batch = cache[idx]
        x_in = batch[:, :in_chans].float()

        _, g_grid, _ = vision.encode_with_spatial(x_in)
        pred = recon(g_grid)
        if pred.shape[-2:] != x_in.shape[-2:]:
            pred = F.interpolate(pred, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.mse_loss(pred, x_in)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(recon.parameters()), max_norm=1.0
        )
        opt.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if step % 20 == 0 or step == n_steps - 1:
            elapsed = time.time() - t_start
            etr = elapsed / (step + 1) * (n_steps - step - 1) if step > 0 else 0
            print(f"  step {step:4d}/{n_steps}  loss={loss_val:.6f}  elapsed={elapsed:.0f}s  eta={etr:.0f}s")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s ({elapsed/n_steps:.1f}s/step)")
    return loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-chans", type=int, required=True, choices=[3, 4])
    parser.add_argument("--device", type=str, default="npu:1")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    history = run_overfit(
        in_chans=args.in_chans, device=args.device,
        n_samples=args.n_samples, n_steps=args.n_steps,
    )

    out_path = args.output or f"overfit_{args.in_chans}ch.json"
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"in_chans": args.in_chans, "loss": history}, f, indent=2)
    print(f"Loss saved to {out_path}")
    print(f"Final={history[-1]:.6f}  Min={min(history):.6f}  Reduction={(history[0]-history[-1])/history[0]*100:.1f}%")


if __name__ == "__main__":
    main()

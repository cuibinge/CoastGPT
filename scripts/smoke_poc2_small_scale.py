#!/usr/bin/env python3
"""Small-scale PoC-2 training: ~2k train, ~500 val, 5 epochs, real vision encoder."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.semantic_head import LandcoverSemanticHead
from Dataset.landcover_dataset import (
    LandcoverSemanticDataset, landcover_collate_fn, split_by_source_image,
)
from Dataset.landcover_tile_grouping import (
    scan_landcover_directories, group_tiles_by_spatial_key, build_merged_samples,
)
from Dataset.landcover_label_map import IGNORE_INDEX, num_classes as get_num_classes
from scripts.poc_stage_semantic import (
    compute_loss, compute_observed_pixel_metrics, clean_vision_state_dict,
)

OUTPUT_DIR = Path("outputs/poc2_small_scale")
SUBSET_TRAIN = 2000
SUBSET_VAL = 500
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
IMAGE_SIZE = 224


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    try:
        import torch_npu
        device = torch.device("npu:0")
    except Exception:
        device = torch.device("cpu")
    print(f"Device: {device}")

    num_classes = 25

    # ---- Data ----
    print("\n=== Building Dataset ===")
    raw = scan_landcover_directories()
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)
    train_samples, val_samples = split_by_source_image(merged, val_ratio=0.2, seed=42)

    train_ds = LandcoverSemanticDataset(train_samples[:SUBSET_TRAIN])
    val_ds = LandcoverSemanticDataset(val_samples[:SUBSET_VAL])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=landcover_collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=landcover_collate_fn,
    )

    # ---- Vision Encoder ----
    print("\n=== Building Vision Encoder ===")
    from ml_collections import ConfigDict

    vis_cfg = ConfigDict({
        "rgb_vision": {
            "arch": "dual",
            "global_encoder_name": "dinov3_vitl16",
            "local_source": "dino",
            "local_encoder_name": "convnext_base",
            "local_ckpt_path": "./dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
            "freeze_global": True,
            "freeze_local": True,
            "global_ckpt_path": "./dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            "input_size": [224, 224],
            "physical_prompt_dim": 4096,
            "patch_dropout": 0.0,
            "input_patchnorm": False,
            "tune_pooler": False,
            "attn_pooler": {"num_query": 144, "num_attn_heads": 16, "num_layers": 6},
        },
        "alignment_dim": 1024,
    })
    vision = DualVisionEncoder(vis_cfg)

    ckpt = torch.load("./output/stage2/checkpoints/iter_2879_consolidated.pt", map_location="cpu")
    if "vision_ckpt" in ckpt:
        sd = ckpt["vision_ckpt"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    sd = clean_vision_state_dict(sd)
    vision.load_state_dict(sd, strict=False)
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False
    print("Vision encoder loaded and frozen.")

    # ---- FPN + Semantic Head ----
    print("\n=== Building FPN + Semantic Head ===")
    fpn = FPNNeck(in_channels=[128, 256, 512, 1024], out_channels=256).to(device)
    sem_head = LandcoverSemanticHead(in_channels=256, num_classes=num_classes, output_size=(224, 224)).to(device)

    trainable = list(fpn.parameters()) + list(sem_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # ---- Training ----
    print(f"\n{'='*60}")
    print(f"Small-scale training: {EPOCHS} epochs, {BATCH_SIZE} batch, {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        fpn.train()
        sem_head.train()
        total_losses = []

        for batch_idx, (images, targets, _metas) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                _, _, pyramid_raw = vision.encode_with_spatial(images)
            c4, c8, c16, c32 = pyramid_raw
            p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
            logits = sem_head(p1, p2, p3, p4)

            loss_total, loss_ce, loss_dice = compute_loss(logits, targets)
            loss_total.backward()
            opt.step()
            total_losses.append(loss_total.item())

            if (batch_idx + 1) % 20 == 0:
                avg_loss = np.mean(total_losses[-20:])
                print(f"Epoch {epoch} | Step {batch_idx+1:4d} | Loss: {avg_loss:.4f}")

        avg_train_loss = float(np.mean(total_losses))
        print(f"Epoch {epoch} complete | Avg Loss: {avg_train_loss:.4f}")

        # Validate (on a subset of val)
        if epoch % 2 == 0 or epoch == EPOCHS:
            fpn.eval()
            sem_head.eval()
            all_metrics = []
            with torch.no_grad():
                for batch_idx, (images, targets, _metas) in enumerate(val_loader):
                    if batch_idx >= 25:  # ~100 val samples
                        break
                    images = images.to(device)
                    targets = targets.to(device)
                    _, _, pyramid_raw = vision.encode_with_spatial(images)
                    c4, c8, c16, c32 = pyramid_raw
                    p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
                    logits = sem_head(p1, p2, p3, p4)

                    for i in range(images.shape[0]):
                        m = compute_observed_pixel_metrics(
                            logits[i:i+1], targets[i:i+1], num_classes
                        )
                        all_metrics.append(m)

            miou = float(np.mean([m["mIoU"] for m in all_metrics]))
            pix_acc = float(np.mean([m["pixel_accuracy"] for m in all_metrics]))
            print(f"  Val mIoU: {miou:.4f}  |  Pixel Acc: {pix_acc:.4f}")

    # ---- Save checkpoint ----
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({
        "epoch": EPOCHS, "fpn": fpn.state_dict(), "sem_head": sem_head.state_dict(),
        "num_classes": num_classes, "ignore_index": IGNORE_INDEX,
    }, str(ckpt_dir / "small_scale_final.pt"))

    print(f"\nSmall-scale training complete. Checkpoint: {ckpt_dir / 'small_scale_final.pt'}")
    print(f"Final train loss: {avg_train_loss:.4f}")


if __name__ == "__main__":
    main()

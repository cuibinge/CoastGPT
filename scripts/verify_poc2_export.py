#!/usr/bin/env python3
"""Quick verification: load small-scale checkpoint, export overlays + GeoJSON on 4 val samples."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

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
from Dataset.landcover_label_map import IGNORE_INDEX, train_id_to_dlmc, num_classes as get_num_classes
from scripts.poc_stage_semantic import (
    _save_overlays, _export_geojson_samples, clean_vision_state_dict,
    compute_loss, compute_observed_pixel_metrics,
)
from ml_collections import ConfigDict

OUTPUT_DIR = Path("outputs/poc2_export_verify")
NUM_SAMPLES = 4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "overlays").mkdir(exist_ok=True)
    (OUTPUT_DIR / "geojson").mkdir(exist_ok=True)

    try:
        import torch_npu
        device = torch.device("npu:0")
    except Exception:
        device = torch.device("cpu")
    print(f"Device: {device}")

    num_classes = 25

    # ---- Data ----
    print("Loading data...")
    raw = scan_landcover_directories()
    groups = group_tiles_by_spatial_key(raw)
    merged = build_merged_samples(raw, groups)
    _, val_samples = split_by_source_image(merged, val_ratio=0.2, seed=42)

    val_ds = LandcoverSemanticDataset(val_samples[:NUM_SAMPLES])
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=NUM_SAMPLES, shuffle=False, collate_fn=landcover_collate_fn,
    )

    # ---- Vision Encoder ----
    print("Building vision encoder...")
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
    print("Vision encoder loaded.")

    # ---- FPN + Semantic Head ----
    print("Building FPN + Semantic Head...")
    fpn = FPNNeck(in_channels=[128, 256, 512, 1024], out_channels=256).to(device)
    sem_head = LandcoverSemanticHead(in_channels=256, num_classes=num_classes, output_size=(224, 224)).to(device)

    sem_ckpt = torch.load("outputs/poc2_small_scale/checkpoints/small_scale_final.pt", map_location="cpu")
    fpn.load_state_dict(sem_ckpt["fpn"])
    sem_head.load_state_dict(sem_ckpt["sem_head"])
    fpn.eval()
    sem_head.eval()
    print("FPN + Semantic Head loaded.")

    # ---- Inference ----
    print("Running inference...")
    batch = next(iter(val_loader))
    images, targets, metas = batch
    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        _, _, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
        logits = sem_head(p1, p2, p3, p4)

    loss_total, loss_ce, loss_dice = compute_loss(logits, targets)
    print(f"Loss: total={loss_total.item():.4f}, ce={loss_ce.item():.4f}, dice={loss_dice.item():.4f}")

    # Per-sample metrics
    for i in range(NUM_SAMPLES):
        m = compute_observed_pixel_metrics(logits[i:i+1], targets[i:i+1], num_classes)
        print(f"  Sample {i}: mIoU={m['mIoU']:.4f}, pix_acc={m['pixel_accuracy']:.4f}, "
              f"classes={list(m['per_class_IoU'].keys())}")

    # ---- Export overlays ----
    print("Exporting overlays...")
    # Build correct meta structure for _save_overlays
    _save_overlays(images, targets, logits, metas, epoch=5, batch_idx=0,
                   output_dir=OUTPUT_DIR / "overlays", num_samples=NUM_SAMPLES)
    print(f"  Overlays saved to {OUTPUT_DIR / 'overlays'}")

    # ---- Export GeoJSON ----
    print("Exporting GeoJSON...")
    _export_geojson_samples(logits, metas, epoch=5, batch_idx=0,
                            output_dir=OUTPUT_DIR / "geojson", num_samples=NUM_SAMPLES)
    print(f"  GeoJSON saved to {OUTPUT_DIR / 'geojson'}")

    # ---- Summary ----
    print(f"\nExport verification complete.")
    print(f"  Overlays: {list((OUTPUT_DIR / 'overlays').glob('*.png'))}")
    print(f"  GeoJSON:  {list((OUTPUT_DIR / 'geojson').glob('*.geojson'))}")


if __name__ == "__main__":
    main()

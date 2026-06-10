#!/usr/bin/env python3
"""Quick stratified eval: 100 samples per size, unbuffered output."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# CRITICAL: disable stdout buffering so progress is visible
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
from Dataset.landcover_label_map import train_id_to_dlmc, dlmc_to_train_id, IGNORE_INDEX
from scripts.poc_stage_semantic import compute_observed_pixel_metrics, clean_vision_state_dict
from ml_collections import ConfigDict

OUTPUT_DIR = Path("outputs/poc2_stratified_eval")
NUM_PER_SIZE = 100
THIN_CLASSES = ["沟渠", "农村道路", "城镇村道路用地"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    by_size = defaultdict(list)
    for s in val_samples:
        sz = tuple(s["original_size"])
        by_size[sz].append(s)
    for sz, samples in sorted(by_size.items()):
        print(f"  Val size {sz}: {len(samples)} total, using {min(NUM_PER_SIZE, len(samples))}")

    # ---- Model ----
    print("Building model...")
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
    print("  Initializing DualVisionEncoder...")
    vision = DualVisionEncoder(vis_cfg)
    print("  Loading stage2 checkpoint (3.3GB)...")
    ckpt = torch.load("./output/stage2/checkpoints/iter_2879_consolidated.pt", map_location="cpu")
    sd = ckpt.get("vision_ckpt", ckpt.get("model", ckpt))
    sd = clean_vision_state_dict(sd)
    vision.load_state_dict(sd, strict=False)
    print("  Moving to NPU...")
    vision = vision.to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False
    print("  Vision encoder ready.")

    fpn = FPNNeck(in_channels=[128, 256, 512, 1024], out_channels=256).to(device)
    sem_head = LandcoverSemanticHead(in_channels=256, num_classes=num_classes, output_size=(224, 224)).to(device)
    sem_ckpt = torch.load("outputs/poc2_small_scale/checkpoints/small_scale_final.pt", map_location="cpu")
    fpn.load_state_dict(sem_ckpt["fpn"])
    sem_head.load_state_dict(sem_ckpt["sem_head"])
    fpn.eval()
    sem_head.eval()
    print("  Model ready.")

    # =====================================================================
    # Size-stratified evaluation
    # =====================================================================
    print("\n" + "=" * 60)
    print("Size-Stratified Evaluation")
    print("=" * 60)

    all_results = {}

    for sz, samples in sorted(by_size.items()):
        subset = samples[:NUM_PER_SIZE]
        ds = LandcoverSemanticDataset(subset)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=8, shuffle=False, collate_fn=landcover_collate_fn,
        )

        size_metrics = []
        per_class_metrics = defaultdict(list)
        n_batches = len(loader)
        print(f"\n  Size {sz[0]}x{sz[1]}: {len(subset)} samples, {n_batches} batches")

        with torch.no_grad():
            for batch_idx, (images, targets, metas) in enumerate(loader):
                images = images.to(device)
                targets = targets.to(device)

                _, _, pyramid_raw = vision.encode_with_spatial(images)
                c4, c8, c16, c32 = pyramid_raw
                p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
                logits = sem_head(p1, p2, p3, p4)

                for i in range(images.shape[0]):
                    m = compute_observed_pixel_metrics(logits[i:i+1], targets[i:i+1], num_classes)
                    size_metrics.append(m)
                    for cls_name, iou_val in m["per_class_IoU"].items():
                        per_class_metrics[cls_name].append(iou_val)

                if (batch_idx + 1) % 5 == 0:
                    print(f"    Batch {batch_idx+1}/{n_batches}")

        n = len(size_metrics)
        miou = float(np.mean([m["mIoU"] for m in size_metrics]))
        pix_acc = float(np.mean([m["pixel_accuracy"] for m in size_metrics]))

        cls_summary = {}
        for cls_name, ious in sorted(per_class_metrics.items()):
            cls_summary[cls_name] = {
                "mean_IoU": float(np.mean(ious)),
                "count": len(ious),
            }

        all_results[f"{sz[0]}x{sz[1]}"] = {
            "n_samples": n,
            "mIoU": miou,
            "pixel_accuracy": pix_acc,
            "per_class": cls_summary,
        }

        print(f"  -> mIoU={miou:.4f}, pix_acc={pix_acc:.4f}")
        for cls_name in THIN_CLASSES:
            if cls_name in cls_summary:
                print(f"     {cls_name}: IoU={cls_summary[cls_name]['mean_IoU']:.4f} "
                      f"(n={cls_summary[cls_name]['count']})")

    # =====================================================================
    # High-res ablation (label-only), quick version on 30 samples
    # =====================================================================
    print("\n" + "=" * 60)
    print("High-Res Rasterization Ablation (label-only, 30 samples)")
    print("=" * 60)

    size_512_samples = by_size.get((512, 512), [])[:30]
    if not size_512_samples:
        for sz, samples in by_size.items():
            if 512 in sz:
                size_512_samples = samples[:30]
                break

    if size_512_samples:
        from PIL import Image
        from Dataset.rasterize_geojson import rasterize_features_to_target

        per_class_ious = defaultdict(list)
        pixel_change_rates = []

        for idx, sample in enumerate(size_512_samples):
            features = sample["features"]
            bounds = tuple(sample["tile_bounds_wgs84"])

            target_224, _ = rasterize_features_to_target(features, bounds, (224, 224))
            target_224_to_512 = np.array(
                Image.fromarray(target_224.astype(np.uint8)).resize((512, 512), Image.NEAREST),
                dtype=np.int64,
            )
            target_512, _ = rasterize_features_to_target(features, bounds, (512, 512))

            for c in range(1, num_classes):
                gt_c = (target_512 == c)
                pred_c = (target_224_to_512 == c)
                if gt_c.sum() == 0 and pred_c.sum() == 0:
                    continue
                intersection = (gt_c & pred_c).sum()
                union = (gt_c | pred_c).sum()
                per_class_ious[c].append(intersection / max(union, 1))

            both_valid = (target_512 != IGNORE_INDEX) | (target_224_to_512 != IGNORE_INDEX)
            if both_valid.sum() > 0:
                changed = (target_512[both_valid] != target_224_to_512[both_valid]).sum()
                pixel_change_rates.append(changed / both_valid.sum())

        print(f"  Overall pixel disagreement: {np.mean(pixel_change_rates)*100:.2f}%")

        thin_results = {}
        for cls_name in THIN_CLASSES:
            train_id = dlmc_to_train_id(cls_name)
            if train_id and train_id in per_class_ious:
                ious = per_class_ious[train_id]
                print(f"  {cls_name}: 224-vs-512 IoU = {np.mean(ious):.4f} (n={len(ious)})")
                thin_results[cls_name] = {
                    "mean_IoU_224_vs_512": float(np.mean(ious)),
                    "n_samples": len(ious),
                }

        ablation_results = {
            "overall_pixel_disagreement_rate": float(np.mean(pixel_change_rates)),
            "thin_class_analysis": thin_results,
            "per_class_comparison": [
                {"class_name": train_id_to_dlmc(c), "mean_IoU_224_vs_512": float(np.mean(ious)), "n_samples": len(ious)}
                for c, ious in sorted(per_class_ious.items(), key=lambda x: np.mean(x[1]))
            ],
        }
    else:
        ablation_results = {"error": "No 512-size samples available"}
        print("  No 512-size samples available.")

    # ---- Save ----
    stratified_path = OUTPUT_DIR / "size_stratified_metrics.json"
    with open(stratified_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if size_512_samples:
        ablation_path = OUTPUT_DIR / "rasterization_ablation_512.json"
        with open(str(ablation_path), "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, ensure_ascii=False, indent=2)

    metrics_json = {
        "by_original_size": all_results,
        "rasterization_ablation_512": ablation_results,
    }
    with open(str(OUTPUT_DIR / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PoC-2 Step 2: GF6 Hard Example Oversampling Fine-tuning.

Loads epoch 40 checkpoint, creates weighted sampler that oversamples
GF6 + dense tiles, fine-tunes for 20 epochs with reduced LR.

Key changes from poc_stage_one_det.py:
  - WeightedRandomSampler: GF6 ×3 weight, density bonus
  - Resume from checkpoint
  - Lower LR (5e-5 vs 1e-4)
  - RPN proposal recall diagnostic in validation
"""
import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dataset.aqua_poc_dataset import AquaPoCDataset, poc_collate_fn
from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn
from Models.dual_vision_encoder import DualVisionEncoder
# Lightweight validation only — no overlay/GeoJSON export during training

# Reuse helper functions from poc_stage_one_det
import importlib.util as _u
_s = _u.spec_from_file_location("poc", _REPO_ROOT / "scripts" / "poc_stage_one_det.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
load_config = _m.load_config
build_vision_encoder = _m.build_vision_encoder
resolve_device = _m.resolve_device
smoke_test_npu_ops = _m.smoke_test_npu_ops
smoke_test_maskrcnn = _m.smoke_test_maskrcnn
save_checkpoint = _m.save_checkpoint
train_epoch = _m.train_epoch
validate = _m.validate


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def compute_sample_weights(manifest_path: str, gf6_weight: float = 3.0,
                           density_power: float = 0.5) -> List[float]:
    """Compute per-sample weights for WeightedRandomSampler.

    weight = sensor_weight * (1 + density_power * n_gt / max_n_gt)

    Args:
        manifest_path: path to train.json
        gf6_weight: base weight multiplier for GF6 tiles
        density_power: strength of density bonus
    """
    with open(manifest_path) as f:
        samples = json.load(f)

    max_gt = max(s.get("num_features", 1) for s in samples)
    weights = []
    for s in samples:
        sensor = s.get("sensor", "")
        base = gf6_weight if "GF6" in sensor else 1.0
        n_gt = s.get("num_features", 1)
        density_bonus = 1.0 + density_power * (n_gt / max(max_gt, 1))
        weights.append(base * density_bonus)

    return weights


@torch.no_grad()
def validate_lightweight(
    model, dataloader, device, epoch,
    score_thresh=0.5, rpn_samples: int = 200,
) -> dict:
    """Lightweight validation: box counts + RPN proposal recall. No overlay images."""
    model.eval()

    total_gt_boxes = 0
    total_pred_boxes = 0
    num_tiles = 0

    # Full val set: count boxes
    for images, targets, _metas in dataloader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        total_gt_boxes += sum(len(t["boxes"]) for t in targets)
        total_pred_boxes += sum((o["scores"] >= score_thresh).sum().item() for o in outputs)
        num_tiles += len(images)

    # RPN proposal recall on subset
    rpn_stats = {"gf6": {"total_gt": 0, "covered": 0},
                 "non_gf6": {"total_gt": 0, "covered": 0}}
    n_done = 0

    for images, targets, metas in dataloader:
        if n_done >= rpn_samples:
            break
        for img_t, tgt, meta in zip(images, targets, metas):
            if n_done >= rpn_samples:
                break
            gt_boxes = tgt["boxes"].numpy()
            if len(gt_boxes) == 0:
                continue

            img_dev = img_t.to(device)
            images_list, _ = model.transform([img_dev], None)
            features = model.backbone(images_list.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            proposals, _ = model.rpn(images_list, features, None)
            props = proposals[0].cpu().numpy()

            covered = 0
            for gt in gt_boxes:
                for prop in props:
                    if compute_iou(gt, prop) >= 0.5:
                        covered += 1
                        break

            is_gf6 = "GF6" in meta.get("sensor", "")
            key = "gf6" if is_gf6 else "non_gf6"
            rpn_stats[key]["total_gt"] += len(gt_boxes)
            rpn_stats[key]["covered"] += covered
            n_done += 1

    for key in rpn_stats:
        s = rpn_stats[key]
        s["recall"] = s["covered"] / max(s["total_gt"], 1)

    print(f"  Val epoch {epoch}: GT={total_gt_boxes}, Pred={total_pred_boxes}, Tiles={num_tiles}")
    print(f"  RPN diag (subset of {n_done} tiles):")
    for key, label in [("gf6", "GF6"), ("non_gf6", "non-GF6")]:
        s = rpn_stats[key]
        print(f"    {label}: RPN rec={s['recall']:.4f} ({s['covered']}/{s['total_gt']})")

    return {"total_gt_boxes": total_gt_boxes, "total_pred_boxes": total_pred_boxes,
            "num_tiles": num_tiles, "rpn_diag": rpn_stats, "epoch": epoch}


def main():
    parser = argparse.ArgumentParser(description="PoC-2 Step 2: GF6 Oversampling")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "poc_aqua_instance.yaml"))
    parser.add_argument("--device", default="npu")
    parser.add_argument("--resume", default="outputs/poc_aqua_full/checkpoints/epoch_040.pt")
    parser.add_argument("--gf6-weight", type=float, default=3.0,
                        help="Weight multiplier for GF6 tiles")
    parser.add_argument("--density-power", type=float, default=0.5,
                        help="Density bonus strength (0=off)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    output_dir = Path(cfg["experiment"]["output_dir"])
    poc2_dir = output_dir.parent / "poc2_gf6_oversample"
    poc2_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {poc2_dir}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model (same as PoC-1)
    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"]).to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)
    adapter = DualVisionFPNBackboneAdapter(vision, fpn).to(device)

    mrcnn_cfg = cfg["model"]["mask_rcnn"]
    model = build_aqua_maskrcnn(
        adapter, num_classes=mrcnn_cfg["num_classes"],
        anchor_sizes=tuple(tuple(s) for s in cfg["model"]["anchors"]["sizes"]),
        aspect_ratios=tuple(tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"]),
        rpn_pre_nms_top_n_train=mrcnn_cfg.get("rpn_pre_nms_top_n_train", 512),
        rpn_post_nms_top_n_train=mrcnn_cfg.get("rpn_post_nms_top_n_train", 128),
        rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
        rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
        rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
        box_score_thresh=mrcnn_cfg.get("box_score_thresh", 0.05),
        box_nms_thresh=mrcnn_cfg.get("box_nms_thresh", 0.5),
        box_detections_per_img=mrcnn_cfg.get("box_detections_per_img", 50),
        image_mean=mrcnn_cfg.get("image_mean", [0.0, 0.0, 0.0]),
        image_std=mrcnn_cfg.get("image_std", [1.0, 1.0, 1.0]),
        min_size=mrcnn_cfg.get("min_size", 224),
        max_size=mrcnn_cfg.get("max_size", 224),
    ).to(device)

    # Load checkpoint
    resume_path = args.resume
    print(f"Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    fpn.load_state_dict(ckpt["fpn_state_dict"])
    model.load_state_dict(ckpt["maskrcnn_state_dict"])
    start_epoch = ckpt.get("epoch", 40)

    fpn_params = sum(p.numel() for p in fpn.parameters())
    head_params = sum(p.numel() for p in model.parameters()) - fpn_params - sum(p.numel() for p in vision.parameters())
    print(f"FPN: {fpn_params:,}  |  Head: {head_params:,}")

    # Data
    data_cfg = cfg["data"]
    train_manifest = str(_REPO_ROOT / data_cfg["train_manifest"])
    val_manifest = str(_REPO_ROOT / data_cfg["val_manifest"])
    data_root = data_cfg.get("data_root", "/home/ma-user/work/Stage3Data/养殖区")

    train_ds = AquaPoCDataset(
        manifest_path=train_manifest, data_root=data_root,
        image_size=data_cfg.get("image_size", 224),
    )
    val_ds = AquaPoCDataset(
        manifest_path=val_manifest, data_root=data_root,
        image_size=data_cfg.get("image_size", 224),
    )
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Weighted sampler
    weights = compute_sample_weights(train_manifest, args.gf6_weight, args.density_power)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
    )
    print(f"Sample weights: GF6 ×{args.gf6_weight}, density_power={args.density_power}")
    gf6_w = [w for w, s in zip(weights, train_ds.samples) if "GF6" in s.get("sensor", "")]
    non_w = [w for w, s in zip(weights, train_ds.samples) if "GF6" not in s.get("sensor", "")]
    print(f"  Weight range: [{min(weights):.2f}, {max(weights):.2f}]")
    print(f"  GF6 mean weight: {np.mean(gf6_w):.2f}  |  non-GF6 mean weight: {np.mean(non_w):.2f}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=poc_collate_fn, sampler=sampler, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=data_cfg.get("num_workers", 2), collate_fn=poc_collate_fn,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0001)
    # Load optimizer state if available
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint")
        except Exception:
            print("Could not load optimizer state (LR changed), starting fresh")

    num_epochs = args.epochs
    print(f"\n{'='*60}")
    print(f"PoC-2 Step 2: GF6 Oversampling Fine-tuning")
    print(f"Resume epoch: {start_epoch}  |  Additional epochs: {num_epochs}")
    print(f"LR: {args.lr}  |  GF6 weight: {args.gf6_weight}")
    print(f"Output: {poc2_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        actual_epoch = start_epoch + epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, device, actual_epoch,
            log_interval=cfg["train"].get("log_interval", 10),
            max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            val_stats = validate_lightweight(
                model, val_loader, device, actual_epoch,
                score_thresh=0.5, rpn_samples=200,
            )
            save_checkpoint(fpn, model, optimizer, actual_epoch, str(poc2_dir))

    print(f"\nDone. Outputs in {poc2_dir}")


if __name__ == "__main__":
    main()

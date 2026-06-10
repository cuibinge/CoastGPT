#!/usr/bin/env python3
"""
Quick test: GF6 RPN recall at different input sizes (224, 320, 384, 448).
Uses existing epoch 40 checkpoint without any retraining.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Models.det_head import DualVisionFPNBackboneAdapter, FPNNeck, build_aqua_maskrcnn
from utils.mask_utils import binary_mask_to_instances

import importlib.util as _u
_s = _u.spec_from_file_location("poc", _REPO_ROOT / "scripts" / "poc_stage_one_det.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
build_vision_encoder = _m.build_vision_encoder
load_config = _m.load_config


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def main():
    device = torch.device("npu:0")
    cfg = load_config("configs/poc_aqua_instance.yaml")

    vis_cfg = ConfigDict({
        "rgb_vision": cfg["model"]["rgb_vision"],
        "alignment_dim": cfg["model"].get("alignment_dim", 768),
    })
    vision = build_vision_encoder(vis_cfg, cfg["model"]["vision_checkpoint"]).to(device)
    fpn = FPNNeck(
        in_channels=cfg["model"]["fpn"]["in_channels"],
        out_channels=cfg["model"]["fpn"]["out_channels"],
    ).to(device)
    adapter = DualVisionFPNBackboneAdapter(vision, fpn).to(device)

    mrcnn_cfg = cfg["model"]["mask_rcnn"]
    anchor_sizes = tuple(tuple(s) for s in cfg["model"]["anchors"]["sizes"])
    aspect_ratios = tuple(tuple(a) for a in cfg["model"]["anchors"]["aspect_ratios"])

    with open("data/poc_aqua_full/val.json") as f:
        manifest = json.load(f)
    gf6_samples = [s for s in manifest if s.get("sensor") == "GF6"]
    print(f"GF6 tiles: {len(gf6_samples)}")

    data_root = Path("/home/ma-user/work/Stage3Data/养殖区")

    input_sizes = [224, 320, 384, 448]
    results = {}

    for imsize in input_sizes:
        print(f"\n{'='*60}")
        print(f"Input size: {imsize}")
        print(f"{'='*60}")

        # Build model with this input size
        model = build_aqua_maskrcnn(
            adapter, num_classes=mrcnn_cfg["num_classes"],
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            rpn_pre_nms_top_n_test=mrcnn_cfg.get("rpn_pre_nms_top_n_test", 256),
            rpn_post_nms_top_n_test=mrcnn_cfg.get("rpn_post_nms_top_n_test", 64),
            rpn_nms_thresh=mrcnn_cfg.get("rpn_nms_thresh", 0.7),
            box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
            image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
            min_size=imsize, max_size=imsize,
        ).to(device)

        ckpt = torch.load("outputs/poc_aqua_full/checkpoints/epoch_040.pt", map_location=device)
        fpn.load_state_dict(ckpt["fpn_state_dict"])
        model.load_state_dict(ckpt["maskrcnn_state_dict"])
        model.eval()

        total_gt = 0
        covered = 0
        n_tiles = 0

        for sample in gf6_samples:
            img_path = data_root / sample["image_path"]
            binary_path = (
                data_root / sample["binary_label_path"]
                if sample.get("binary_label_path") else None
            )
            if not binary_path or not binary_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            image_resized = image.resize((imsize, imsize), Image.BILINEAR)
            img_t = torch.from_numpy(np.array(image_resized, dtype=np.float32) / 255.0)
            img_t = img_t.permute(2, 0, 1).to(device)

            binary = Image.open(binary_path).convert("L")
            binary_resized = binary.resize((imsize, imsize), Image.NEAREST)
            binary_np = np.array(binary_resized)
            _, gt_boxes, _ = binary_mask_to_instances(binary_np, min_area=8, connectivity=4)
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            if len(gt_boxes) == 0:
                continue

            with torch.no_grad():
                images_list, _ = model.transform([img_t], None)
                features = model.backbone(images_list.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])
                proposals, _ = model.rpn(images_list, features, None)
            props = proposals[0].cpu().numpy()

            for gt in gt_boxes:
                total_gt += 1
                for prop in props:
                    if compute_iou(gt, prop) >= 0.5:
                        covered += 1
                        break
            n_tiles += 1

            if (n_tiles) % 50 == 0:
                rec = covered / max(total_gt, 1)
                print(f"  [{n_tiles}/{len(gf6_samples)}] RPN rec={rec:.4f} ({covered}/{total_gt})")

        rec = covered / max(total_gt, 1)
        results[imsize] = {"total_gt": total_gt, "covered": covered, "recall": rec}
        print(f"  FINAL: RPN recall={rec:.4f} ({covered}/{total_gt}) on {n_tiles} tiles")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: GF6 RPN Recall vs Input Size")
    print(f"{'='*60}")
    print(f"{'Size':>8} {'RPN Recall':>12} {'Coverage':>15}")
    print("-" * 38)
    for imsize in input_sizes:
        r = results[imsize]
        print(f"{imsize:>8} {r['recall']:>12.4f} {r['covered']:>6}/{r['total_gt']:>6}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PoC-4: LLM Fallback + Fusion — Single-image inference script.

Loads separate checkpoints for each predictor, builds FusionPipeline,
runs inference on a single image, and saves the merged GeoJSON.

Usage:
    python scripts/poc_stage_fusion.py \
        -c Configs/poc4_fusion.yaml \
        --image-file path/to/image.jpg \
        --prompt "[DET] 请检测图中的海岸线和海水养殖区" \
        --output result.json \
        --georef path/to/georef.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image
from Models.coastgpt import CoastGPT
from Models.dual_vision_encoder import DualVisionEncoder
from Models.edge_head import SingleScaleEdgeHead
from Models.fpn_neck import FPNNeck
from Models.fusion_pipeline import FusionConfig, FusionPipeline
from Models.fusion_predictors import (
    EdgePredictor,
    InstancePredictor,
    LLMPredictor,
    LLMTextPredictor,
    SemanticPredictor,
)
from Models.semantic_head import LandcoverSemanticHead
from Models.det_head import DualVisionFPNBackboneAdapter, build_aqua_maskrcnn
from Dataset.build_transform import build_vlp_transform
from Trainer.utils.config_parser import ConfigArgumentParser


def _load_vision_encoder(config: dict, device: torch.device, dtype: torch.dtype) -> DualVisionEncoder:
    """Load frozen DualVisionEncoder from checkpoint."""
    encoder = DualVisionEncoder(config)
    ckpt_path = config.get("checkpoints", {}).get("vision_ckpt")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Handle structured checkpoint format
        if "vision_ckpt" in ckpt:
            vision_state = ckpt["vision_ckpt"]
        elif "model" in ckpt:
            vision_state = ckpt["model"]
        else:
            vision_state = ckpt
        # Strip prefixes
        cleaned = {}
        for k, v in vision_state.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("vision."):
                new_k = new_k[len("vision."):]
            cleaned[new_k] = v
        encoder.load_state_dict(cleaned, strict=False)
        print(f"Loaded vision encoder from {ckpt_path}")
    encoder.to(device).to(dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _load_label_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = ConfigArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="poc4_output.json")
    parser.add_argument("--georef", type=str, default=None, help="Path to georef JSON")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--save-diagnostics", type=str, default=None)
    args = parser.parse_args(wandb=False)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.float16

    # --- Load vision encoder (shared) ---
    print("Loading vision encoder...")
    vision_encoder = _load_vision_encoder(cfg, device, dtype)

    # --- Load FPN (shared) ---
    print("Loading FPN...")
    fpn_cfg = cfg.get("model", {}).get("fpn", {})
    fpn = FPNNeck(
        in_channels=fpn_cfg.get("in_channels", [128, 256, 512, 1024]),
        out_channels=fpn_cfg.get("out_channels", 256),
    )
    fpn.to(device).to(dtype)

    # --- Load label_map ---
    label_map_path = cfg.get("fusion", {}).get("gating", {}).get("label_map_path", "Configs/label_map.json")
    label_map = _load_label_map(label_map_path)

    pred_cfg = cfg.get("predictors", {})

    # --- Build InstancePredictor ---
    print("Building InstancePredictor...")
    inst_ckpt = cfg.get("checkpoints", {}).get("instance_ckpt")
    adapter = DualVisionFPNBackboneAdapter(vision_encoder, fpn)
    mask_rcnn = build_aqua_maskrcnn(adapter, num_classes=2)
    if inst_ckpt and Path(inst_ckpt).exists():
        ckpt = torch.load(inst_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        mask_rcnn.load_state_dict(state, strict=False)
        print(f"  Loaded instance head from {inst_ckpt}")
    mask_rcnn.to(device).to(dtype)
    inst_cfg = pred_cfg.get("instance", {})
    instance_predictor = InstancePredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        mask_rcnn=mask_rcnn,
        score_thresh=inst_cfg.get("score_thresh", 0.5),
        mask_thresh=inst_cfg.get("mask_thresh", 0.5),
        min_area_px=inst_cfg.get("min_area_px", 8.0),
        class_name=inst_cfg.get("class_name", "海水养殖区"),
        device=args.device,
    )

    # --- Build SemanticPredictor ---
    print("Building SemanticPredictor...")
    sem_ckpt = cfg.get("checkpoints", {}).get("semantic_ckpt")
    sem_class_names = {}
    sem_classes = label_map.get("branch_classes", {}).get("semantic", {})
    for idx_str, name in sem_classes.items():
        if name and name != "background":
            sem_class_names[int(idx_str)] = name
    num_sem_classes = max(sem_class_names.keys()) + 1 if sem_class_names else 25
    sem_head = LandcoverSemanticHead(
        in_channels=256,
        num_classes=num_sem_classes,
    )
    if sem_ckpt and Path(sem_ckpt).exists():
        ckpt = torch.load(sem_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        sem_head.load_state_dict(state, strict=False)
        print(f"  Loaded semantic head from {sem_ckpt}")
    sem_head.to(device).to(dtype)
    sem_cfg = pred_cfg.get("semantic", {})
    semantic_predictor = SemanticPredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        semantic_head=sem_head,
        class_names=sem_class_names,
        min_area_px=sem_cfg.get("min_area_px", 50.0),
        simplify_epsilon=sem_cfg.get("simplify_epsilon", 1.0),
        device=args.device,
    )

    # --- Build EdgePredictor ---
    print("Building EdgePredictor...")
    edge_ckpt = cfg.get("checkpoints", {}).get("edge_ckpt")
    edge_head = SingleScaleEdgeHead(output_size=(224, 224))
    if edge_ckpt and Path(edge_ckpt).exists():
        ckpt = torch.load(edge_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        edge_head.load_state_dict(state, strict=False)
        print(f"  Loaded edge head from {edge_ckpt}")
    edge_head.to(device).to(dtype)
    edge_cfg = pred_cfg.get("edge", {})
    edge_predictor = EdgePredictor(
        vision_encoder=vision_encoder,
        fpn_neck=fpn,
        edge_head=edge_head,
        threshold=edge_cfg.get("threshold", 0.6),
        min_area=edge_cfg.get("min_area", 8),
        min_length=edge_cfg.get("min_length", 10),
        max_components=edge_cfg.get("max_components", 5),
        simplify_epsilon=edge_cfg.get("simplify_epsilon", 0.1),
        class_name=edge_cfg.get("class_name", "海岸线"),
        device=args.device,
    )

    # --- Build LLM Predictors ---
    print("Loading CoastGPT for LLM predictors...")
    llm_ckpt = cfg.get("checkpoints", {}).get("llm_ckpt")
    # Use ml_collections ConfigDict for CoastGPT constructor
    from ml_collections import ConfigDict
    coastgpt_config = ConfigDict(cfg.get("model", {}))
    coastgpt_config.accelerator = cfg.get("accelerator", "npu")
    coastgpt_config.stage = 0  # eval mode

    coastgpt = CoastGPT(coastgpt_config)
    if llm_ckpt and Path(llm_ckpt).exists():
        ckpt = torch.load(llm_ckpt, map_location="cpu")
        if hasattr(coastgpt, "custom_load_state_dict"):
            coastgpt.custom_load_state_dict(llm_ckpt)
        else:
            state = ckpt.get("model", ckpt)
            coastgpt.load_state_dict(state, strict=False)
        print(f"  Loaded LLM from {llm_ckpt}")
    coastgpt.to(device).to(dtype)
    coastgpt.eval()
    tokenizer = coastgpt.language.tokenizer

    llm_cfg = pred_cfg.get("llm", {})
    llm_predictor = LLMPredictor(
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        config=coastgpt_config,
        max_new_tokens=llm_cfg.get("max_new_tokens", 1024),
        device=args.device,
    )

    text_cfg = pred_cfg.get("llm_text", {})
    llm_text_predictor = LLMTextPredictor(
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        config=coastgpt_config,
        max_new_tokens=text_cfg.get("max_new_tokens", 512),
        device=args.device,
    )

    # --- Build FusionPipeline ---
    fusion_cfg_raw = cfg.get("fusion", {})
    fusion_config = FusionConfig(
        parser_max_new_tokens=fusion_cfg_raw.get("parser", {}).get("max_new_tokens", 128),
        parser_confidence_threshold=fusion_cfg_raw.get("parser", {}).get("confidence_threshold", 0.3),
        label_map_path=fusion_cfg_raw.get("gating", {}).get("label_map_path", "Configs/label_map.json"),
        empty_target_policy=fusion_cfg_raw.get("gating", {}).get("empty_target_policy", "error"),
        sliver_min_area_deg=fusion_cfg_raw.get("validation", {}).get("sliver_min_area_deg", 1e-10),
        sliver_min_length_deg=fusion_cfg_raw.get("validation", {}).get("sliver_min_length_deg", 1e-6),
        llm_min_confidence=fusion_cfg_raw.get("validation", {}).get("llm_min_confidence", 0.3),
        dedup_internal_thresholds=fusion_cfg_raw.get("dedup", {}).get("internal_thresholds", {}),
        dedup_cross_thresholds=fusion_cfg_raw.get("dedup", {}).get("cross_thresholds", {}),
    )

    pipeline = FusionPipeline(
        predictors={
            "instance": instance_predictor,
            "semantic": semantic_predictor,
            "edge": edge_predictor,
            "llm": llm_predictor,
            "llm_text": llm_text_predictor,
        },
        config=fusion_config,
        label_map=label_map,
        coastgpt_model=coastgpt,
        tokenizer=tokenizer,
        device=args.device,
    )

    # --- Load image ---
    print(f"Loading image: {args.image_file}")
    image = Image.open(args.image_file).convert("RGB")
    transform = build_vlp_transform(coastgpt_config, is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)

    # --- Load georef ---
    georef = {"source_crs": "EPSG:4326"}
    if args.georef:
        with open(args.georef, "r") as f:
            georef.update(json.load(f))

    # --- Run fusion ---
    print(f"Running fusion with prompt: {args.prompt}")
    fc, diagnostics = pipeline.run(image_tensor, args.prompt, georef)

    # --- Save output ---
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"Saved GeoJSON ({len(fc.get('features', []))} features) to {args.output}")

    # --- Save diagnostics ---
    if args.save_diagnostics:
        with open(args.save_diagnostics, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved diagnostics to {args.save_diagnostics}")

    # --- Print summary ---
    print("\n--- Fusion Summary ---")
    print(f"  Prefix: {diagnostics.get('prefix')}")
    parse_info = diagnostics.get("parse", {})
    print(f"  Parse: task={parse_info.get('task_type')}, classes={parse_info.get('target_classes')}, source={parse_info.get('source')}")
    dispatch_info = diagnostics.get("dispatch", {})
    known_info = dispatch_info.get("known", {})
    for branch, classes in known_info.items():
        if classes:
            print(f"  Known [{branch}]: {classes}")
    unknown_info = dispatch_info.get("unknown", [])
    if unknown_info:
        print(f"  Unknown: {unknown_info}")
    print(f"  Det features: {diagnostics.get('det_feature_count', 0)}")
    print(f"  LLM features (before dedup): {diagnostics.get('llm_feature_count_before_dedup', 0)}")
    dedup_cross = diagnostics.get("dedup_cross", {})
    print(f"  Final features: {diagnostics.get('final_feature_count', 0)}")
    print(f"  Cross-dedup removed: {dedup_cross.get('n_cross_removed', 0)}")


if __name__ == "__main__":
    main()

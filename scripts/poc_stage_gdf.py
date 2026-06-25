#!/usr/bin/env python3
"""
PoC-3 A3: Gravitational Distance Field Training Script.

Single-NPU entry point for A3 experiments:
  - A3-0: GDF from scratch
  - A3-1: A2 warm-start FPN + GDF head

Usage:
    python scripts/poc_stage_gdf.py -c configs/poc3_a3_1_gdf_warmstart.yaml --device npu
    python scripts/poc_stage_gdf.py -c configs/poc3_a3_1_gdf_warmstart.yaml --overfit 2 --epochs 1
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.gdf_head import SingleScaleGDFHead
from Dataset.coastline_dataset import (
    CoastlineEdgeDataset,
    coastline_collate_fn,
    build_coastline_manifest,
)
from utils.gdf_target import build_gdf_target
from utils.gdf_losses import gdf_loss
from utils.gdf_postprocess import endpoint_voting, decode_field
from utils.coastline_metrics import compute_all_edge_metrics


# =============================================================================
# Config loading
# =============================================================================


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, "r", encoding="utf-8") as f:
        return ConfigDict(yaml.safe_load(f))


# =============================================================================
# Vision encoder
# =============================================================================


def clean_vision_sd(state_dict: dict) -> dict:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("vision."):
            nk = nk[len("vision."):]
        cleaned[nk] = v
    return cleaned


def build_vision(cfg: ConfigDict) -> DualVisionEncoder:
    print("Building DualVisionEncoder...")
    model_cfg = ConfigDict(cfg.get("model", {}))
    vision = DualVisionEncoder(model_cfg)
    ckpt_path = cfg.get("model.vision_checkpoint",
                         "./output/stage2/checkpoints/iter_2879_consolidated.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("vision_ckpt", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    vision.load_state_dict(clean_vision_sd(sd), strict=False)
    return vision


# =============================================================================
# Dataset wrapper with GDF targets
# =============================================================================


class GDFDataset(torch.utils.data.Dataset):
    """Wraps CoastlineEdgeDataset to produce GDF targets instead of edge maps."""

    def __init__(self, tiles, max_radius=32, image_size=224):
        self.edge_ds = CoastlineEdgeDataset(
            tiles, line_width=1,  # width=1 centerline for field source
            soft_edge_sigma=0.0, soft_edge_radius=0,
            image_size=image_size,
        )
        self.max_radius = max_radius

    def __len__(self):
        return len(self.edge_ds)

    def __getitem__(self, idx):
        sample = self.edge_ds[idx]
        image = sample["image"]          # [3, H, W]
        edge_center = sample["target"][0].numpy()  # [H, W]

        gdf_target, loss_mask = build_gdf_target(edge_center, self.max_radius)

        return {
            "image": image,
            "gdf_target": torch.from_numpy(gdf_target),
            "loss_mask": torch.from_numpy(loss_mask),
            "meta": sample["meta"],
        }


def gdf_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch])
    targets = torch.stack([x["gdf_target"] for x in batch])
    masks = torch.stack([x["loss_mask"] for x in batch])
    metas = [x["meta"] for x in batch]
    return images, targets, masks, metas


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    fpn: FPNNeck,
    gdf_head: SingleScaleGDFHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str,
    best_f1: float = 0.0,
):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "fpn": fpn.state_dict(),
        "gdf_head": gdf_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1,
    }, str(ckpt_path))
    print(f"Checkpoint saved to {ckpt_path}")


# =============================================================================
# Training
# =============================================================================


def train_epoch(
    fpn: FPNNeck,
    gdf_head: SingleScaleGDFHead,
    vision: DualVisionEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
    max_grad_norm: float = 1.0,
    lambda_vec: float = 1.0,
    lambda_dist: float = 0.5,
    lambda_valid: float = 0.5,
    lambda_cons: float = 0.0,
) -> float:
    vision.eval()
    fpn.train()
    gdf_head.train()

    total_loss = 0.0
    steps = 0

    for batch_idx, (images, targets, masks, metas) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            _, g_grid, pyramid_raw = vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if fpn.has_vit else None
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=vit_feat)
        out = gdf_head(p1, p2, p3, p4)
        field_pred = out["field"]

        loss_dict = gdf_loss(
            field_pred, targets, loss_mask=masks,
            lambda_vec=lambda_vec, lambda_dist=lambda_dist,
            lambda_valid=lambda_valid, lambda_cons=lambda_cons,
        )
        loss = loss_dict["total"]
        loss.backward()

        if max_grad_norm > 0:
            trainable = list(fpn.parameters()) + list(gdf_head.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / steps
            print(
                f"Epoch {epoch:3d} | Step {batch_idx+1:4d} | "
                f"Loss: {avg:.4f} | vec={loss_dict['loss_vec'].item():.3f} "
                f"dist={loss_dict['loss_dist'].item():.3f} "
                f"valid={loss_dict['loss_valid'].item():.3f}"
            )

    avg_loss = total_loss / max(steps, 1)
    print(f"Epoch {epoch:3d} done | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# =============================================================================
# Validation
# =============================================================================


@torch.no_grad()
def validate(
    vision: DualVisionEncoder,
    fpn: FPNNeck,
    gdf_head: SingleScaleGDFHead,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    max_radius: int = 32,
    valid_threshold: float = 0.5,
    vote_tau: float = 16.0,
    max_batches: int = 0,
    threshold_values: Optional[List[float]] = None,
) -> dict:
    vision.eval()
    fpn.eval()
    gdf_head.eval()

    do_sweep = threshold_values is not None and len(threshold_values) > 1

    val_dir = Path(output_dir) / "vis" / f"val_epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    all_vote_maps = []
    all_targets = []
    all_metas = []

    for batch_idx, (images, targets, masks, metas) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)

        _, g_grid, pyramid_raw = vision.encode_with_spatial(images)
        c4, c8, c16, c32 = pyramid_raw
        vit_feat = g_grid if fpn.has_vit else None
        p1, p2, p3, p4 = fpn(c4, c8, c16, c32, vit_feat=vit_feat)
        out = gdf_head(p1, p2, p3, p4)

        field_np = out["field"].cpu().numpy()
        # Extract edge target from GDF target (channel 3 = valid mask as edge indicator)
        edge_target = (targets[:, 2].cpu().numpy() < 0.01).astype(np.float32)  # log_dist < 0.01 = on edge

        for i in range(len(field_np)):
            vote_map = endpoint_voting(
                field_np[i], max_radius=max_radius,
                valid_threshold=valid_threshold, tau=vote_tau,
                smooth_sigma=1.0,
            )
            all_vote_maps.append(vote_map)
            all_targets.append(edge_target[i])
            all_metas.append(metas[i])

    if not all_vote_maps:
        return {}

    # Threshold sweep for best F1
    if do_sweep:
        sweep = {}
        for t in threshold_values:
            all_m = [compute_all_edge_metrics(v, g, threshold=t)
                      for v, g in zip(all_vote_maps, all_targets)]
            agg = _agg(all_m)
            sweep[t] = agg.get("buffered_f1_1px", 0.0)
        best_t = max(sweep, key=sweep.get)
        best_f1 = sweep[best_t]
        print(f"Threshold sweep (epoch {epoch}):")
        for t in threshold_values:
            m = "*" if t == best_t else " "
            print(f"  {m} t={t:.2f}: F1@1px={sweep[t]:.4f}")
    else:
        best_t = valid_threshold
        all_m = [compute_all_edge_metrics(v, g, threshold=0.3)
                  for v, g in zip(all_vote_maps, all_targets)]
        agg = _agg(all_m)
        best_f1 = agg.get("buffered_f1_1px", 0.0)

    print(f"Validation epoch {epoch}: F1@1px={best_f1:.4f} @ t={best_t:.2f}, "
          f"n={len(all_vote_maps)}")
    return {"buffered_f1_1px": best_f1, "best_threshold": float(best_t)}


def _agg(metrics_list):
    if not metrics_list:
        return {}
    agg = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if vals:
            agg[key] = float(np.mean(vals))
    return agg


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PoC-3 A3 GDF Training")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overfit", type=int, default=0,
                        help="Number of samples for overfit test")
    parser.add_argument("--build-manifest-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output or cfg.get("experiment.output_dir", "outputs/poc3_gdf/a3_0")
    device_str = args.device or cfg.get("train.device", "npu")

    if device_str == "npu":
        try:
            import torch_npu  # noqa: F401
            device = torch.device("npu:0")
        except Exception:
            device = torch.device("cpu")
    elif device_str == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build manifest
    manifest_path = cfg.get("data.manifest_path",
                             str(Path(output_dir) / "coastline_manifest.json"))
    manifest = build_coastline_manifest(
        data_roots=list(cfg.get("data.roots", [])),
        output_path=manifest_path,
        val_ratio=float(cfg.get("data.val_ratio", 0.2)),
        seed=int(cfg.get("data.val_split_seed", 42)),
    )
    if args.build_manifest_only:
        print("Manifest built. Exiting.")
        return

    # Build vision encoder
    vision = build_vision(cfg)
    vision = vision.to(device).eval()
    for p in vision.parameters():
        p.requires_grad = False
    print("Vision encoder frozen ✓")

    # Build FPN
    fpn_cfg = cfg.get("model.fpn", {})
    fpn = FPNNeck(
        in_channels=list(fpn_cfg.get("in_channels", [128, 256, 512, 1024])),
        out_channels=int(fpn_cfg.get("out_channels", 256)),
        vit_in_channels=int(fpn_cfg.get("vit_in_channels", 1024)),
    )
    fpn = fpn.to(device)

    # Build GDF Head
    gdf_head = SingleScaleGDFHead(
        in_channels=int(cfg.get("model.gdf_head.in_channels", 256)),
        decoder_channels=tuple(cfg.get("model.gdf_head.decoder_channels", [256, 128])),
        output_size=tuple(cfg.get("model.gdf_head.output_size", [224, 224])),
    )
    gdf_head = gdf_head.to(device)

    # Warm-start FPN from A2 checkpoint
    a2_ckpt_path = cfg.get("model.init.load_a2_checkpoint")
    if a2_ckpt_path and Path(a2_ckpt_path).exists():
        a2 = torch.load(a2_ckpt_path, map_location="cpu")
        if "model" in a2:
            model_sd = a2["model"]
            fpn_sd = {k[4:]: v for k, v in model_sd.items() if k.startswith("fpn.")}
        elif "fpn" in a2:
            fpn_sd = a2["fpn"]
        else:
            fpn_sd = {}
        if fpn_sd:
            fpn.load_state_dict(fpn_sd)
            print(f"Warm-started FPN from {a2_ckpt_path}")

    total_params = sum(p.numel() for p in fpn.parameters()) + \
                   sum(p.numel() for p in gdf_head.parameters())
    print(f"Params: {total_params:,}")

    # Datasets
    image_size = int(cfg.get("data.image_size", 224))
    max_radius = int(cfg.get("field.max_radius_px", 32))

    train_tiles = manifest["train"]
    val_tiles = manifest["val"]
    if args.overfit > 0:
        train_tiles = train_tiles[:args.overfit]
        val_tiles = train_tiles
        print(f"Overfit mode: {len(train_tiles)} samples")

    train_ds = GDFDataset(train_tiles, max_radius=max_radius, image_size=image_size)
    val_ds = GDFDataset(val_tiles, max_radius=max_radius, image_size=image_size)

    batch_size = args.batch_size or int(cfg.get("train.batch_size", 2))
    num_workers = int(cfg.get("data.num_workers", 2))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=gdf_collate_fn,
        pin_memory=(device_str != "cpu"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=gdf_collate_fn,
    )
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val")

    # Optimizer
    lr_fpn = float(cfg.get("train.lr_fpn", 2e-5))
    lr_gdf = float(cfg.get("train.lr_gdf_head", 1e-4))
    wd = float(cfg.get("train.weight_decay", 1e-4))
    optimizer = torch.optim.AdamW([
        {"params": fpn.parameters(), "lr": lr_fpn},
        {"params": gdf_head.parameters(), "lr": lr_gdf},
    ], weight_decay=wd)

    # Training params
    epochs = args.epochs or int(cfg.get("train.epochs", 20))
    log_interval = int(cfg.get("train.log_interval", 20))
    val_interval = int(cfg.get("train.val_interval", 5))
    save_interval = int(cfg.get("train.save_interval", 5))
    max_grad_norm = float(cfg.get("train.max_grad_norm", 1.0))

    # Loss params
    l_vec = float(cfg.get("loss.lambda_vec", 1.0))
    l_dist = float(cfg.get("loss.lambda_dist", 0.5))
    l_valid = float(cfg.get("loss.lambda_valid", 0.5))
    l_cons = float(cfg.get("loss.lambda_consistency", 0.0))

    # Voting params
    valid_thresh = float(cfg.get("voting.valid_threshold", 0.5))
    vote_tau = float(cfg.get("voting.tau", 16.0))

    threshold_values = None
    if cfg.get("voting.threshold_sweep", False):
        threshold_values = list(cfg.get("voting.threshold_values", []))

    print(f"Training: {epochs} epochs, batch={batch_size}, "
          f"lr_fpn={lr_fpn}, lr_gdf={lr_gdf}, R={max_radius}")

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(
            fpn, gdf_head, vision, train_loader, optimizer, device,
            epoch=epoch, log_interval=log_interval, max_grad_norm=max_grad_norm,
            lambda_vec=l_vec, lambda_dist=l_dist,
            lambda_valid=l_valid, lambda_cons=l_cons,
        )

        if epoch % val_interval == 0:
            metrics = validate(
                vision, fpn, gdf_head, val_loader, device,
                output_dir=output_dir, epoch=epoch,
                max_radius=max_radius,
                valid_threshold=valid_thresh, vote_tau=vote_tau,
                max_batches=0,
                threshold_values=threshold_values,
            )
            if metrics:
                f1 = metrics.get("buffered_f1_1px", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    save_checkpoint(fpn, gdf_head, optimizer, 0, output_dir, best_f1)
                    print(f"  New best F1@1px: {best_f1:.4f}")

        if epoch % save_interval == 0:
            save_checkpoint(fpn, gdf_head, optimizer, epoch, output_dir, best_f1)

    save_checkpoint(fpn, gdf_head, optimizer, epochs, output_dir, best_f1)
    print(f"Training complete. Best F1@1px: {best_f1:.4f}")


if __name__ == "__main__":
    main()

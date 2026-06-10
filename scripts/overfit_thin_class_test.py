#!/usr/bin/env python3
"""Overfit test v2: pre-compute encoder features, then train FPN+Head rapidly."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch, numpy as np, yaml, random, pickle
from PIL import Image
from ml_collections import ConfigDict

from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck
from Models.semantic_head import LandcoverSemanticHead
from Dataset.landcover_tile_grouping import scan_landcover_directories, group_tiles_by_spatial_key, build_merged_samples
from Dataset.landcover_dataset import IGNORE_INDEX
from Dataset.landcover_label_map import train_id_to_dlmc, dlmc_to_train_id

CACHE_FILE = '/tmp/overfit_features.pt'

# --- Load config and model (once) ---
with open('configs/poc2_landcover_semantic.yaml') as f:
    cfg = yaml.safe_load(f)

config = ConfigDict({
    'rgb_vision': ConfigDict(cfg['model']['rgb_vision']),
    'alignment_dim': cfg['model']['alignment_dim'],
})

vision = DualVisionEncoder(config)
vision.eval()
vc = torch.load(cfg['model']['vision_checkpoint'], map_location='cpu')
vs = vision.state_dict()
for k, v in vc.items():
    if k in vs: vs[k] = v.copy()
vision.load_state_dict(vs, strict=False)
for p in vision.parameters():
    p.requires_grad = False
print("Vision encoder loaded.")

# --- Select thin class tiles ---
raw = scan_landcover_directories()
groups = group_tiles_by_spatial_key(raw)
merged = build_merged_samples(raw, groups)

thin_ids = {dlmc_to_train_id(n) for n in ['公路用地', '沟渠', '农村道路', '城镇村道路用地']}
thin_tiles = []
for s in merged:
    tids = set(s['train_ids'])
    if tids & thin_ids:
        cache_path = f'data/landcover_target_cache/{s["sample_id"]}.pt'
        d = torch.load(cache_path, map_location='cpu')
        t = d['target'].numpy()
        for tid in (tids & thin_ids):
            cnt = int((t == tid).sum())
            if 5 <= cnt <= 500:
                thin_tiles.append((s, cnt, tid))
                break

random.seed(42)
thin_tiles.sort(key=lambda x: x[1])
selected = thin_tiles[:8]
print(f"Selected {len(selected)} tiles for overfit test:")
for s, cnt, tid in selected:
    name = train_id_to_dlmc(tid)
    print(f"  {s['sample_id']}: {name}(id={tid}) = {cnt}px")

# --- Pre-compute or load cached features ---
if Path(CACHE_FILE).exists():
    data = torch.load(CACHE_FILE, map_location='cpu')
    pyramid_caches = data['pyramid_caches']
    targets = data['targets']
    print(f"Loaded cached features from {CACHE_FILE}")
else:
    images_list = []
    targets_list = []
    for s, cnt, tid in selected:
        img = Image.open(s['image_path']).convert('RGB').resize((224, 224), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        d = torch.load(f'data/landcover_target_cache/{s["sample_id"]}.pt', map_location='cpu')
        gt = d['target'].clone().detach() if isinstance(d['target'], torch.Tensor) else torch.from_numpy(d['target'].astype(np.int64))
        images_list.append(img_t)
        targets_list.append(gt)

    images = torch.stack(images_list)
    targets = torch.stack(targets_list)
    print(f"Computing encoder features for {images.shape[0]} images...")
    with torch.no_grad():
        _, _, pyramid_raw = vision.encode_with_spatial(images)
    pyramid_caches = tuple(p.detach().cpu() for p in pyramid_raw)
    torch.save({'pyramid_caches': pyramid_caches, 'targets': targets}, CACHE_FILE)
    print(f"Cached features to {CACHE_FILE}")

targets = targets if 'targets' in dir() else data['targets']
print(f"Targets: {targets.shape}")

# --- Train FPN + Head (fast, no encoder forward) ---
fpn = FPNNeck(in_channels=[128, 256, 512, 1024], out_channels=256)
sem_head = LandcoverSemanticHead(in_channels=256, num_classes=25, output_size=(224, 224))

# Try to load best checkpoint for warm-start
best_path = 'outputs/poc2b_cond_bce_030_t50/checkpoints/best_miou.pt'
use_warmstart = Path(best_path).exists()
if use_warmstart:
    best = torch.load(best_path, map_location='cpu')
    fpn.load_state_dict(best['fpn'])
    sem_head.load_state_dict(best['sem_head'])
    print("Warm-started from best_miou checkpoint")
else:
    print("Training from scratch")

fpn.train(); sem_head.train()

optimizer = torch.optim.AdamW(
    list(fpn.parameters()) + list(sem_head.parameters()), lr=1e-3, weight_decay=0.0
)

print("\n=== Overfit training (2000 iterations) ===")
for it in range(2001):
    optimizer.zero_grad()
    c4, c8, c16, c32 = pyramid_caches
    p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
    logits = sem_head(p1, p2, p3, p4)
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    loss.backward()
    optimizer.step()

    if it % 200 == 0:
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            print(f"\n--- iter {it}: loss={loss.item():.4f} ---")
            for i, (s, cnt, tid) in enumerate(selected):
                name = train_id_to_dlmc(tid)
                mask = targets[i] == tid
                n_gt = mask.sum().item()
                correct = (pred[i][mask] == tid).sum().item()
                valid = targets[i] != IGNORE_INDEX
                all_correct = (pred[i][valid] == targets[i][valid]).sum().item()
                all_n = valid.sum().item()
                own_p = probs[i, tid][mask].mean().item()
                bg_p = probs[i, 0][mask].mean().item()
                argmax_class = pred[i][mask].mode().values.item() if n_gt > 0 else -1
                argmax_name = train_id_to_dlmc(int(argmax_class)) if argmax_class > 0 else ('bg' if argmax_class == 0 else 'none')
                print(f"  [{s['sample_id']}] {name}({n_gt}px): correct={correct}/{n_gt} own_p={own_p:.4f} bg_p={bg_p:.4f} argmax→{argmax_name} acc={100*all_correct/max(all_n,1):.1f}%")

print("\nDone.")

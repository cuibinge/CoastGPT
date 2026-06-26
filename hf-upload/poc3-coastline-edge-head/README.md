---
language: en
tags:
- remote-sensing
- coastline-detection
- edge-detection
- earth-observation
- gf1
- gf2
- coastal
license: other
datasets:
- CoastGPT-coastline-v1
metrics:
- buffered-f1
- chamfer-distance
model-index:
- name: CoastGPT PoC-3 Edge Head
  results:
  - task:
      type: edge-detection
      name: Coastline Detection
    dataset:
      name: Coastline GF1/GF2 tiles
      type: CoastGPT-coastline-v1
    metrics:
    - type: buffered-f1-1px
      value: 0.554
    - type: buffered-f1-3px
      value: 0.782
    - type: chamfer-distance-px
      value: 8.1
---

# CoastGPT PoC-3 Coastline Edge Head

Single-scale coastline edge detection head for the CoastGPT framework.

## Architecture

```
4-band TIF (R,G,B,NIR) → ConvNeXt Base (frozen) + DINOv3 ViT-L/16 (frozen)
  → FPN Neck (3.7M params) → SingleScaleEdgeHead (2.66M params)
  → sigmoid → threshold → skeleton → path → GeoJSON LineString
```

## Usage

```python
import torch
from Models.edge_head import SingleScaleEdgeHead
from Models.fpn_neck import FPNNeck

# Load weights
ckpt = torch.load("pytorch_model.bin", map_location="cpu")
fpn = FPNNeck([128, 256, 512, 1024], 256, vit_in_channels=1024)
edge_head = SingleScaleEdgeHead(256, output_size=(224, 224))
fpn.load_state_dict(ckpt["fpn"])
edge_head.load_state_dict(ckpt["edge_head"])

# Inference (vision encoder must be loaded separately)
# See scripts/poc3_edge_infer.py for full pipeline
```

## Performance

| Metric | Value |
|--------|-------|
| Buffered F1@1px | 0.554 |
| Buffered F1@3px | 0.782 |
| Chamfer Distance | 8.1 px |
| Hausdorff 95% | 36.7 px |

**By shoreline type:**

| Type | Threshold | F1@1px | F1@3px |
|------|-----------|--------|--------|
| 生物岸线 | 0.60 | 0.533 | 0.765 |
| 盐田围堤 | 0.70 | 0.586 | 0.806 |

## Training

- 701 training / 253 validation tiles
- 224×224 input, 4-band GF1/GF2 TIF
- Focal(α=0.75, γ=2.0) + Dice loss
- 12 epochs, warm-started from A1-relabel epoch 10
- GeoJSON-derived unified labels (P3-L pipeline)

## Limitations

- Requires frozen DINOv3 ViT-L/16 and ConvNeXt Base backbones (not included)
- 224×224 input resolution
- Trained on GF1/GF2 Chinese coastal imagery
- Precision@1px=0.46 — false positives on non-coastline linear features

## Citation

CoastGPT PoC-3. Internal research project. Contact authors for usage.

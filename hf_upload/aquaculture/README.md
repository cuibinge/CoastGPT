---
language: zh
license: cc-by-nc-4.0
tags:
- remote-sensing
- coastal
- aquaculture
- instance-segmentation
- mask-rcnn
- gf1
- gf2
- gf6
metrics:
- mAP
- IoU
---

# CoastGPT Aquaculture Detection Head

养殖区实例分割头 — Mask R-CNN with FPN neck, trained on GF-1/GF-2/GF-6 multi-sensor aquaculture imagery.

## Architecture

```
DualVisionEncoder (External)
    │
    ├── C2 [128ch] → FPN lateral → P2 [256ch]
    ├── C3 [256ch] → FPN lateral → P3 [256ch]
    ├── C4 [512ch] → FPN lateral → P4 [256ch]
    └── C5 [1024ch]→ FPN lateral → P5 [256ch]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              RPN Head                        ROI Heads
              (region proposal)               (box + mask prediction)
```

## Usage

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load head weights
state_dict = torch.load("pytorch_model.bin", map_location="cpu")

# Build model with torchvision Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
# Replace head weights
model.rpn.load_state_dict(
    {k[len('rpn.head.'):]: v for k, v in state_dict['state_dict'].items() if k.startswith('rpn.')},
    strict=False
)
model.roi_heads.load_state_dict(
    {k: v for k, v in state_dict['state_dict'].items() if k.startswith('roi_heads.')},
    strict=False
)

# The backbone (DualVisionEncoder) must be loaded separately:
# from Models.dual_vision_encoder import DualVisionEncoder
# vision = DualVisionEncoder(config)
# vision.load_state_dict(vision_ckpt)
# model.backbone = CustomBackbone(vision, fpn)
```

**Important:** This package contains only the detection head weights (FPN + RPN + ROI heads). The DualVisionEncoder backbone must be loaded separately from the CoastGPT model checkpoint.

## Components

| Component | Keys | Description |
|-----------|------|-------------|
| FPN | 16 | Feature Pyramid Network (lateral + smooth convs, 4 levels) |
| RPN Head | 6 | Region Proposal Network (conv + cls + bbox) |
| Box Head | 4 | Two-layer MLP for bounding box prediction |
| Box Predictor | 4 | Classification + regression heads |
| Mask Head | 8 | 4 conv layers for mask features |
| Mask Predictor | 4 | Deconv + logits for per-instance mask |

## Classes

| ID | Class |
|----|-------|
| 0 | background |
| 1 | aquaculture (养殖区) |

## Training Data

- **Sensors:** GF-1 PMS2, GF-2 PMS2, GF-6 PMS
- **Resolution:** 0.8m ~ 8m (multi-size training)
- **Tiles:** ~1,800 4-band TIF tiles
- **Labels:** GeoJSON polygon annotations converted to instance masks

## Dependencies

- PyTorch >= 2.0
- torchvision >= 0.15
- CoastGPT DualVisionEncoder (external)

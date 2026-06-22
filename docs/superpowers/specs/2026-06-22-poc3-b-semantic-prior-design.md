# PoC-3 P3-B: Semantic Prior for Coastline Detection

> 日期: 2026-06-22
> 状态: design approved → implementing
> 阶段: PoC-3 / P3-B

## Background

A0/A1/A2/A2b/A3 all failed to solve "is this edge a coastline or not?":
- A2 (edge+Dice+DS, 224): F1=0.039, FP median dist=51px
- A2b (edge+Dice+DS, 448): F1=0.023, FP median dist=137px
- A3 (GDF vector field): F1=0.024, FP median dist=48px (marginal)
- Oracle ceiling stuck at 0.032-0.048 across all approaches

Conclusion: **geometric/edge-only approaches cannot substitute for semantic discrimination.**
The model needs an explicit "sea vs land" signal.

## P3-B1: Oracle Sea/Land Mask → Coastline Upper Bound

### Goal
Verify: if we had a perfect sea/land mask, would coastline detection be solved?

### Method
1. Read coastline GeoJSON LineString for each tile
2. Rasterize lines to binary edge map
3. Flood-fill from tile border → sea side (1) vs land side (0)
4. Extract boundary: `|sobel(sea_mask)| > 0`
5. Evaluate against binary edge GT (standard metrics)

### Success criterion
F1@1px > 0.15 (vs A2 0.039) → semantic direction is correct.

### Files
- `scripts/poc3_b1_oracle_sea_land.py` — one-shot evaluation script

---

## P3-B2: Train Sea/Land Semantic Segmentation → Boundary

### Architecture

```
ViT-FPN (frozen ConvNeXt + trainable FPN, A2 warm-start)
  → SemanticHead (3-class: land=0, sea=1, ignore=255)
  → CrossEntropy loss
  → Inference: argmax → sea/land binary → boundary operator → coastline
```

### Sea/Land class mapping from 24 DLMC classes

Sea (class 1):
- 河流水面(8), 坑塘水面(9), 沟渠(6), 内陆滩涂(11),
- 沿海滩涂(22), 盐田(23), 养殖坑塘(24)

Land (class 0):
- All other 17 classes

Ignore (255):
- Background (0)
- Spec-only classes (no training data)
- Pixels where DLMC label is ambiguous near coastline

### Coastline extraction from semantic output

```
sea_logit = softmax(output)[:, 1]           # P(sea)
sea_mask = argmax(output) == 1              # binary sea
coastline_edge = binary_dilation(sea_mask) != binary_erosion(sea_mask)  # boundary
// or: sobel(sea_mask) > 0
```

### Training

- Dataset: landcover 33k tiles → remap to sea/land/ignore
- FPN: warm-start from A2-224
- SemanticHead: random init, 3-class output
- Loss: CrossEntropy(ignore_index=255)
- Metrics: sea IoU, land IoU, boundary F1, FP distance, oracle ceiling
- Budget: 10-20 epochs, single NPU

### Files
- `Dataset/semantic_sea_land_dataset.py` — remap landcover → sea/land
- `scripts/poc3_b2_semantic_coastline.py` — train/eval script
- `configs/poc3_b2_semantic_coastline.yaml`

### Success criteria (vs A2 edge baseline)

| Metric | A2 edge | P3-B2 target |
|--------|---------|-------------|
| F1@1px | 0.039 | > 0.06 |
| Oracle ceiling | 0.048 | > 0.08 |
| FP median dist | 51.5px | < 40px |
| Sea IoU | N/A | > 0.80 |

---

## Implementation Order

1. P3-B1 Oracle: GeoJSON → flood-fill → boundary evaluation (2h)
2. P3-B2 Semantic: landcover remap → train → boundary eval (1d)

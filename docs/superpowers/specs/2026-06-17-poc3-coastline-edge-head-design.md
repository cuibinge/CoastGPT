# PoC-3: 海岸线 Edge Head 设计文档

> 日期: 2026-06-17
> 状态: 设计定稿
> 前置: PoC-2b 土地覆盖 Semantic Head 已完成；ViT-FPN 架构可复用
> 下一阶段: PoC-4 LLM fallback + 融合

---

## 1. 目标与范围

在冻结的 DualVisionEncoder + ViT-FPN 之上，实现海岸线 Edge Head，完成：

```
GeoJSON LineString/MultiLineString GT (WGS84)
  → WGS84 → source CRS → model pixel space
  → densify → rasterize / soft edge target
  → Edge Head 训练
  → edge heatmap
  → threshold → skeletonize → graph extraction → polyline
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation
```

### 不做

- 不接 LLM fallback / gating
- 不接 Stage 4 联合训练
- 不接 EpochBasedTrainer / DeepSpeed 分布式
- 不修改 CoastGPT 主训练路径
- 不做 A3 dilation context / A5 residual refinement（皆为首轮可选消融）

---

## 2. 核心设计决策

### 2.1 二分类 edge vs background

7 种海岸线子类型（砂质/基岩/生物/河口/港口/建设围堤/盐田围堤）合并为单一 edge channel。输出 binary edge heatmap → skeleton → polyline。

### 2.2 复用 ViT-FPN

DINOv3 ViT `g_grid` 通过 `vit_proj` 注入 FPN P3。海岸线比土地覆盖更依赖全局海陆边界走向，ViT 全局语义对连续性至关重要。

### 2.3 Edge vs Semantic 差异

| 维度 | Semantic (PoC-2b) | Edge (PoC-3) |
|---|---|---|
| 前景像素比例 | 10-60% | 0.5-3%，极端不平衡 |
| 空间精度要求 | 区域级 IoU | 像素级，1px 偏移可致断线 |
| 拓扑敏感性 | 低 | 高，断线 → 碎片化 LineString |
| 输出几何类型 | Polygon / MultiPolygon | LineString / MultiLineString |

---

## 3. 阶段路线

主线: **A0 → A1 → A2 → A4**，每阶段只引入一个主要变量。

```
A0: Single-scale BCE+Dice + 最小后处理 → 闭环验证
A1: Focal+Dice + threshold sweep → 正式 baseline
A2: HED-style 多尺度 side-output 深监督 → 主模型候选
A4: Topology-aware postprocess → 最终候选方案
```

A3 (dilation context) 和 A5 (residual refinement) 不进入首轮主线。

最终候选方案:

```
ViT-FPN
  + Multi-scale Edge Head (HED-style)
  + Deep-supervised Focal/Dice
  + soft edge target
  + threshold sweep
  + topology-aware skeleton graph postprocess
  + LineString/MultiLineString GeoJSON validation
```

### 3.1 Gate 机制

**A0 → A1**: GT overlay 正确、round-trip 误差达标、loss 下降、heatmap 非全背景、skeleton/polyline 可导出、GeoJSON 可 parse、coordinate-in-tile rate = 100%

**A1 → A2**: Focal+Dice 相比 A0 提升 edge recall 或 buffered-F1、threshold sweep 存在稳定最优区间、false positive 不失控

**A2 → A4**: fused heatmap 不低于 A1、side outputs 可解释、buffered-F1@3px 达到可用水平、skeleton 非碎片化

**A4 → PoC-3 通过**: buffered-F1@1px ≥ 0.70、buffered-F1@3px ≥ 0.85、average offset < 10m、geometry valid rate ≥ 95%

任一 gate 不通过，优先回溯 pipeline/数据问题，不急于进入下一阶段。

---

## 4. 数据与 GT 生成

### 4.1 数据来源

`/home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级+二级`

- Level 1: 289 tiles, merged 海岸线类别, 256×256
- Level 2: 665 tiles, 7 种子类型, 128×256 混合
- 总计 954 tiles，全部同时有 Binary TIF + GeoJSON

### 4.2 GT 生成流程

```
GeoJSON coordinates (WGS84)
  → pyproj Transformer: EPSG:4326 → source_crs
  → inverse model_transform
  → model pixel coordinates (224×224)
  → clip to tile bounds
  → densify polyline (max_step=0.5px)
  → draw edge map / soft distance target
```

### 4.3 像素坐标约定

| 项 | 约定 |
|---|---|
| 训练/推理统一空间 | model input pixel space 224×224 |
| LineString 点坐标 | pixel center |
| 坐标转换精度 | float32 |
| tile 外线段 | clip to tile bounds |
| 空 GT | 允许，输出 features=[] |

### 4.4 训练标签与评估标签分离

| 标签 | 用途 | 生成方式 |
|---|---|---|
| `edge_center_1px` | evaluation / skeleton | draw polyline width=1 |
| `edge_train_target` | training loss | width=3 hard band 或 soft distance target |

A0 使用 hard width=3。A1 起推荐使用 soft edge target:

```
distance = distance_transform_to_polyline
edge_soft = exp(-(distance^2) / (2 * sigma^2))
edge_soft[distance > radius] = 0
```

推荐参数: `soft_sigma_px=1.0, soft_radius_px=3.0`

### 4.5 Densify Polyline

GeoJSON 顶点稀疏时直接 rasterize 易造成像素级断裂。所有 polyline 在 rasterize 前 densify:

```python
def densify_polyline(points, max_step=0.5):
    dense = []
    for p0, p1 in zip(points[:-1], points[1:]):
        dense.append(p0)
        dist = euclidean_distance(p0, p1)
        n = max(1, int(math.ceil(dist / max_step)))
        for i in range(1, n):
            t = i / n
            dense.append(lerp(p0, p1, t))
    dense.append(points[-1])
    return dense
```

### 4.6 数据划分

按 coastal segment / source image 分组划分 train/val，同一海岸段所有 tile 必须在同一 split。禁止随机 tile 划分（空间泄漏）。

### 4.7 Edge Dataset 输出 Contract

```python
sample = {
    "image": Tensor[3, 224, 224],
    "target": {
        "edge": FloatTensor[1, 224, 224],  # binary heatmap or soft target
    },
    "meta": {
        "sample_id": str,
        "image_path": str,
        "source_crs": str,
        "original_transform": list[float],
        "model_transform": list[float],
        "original_size": [int, int],
        "model_input_size": [224, 224],
        "resize_scale": [float, float],
        "has_edge": bool,
        "num_linestrings": int,
    }
}
```

---

## 5. Edge Head 架构

### 5.1 共享部分: ViT-FPN

```text
DualVisionEncoder
  ├─ ConvNeXt pyramid_raw: c4/c8/c16/c32
  └─ DINOv3 ViT g_grid [B,1024,14,14]

ViT-FPN (vit_in_channels=1024)
  ├─ c4/c8/c16/c32 → lateral + top-down
  ├─ g_grid → vit_proj → inject into P3
  └─ output: P1[256,56], P2[256,28], P3[256,14], P4[256,7]
```

FPN 参数 ~3.7M (含 vit_proj)，全部训练。Vision encoder 冻结。

### 5.2 A0/A1: Single-Scale Edge Head

```text
P1 [B,256,56,56]
P2 [B,256,28,28] → upsample 56
P3 [B,256,14,14] → upsample 56
P4 [B,256,7,7]   → upsample 56

concat → [B,1024,56,56]
  → 3×3 Conv 1024→256 + BN + ReLU
  → 3×3 Conv 256→128 + BN + ReLU
  → 1×1 Conv 128→1
  → edge_logit_56 [B,1,56,56]
  → bilinear upsample → edge_logit_224 [B,1,224,224]
```

Loss 在 224×224 上计算（避免 56×56 低分辨率细线 aliasing）。

### 5.3 A2: Multi-Scale Deep Supervision Edge Head

```text
P1 → side1_conv → side1_logit_56 → upsample 224
P2 → side2_conv → side2_logit_28 → upsample 224
P3 → side3_conv → side3_logit_14 → upsample 224
P4 → side4_conv → side4_logit_7  → upsample 224

fused_logit = 1×1 Conv(concat(side1_224, side2_224, side3_224, side4_224))
```

所有 side loss 在 upsample 到 224 后计算。禁止在低分辨率上用 nearest/average 下采样 GT（细线会消失；若必须，用 max-pooling）。

---

## 6. Loss 设计

### 6.1 A0: BCE + Dice

```
L_edge = BCEWithLogits(logit, target) + DiceLoss(sigmoid(logit), target)
```

仅用于 pipeline 验证，不作性能结论。

### 6.2 A1: Focal + Dice（正式 baseline）

```
L_edge = λ_focal * FocalLoss(logit, target; α=0.75, γ=2.0)
       + λ_dice  * SoftDiceLoss(sigmoid(logit), target)
```

`gamma=2.0` 降低背景主导，`alpha=0.75` 提升前景权重。前景 < 1% 时尝试 `alpha=0.85`。

### 6.3 A2: Deep-supervised Loss

```
L_edge =
  1.0 * L_fused_224
+ 0.5 * L_side1_224
+ 0.3 * L_side2_224
+ 0.2 * L_side3_224
+ 0.1 * L_side4_224

其中每个 L = FocalLoss + DiceLoss
```

---

## 7. 后处理

### 7.1 最小后处理 (A0)

```
sigmoid heatmap → fixed threshold=0.5 → skeletonize → simple path extraction → pixel→WGS84 → GeoJSON LineString
```

### 7.2 Threshold Sweep (A1 起)

验证集 sweep: `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]`

按 `buffered_f1_1px` 选择最佳阈值。初始默认值: 0.30。

### 7.3 完整拓扑后处理 (A4)

```
sigmoid heatmap
  → threshold (from sweep)
  → remove small components (min_area=8px)
  → morphological closing (kernel=3)
  → gap bridge (max_gap=3px)
  → skeletonize
  → graph extraction
  → spur pruning (max_length=4px)
  → extract valid paths (length ≥ max(10px, 0.15 × longest))
  → top-k = 5
  → Douglas-Peucker simplify (epsilon=1.0px)
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation

if len(valid_paths) == 0: output features=[]
elif len(valid_paths) == 1: output LineString
else: output MultiLineString
```

不强制只保留最长路径——岛岸线、河口等多段岸线应保留。

---

## 8. 评估

### 8.1 主指标

| 指标 | 目标 |
|---|---|
| buffered-F1 @ 1px | ≥ 0.70 |
| buffered-F1 @ 3px | ≥ 0.85 |
| average offset | < 10m |
| GeoJSON parse success rate | > 99% |
| GeoJSON schema valid rate | > 99% |
| geometry valid rate | ≥ 95% |
| coordinate-in-tile rate | 100% |

### 8.2 诊断指标

| 指标 | 目的 |
|---|---|
| pixel precision/recall/F1 | edge map 二值质量 |
| Chamfer distance | 平均几何偏移 |
| Hausdorff distance | 最坏偏移/断裂情况 |
| connected component count error | 拓扑碎片化 |
| predicted_length / GT length | 漏线或假阳性过长 |
| endpoint distance | tile 中间断裂检测 |

### 8.3 可视化输出

```
outputs/poc3_edge/vis/
  {sample_id}_image.png
  {sample_id}_gt_center.png
  {sample_id}_gt_train_target.png
  {sample_id}_pred_heatmap.png
  {sample_id}_pred_binary.png
  {sample_id}_pred_skeleton.png
  {sample_id}_gt_pred_overlay.png
```

---

## 9. 训练配置

```yaml
model:
  backbone: dual_vision_encoder
  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256
    vit_in_channels: 1024      # ViT-FPN

  edge_head:
    type: single_scale         # A0/A1; A2 改为 multi_scale
    out_channels: 1
    decoder_channels: [256, 128]
    deep_supervision: false    # A2 开启

train:
  input_size: 224
  batch_size: 2
  accum_steps: 8
  epochs: 40
  lr_fpn: 5.0e-5
  lr_edge_head: 1.0e-4
  weight_decay: 1.0e-4
  freeze_vision: true
  precision: bf16              # 坐标转换除外

loss:                          # A1 起
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0

gt:
  line_width_train: 3
  line_width_eval: 1
  soft_edge: false             # A2 起改为 true
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
  densify_max_step_px: 0.5

postprocess:
  threshold_values: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  threshold_select_metric: buffered_f1_1px
  min_component_area: 8
  close_kernel: 3
  max_gap_bridge_px: 3
  min_line_length_px: 10
  spur_prune_length_px: 4
  douglas_peucker_epsilon_px: 1.0
  max_components: 5
```

---

## 10. 代码落点

```
Models/
  edge_head.py                 # SingleScaleEdgeHead, MultiScaleEdgeHead

Dataset/
  coastline_dataset.py         # GeoJSON LineString parser, WGS84→pixel, densify, GT generation

scripts/
  poc_stage_edge.py            # train/eval entrypoint

utils/
  edge_losses.py               # FocalLoss, SoftDiceLoss, DeepSupervisedEdgeLoss
  edge_postprocess.py          # threshold, skeletonize, graph, spur prune, simplify
  coastline_metrics.py         # buffered-F1, Chamfer, Hausdorff, length ratio

configs/
  poc3_edge_a0_closure.yaml
  poc3_edge_a1_focal_dice.yaml
  poc3_edge_a2_multiscale.yaml
  poc3_edge_a4_topology.yaml
```

### 模块依赖

```
poc_stage_edge.py
  ├── Models/dual_vision_encoder.py (frozen)
  ├── Models/fpn_neck.py (ViT-FPN, 复用)
  ├── Models/edge_head.py (新增)
  ├── Dataset/coastline_dataset.py (新增)
  ├── utils/edge_losses.py (新增)
  ├── utils/edge_postprocess.py (新增)
  ├── utils/coastline_metrics.py (新增)
  ├── utils/georef_transform.py (复用)
  └── utils/geojson_builder.py (复用)
```

---

## 11. 通过标准

### 11.1 A0 最小闭环

- [ ] GeoJSON LineString/MultiLineString 正确解析
- [ ] WGS84 ↔ pixel round-trip 误差达标
- [ ] edge GT overlay 与原图海岸线对齐
- [ ] loss 正常下降
- [ ] predicted heatmap 非全背景
- [ ] skeletonize 可生成有效 polyline
- [ ] GeoJSON 输出可 parse
- [ ] coordinate-in-tile rate = 100%

### 11.2 PoC-3 最终通过

- [ ] buffered-F1@1px ≥ 0.70
- [ ] buffered-F1@3px ≥ 0.85
- [ ] average offset < 10m
- [ ] GeoJSON parse success rate > 99%
- [ ] GeoJSON schema valid rate > 99%
- [ ] geometry valid rate ≥ 95%
- [ ] coordinate-in-tile rate = 100%
- [ ] 空输出 features=[] 正确处理
- [ ] 可视化 overlay 与原图基本对齐

---

## 12. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 前景像素极少 (0.5-3%) | BCE 被背景淹没 | Focal+Dice；控制 empty negative 比例 |
| GT 线过细 (1px) | 训练不稳定 | width=3 hard band 或 soft edge target |
| GeoJSON 顶点稀疏 | rasterize 后断裂 | densify polyline, max_step≤0.5px |
| threshold 固定不合理 | recall/precision 失衡 | 验证集 threshold sweep |
| skeleton 毛刺多 | LineString 碎片化 | spur pruning + min length filter |
| 只保留最长线 | 岛岸线/多段岸线丢失 | MultiLineString, top-k paths |
| 低分辨率 side loss 细线消失 | 多尺度监督无效 | 所有 side logits upsample 到 224 再算 loss |
| 空间泄漏 | 评估虚高 | 按 coastal segment 分组划分 |

---

## 13. 实验命名与目录

```
poc3_a0_bce_dice_closure
poc3_a1_focal_dice_sweep
poc3_a2_multiscale_deepsup
poc3_a4_topology_postprocess

outputs/poc3_edge/
  a0_bce_dice_closure/
  a1_focal_dice_sweep/
  a2_multiscale_deepsup/
  a4_topology_postprocess/
  ablations/
```

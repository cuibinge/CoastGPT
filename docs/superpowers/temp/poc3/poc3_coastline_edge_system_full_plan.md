# PoC-3 海岸线 Edge Detection System 总设计与落地执行计划

> 日期：2026-06-17  
> 版本：v1.0  
> 状态：可执行设计稿  
> 适用阶段：CoastGPT 检测头 + LLM 混合架构 PoC-3  
> 任务范围：海岸线 Edge Head，从遥感 tile 中提取海岸线并输出 GeoJSON `LineString` / `MultiLineString`  
> 技术路线：冻结 DualVisionEncoder + ViT-FPN + Edge Head + skeleton/graph topology postprocess  
> 关键原则：PoC-3 不是单纯 semantic segmentation，而是 **edge detection + geospatial topology reconstruction**

---

## 0. 执行结论

PoC-3 推荐采用如下主线：

```text
A0: Single-scale Edge Head + BCE/Dice + minimal postprocess
  → A1: Focal/Dice + threshold sweep
  → A2: Multi-scale deep-supervised Edge Head
  → A4: Topology-aware postprocess
```

最终候选方案：

```text
ViT-FPN
  + Multi-scale Edge Head
  + Deep-supervised Focal/Dice
  + soft edge target
  + threshold sweep
  + skeleton graph postprocess
  + LineString / MultiLineString GeoJSON validation
```

A3 / A5 不进入首轮主线：

```text
A3: dilation context module       # 可选消融
A5: residual edge refinement      # 可选消融
```

核心执行要求：

```text
1. A0 必须先跑通 GeoJSON → edge GT → heatmap → skeleton → GeoJSON 的闭环。
2. A1 必须加入 threshold sweep，不能继续固定 threshold=0.5。
3. A2 所有 side output 必须 upsample 到 224×224 后再计算 loss。
4. A4 是完整 topology-aware postprocess，不是后处理首次出现；A0 就要有 minimal postprocess。
5. 所有阶段必须 gate-driven，禁止在 A0 未通过时进入 A1/A2/A4。
```

---

## 1. 系统目标

PoC-3 的目标是完成 **海岸线 Edge Head 闭环**：

```text
GeoJSON LineString / MultiLineString GT
  → WGS84 → source CRS → model pixel space
  → densify polyline
  → rasterize edge target
  → Edge Head 训练
  → edge heatmap
  → threshold / skeletonize / graph extraction
  → polyline reconstruction
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation
```

PoC-3 不是简单的 binary semantic segmentation。它的实际目标是：

> 从极稀疏、拓扑敏感、像素级精度要求的边缘响应中恢复 GIS 可用的线状几何。

---

## 2. Edge Detection vs Semantic Segmentation 的关键差异

| 维度 | Semantic Segmentation PoC-2b | Edge Detection PoC-3 |
|---|---|---|
| 前景像素比例 | 10–60% | 0.5–3%，极端不平衡 |
| 空间精度要求 | 区域级，IoU 容错较高 | 像素级，1px 偏移可能导致断线 |
| 拓扑敏感性 | 较低，region 可容错 | 很高，断线会导致碎片化 LineString |
| 上下文依赖 | 局部纹理 + 类别语义 | 全局连续性，海岸线通常贯穿 tile |
| 输出几何 | Polygon / MultiPolygon | LineString / MultiLineString |
| 后处理重点 | connected components + contour | skeleton + graph + topology pruning |
| 主要失败模式 | 类别混淆 / 区域漏检 | 断线、毛刺、碎片、偏移、拓扑错误 |

因此，PoC-3 的优化重点不是单纯 mIoU，而是：

```text
edge recall
+ pixel-level alignment
+ buffered-F1
+ topology continuity
+ GeoJSON validity
```

---

## 3. 总体架构

### 3.1 Backbone：DualVisionEncoder

沿用现有 DualVisionEncoder：

```text
DualVisionEncoder
  ├── DINOv3 ViT-L16
  │     └── g_grid: global semantic context
  └── ConvNeXt Base
        └── pyramid_raw: c4/c8/c16/c32 local multi-scale features
```

输出：

```python
image_seq, g_grid, pyramid_raw = vision.encode_with_spatial(images)
c4, c8, c16, c32 = pyramid_raw
```

### 3.2 Neck：ViT-FPN

PoC-2b 已验证 ViT-FPN 对土地覆盖有效。PoC-3 应继续复用 ViT-FPN：

```text
ConvNeXt c4/c8/c16/c32
  + DINOv3 ViT g_grid
  → FPN P1/P2/P3/P4
```

推荐：

```text
P1: [B, 256, 56, 56]  # high-resolution edge detail
P2: [B, 256, 28, 28]  # local structure
P3: [B, 256, 14, 14]  # ViT global semantic fusion
P4: [B, 256, 7, 7]    # coarse coastline layout
```

### 3.3 Edge Head

PoC-3 包含两个主线 Edge Head：

```text
SingleScaleEdgeHead   # A0/A1
MultiScaleEdgeHead    # A2/A4 final candidate
```

---

## 4. 阶段路线

### 4.1 总路线

```text
P3-A0: Pipeline Closure
  → P3-A1: Class Imbalance Baseline
  → P3-A2: Multi-scale Edge Supervision
  → P3-A4: Topology-aware Postprocess
```

### 4.2 阶段定义

| 阶段 | 名称 | 架构 | Loss | 后处理 | 目的 | 是否最终候选 |
|---|---|---|---|---|---|---|
| A0 | Pipeline Closure | Single-scale | BCE + Dice | minimal | 跑通闭环 | 否 |
| A1 | Imbalance Baseline | Single-scale | Focal + Dice | threshold sweep | 解决前景稀疏 | 是 |
| A2 | Multi-scale Supervision | HED-style side outputs | deep-supervised Focal + Dice | threshold sweep | 提升定位与连续性 | 是，主模型 |
| A4 | Topology Postprocess | A2 模型 | 不变 | full topology graph | 稳定 GeoJSON 输出 | 是，最终候选 |
| A3 | Dilation Context | optional | 同 A2 | 同 A4 | 扩大感受野 | 可选消融 |
| A5 | Residual Refine | optional | 同 A2 | 同 A4 | 细化边缘位置 | 可选消融 |

---

## 5. P3-A0：Pipeline Closure

### 5.1 目标

A0 的目标不是性能，而是验证：

```text
GeoJSON label
  → pixel edge target
  → model train
  → heatmap
  → skeleton
  → LineString GeoJSON
```

完整链路成立。

A0 不能用于最终性能结论。

### 5.2 模型

```text
ViT-FPN
  → SingleScaleEdgeHead
```

SingleScaleEdgeHead：

```text
P1 [B,256,56,56]
P2 [B,256,28,28] → upsample 56
P3 [B,256,14,14] → upsample 56
P4 [B,256,7,7]   → upsample 56
concat → [B,1024,56,56]
3×3 Conv 1024→256 + BN + ReLU
3×3 Conv 256→128 + BN + ReLU
1×1 Conv 128→1
bilinear upsample → [B,1,224,224]
```

### 5.3 Loss

```text
L_edge = BCEWithLogits(edge_logit, edge_target)
       + DiceLoss(sigmoid(edge_logit), edge_target)
```

### 5.4 GT

A0 使用 hard edge band：

```yaml
gt:
  line_width_train: 3
  line_width_eval: 1
  densify_max_step_px: 0.5
```

### 5.5 Minimal postprocess

A0 必须包含最小后处理：

```text
sigmoid
  → fixed threshold=0.5
  → binary mask
  → remove small components
  → skeletonize
  → simple connected component path extraction
  → Douglas-Peucker simplify
  → pixel → WGS84
  → GeoJSON LineString
```

### 5.6 A0 Gate

A0 必须全部通过：

```text
1. 数据扫描成功，tile count > 0
2. image 可加载
3. GeoJSON 或 binary label 可加载
4. GT edge fg_ratio 合理，建议 0.5%–5%
5. GT overlay 与原图海岸线视觉对齐
6. WGS84 ↔ pixel round-trip 误差可控
7. loss 能下降
8. prediction 不是全背景
9. skeleton 能生成有效 polyline
10. GeoJSON 可 parse
11. geometry valid
12. coordinate-in-tile rate = 100%
```

若 A0 失败，禁止进入 A1。优先 debug：

```text
GT rasterization
CRS / affine transform
tile bounds
densify
line width
```

---

## 6. P3-A1：Class Imbalance Baseline

### 6.1 目标

解决 edge 前景极稀疏问题：

```text
edge foreground ratio ≈ 0.5%–3%
```

BCE 容易被 easy background 淹没，因此 A1 引入：

```text
Focal Loss + Dice Loss
```

### 6.2 模型

保持 SingleScaleEdgeHead 不变。A1 的唯一主要变量是 loss + threshold sweep。

### 6.3 Loss

```text
L_edge = λ_focal * FocalLoss(logit, target; alpha=0.75, gamma=2.0)
       + λ_dice  * SoftDiceLoss(sigmoid(logit), target)
```

推荐：

```yaml
loss:
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0
```

若前景低于 1%，可尝试：

```yaml
focal_alpha: 0.85
```

但不建议同时叠加强 `pos_weight`，否则容易出现厚边和假阳性海岸带。

### 6.4 Threshold sweep

A1 必须加入 threshold sweep。  
Focal Loss 会改变 heatmap calibration，如果继续固定 0.5，可能错误低估 A1 效果。

```yaml
postprocess:
  threshold_values:
    - 0.10
    - 0.15
    - 0.20
    - 0.25
    - 0.30
    - 0.35
    - 0.40
    - 0.45
    - 0.50
    - 0.55
    - 0.60
    - 0.65
    - 0.70
  threshold_select_metric: buffered_f1_1px
```

### 6.5 A1 Gate

A1 通过条件：

```text
1. 相比 A0，edge recall 或 buffered-F1 提升
2. threshold sweep 存在稳定最优区间
3. false positive 不失控
4. heatmap 边缘响应覆盖主岸线
5. predicted length / GT length 不严重失衡
```

若 A1 无提升，先排查：

```text
1. GT 是否错位
2. line_width_train 是否过窄
3. soft edge target 是否必要
4. positive/negative 采样比例是否错误
5. threshold 是否过高
6. postprocess 是否过度删除组件
```

---

## 7. P3-A2：Multi-scale Edge Supervision

### 7.1 目标

A2 解决海岸线任务的第二核心难点：

```text
细粒度定位 + 全局连续性
```

### 7.2 架构

采用 HED-style side outputs：

```text
P1 → side1_logit_56 → upsample 224
P2 → side2_logit_28 → upsample 224
P3 → side3_logit_14 → upsample 224
P4 → side4_logit_7  → upsample 224

fused_logit = 1×1 Conv(concat(side1_224, side2_224, side3_224, side4_224))
```

输出：

```python
{
    "fused": fused_logit_224,
    "side1": side1_logit_224,
    "side2": side2_logit_224,
    "side3": side3_logit_224,
    "side4": side4_logit_224,
}
```

### 7.3 Side output 作用

| 分支 | 来源 | 作用 |
|---|---|---|
| side1 | P1 | 细粒度边缘定位 |
| side2 | P2 | 局部岸线结构 |
| side3 | P3 | ViT 注入后的全局语义 |
| side4 | P4 | 宏观海陆布局和长线走向 |
| fused | side concat | 最终 heatmap |

### 7.4 Critical rule

所有 side output 必须 upsample 到 224×224 后再计算 loss：

```text
side_logit_native
  → bilinear upsample to 224×224
  → compute Focal + Dice against 224×224 GT
```

禁止直接在低分辨率下用 nearest/average 下采样 GT 计算细线 loss。  
如果未来必须在原生分辨率计算 side loss，GT 下采样必须使用 max-pooling。

### 7.5 Loss

```text
L_edge =
  1.0 * L_fused_224
+ 0.5 * L_side1_224
+ 0.3 * L_side2_224
+ 0.2 * L_side3_224
+ 0.1 * L_side4_224
```

每个 `L_*` 为：

```text
FocalLoss + DiceLoss
```

推荐：

```yaml
loss:
  type: deep_supervised_focal_dice
  fused_weight: 1.0
  side_weights:
    side1: 0.5
    side2: 0.3
    side3: 0.2
    side4: 0.1
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0
```

### 7.6 A2 Gate

A2 通过条件：

```text
1. fused heatmap 不低于 A1
2. side1 捕获细节边缘
3. side3 / side4 对大体岸线走向有响应
4. buffered-F1@3px 达到可用水平
5. component fragmentation 不明显恶化
6. predicted length / GT length 不严重失衡
```

若 A2 不如 A1，优先排查：

```text
1. side loss 权重是否过大
2. P4 side loss 是否过度粗化
3. side logits 是否全部正确 upsample 到 224
4. fused head 是否学到有效融合
5. concat 顺序是否错误
6. BN 是否 collapse
```

---

## 8. P3-A4：Topology-aware Postprocess

### 8.1 目标

A4 不改变训练主线，它负责将 heatmap 稳定转为合法 GIS geometry：

```text
edge heatmap → binary mask → skeleton → graph → LineString / MultiLineString
```

### 8.2 完整后处理流程

```text
edge_logit_224
  → sigmoid heatmap
  → threshold sweep
  → remove small components
  → morphological closing / gap bridge
  → skeletonize
  → graph extraction
  → spur pruning
  → extract valid paths
  → Douglas-Peucker simplify
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation
```

### 8.3 参数建议

```yaml
postprocess:
  threshold_sweep: true
  threshold_select_metric: buffered_f1_1px
  threshold_default: 0.30

  min_component_area: 8
  close_kernel: 3
  max_gap_bridge_px: 3
  min_line_length_px: 10
  spur_prune_length_px: 4
  douglas_peucker_epsilon_px: 1.0
  max_components: 5
```

### 8.4 LineString / MultiLineString 输出规则

不要强制只保留最长路径。  
海岸线 tile 可能包含岛岸线、河口、多段岸线。

推荐：

```text
valid_components = components with length >= max(10px, 0.15 * longest_length)
valid_components = top_k(valid_components, k=5)

if len(valid_components) == 0:
    output features=[]
elif len(valid_components) == 1:
    output LineString
else:
    output MultiLineString
```

输出要求：

```text
1. 每条 LineString 至少 2 个点
2. 不允许连续重复点
3. 坐标必须在 tile bounds 内
4. 过短 geometry 过滤
5. 空结果合法输出 features=[]
```

### 8.5 A4 Gate / PoC-3 正式通过标准

```text
1. buffered-F1@1px ≥ 0.70
2. buffered-F1@3px ≥ 0.85
3. average offset < 10m
4. Chamfer / Hausdorff 无明显异常
5. component count error 可控
6. predicted length / GT length 合理
7. GeoJSON parse success rate > 99%
8. GeoJSON schema valid rate > 99%
9. geometry valid rate ≥ 95%
10. coordinate-in-tile rate = 100%
11. empty output features=[] 正确处理
12. 无岸线 tile false positive rate 可控
```

---

## 9. Ground Truth 生成策略

### 9.1 支持标签来源

优先：

```text
GeoJSON LineString / MultiLineString
```

可选 fallback：

```text
Binary TIF edge mask
```

### 9.2 GeoJSON 到 pixel edge target

流程：

```text
GeoJSON coordinates in WGS84
  → pyproj Transformer: EPSG:4326 → source_crs
  → inverse model_transform
  → model pixel coordinates in 224×224
  → clip to tile bounds
  → densify polyline
  → draw edge map
```

### 9.3 Densify

GeoJSON 顶点可能稀疏，直接 rasterize 会断裂。  
必须 densify：

```yaml
gt:
  densify_max_step_px: 0.5
```

伪代码：

```python
def densify_polyline(points, max_step=0.5):
    dense = []
    for p0, p1 in zip(points[:-1], points[1:]):
        dense.append(p0)
        dist = euclidean_distance(p0, p1)
        n = max(1, ceil(dist / max_step))
        for i in range(1, n):
            t = i / n
            dense.append(lerp(p0, p1, t))
    dense.append(points[-1])
    return dense
```

### 9.4 训练标签与评估标签分离

| 标签 | 用途 | 生成方式 |
|---|---|---|
| `edge_center_1px` | evaluation / skeleton GT | draw polyline width=1 |
| `edge_train_target` | training loss | width=3 hard band 或 soft edge target |

### 9.5 Soft edge target

A1/A2 推荐切换到 soft edge target：

```text
distance = distance_transform_to_polyline
edge_soft = exp(-(distance^2) / (2 * sigma^2))
edge_soft[distance > radius] = 0
```

推荐：

```yaml
gt:
  line_width_train: 3
  line_width_eval: 1
  soft_edge: true
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
  densify_max_step_px: 0.5
```

---

## 10. 数据划分与采样

### 10.1 划分原则

不能随机 tile 划分。  
海岸线具有强空间连续性，相邻 tile 高度相关，随机划分会造成空间泄漏。

推荐：

```text
按 coastal segment / source image 分组划分 train / val
同一海岸段或同一源图像的所有 tile 必须落在同一 split
```

### 10.2 样本类型

| 样本类型 | 说明 | 作用 |
|---|---|---|
| positive-crossing | 海岸线穿过 tile | 主训练样本 |
| near-coast negative | tile 内无线，但靠近海岸 | 抑制近岸假阳性 |
| empty inland/sea | 完全无岸线 | 抑制无目标乱输出 |

建议采样比例：

```yaml
sampler:
  positive_crossing: 0.75
  near_coast_negative: 0.20
  empty_negative: 0.05
```

PoC-3 A0 可先只使用 positive samples 跑通闭环。  
PoC-4 / Stage 4 前必须加入 negative samples，否则 known class gating 下可能在无海岸线 tile 上输出虚假 LineString。

---

## 11. 评估指标

### 11.1 主指标

| 指标 | 目标 |
|---|---:|
| buffered-F1 @ 1px | ≥ 0.70 |
| buffered-F1 @ 3px | ≥ 0.85 |
| average offset | < 10m |

主排序：

```text
primary: buffered-F1@1px
secondary: Chamfer distance
tie-breaker:
  - connected component count error
  - predicted length / GT length
  - geometry valid rate
```

### 11.2 Pixel metrics

```text
pixel precision
pixel recall
pixel F1
foreground ratio pred
foreground ratio GT
```

### 11.3 Geometry metrics

```text
Chamfer distance
Hausdorff distance
average offset in meters
endpoint distance
```

### 11.4 Topology metrics

```text
connected component count error
predicted length / GT length
fragmentation ratio
spur count
```

### 11.5 GeoJSON validity metrics

```text
GeoJSON parse success rate
GeoJSON schema valid rate
geometry valid rate
coordinate-in-tile rate
empty-output correctness
```

---

## 12. 代码落点

### 12.1 文件规划

```text
utils/
  edge_losses.py
    - SoftDiceLoss
    - BinaryFocalLoss
    - edge_bce_dice_loss
    - edge_focal_dice_loss
    - DeepSupervisedEdgeLoss

Models/
  edge_head.py
    - SingleScaleEdgeHead
    - MultiScaleEdgeHead
    - optional ResidualEdgeRefineBlock

utils/
  edge_postprocess.py
    - heatmap_to_binary
    - binary_to_skeleton
    - skeleton_to_paths
    - graph extraction
    - spur pruning
    - Douglas-Peucker simplify
    - paths_to_geojson
    - postprocess_edge

utils/
  coastline_metrics.py
    - pixel_edge_metrics
    - buffered_f1
    - chamfer_distance
    - hausdorff_distance
    - length_ratio
    - component_count_error

Dataset/
  coastline_dataset.py
    - CoastlineEdgeDataset
    - manifest builder
    - GeoJSON parser
    - WGS84 → pixel
    - densify
    - edge target generation

configs/
  poc3_edge_a0_closure.yaml
  poc3_edge_a1_focal_dice.yaml
  poc3_edge_a2_multiscale.yaml
  poc3_edge_a4_topology.yaml

scripts/
  poc_stage_edge.py
    - train/eval entrypoint
    - single NPU
    - frozen vision encoder
    - FPN + Edge Head training
    - overlay export
    - GeoJSON export
```

### 12.2 实验目录

```text
outputs/poc3_edge/
  a0_bce_dice_closure/
    checkpoints/
    vis/
    geojson/
    metrics.json

  a1_focal_dice_sweep/
    checkpoints/
    threshold_sweep/
    vis/
    geojson/
    metrics.json

  a2_multiscale_deepsup/
    checkpoints/
    side_outputs/
    vis/
    geojson/
    metrics.json

  a4_topology_postprocess/
    threshold_sweep/
    graph_debug/
    final_geojson/
    final_metrics.json

  ablations/
    a3_dilation_context/
    a5_residual_refine/
```

---

## 13. 推荐实验矩阵

| 实验 | 架构 | Loss | GT | 后处理 | 目的 |
|---|---|---|---|---|---|
| E0 | single-scale | BCE + Dice | hard width=3 | threshold=0.5 | pipeline sanity |
| E1 | single-scale | Focal + Dice | hard width=3 | threshold sweep | 不平衡修复 |
| E2 | single-scale | Focal + Dice | soft edge | threshold sweep | 降低标注偏移敏感性 |
| E3 | multi-scale side outputs | Focal + Dice | soft edge | threshold sweep | 主候选模型 |
| E4 | E3 | Focal + Dice | soft edge | topology postprocess | 主候选输出方案 |
| E5 | E4 + dilation | Focal + Dice | soft edge | topology postprocess | 上下文增强消融 |
| E6 | E4 + residual refine | Focal + Dice | soft edge | topology postprocess | 边缘细化消融 |

推荐推进：

```text
E0 → E1 → E2 → E3 → E4
```

若 E4 达标，E5/E6 可不做。

---

## 14. 配置模板

### 14.1 A0 config

```yaml
experiment:
  name: poc3_a0_bce_dice_closure
  output_dir: outputs/poc3_edge/a0_bce_dice_closure
  seed: 42
  stage: P3-A0

data:
  roots:
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches
  manifest_path: outputs/poc3_edge/coastline_manifest.json
  image_size: 224
  val_ratio: 0.2
  val_split_seed: 42
  num_workers: 2
  line_width_train: 3
  line_width_eval: 1
  densify_max_step_px: 0.5

model:
  backbone: dual_vision_encoder
  fpn: vit_fpn
  vit_fusion: true
  edge_head:
    type: single_scale
    in_channels: 256
    decoder_channels: [256, 128]
    output_size: [224, 224]

train:
  task: edge_only
  device: npu
  epochs: 20
  batch_size: 2
  lr_fpn: 5.0e-5
  lr_edge_head: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  freeze_vision: true
  precision: bf16

loss:
  type: bce_dice

postprocess:
  mode: minimal
  threshold: 0.5
  min_component_area: 8
  min_line_length_px: 10
  max_components: 5
  simplify_epsilon_px: 1.0

eval:
  export_overlay: true
  export_geojson: true
  threshold_sweep: false
  max_overlay_samples: 12
```

### 14.2 A1 config delta

```yaml
experiment:
  name: poc3_a1_focal_dice_sweep
  output_dir: outputs/poc3_edge/a1_focal_dice_sweep
  stage: P3-A1

loss:
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0

postprocess:
  mode: threshold_sweep
  threshold_values: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  threshold_select_metric: buffered_f1_1px
```

### 14.3 A2 config delta

```yaml
experiment:
  name: poc3_a2_multiscale_deepsup
  output_dir: outputs/poc3_edge/a2_multiscale_deepsup
  stage: P3-A2

model:
  edge_head:
    type: multi_scale
    side_outputs: [P1, P2, P3, P4]
    fused_output: true
    output_size: [224, 224]

loss:
  type: deep_supervised_focal_dice
  fused_weight: 1.0
  side_weights:
    side1: 0.5
    side2: 0.3
    side3: 0.2
    side4: 0.1
  focal_alpha: 0.75
  focal_gamma: 2.0
```

### 14.4 A4 config delta

```yaml
experiment:
  name: poc3_a4_topology_postprocess
  output_dir: outputs/poc3_edge/a4_topology_postprocess
  stage: P3-A4

postprocess:
  mode: topology
  threshold_sweep: true
  threshold_select_metric: buffered_f1_1px
  min_component_area: 8
  close_kernel: 3
  max_gap_bridge_px: 3
  skeletonize: true
  graph_extract: true
  spur_prune_length_px: 4
  min_line_length_px: 10
  douglas_peucker_epsilon_px: 1.0
  max_components: 5

geometry:
  allow_linestring: true
  allow_multilinestring: true
  empty_output_allowed: true
```

---

## 15. Gate-driven Execution DAG

### 15.1 DAG

```text
Pre-flight sanity
  ↓
A0 dataset + GT overlay check
  ↓
A0 2-sample overfit
  ↓
A0 full closure
  ↓
Gate A0
  ↓
A1 Focal/Dice + threshold sweep
  ↓
Gate A1
  ↓
A2 Multi-scale deep supervision
  ↓
Gate A2
  ↓
A4 Topology-aware postprocess
  ↓
Gate A4
  ↓
PoC-3 pass / handoff to PoC-4
```

### 15.2 Pre-flight sanity

必须先做非训练检查：

```text
1. dataset scan
2. manifest build
3. image load
4. GeoJSON load
5. binary label fallback load
6. WGS84 → pixel transform
7. edge target generation
8. GT overlay export
9. foreground ratio统计
```

### 15.3 A0 overfit test

推荐：

```yaml
overfit:
  num_samples: 2
  iterations: 200
  batch_size: 2
```

通过：

```text
1. loss 明显下降
2. prediction 从全背景变成有边缘响应
3. heatmap 与 GT 大体重合
4. 无 NaN / inf
```

### 15.4 Gate summary

| Gate | 必须满足 |
|---|---|
| Gate A0 | GT 正确、loss 下降、GeoJSON 闭环 |
| Gate A1 | recall / buffered-F1 优于 A0，threshold sweep 有稳定最优 |
| Gate A2 | fused ≥ A1，side outputs 有解释性，碎片化不恶化 |
| Gate A4 | buffered-F1、geometry validity、coordinate-in-tile 达标 |

---

## 16. Debug Playbook

### 16.1 根因优先级

PoC-3 失败时，优先假设不是模型问题：

```text
1. CRS / pixel transform 错误          40%
2. GT rasterization 不一致            25%
3. threshold 固定或 calibration 错误   15%
4. skeleton / graph postprocess bug    10%
5. model capacity / feature 问题        10%
```

### 16.2 黄金 debug 路径

```text
1. GT overlay
2. pixel foreground ratio
3. model heatmap min/max/mean
4. threshold sweep curve
5. pred binary mask
6. skeleton result
7. graph connectivity
8. LineString / MultiLineString output
9. GeoJSON validity
10. metric sanity
```

### 16.3 常见失败模式

| 症状 | 优先怀疑 | 修复 |
|---|---|---|
| GT 不贴边 | CRS / affine 错 | 检查 source_crs、model_transform、resize_georef |
| fg_ratio < 0.1% | line_width 太窄或 rasterize 断裂 | densify、line_width=3、soft target |
| loss 不下降 | GT 错或全背景 | 可视化 GT、overfit 2 samples |
| prediction 全黑 | BCE 背景淹没 | A1 Focal、降低 threshold |
| FP 爆炸 | threshold 太低 / negative 不足 | threshold sweep、加入 near-coast negative |
| skeleton 断裂 | threshold 太高 / GT 断裂 | 降 threshold、gap bridge |
| 毛刺很多 | noise / pruning 不足 | spur pruning、min_component_area |
| 多段岸线丢失 | 只保留 longest path | 支持 MultiLineString top-k |
| GeoJSON invalid | 重复点 / NaN / 越界 | remove duplicate、clamp、schema validation |
| buffered-F1 低但视觉正常 | metric 或 GT centerline 问题 | 检查 eval GT、buffer radius、坐标一致性 |

---

## 17. 可视化与日志

### 17.1 每轮保存可视化

```text
outputs/poc3_edge/<stage>/vis/
  sample_id_image.png
  sample_id_gt_center.png
  sample_id_gt_train_target.png
  sample_id_pred_heatmap.png
  sample_id_pred_binary.png
  sample_id_pred_skeleton.png
  sample_id_gt_pred_overlay.png
  sample_id_geojson_overlay.png
```

### 17.2 必须记录的 model stats

```text
heatmap_min
heatmap_max
heatmap_mean
heatmap_entropy
pred_fg_ratio
gt_fg_ratio
loss_total
loss_focal / loss_bce
loss_dice
```

### 17.3 必须记录的 topology stats

```text
num_components_before_filter
num_components_after_filter
skeleton_pixel_count
path_count
path_length_mean
path_length_max
spur_count
predicted_length_px
gt_length_px
length_ratio
```

### 17.4 必须记录的 GeoJSON stats

```text
parse_success
schema_valid
geometry_valid
coordinate_in_tile
num_features
geometry_type
empty_output
```

---

## 18. 命令级执行建议

### 18.1 Build manifest only

```bash
cd /home/ma-user/work/CoastGPT

python scripts/poc_stage_edge.py \
  --config configs/poc3_edge_a0_closure.yaml \
  --build-manifest-only
```

### 18.2 Dataset smoke test

```bash
python Dataset/coastline_dataset.py
```

### 18.3 Loss smoke test

```bash
python utils/edge_losses.py
```

### 18.4 Edge head shape test

```bash
python Models/edge_head.py
```

### 18.5 Postprocess smoke test

```bash
python utils/edge_postprocess.py
```

### 18.6 Metrics smoke test

```bash
python utils/coastline_metrics.py
```

### 18.7 A0 CPU dry run

```bash
python scripts/poc_stage_edge.py \
  --config configs/poc3_edge_a0_closure.yaml \
  --device cpu \
  --epochs 1 \
  --batch-size 2
```

### 18.8 A0 NPU full run

```bash
python scripts/poc_stage_edge.py \
  --config configs/poc3_edge_a0_closure.yaml \
  --device npu
```

---

## 19. 保存策略

```text
outputs/poc3_edge/<stage>/
  checkpoints/
    epoch_005.pt
    epoch_010.pt
    epoch_020.pt
    best.pt

  configs/
    resolved_config.yaml

  metrics/
    train_loss.csv
    val_metrics.csv
    threshold_sweep.csv
    final_metrics.json

  vis/
    overlays/

  geojson/
    predictions/

  debug/
    gt_overlay/
    skeleton/
    graph/
```

保存内容：

```text
FPN weights
Edge Head weights
optimizer state
epoch
best metric
threshold chosen
postprocess config
label manifest hash
```

Vision encoder 冻结，不保存 base weights。

---

## 20. 与 PoC-4 / PoC-5 的接口

PoC-3 通过后，为 PoC-4 提供标准输出：

```python
edge_result = {
    "class": "海岸线",
    "geometry_type": "LineString" or "MultiLineString",
    "confidence": float,
    "pixel_geometry": ...,
    "wgs84_geometry": ...,
    "valid": bool,
    "debug": {
        "threshold": float,
        "num_components": int,
        "length_px": float,
        "buffered_f1_1px": float,
    }
}
```

Known class gating：

```text
prompt 包含 "海岸线"
  → 调用 Edge Head
  → 输出检测头 GeoJSON
  → 不调用 LLM 生成已知海岸线
```

LLM fallback 只处理 unknown class，并且必须经过：

```text
JSON parse
GeoJSON schema validation
geometry validity
coordinate-in-tile check
geometry type policy
area/length abnormal filtering
```

---

## 21. 最终通过标准

PoC-3 最终通过条件：

```text
1. A0 闭环通过
2. A1 相比 A0 有稳定收益
3. A2 fused heatmap 不低于 A1
4. A4 topology postprocess 输出稳定
5. buffered-F1@1px ≥ 0.70
6. buffered-F1@3px ≥ 0.85
7. average offset < 10m
8. GeoJSON parse success rate > 99%
9. GeoJSON schema valid rate > 99%
10. geometry valid rate ≥ 95%
11. coordinate-in-tile rate = 100%
12. empty-output features=[] 正确
13. LineString / MultiLineString 类型输出正确
14. 可视化 overlay 与原图基本对齐
```

---

## 22. 一句话总结

PoC-3 的本质不是单纯训练一个边缘检测模型，而是：

> **基于 ViT-FPN 的海岸线边缘推理 + GIS 拓扑重建系统。**

模型负责提供高质量 heatmap，  
GT 管线负责保证监督正确，  
后处理负责恢复拓扑连续线，  
GeoJSON 校验负责保证输出可用。

最终系统应以：

```text
edge accuracy
+ topology continuity
+ coordinate correctness
+ GeoJSON validity
```

共同作为成功标准。

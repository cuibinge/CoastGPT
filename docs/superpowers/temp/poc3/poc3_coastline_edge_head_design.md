# PoC-3 海岸线 Edge Head 设计方案

> 日期：2026-06-17  
> 状态：设计稿  
> 适用阶段：CoastGPT 检测头 + LLM 混合架构 PoC-3  
> 前置依赖：PoC-2b 土地覆盖 Semantic Head 已完成；ViT-FPN 架构可复用  
> 核心目标：完成海岸线 Edge Detection 闭环，并验证细线目标在 GeoJSON 输出中的空间精度与拓扑稳定性

---

## 1. 背景与目标

PoC-3 面向 **海岸线检测**，任务目标不是输出区域 polygon，而是从遥感 tile 中提取连续的海岸线，并导出为 GeoJSON `LineString` 或 `MultiLineString`。

完整闭环如下：

```text
GeoJSON LineString / MultiLineString GT
  → WGS84 → source CRS → model pixel space
  → rasterize / distance-transform edge map
  → Edge Head 训练
  → edge heatmap
  → threshold / topology postprocess / skeletonize
  → polyline extraction
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation
```

PoC-3 的关键不是单纯做 binary segmentation，而是解决海岸线任务特有的几个问题：

| 维度 | Semantic Segmentation PoC-2b | Edge Detection PoC-3 |
|---|---|---|
| 前景像素比例 | 10–60% | 0.5–3%，极端不平衡 |
| 空间精度要求 | 区域级，IoU 容错较高 | 像素级，1px 偏移可能导致断线 |
| 拓扑敏感性 | 较低，region 可容错 | 很高，断线会导致碎片化 LineString |
| 上下文依赖 | 局部纹理 + 类别语义 | 全局连续性，海岸线通常贯穿 tile |
| 输出几何 | Polygon / MultiPolygon | LineString / MultiLineString |

因此，PoC-3 应采用 **先闭环、再增强** 的策略：  
先用最小 Edge Head 跑通数据、训练、后处理和 GeoJSON 输出，再针对 edge-specific 问题引入 Focal Loss、多尺度深监督和拓扑后处理。

---

## 2. 设计原则

### 2.1 复用 PoC-2b 的 ViT-FPN

PoC-2b 已证明 ViT-FPN 对土地覆盖有效，其中 DINOv3 ViT `g_grid` 通过 `vit_proj` 注入 FPN P3，提供全局语义上下文。海岸线比土地覆盖更依赖全局连续性，因此 PoC-3 应优先复用这套 ViT-FPN，而不是退回纯 ConvNeXt FPN。

推荐主干：

```text
DualVisionEncoder
  ├─ ConvNeXt pyramid_raw: c4/c8/c16/c32
  └─ DINOv3 ViT g_grid

ViT-FPN
  ├─ ConvNeXt local pyramid
  └─ ViT global feature injected into P3

Edge Head
  └─ edge heatmap → skeleton / polyline
```

### 2.2 A0 只做 pipeline validation，不作为性能结论

单尺度 `BCE + Dice` 可以作为最小闭环，但不能代表 PoC-3 的最终能力。  
由于海岸线前景像素极少，BCE 很容易被背景主导，A0 结果只能用于确认：

- GT 生成是否正确；
- loss 是否能下降；
- heatmap 是否大体对齐；
- skeleton / GeoJSON 输出是否能跑通；
- 坐标转换和校验是否正常。

正式 baseline 应从 `Focal + Dice` 开始。

### 2.3 后处理是 PoC-3 成败关键

海岸线最终输出不是 heatmap，而是 LineString / MultiLineString。  
因此模型输出之后必须有稳定的 topology postprocess，包括：

- threshold sweep；
- remove small components；
- gap linking；
- skeletonize；
- graph extraction；
- spur pruning；
- top-k line extraction；
- Douglas-Peucker 简化；
- GeoJSON geometry validation。

---

## 3. 阶段路线

| 阶段 | 名称 | 架构 | Loss | 目的 | 是否作为最终候选 |
|---|---|---|---|---|---|
| P3-A0 | Baseline closure | Single-scale Edge Head | BCE + Dice | 跑通数据、训练、坐标、GeoJSON 闭环 | 否 |
| P3-A1 | Imbalance fix | Single-scale Edge Head | Focal + Dice | 解决极端前景/背景不平衡 | 是 |
| P3-A2 | Multi-scale edge | HED-style side outputs | side Focal+Dice + fused Focal+Dice | 兼顾细节边缘和全局岸线走向 | 是，主推 |
| P3-A3 | Context refinement | Dilation conv / ASPP-lite | 同 A2 | 扩大感受野，提升长线连续性 | 可选 |
| P3-A4 | Topology postprocess | 不改网络 | 不变 | 减少断线、毛刺、碎片化输出 | 必做 |
| P3-A5 | Residual refinement | Residual edge refine block | 同 A2/A3 | 细化边缘位置 | 可选消融 |

推荐最终候选为：

```text
P3-A2 + P3-A4
```

即 **多尺度深监督 Edge Head + Focal/Dice + topology postprocess**。

---

## 4. 数据与 GT 生成

### 4.1 输入数据

海岸线数据来源：

```text
海岸线 / RS-海岸线一级+二级
```

任务分支：

```json
{
  "task": "DET",
  "branch": "edge",
  "geometry_type": "LineString",
  "known_classes": ["海岸线"]
}
```

支持几何类型：

- `LineString`
- `MultiLineString`

若数据中存在 `Polygon` 边界表达，需要在数据预处理阶段显式转换为 coastline polyline，不应在训练阶段隐式猜测。

---

### 4.2 GeoJSON 到 model pixel

GT 转换流程：

```text
GeoJSON coordinates in WGS84
  → pyproj Transformer: EPSG:4326 → source_crs
  → inverse model_transform
  → model pixel coordinates in 224×224
  → clip to tile bounds
  → densify polyline
  → draw edge map
```

核心约定：

| 项 | 约定 |
|---|---|
| 训练/推理统一空间 | model input pixel space，224×224 |
| LineString 点坐标 | pixel center 坐标 |
| 坐标转换 | 使用 `model_transform` |
| CRS 转换精度 | float32 以上，坐标模块禁用 bf16 |
| tile 外线段 | clip 到 tile bounds |
| 空 GT | 允许，输出 `features=[]` |

---

### 4.3 Densify polyline

如果 GeoJSON 顶点稀疏，直接 rasterize 容易造成像素级断裂。  
因此所有 polyline 在 rasterize 前需要 densify。

推荐参数：

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
        n = max(1, int(math.ceil(dist / max_step)))
        for i in range(1, n):
            t = i / n
            dense.append(lerp(p0, p1, t))
    dense.append(points[-1])
    return dense
```

---

### 4.4 训练标签与评估标签分离

海岸线训练不建议只用 1px hard centerline。  
1px GT 对轻微标注偏差过于敏感，容易导致模型训练不稳定。

建议生成两类标签：

| 标签 | 用途 | 生成方式 |
|---|---|---|
| `edge_center_1px` | evaluation / skeleton target | draw polyline width=1 |
| `edge_train_target` | training loss | width=3 hard band 或 soft distance target |

推荐从 A1 起使用 soft edge target：

```text
distance = distance_transform_to_polyline
edge_soft = exp(-(distance^2) / (2 * sigma^2))
edge_soft[distance > radius] = 0
```

推荐参数：

```yaml
gt:
  line_width_eval: 1
  line_width_train: 3
  soft_edge: true
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
```

A0 可先使用 hard width=3，降低 pipeline 复杂度。

---

## 5. Edge Head 架构

### 5.1 A0/A1：Single-scale Edge Head

结构与 PoC-2b semantic decoder 保持相近，但输出为 1-channel binary logit。

```text
输入:
  P1 [B,256,56,56]
  P2 [B,256,28,28]
  P3 [B,256,14,14]
  P4 [B,256,7,7]

处理:
  P2/P3/P4 bilinear upsample → 56×56
  concat(P1,P2,P3,P4) → [B,1024,56,56]
  3×3 Conv 1024→256 + BN + ReLU
  3×3 Conv 256→128 + BN + ReLU
  1×1 Conv 128→1 → edge_logit_56 [B,1,56,56]
  bilinear upsample → edge_logit_224 [B,1,224,224]

输出:
  edge_logit_224
```

训练 loss 建议在 224×224 上计算。  
不要只在 56×56 上计算 loss，因为细线目标在低分辨率下会发生 aliasing，导致模型学习粗带状响应而不是中心线。

---

### 5.2 A2：Multi-scale Deep Supervision Edge Head

采用 HED-style side outputs。

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
    "side4": side4_logit_224
}
```

Side output 作用：

| 分支 | 分辨率 | 作用 |
|---|---:|---|
| side1 / P1 | 56×56 | 细粒度边缘定位 |
| side2 / P2 | 28×28 | 局部岸线结构 |
| side3 / P3 | 14×14 | ViT 注入后的全局语义 |
| side4 / P4 | 7×7 | 宏观海陆布局和长线走向 |
| fused | 224×224 | 最终 heatmap |

注意：  
side loss 建议全部 upsample 到 224 后和同一份 GT 计算。  
如果未来要在原生分辨率算 side loss，GT 下采样必须用 max-pooling，而不是 nearest 或 average，否则细线会消失。

---

### 5.3 A3：Dilation Context Module

A3 是可选增强，不应阻塞 A1/A2。

轻量版本：

```text
concat(P1,P2,P3,P4)
  → 3×3 Conv dilation=1
  → 3×3 Conv dilation=2
  → 1×1 Conv
  → edge logit
```

多分支版本：

```text
3×3 Conv dilation=1
3×3 Conv dilation=2
3×3 Conv dilation=4
concat → 1×1 fuse
```

作用：

- 不降低空间分辨率；
- 扩大感受野；
- 有助于追踪贯穿 tile 的连续岸线；
- 可能增加 false positive，需要和 threshold sweep 一起评估。

---

### 5.4 A5：Residual Edge Refinement Block

该模块可选，建议只作为后期消融。

```text
edge_logit_224
  → 3×3 Conv 1→16 + ReLU
  → 3×3 Conv 16→1
  → delta_logit
  → refined_logit = edge_logit_224 + delta_logit
```

不建议在首轮加入该模块，原因：

- 增加变量；
- 可能过拟合标注噪声；
- 对 PoC-3 首要问题，即极端不平衡和拓扑断裂，不是最高优先级。

---

## 6. Loss 设计

### 6.1 A0：BCE + Dice

仅用于 pipeline validation。

```text
L_edge = BCEWithLogits(edge_logit, edge_target)
       + DiceLoss(sigmoid(edge_logit), edge_target)
```

通过标准：

- loss 能下降；
- heatmap 不是全背景；
- 可视化基本对齐；
- skeleton / GeoJSON 输出链路跑通。

---

### 6.2 A1：Focal + Dice

正式 baseline。

```text
L_edge = λ_focal * FocalLoss(logit, target; alpha, gamma)
       + λ_dice  * SoftDiceLoss(sigmoid(logit), target)
```

推荐初始参数：

```yaml
loss:
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0
```

说明：

- `gamma=2.0`：降低大量 easy background 的权重；
- `alpha=0.75`：提高前景边缘像素权重；
- 当前不建议叠加强 `pos_weight`，否则容易预测厚边和假阳性海岸带；
- 若前景像素低于 1%，可尝试 `alpha=0.85`。

---

### 6.3 A2：Side-output Deep Supervision Loss

```text
L_edge =
  1.0 * L_fused_224
+ 0.5 * L_side1_224
+ 0.3 * L_side2_224
+ 0.2 * L_side3_224
+ 0.1 * L_side4_224
```

其中每个 `L_*` 使用：

```text
L = FocalLoss + DiceLoss
```

推荐配置：

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

---

## 7. 后处理设计

### 7.1 基本流程

```text
edge_logit_224
  → sigmoid heatmap
  → threshold
  → remove small components
  → morphological closing / gap bridge
  → skeletonize
  → graph extraction
  → prune short spurs
  → extract valid paths
  → Douglas-Peucker simplify
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
```

---

### 7.2 阈值搜索

不要固定 `threshold=0.5`。  
edge heatmap 的最佳阈值会随 loss、GT 宽度、soft target 和数据分布变化。

验证集 sweep：

```yaml
postprocess:
  threshold_sweep: true
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

初始经验默认值：

```yaml
postprocess:
  threshold_default: 0.30
```

---

### 7.3 连通性修复

推荐参数：

```yaml
postprocess:
  min_component_area: 8
  close_kernel: 3
  max_gap_bridge_px: 3
  min_line_length_px: 10
  spur_prune_length_px: 4
  douglas_peucker_epsilon_px: 1.0
  max_components: 5
```

说明：

| 参数 | 作用 |
|---|---|
| `min_component_area` | 删除孤立噪声点 |
| `close_kernel` | 连接近邻小断裂 |
| `max_gap_bridge_px` | 对 skeleton graph 做短缺口桥接 |
| `min_line_length_px` | 删除过短 LineString |
| `spur_prune_length_px` | 删除骨架毛刺 |
| `douglas_peucker_epsilon_px` | 简化线段，减少冗余点 |
| `max_components` | 限制 MultiLineString 组件数 |

---

### 7.4 LineString / MultiLineString 输出策略

不要强制只保留最长连通路径。  
某些 tile 可能包含岛岸线、河口或多个海岸片段。

推荐策略：

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

- 每条 LineString 至少 2 个点；
- 不允许连续重复点；
- 坐标必须在 tile bounds 内；
- 过短 geometry 过滤；
- 空结果合法输出 `features=[]`。

---

## 8. 评估指标

### 8.1 主指标

| 指标 | 目标 |
|---|---:|
| buffered-F1 @ 1px | ≥ 0.70 |
| buffered-F1 @ 3px | ≥ 0.85 |
| average offset | < 10m |
| GeoJSON parse success rate | > 99% |
| GeoJSON schema valid rate | > 99% |
| geometry valid rate | ≥ 95% |
| coordinate-in-tile rate | 100% |

---

### 8.2 诊断指标

| 指标 | 目的 |
|---|---|
| pixel precision / recall / F1 | 检查 edge map 二值预测质量 |
| Chamfer distance | 平均几何偏移 |
| Hausdorff distance | 最坏偏移 / 断裂情况 |
| connected component count error | 拓扑碎片化诊断 |
| predicted length / GT length | 检查漏线或假阳性过长 |
| endpoint distance | 检查海岸线是否在 tile 中间断裂 |
| empty tile false positive rate | 检查无岸线 tile 上是否乱输出 |

主排序建议：

```text
primary: buffered-F1@1px
secondary: Chamfer distance
tie-breaker:
  - connected component count error
  - predicted length / GT length
  - geometry valid rate
```

---

## 9. 数据划分与采样

### 9.1 划分原则

不能随机 tile 划分。  
海岸线具有强空间连续性，相邻 tile 高度相关，随机划分会造成空间泄漏。

推荐：

```text
按 coastal segment / source image 分组划分 train / val
同一海岸段的所有 tile 必须落在同一 split
```

---

### 9.2 样本类型

| 样本类型 | 说明 | 作用 |
|---|---|---|
| positive-crossing | 海岸线穿过 tile | 主训练样本 |
| near-coast negative | tile 内无线，但靠近海岸 | 抑制近岸假阳性 |
| empty inland/sea | 完全无岸线 | 抑制无目标乱输出 |

初始采样比例：

```yaml
sampler:
  positive_crossing: 0.75
  near_coast_negative: 0.20
  empty_negative: 0.05
```

如果当前海岸线数据几乎全部为 positive tile，则 PoC-3 首轮可以不引入 empty negative。  
但进入 PoC-4/Stage 4 之前必须加入无海岸线样本，否则 known class gating 下可能在无岸线 tile 上输出虚假 LineString。

---

## 10. 实验矩阵

| 实验 | 架构 | Loss | GT | 后处理 | 目的 |
|---|---|---|---|---|---|
| E0 | single-scale | BCE + Dice | hard width=3 | threshold=0.5 | pipeline sanity |
| E1 | single-scale | Focal + Dice | hard width=3 | threshold sweep | 不平衡修复 |
| E2 | single-scale | Focal + Dice | soft edge | threshold sweep | 降低标注偏移敏感性 |
| E3 | multi-scale side outputs | Focal + Dice | soft edge | threshold sweep | 主候选模型 |
| E4 | E3 | Focal + Dice | soft edge | topology postprocess | 主候选输出方案 |
| E5 | E4 + dilation | Focal + Dice | soft edge | topology postprocess | 上下文增强消融 |
| E6 | E4 + residual refine | Focal + Dice | soft edge | topology postprocess | 边缘细化消融 |

推荐推进顺序：

```text
E0 → E1 → E2 → E3 → E4
```

若 E4 已达标，E5/E6 可不做或仅作为提升实验。

---

## 11. 训练配置建议

```yaml
model:
  backbone: dual_vision_encoder
  fpn: vit_fpn
  vit_fusion: true

  edge_head:
    type: edge_head
    out_channels: 1
    decoder_channels: [256, 128]
    deep_supervision: false    # E3 开启
    dilation: false            # E5 开启
    residual_refine: false     # E6 开启

train:
  task: edge_only
  input_size: 224
  batch_size_per_gpu: 2
  accum_steps: 8
  epochs: 40
  lr_fpn: 5.0e-5
  lr_edge_head: 1.0e-4
  weight_decay: 1.0e-4
  freeze_vision: true
  freeze_llm: true
  precision: bf16              # 坐标转换模块除外

loss:
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0

gt:
  line_width_train: 3
  line_width_eval: 1
  soft_edge: true
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
  densify_max_step_px: 0.5

postprocess:
  threshold_sweep: true
  threshold_select_metric: buffered_f1_1px
  threshold_default: 0.30
  threshold_values: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  min_component_area: 8
  close_kernel: 3
  max_gap_bridge_px: 3
  min_line_length_px: 10
  spur_prune_length_px: 4
  douglas_peucker_epsilon_px: 1.0
  max_components: 5

eval:
  primary_metric: buffered_f1_1px
  metrics:
    - pixel_precision
    - pixel_recall
    - pixel_f1
    - buffered_f1_1px
    - buffered_f1_3px
    - chamfer_distance_px
    - hausdorff_distance_px
    - average_offset_m
    - component_count_error
    - predicted_length_gt_length_ratio
    - endpoint_distance_px
    - geojson_parse_success_rate
    - geojson_schema_valid_rate
    - geometry_valid_rate
    - coordinate_in_tile_rate
```

---

## 12. 代码落点建议

建议新增或修改以下文件：

```text
Models/
  edge_head.py
    - SingleScaleEdgeHead
    - MultiScaleEdgeHead
    - ResidualEdgeRefineBlock

scripts/
  poc_stage_edge.py
    - train / eval entrypoint
    - loss routing: edge only
    - checkpoint save / resume

data/
  coastline_dataset.py
    - GeoJSON LineString/MultiLineString parser
    - WGS84 → model pixel
    - densify
    - edge target generation

utils/
  edge_losses.py
    - FocalLoss
    - SoftDiceLoss
    - DeepSupervisedEdgeLoss

utils/
  edge_postprocess.py
    - threshold
    - connected components
    - gap bridge
    - skeletonize
    - graph extraction
    - spur pruning
    - line extraction
    - Douglas-Peucker simplify

utils/
  coastline_metrics.py
    - buffered-F1
    - Chamfer distance
    - Hausdorff distance
    - length ratio
    - component count error
    - endpoint distance

configs/
  poc3_edge_baseline.yaml
  poc3_edge_focal_dice.yaml
  poc3_edge_multiscale.yaml
```

---

## 13. 可视化与调试输出

每个 eval epoch 保存固定样本可视化：

```text
outputs/poc3_edge/vis/
  sample_id_image.png
  sample_id_gt_center.png
  sample_id_gt_train_target.png
  sample_id_pred_heatmap.png
  sample_id_pred_binary.png
  sample_id_pred_skeleton.png
  sample_id_gt_pred_overlay.png
  sample_id_geojson_overlay.png
```

必须检查：

| 可视化 | 检查项 |
|---|---|
| image + GT centerline | 坐标转换是否正确 |
| GT train target | soft band 是否过宽 |
| pred heatmap | 是否全背景 / 全前景 |
| pred binary | threshold 是否合理 |
| pred skeleton | 是否断线、毛刺过多 |
| GeoJSON overlay | WGS84 输出是否回贴正确 |

---

## 14. 通过标准

### 14.1 A0 最小闭环通过标准

```text
1. GeoJSON LineString/MultiLineString 能正确解析
2. WGS84 → pixel → WGS84 round-trip 误差达标
3. edge GT 可视化与原图海岸线对齐
4. loss 能下降
5. predicted heatmap 非全背景
6. skeletonize 可生成有效 polyline
7. GeoJSON 输出可 parse
8. coordinate-in-tile rate = 100%
```

A0 不要求达到最终 F1 指标。

---

### 14.2 正式 PoC-3 通过标准

```text
1. buffered-F1@1px ≥ 0.70
2. buffered-F1@3px ≥ 0.85
3. average offset < 10m
4. GeoJSON parse success rate > 99%
5. GeoJSON schema valid rate > 99%
6. geometry valid rate ≥ 95%
7. coordinate-in-tile rate = 100%
8. 空输出 features=[] 可正确处理
9. 无岸线 tile 的 false positive rate 可控
10. 可视化 overlay 与原图基本对齐
```

---

## 15. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 前景像素极少 | BCE 被背景淹没，模型全背景 | 使用 Focal + Dice；控制 empty negative 比例 |
| GT 线过细 | 训练不稳定，对标注偏移过敏 | 使用 width=3 或 soft edge target |
| GeoJSON 顶点稀疏 | rasterize 后断裂 | densify polyline，max_step≤0.5px |
| threshold 固定不合理 | recall/precision 失衡 | 验证集 threshold sweep |
| skeleton 毛刺多 | LineString 质量差 | spur pruning + min length filter |
| 只保留最长线 | 岛岸线/多段岸线丢失 | 输出 MultiLineString，保留 top-k valid paths |
| 空 tile 乱输出 | PoC-4/Stage 4 gating 误报 | 加入 near-coast negative 和 empty negative |
| 低分辨率 side loss 细线消失 | 多尺度监督无效 | side logits upsample 到 224 再算 loss |
| 过强 dilation | 假阳性岸线带增加 | A3 仅作为消融，需配合 precision 指标评估 |

---

## 16. 最终建议

PoC-3 推荐主线：

```text
E0: single-scale BCE+Dice 跑通闭环
  → E1: Focal+Dice 作为正式 baseline
  → E2: soft edge target 降低标注偏移敏感性
  → E3: multi-scale side-output deep supervision
  → E4: threshold sweep + topology postprocess
```

最终候选方案：

```text
ViT-FPN
  + Multi-scale Edge Head
  + Focal/Dice deep supervision
  + soft edge target
  + threshold sweep
  + skeleton graph postprocess
  + LineString/MultiLineString GeoJSON validation
```

不建议一开始就加入 dilation 或 residual refinement。  
它们应作为 E5/E6 消融，在 E4 仍未达标或需要进一步提升连续性时再引入。

---

## 17. 与后续 PoC 的关系

PoC-3 通过后，进入：

```text
PoC-4: LLM fallback + known/unknown gating + 多几何类型融合
PoC-5: Stage 4 联合训练
```

PoC-3 需要为 PoC-4 提供稳定接口：

```python
edge_result = {
    "class": "海岸线",
    "geometry_type": "LineString" | "MultiLineString",
    "confidence": float,
    "pixel_geometry": ...,
    "wgs84_geometry": ...,
    "valid": bool,
    "metrics_debug": {
        "threshold": float,
        "num_components": int,
        "length_px": float
    }
}
```

known class gating 中：

```text
prompt 目标包含 "海岸线"
  → 调用 Edge Head
  → 输出检测头 GeoJSON
  → 不调用 LLM 生成已知海岸线
```

LLM fallback 只处理 unknown class，且必须经过 GeoJSON schema、geometry validity、coordinate-in-tile 和 geometry type 校验后才能进入最终 FeatureCollection。

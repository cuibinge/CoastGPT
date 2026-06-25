# PoC-3 阶段路线确认：A0 → A1 → A2 → A4

> 日期：2026-06-17  
> 状态：设计决策确认稿  
> 适用范围：CoastGPT PoC-3 海岸线 Edge Head  
> 主题：确认 PoC-3 是否采用 `A0 → A1 → A2 → A4` 的推进顺序  
> 结论：推荐采用该主线；但 A4 的基础后处理能力应从 A0 起保留，A4 阶段再升级为完整 topology-aware postprocess。

---

## 1. 决策结论

PoC-3 推荐采用如下阶段路线：

```text
A0: BCE + Dice 单尺度闭环
  → A1: Focal + Dice 解决前景稀疏
  → A2: 多尺度深监督解决细边缘 + 全局连续性
  → A4: 拓扑后处理稳定 GeoJSON LineString / MultiLineString
```

其中：

```text
A2 + A4 = 最终候选方案
A3 / A5 = 可选消融，不进入首轮主线
```

该路线合理，因为每一阶段只引入一个主要变量，符合 PoC 的最小可验证原则：

| 阶段 | 主要新增变量 | 验证目标 |
|---|---|---|
| A0 | 最小单尺度 Edge Head + BCE/Dice | 验证 GT、训练、坐标、GeoJSON 闭环 |
| A1 | Focal Loss + threshold sweep | 验证极端类别不平衡是否是主瓶颈 |
| A2 | HED-style 多尺度 side outputs | 验证细粒度定位与全局岸线连续性 |
| A4 | topology-aware postprocess | 稳定 heatmap → LineString / MultiLineString 输出 |

最终建议：

```text
保留 A0 → A1 → A2 → A4 顺序。
但不要把 A4 理解为“最后才开始做后处理”。
正确做法是：
  - A0 就包含最小后处理，保证闭环；
  - A4 再升级为完整拓扑后处理。
```

---

## 2. 为什么该推进顺序合理

### 2.1 A0 必须先做闭环

PoC-3 的第一风险不是模型性能，而是完整链路是否正确：

```text
GeoJSON LineString / MultiLineString
  → WGS84 → source CRS → model pixel
  → edge GT
  → Edge Head 训练
  → heatmap
  → skeleton / polyline
  → pixel → WGS84
  → GeoJSON
  → schema / geometry validation
```

如果 A0 不先跑通，后续 Focal Loss、多尺度深监督和拓扑后处理的效果都无法解释。

A0 的目标不是达标，而是验证：

- GeoJSON 解析正确；
- WGS84 ↔ pixel 坐标转换正确；
- densify / rasterize 后 GT 与图像对齐；
- loss 能下降；
- 模型不是全背景输出；
- skeletonize 能生成 polyline；
- GeoJSON 能合法输出；
- coordinate-in-tile rate 达到 100%。

因此 A0 必须存在。

---

### 2.2 A1 紧跟 A0 是必要的

海岸线 Edge Detection 与 PoC-2b Semantic Segmentation 的最大差异之一是前景像素比例极低：

```text
Semantic foreground: 10–60%
Edge foreground:    0.5–3%
```

在这种前景比例下，纯 BCE 很容易被背景像素主导，模型可能学成：

```text
降低整体 loss 的最简单方式 = 全背景或极弱边缘响应
```

因此 A1 应在 A0 之后立刻引入：

```text
Focal Loss + Dice Loss
```

推荐形式：

```text
L_edge = λ_focal * FocalLoss(logit, target; alpha=0.75, gamma=2.0)
       + λ_dice  * SoftDiceLoss(sigmoid(logit), target)
```

A1 是正式 baseline，不应继续把 A0 的 BCE+Dice 结果作为性能判断依据。

---

### 2.3 A2 再引入多尺度深监督

A2 的目标是解决海岸线的第二个核心难点：

```text
细节定位 + 全局连续性
```

PoC-3 需要同时关注：

| FPN 层级 | 作用 |
|---|---|
| P1 | 细粒度边缘定位 |
| P2 | 局部海陆边界结构 |
| P3 | ViT 注入后的全局语义 |
| P4 | 宏观海陆布局和长线走向 |

因此 A2 采用 HED-style side outputs 是合理的：

```text
P1 → side1_logit_56 → upsample 224
P2 → side2_logit_28 → upsample 224
P3 → side3_logit_14 → upsample 224
P4 → side4_logit_7  → upsample 224

fused_logit = 1×1 Conv(concat(side1, side2, side3, side4))
```

Loss：

```text
L_edge =
  1.0 * L_fused_224
+ 0.5 * L_side1_224
+ 0.3 * L_side2_224
+ 0.2 * L_side3_224
+ 0.1 * L_side4_224
```

注意：

```text
所有 side loss 都应 upsample 到 224×224 后再计算。
不要在 7×7 / 14×14 / 28×28 原生分辨率上直接用 nearest/average 下采样 GT 计算细线 loss。
```

原因是细线在低分辨率下容易消失或 aliasing。  
如果未来必须在原生分辨率计算 side loss，GT 下采样必须使用 max-pooling。

---

### 2.4 A4 是最终输出质量稳定化

A4 不改变网络训练主线，它解决的是：

```text
heatmap → binary mask → skeleton → graph → LineString/MultiLineString
```

的几何稳定性问题。

如果没有 A4，即使 heatmap 看起来合理，也可能输出：

- 断裂线；
- 过多毛刺；
- 过短碎片；
- 单 tile 多条岸线被强行合并；
- 岛岸线被最长路径策略丢弃；
- LineString 点过密；
- geometry invalid；
- coordinate 超出 tile bounds。

因此 A4 是最终候选必需阶段。

推荐最终后处理：

```text
sigmoid heatmap
  → threshold sweep
  → remove small components
  → morphological closing / gap bridge
  → skeletonize
  → graph extraction
  → spur pruning
  → extract top-k valid paths
  → Douglas-Peucker simplify
  → pixel → WGS84
  → GeoJSON LineString / MultiLineString
  → schema + geometry validation
```

---

## 3. 关键修正：A4 不应完全后置

阶段路线应写成：

```text
A0: minimal postprocess
A1: threshold sweep
A2: multi-scale deep supervision
A4: full topology-aware postprocess
```

而不是：

```text
A0/A1/A2 完全不做后处理
最后 A4 才开始 skeletonize 和 GeoJSON
```

原因：

PoC-3 的闭环对象不是 heatmap，而是 GeoJSON 线要素。  
所以从 A0 起就至少需要最小后处理，保证端到端可运行。

---

## 4. 推荐阶段定义

### 4.1 P3-A0：Pipeline Closure

目标：  
验证数据、训练、坐标、GeoJSON 输出闭环。

配置：

```yaml
stage: P3-A0
name: pipeline_closure

model:
  edge_head: single_scale

loss:
  type: bce_dice

gt:
  target_type: hard_band
  line_width_train: 3
  line_width_eval: 1
  densify_max_step_px: 0.5

postprocess:
  mode: minimal
  threshold: 0.5
  skeletonize: true
  export_geojson: true
```

A0 必须包含：

```text
sigmoid
  → fixed threshold=0.5
  → skeletonize
  → simple path extraction
  → pixel→WGS84
  → GeoJSON LineString
```

A0 不要求高 F1。  
A0 的结果不能作为 PoC-3 性能结论。

A0 通过条件：

```text
1. GT overlay 正确
2. loss 正常下降
3. prediction 不是全背景
4. skeleton 能生成有效 polyline
5. GeoJSON 可 parse
6. coordinate-in-tile rate = 100%
```

---

### 4.2 P3-A1：Class Imbalance Baseline

目标：  
解决海岸线 edge 前景稀疏导致的 BCE 背景淹没问题。

配置：

```yaml
stage: P3-A1
name: focal_dice_baseline

model:
  edge_head: single_scale

loss:
  type: focal_dice
  focal_alpha: 0.75
  focal_gamma: 2.0
  lambda_focal: 1.0
  lambda_dice: 1.0

gt:
  target_type: hard_band_or_soft_edge
  line_width_train: 3
  line_width_eval: 1
  soft_edge: optional
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
  densify_max_step_px: 0.5

postprocess:
  mode: threshold_sweep
  threshold_values: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  threshold_select_metric: buffered_f1_1px
```

A1 必须加入 threshold sweep。  
原因是 Focal Loss 会改变 heatmap calibration，继续固定 threshold=0.5 可能错误低估 A1 效果。

A1 通过条件：

```text
1. 相比 A0，edge recall 或 buffered-F1 明显提升
2. threshold sweep 存在稳定最优区间
3. false positive 不失控
4. heatmap 边缘响应与 GT 对齐
```

---

### 4.3 P3-A2：Multi-scale Edge Supervision

目标：  
验证多尺度深监督是否能同时提升：

- 细粒度边缘定位；
- 长岸线连续性；
- 低纹理区域的边界追踪；
- fused heatmap 稳定性。

配置：

```yaml
stage: P3-A2
name: multiscale_deep_supervision

model:
  edge_head: multi_scale
  side_outputs:
    - P1
    - P2
    - P3
    - P4
  fused_output: true

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

gt:
  target_type: soft_edge
  line_width_eval: 1
  soft_sigma_px: 1.0
  soft_radius_px: 3.0
  densify_max_step_px: 0.5

postprocess:
  mode: threshold_sweep
```

A2 通过条件：

```text
1. fused heatmap 不劣于 A1
2. side1 捕获细节边缘
3. side3/side4 对大体岸线走向有响应
4. buffered-F1@3px 达到可用水平
5. component fragmentation 不明显恶化
```

如果 A1 没有明显优于 A0，不建议急于进入 A2。  
应先回查：

- GT 是否错位；
- edge target 是否过窄/过宽；
- positive/negative 采样比例是否错误；
- threshold 是否固定过高；
- 后处理是否过度删线。

---

### 4.4 P3-A4：Topology-aware Postprocess

目标：  
将 A2 的 heatmap 稳定转为合法、简洁、连续的 GeoJSON geometry。

配置：

```yaml
stage: P3-A4
name: topology_postprocess

postprocess:
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

LineString / MultiLineString 输出规则：

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

A4 通过条件：

```text
1. buffered-F1@1px ≥ 0.70
2. buffered-F1@3px ≥ 0.85
3. average offset < 10m
4. geometry valid rate ≥ 95%
5. GeoJSON parse/schema valid rate > 99%
6. coordinate-in-tile rate = 100%
7. LineString / MultiLineString 输出符合预期
8. empty tile false positive rate 可控
```

---

## 5. A3 / A5 为什么不进入首轮主线

### 5.1 A3：Dilation Context Module

A3 可选，但不应在首轮主线加入。

原因：

- dilation 可扩大感受野，但可能增加假阳性岸线带；
- A2 已经通过 P3/P4 和 ViT-FPN 提供全局上下文；
- 如果 A1/A2 未解决问题，优先怀疑 GT、loss、后处理，而不是直接加 dilation；
- dilation 会引入额外变量，影响 PoC 判断。

推荐进入条件：

```text
A2 heatmap 局部边缘清晰，但长线连续性不足；
buffered-F1@3px 可用，但 component fragmentation 高；
Chamfer 不高，但断线多。
```

---

### 5.2 A5：Residual Edge Refinement

A5 也不建议进入首轮主线。

原因：

- residual refine 更适合细化边缘位置；
- 它不能优先解决前景稀疏或 topology fragmentation；
- 容易过拟合标注噪声；
- 会让首轮 PoC 变量过多。

推荐进入条件：

```text
A2/A4 已经有连续线，但平均偏移仍偏大；
buffered-F1@3px 高，buffered-F1@1px 不达标；
说明大体岸线正确，但像素级定位不足。
```

---

## 6. 推荐 gate 机制

### 6.1 A0 → A1

进入条件：

```text
1. GT overlay 正确
2. WGS84 ↔ pixel round-trip 误差达标
3. loss 下降
4. heatmap 非全背景
5. skeleton/polyline 可导出
6. GeoJSON 可 parse
7. coordinate-in-tile rate = 100%
```

若不满足，禁止进入 A1。  
因为此时问题不是 loss，而是 pipeline 或数据。

---

### 6.2 A1 → A2

进入条件：

```text
1. Focal+Dice 相比 BCE+Dice 提升 edge recall 或 buffered-F1
2. threshold sweep 后存在稳定最优阈值区间
3. false positive rate 没有明显失控
4. 可视化中边缘响应覆盖主岸线
```

若 A1 相比 A0 无提升，应检查：

```text
- GT 是否错位；
- line_width_train 是否过小；
- soft edge target 是否必要；
- positive/negative 采样比例是否错误；
- threshold 是否不合理；
- postprocess 是否过度删除组件。
```

---

### 6.3 A2 → A4

进入条件：

```text
1. fused heatmap 不低于 A1
2. side outputs 可解释
3. buffered-F1@3px 达到可用水平
4. skeleton 不是大面积碎片化
5. predicted length / GT length 不严重失衡
```

若 A2 不如 A1，应优先检查：

```text
- side loss 权重是否过大；
- P4 side loss 是否过度粗化；
- side logits 是否全部正确 upsample 到 224；
- fused head 是否学到有效融合；
- GT 是否在低分辨率 side loss 中丢失。
```

---

### 6.4 A4 → PoC-3 通过

通过条件：

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
```

---

## 7. 实验命名建议

建议将四个主线实验命名为：

```text
poc3_a0_bce_dice_closure
poc3_a1_focal_dice_sweep
poc3_a2_multiscale_deepsup
poc3_a4_topology_postprocess
```

可选消融：

```text
poc3_a3_dilation_context
poc3_a5_residual_refine
```

推荐 checkpoint 目录：

```text
outputs/poc3_edge/
  a0_bce_dice_closure/
  a1_focal_dice_sweep/
  a2_multiscale_deepsup/
  a4_topology_postprocess/
  ablations/
    a3_dilation_context/
    a5_residual_refine/
```

---

## 8. 最终推荐写法

在 PoC-3 总设计中，阶段路线建议写为：

```text
PoC-3 主线采用 A0 → A1 → A2 → A4：

A0 用 single-scale Edge Head + BCE/Dice + 最小 skeleton/GeoJSON 后处理跑通闭环；
A1 在 A0 闭环正确后引入 Focal/Dice 和 threshold sweep，作为正式 baseline；
A2 在 A1 有效后引入 HED-style 多尺度 side-output 深监督，作为主模型候选；
A4 在 A2 基础上加入 topology-aware postprocess，稳定 LineString/MultiLineString 输出，作为最终候选方案。

A3 dilation context 和 A5 residual refinement 不进入首轮主线，仅在 A2/A4 未达标时作为针对性消融。
```

---

## 9. 最终判断

`A0 → A1 → A2 → A4` 是合理且推荐的 PoC-3 主线。

但需要明确三点：

```text
1. A0 必须包含最小后处理，否则不能证明 edge-to-GeoJSON 闭环；
2. A1 必须加入 threshold sweep，否则可能低估 Focal Loss；
3. A4 是完整拓扑后处理升级，不是后处理首次出现。
```

因此最终执行路线应表述为：

```text
A0: 训练 + 最小后处理闭环
A1: Focal/Dice + threshold sweep
A2: 多尺度深监督
A4: 完整 topology-aware postprocess
```

最终候选：

```text
ViT-FPN
  + Multi-scale Edge Head
  + Deep-supervised Focal/Dice
  + soft edge target
  + threshold sweep
  + topology-aware skeleton graph postprocess
  + LineString/MultiLineString GeoJSON validation
```

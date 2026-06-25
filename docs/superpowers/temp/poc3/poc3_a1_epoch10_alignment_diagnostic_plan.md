# PoC-3 A1 Epoch 10 诊断报告与下一步计划

> 日期：2026-06-18  
> 阶段：PoC-3 / P3-A1  
> 实验：Focal + Dice + threshold sweep  
> 主题：A1 epoch 10 后，判断是否继续训练到 40 epoch，还是先进行 heatmap / alignment / tolerance 诊断  
> 结论：**先诊断，不建议直接继续等到 40 epoch。当前问题不是“模型看不到边缘”，而是“模型已经产生稀疏高响应，但严格像素指标显示存在空间对齐 / 容差问题”。**

---

## 1. 当前结果摘要

A1 epoch 10 结果：

| 指标 | A0 (BCE+Dice) | A1 epoch 5 | A1 epoch 10 |
|---|---:|---:|---:|
| buffered_f1_1px | 0.007 | 0.011 | 0.017 |
| buffered_f1_3px | 0.013 | 0.022 | 0.033 |
| pixel_f1 | 0.004 | 0.006 | 0.008 |
| best threshold | 0.50 | 0.10 | 0.10 |
| loss | 0.903 | 0.770 | 0.690 |

新增 heatmap 诊断统计：

| 指标 | 数值 | 解读 |
|---|---:|---|
| `p_max_mean` | 0.73 | 模型能在部分像素产生强边缘响应 |
| `p_max_median` | 0.91 | 多数 tile 的最强响应很高 |
| `p_mean` | 0.003 | 输出极稀疏，背景抑制强 |
| `GT_fg_ratio` | 0.76% | GT 是典型细线边缘目标 |
| `pred_fg@0.10` | 0.69% | 预测前景密度与 GT 密度接近 |
| `Empty@0.05` | 0/50 | 低阈值下所有 tile 都有响应 |
| `p_max > 0.5` | 72% tiles | 大部分 tile 至少存在高置信边缘点 |

---

## 2. 关键结论

### 2.1 模型不是“看不到”边缘

当前诊断显示：

```text
p_max_median = 0.91
p_mean = 0.003
pred_fg@0.10 = 0.69%
GT_fg = 0.76%
Empty@0.05 = 0%
```

这说明模型已经满足以下条件：

```text
1. 能输出高强度边缘响应；
2. 输出非常稀疏，没有大面积误报；
3. 预测前景密度与 GT 前景密度接近；
4. 低阈值下所有 tile 都非空；
5. loss 持续下降。
```

因此，当前不应继续将问题归因为：

```text
- Focal Loss 无效；
- 模型完全没学到；
- threshold sweep 错误；
- 训练 epoch 不够。
```

更准确的判断是：

> 模型已经学到海岸线相关响应，但 prediction 与 GT 在像素级位置上存在轻微偏移，严格 pixel-F1 / buffered-F1@1px 对这种偏移极端敏感。

---

## 3. 为什么 pixel-F1 / buffered-F1 仍然很低

海岸线是极稀疏目标。  
当前 GT 前景比例只有：

```text
GT_fg = 0.76%
```

在 224×224 tile 中，大约只有数百个前景像素。  
如果预测线与 GT 线相差 1–3px，即使视觉上“沿着海岸线”，严格 pixel overlap 也可能极低。

典型现象：

```text
p_max 很高
pred_fg_ratio 合理
但 pixel_f1 很低
```

这通常意味着：

```text
1. 预测线与 GT 有局部偏移；
2. GT rasterization 和模型输出的 pixel convention 不一致；
3. line_width_train 与 eval centerline 不一致；
4. metric 过严，没有足够 tolerance；
5. 后处理/threshold 不是主因。
```

---

## 4. 当前阶段状态判定

A1 epoch 10 状态：

| 项目 | 状态 | 说明 |
|---|---|---|
| loss convergence | PASS | loss 0.903 → 0.690 |
| heatmap activation | PASS | `p_max_median=0.91` |
| sparsity control | PASS | `p_mean=0.003` |
| foreground density | PASS | `pred_fg@0.10≈GT_fg` |
| empty prediction | PASS | `Empty@0.05=0%` |
| strict pixel alignment | FAIL | pixel_f1 / buffered-F1 仍极低 |
| A1 gate | PARTIAL PASS | heatmap 有效，但 alignment/tolerance 未过 |
| 下一步 | DIAGNOSE | 先做 offset/tolerance/alignment 诊断 |

结论：

```text
A1 不应判定为模型失败。
A1 当前应判定为：heatmap learning pass，pixel alignment fail。
```

---

## 5. 当前不建议继续做什么

暂时不建议：

```text
1. 不建议直接继续训练到 40 epoch；
2. 不建议立即进入 A2 多尺度深监督；
3. 不建议继续加大 focal_alpha / focal_gamma；
4. 不建议盲目调学习率；
5. 不建议只看 pixel_f1 判定模型失败；
6. 不建议直接改后处理参数掩盖对齐问题。
```

原因：

> 如果当前问题是 GT / pixel convention / affine / row-col mismatch，继续训练或上 A2 只会在错误监督上堆复杂度。

---

## 6. 最高优先级诊断：Offset Sweep

### 6.1 目的

判断 prediction 与 GT 是否存在系统性偏移。

### 6.2 方法

对 prediction binary 或 heatmap 进行平移搜索：

```text
dx, dy ∈ [-5, 5] px
```

对每个 offset 计算：

```text
buffered-F1@1px
buffered-F1@3px
Chamfer distance
pixel-F1
```

### 6.3 结果解释

| 结果 | 解释 |
|---|---|
| 某个固定 offset 显著提升 F1 | 存在系统性坐标偏移 |
| 多个方向都无提升 | 不是简单平移偏移 |
| 大 offset 才提升 | georef / resize / affine 可能有尺度或方向错误 |
| offset 提升只在部分 tile 出现 | 局部 GT 或数据源不一致 |
| offset 不提升但 tolerance 提升 | 自然定位误差，不是系统性偏移 |

示例判断：

```text
原始 buffered-F1@1px = 0.017
shift(dx=2, dy=-1) 后 buffered-F1@1px = 0.25+

=> 高概率为系统性像素偏移。
```

---

## 7. 必须检查的代码风险点

### 7.1 `draw_edge_map` 的 row / col 是否写反

如果使用 `skimage.draw.line_aa`，返回值语义是：

```python
rr, cc, val = line_aa(row0, col0, row1, col1)
```

其中：

```text
rr = row indices
cc = col indices
```

正确写法通常应为：

```python
edge_map[rr, cc] = np.maximum(edge_map[rr, cc], val)
```

危险写法：

```python
edge_map[cc, rr] = np.maximum(edge_map[cc, rr], val)
```

如果写成 `edge_map[cc, rr]`，会产生 row/col 交换问题。  
这可能导致 GT 或预测 overlay 出现转置式错位。

### 7.2 pixel coordinate 约定

海岸线 LineString 应明确使用：

```text
pixel center coordinate
```

需要检查：

```text
1. wgs84_to_pixel 得到的是 corner 还是 center；
2. draw_line 前是否做了 0.5px offset；
3. int() 是 floor，不是 round；
4. PIL resize 与 model_transform resize 是否一致；
5. row/col 与 col/row 在所有模块中是否一致。
```

重点关注：

```python
int(r0), int(c0)
```

建议对比：

```python
int()
round()
floor()
center_offset ±0.5
```

但不要盲改，应先用 offset sweep 判断偏移方向。

---

## 8. Tolerance Curve 诊断

### 8.1 目的

判断模型是否只是存在 1–3px 局部定位误差。

### 8.2 方法

计算：

```text
buffered-F1@r, r ∈ {0, 1, 2, 3, 4, 5, 7, 10}
```

### 8.3 结果解释

| 曲线形态 | 解释 |
|---|---|
| 1px 很低，3px 明显升高，5px 很高 | 局部小偏移，模型语义正确 |
| 1px/3px/5px 都低 | GT 或预测语义位置错误 |
| 1px 低但 10px 高 | 大体海岸线方向对，但几何对齐差 |
| 所有半径都低，但 p_max 高 | 模型预测到其他线状纹理，不是海岸线 |
| 0px 低，1–2px 急剧上升 | pixel center/corner 或 rounding 问题 |

### 8.4 建议记录

```text
buffered_f1_0px
buffered_f1_1px
buffered_f1_2px
buffered_f1_3px
buffered_f1_4px
buffered_f1_5px
buffered_f1_7px
buffered_f1_10px
```

---

## 9. Soft Heatmap Metric 诊断

当前 hard threshold + binary overlap 对 1px 线过于苛刻。  
建议加入 soft heatmap metric，判断模型质量是否被 hard binarization 低估。

### 9.1 pred mass near GT

```text
gt_dilated_r3 = dilate(gt_centerline, radius=3)
pred_mass_near_gt = pred_heatmap[gt_dilated_r3].sum() / pred_heatmap.sum()
```

解释：

| 值 | 含义 |
|---|---|
| 高 | 模型热力图质量可用，主要是轻微偏移 |
| 低 | 模型响应不在 GT 附近，可能预测到其他纹理 |

### 9.2 GT covered by pred heatmap

```text
pred_maxpool_r3 = max_pool(pred_heatmap, radius=3)
gt_covered = mean(pred_maxpool_r3[gt_centerline] > threshold)
```

推荐 threshold：

```text
0.05, 0.10, 0.20
```

### 9.3 需要记录

```text
pred_mass_near_gt_r1
pred_mass_near_gt_r3
pred_mass_near_gt_r5
gt_covered_by_pred_r1@0.05
gt_covered_by_pred_r3@0.05
gt_covered_by_pred_r5@0.05
```

---

## 10. GT 口径一致性检查

当前需要分别统计：

```text
GT_center_fg_ratio(width=1)
GT_train_fg_ratio(width=3)
pred_fg_ratio@threshold=0.10
```

因为：

```text
train target = width=3
eval target  = width=1
```

如果口径混用，会误判模型输出密度。

### 10.1 解释方式

| 现象 | 解读 |
|---|---|
| pred_fg 接近 width=1 | 模型输出较细，可能对 eval 有利 |
| pred_fg 接近 width=3 | 模型学了训练带宽，但 eval 会因厚度被惩罚 |
| pred_fg 小于 width=1 | 输出过稀，可能漏线 |
| pred_fg 大于 width=3 | 输出过厚，FP 风险 |

### 10.2 建议记录

```text
gt_fg_width1_mean
gt_fg_width1_median
gt_fg_width3_mean
gt_fg_width3_median
pred_fg@0.10
pred_fg@0.05
pred_fg@0.02
```

---

## 11. Overlay 必须输出

固定 8–16 个样本，输出以下图：

```text
image
GT centerline width=1
GT train target width=3
pred heatmap
pred binary@0.10
pred binary@0.05
pred binary@0.02
GT/pred overlay@0.10
GT/pred overlay@0.05
GT/pred overlay@0.02
```

### 11.1 视觉判断重点

| 观察 | 解释 |
|---|---|
| pred 在海岸线旁边平行偏移 | pixel convention / georef 偏移 |
| pred 与 GT 交叉但不重合 | 局部定位误差 |
| pred 在其他纹理上 | 模型语义错误 |
| pred 沿海岸线但比 GT 更平滑 | GT 噪声或模型合理泛化 |
| pred 断裂但位置对 | topology/postprocess 问题 |
| pred 完全随机 | 模型未学到 |

---

## 12. 当前决策树

```text
A1 epoch 10 diagnostic
  |
  |-- p_max 高 + pred_fg 接近 GT_fg + Empty=0
  |      |
  |      |-- offset sweep 有固定最优偏移
  |      |      → 修 pixel convention / row-col / affine / center-corner
  |      |
  |      |-- tolerance curve 3–5px 明显提升
  |      |      → 模型语义正确，下一步优化定位：
  |      |          1. soft edge target
  |      |          2. A2 multi-scale
  |      |          3. 评估使用合理 buffer
  |      |
  |      |-- tolerance curve 仍低
  |             → 检查模型是否预测到非海岸线纹理；
  |               复查 GT overlay 和 label quality
  |
  |-- p_max 低 / Empty 多
         → 回到 loss / LR / GT width / overfit 诊断
```

---

## 13. 是否进入 A2 的条件

当前暂不建议立即进入 A2。  
A2 进入条件应改为：

```text
A1 heatmap localization sanity pass
```

至少满足：

```text
1. offset sweep 无严重系统性偏移，或偏移已修复；
2. tolerance curve 证明 prediction 在 3–5px 内覆盖 GT；
3. overlay 显示 pred 在真实海岸线附近；
4. pred_fg_ratio 与 GT_fg_ratio 口径一致；
5. hard metric 低的原因已解释清楚。
```

如果上述条件满足，再进入 A2。  
A2 的目标应变成：

```text
从“3–5px 内大体正确”
提升到
“1–3px 内更精确、更连续”
```

---

## 14. 建议的下一步执行顺序

推荐按以下顺序执行：

```text
1. 暂停继续训练到 40 epoch；
2. 检查 draw_edge_map row/col 是否写反；
3. 对 epoch 10 checkpoint 做 offset sweep [-5, 5]；
4. 计算 tolerance curve r=0/1/2/3/4/5/7/10；
5. 分别统计 GT width=1 / width=3 foreground ratio；
6. 输出固定样本 overlay；
7. 增加 soft heatmap metric；
8. 根据诊断结果决定：
   a) 修 pixel convention；
   b) 调整 eval tolerance / metric；
   c) 切 soft edge target；
   d) 进入 A2。
```

---

## 15. 诊断完成后的处理策略

### 15.1 如果发现系统性偏移

处理：

```text
1. 修正 row/col 顺序；
2. 修正 pixel center/corner 约定；
3. 修正 resize_georef；
4. 修正 int/floor/round 策略；
5. 重新生成 GT；
6. 重新跑 A0/A1 sanity。
```

禁止：

```text
不要通过后处理 shift prediction 来“补偿”GT 错误。
```

### 15.2 如果没有系统性偏移，但 3–5px 容差表现好

处理：

```text
1. 保留 A1；
2. 进入 soft edge target；
3. 进入 A2 multi-scale deep supervision；
4. 评估指标保留 buffered-F1@1px，但同时记录 tolerance curve；
5. 后处理阶段允许 gap bridge / skeleton graph 修复。
```

### 15.3 如果 tolerance curve 也很差

处理：

```text
1. 检查 GT overlay；
2. 检查模型是否预测到非海岸线纹理；
3. 检查训练样本 label quality；
4. 做 2-sample overfit；
5. 重新评估 loss 与采样策略。
```

---

## 16. 当前阶段记录建议

建议将当前实验状态记录为：

```text
P3-A1 epoch 10:
  loss_convergence: PASS
  heatmap_activation: PASS
  sparsity_control: PASS
  foreground_density_match: PASS
  empty_prediction_check: PASS
  strict_pixel_alignment: FAIL
  next_action: alignment_tolerance_diagnostics
```

一句话结论：

> A1 已经学到稀疏高置信边缘响应，但严格像素指标仍低；下一步应优先诊断 GT/pred 空间对齐与 metric tolerance，而不是继续盲训或直接上 A2。

---

## 17. 建议新增脚本 / 工具

可以新增：

```text
scripts/diagnose_edge_alignment.py
```

功能：

```text
1. 加载 checkpoint；
2. 跑固定 val subset；
3. 输出 logit/prob 分布；
4. 输出 threshold sweep；
5. 输出 offset sweep；
6. 输出 tolerance curve；
7. 输出 GT width=1 / width=3 fg ratio；
8. 输出 overlay；
9. 保存诊断 JSON。
```

建议输出目录：

```text
outputs/poc3_edge/a1_focal_dice_sweep/diagnostics_epoch_010/
  metrics_distribution.json
  offset_sweep.csv
  tolerance_curve.csv
  fg_ratio_stats.json
  overlays/
```

---

## 18. 诊断指标模板

```json
{
  "checkpoint": "epoch_010.pt",
  "num_samples": 50,
  "prob_distribution": {
    "p_min": 0.0,
    "p_mean": 0.003,
    "p_median": 0.0,
    "p95": 0.01,
    "p99": 0.08,
    "p_max_mean": 0.73,
    "p_max_median": 0.91
  },
  "foreground_ratio": {
    "gt_width1_mean": null,
    "gt_width3_mean": 0.0076,
    "pred_fg_0.10": 0.0069,
    "pred_fg_0.05": null,
    "pred_fg_0.02": null
  },
  "empty_prediction": {
    "threshold_0.10": null,
    "threshold_0.05": 0.0,
    "threshold_0.02": null
  },
  "best_threshold": 0.10,
  "offset_sweep_best": {
    "dx": null,
    "dy": null,
    "buffered_f1_1px": null,
    "buffered_f1_3px": null
  },
  "tolerance_curve": {
    "f1_0px": null,
    "f1_1px": 0.017,
    "f1_2px": null,
    "f1_3px": 0.033,
    "f1_5px": null,
    "f1_7px": null,
    "f1_10px": null
  }
}
```

---

## 19. 最终判断

当前 A1 epoch 10 不应被视为失败。  
它说明：

```text
1. Focal + Dice 有效降低 loss；
2. 模型已经产生强边缘响应；
3. 预测前景密度与 GT 相近；
4. 空预测问题基本消失；
5. 主要矛盾转移到 pixel alignment / tolerance。
```

最终建议：

```text
暂停长训；
先做 alignment + tolerance diagnostics；
确认无坐标/GT bug 后，再决定是否：
  - 调 soft edge target；
  - 进入 A2；
  - 调整后处理；
  - 或修复 GT pipeline。
```

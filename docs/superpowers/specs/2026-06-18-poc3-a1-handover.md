# PoC-3 海岸线 Edge Head — A1 工作总结与交接文档

> 日期: 2026-06-18
> 状态: A0 闭环通过、A1 epoch 10 诊断完成
> 下一步: soft edge target + A2 多尺度深监督
> 关键发现: 模型学会检测边缘但定位到非海岸线纹理，不是 GT pipeline bug

---

## 1. 已完成工作

### 1.1 代码产出（9 个文件，11 次 commit）

| 文件 | 内容 | 状态 |
|---|---|---|
| `utils/edge_losses.py` | SoftDiceLoss, BinaryFocalLoss, BCE+Dice, Focal+Dice, DeepSupervisedEdgeLoss | ✅ |
| `Models/edge_head.py` | SingleScaleEdgeHead (2.66M), MultiScaleEdgeHead (1.18M) | ✅ |
| `utils/edge_postprocess.py` | sigmoid→skeleton→path→simplify→GeoJSON 完整管线 | ✅ |
| `utils/coastline_metrics.py` | buffered-F1, Chamfer, Hausdorff, pixel metrics | ✅ |
| `Dataset/coastline_dataset.py` | CoastlineEdgeDataset, manifest builder, Binary TIF GT 生成 | ✅ |
| `configs/poc3_edge_a0_closure.yaml` | A0 BCE+Dice 配置 | ✅ |
| `configs/poc3_edge_a1_focal_dice.yaml` | A1 Focal+Dice + threshold sweep 配置 | ✅ |
| `scripts/poc_stage_edge.py` | 训练/验证入口（支持 A0/A1，Focal/BCE，threshold sweep） | ✅ |
| `docs/superpowers/plans/2026-06-17-poc3-coastline-edge-head.md` | 完整实现计划 | ✅ |

### 1.2 Bug 修复

| Bug | 影响 | 状态 |
|---|---|---|
| `_extract_tile_name` 只去扩展名，未去 `_Orig_WRZ` variant suffix | Binary TIF 标签全部找不到 (954/954) | ✅ 已修复 |
| `edge_bce_dice_loss` 中 sigmoid 用了未 squeezed 的 `logits` 而非 `logits_flat` | 代码脆弱性 | ✅ 已修复 |
| `SoftDiceLoss`/`BinaryFocalLoss` 每次调用重新实例化 | 不必要内存分配 | ✅ 已修复 |
| `draw_edge_map` 中 `edge_map[cc, rr]` 写反了 row/col | GeoJSON fallback GT 路径错误（非主路径，Binary TIF 不受影响） | ✅ 已修复 |

---

## 2. 实验进度

### 2.1 Pipeline 验证矩阵

| 阶段 | 实验 | 状态 | 关键结果 |
|---|---|---|---|
| Pre-flight | 数据扫描 + GT overlay + round-trip | ✅ PASS | 954 tiles, 100% Binary TIF |
| A0 | BCE+Dice, single-scale, 20 epoch | ✅ PASS | 闭环验证通过，pixel_f1=0.004 |
| A0 Overfit | 2-sample, 200 iter NPU | ✅ PASS | loss 1.76→0.30, pred_fg=0.68% |
| A1 | Focal+Dice, threshold sweep, 40 epoch | ⏸️ epoch 10 暂停 | f1@1px=0.017, loss 0.85→0.69 |

### 2.2 A0 闭环通过标准

```
✅ GeoJSON LineString/MultiLineString 正确解析
✅ WGS84 ↔ pixel round-trip 误差 < 1e-6°
✅ edge GT overlay 与原图对齐（Binary TIF, stem bug 修复后）
✅ loss 下降 (1.476 → 0.903)
✅ heatmap 非全黑 (overfit test pred_fg=0.68%)
✅ skeleton 可生成 polyline
✅ GeoJSON 输出可 parse
✅ coordinate-in-tile rate = 100%
```

### 2.3 A1 epoch 10 结果

| 指标 | A0 (epoch 10) | A1 (epoch 10) |
|---|---|---|
| loss | 0.903 | 0.690 |
| pixel_f1 | 0.004 | 0.008 |
| buffered_f1_1px | 0.007 | 0.017 |
| buffered_f1_3px | 0.013 | 0.033 |
| best threshold | 0.50 | 0.10 |

---

## 3. 诊断结论

### 3.1 Heatmap 分布（epoch 10, 50 val tiles）

```
p_max_mean=0.73, p_max_median=0.91     ← 72% tile 有 >0.5 的强响应
p_mean=0.003                            ← 输出极稀疏，背景抑制好
pred_fg@0.10=0.69%, GT_fg=0.76%        ← 前景密度匹配
Empty@0.05=0%                           ← 所有 tile 有响应
```

### 3.2 Offset sweep [-5,+5]

```
基线 (0,0):  f1=0.0135
最优 (-5,-3): f1=0.0246  (+83%)
```

Group-wise 不一致：生物岸线 best=(+1,-5)、盐田围堤 best=(-5,-5)→**不是全局平移**

### 3.3 Tolerance curve

```
buffer |   f1   | recall
   0px | 0.0007 | 0.002
   1px | 0.0135 | 0.027
   3px | 0.0334 | 0.063
   5px | 0.0485 | 0.093
  10px | 0.0616 | 0.127
```

### 3.4 Soft metrics

```
pred_mass_near_GT @ r=7: 13.5%   ← 仅13.5%的heatmap能量在GT 7px内
GT_covered_by_pred @ r=7: 8.3%   ← 仅8.3%的GT被pred覆盖
```

### 3.5 最终诊断

**模型学会了检测边缘，但检测到了非海岸线纹理（田地边界、道路、建筑边缘等）。**

- p_max=0.91 → 模型确实在检测高置信度边缘
- 但 pred_mass_near_GT@7px=13.5% → 87% 的 heatmap 响应不在海岸线附近
- tolerance curve 拉到 10px 仍仅 f1=0.06 → 不是对齐问题
- ViT-FPN 的全局语义未充分激活 → 224×224 tile 上下文中，模型无法区分海陆边界和其他纹理边界

**结论: 不是 GT pipeline bug。不需要继续修坐标系统。**

---

## 4. 下一步建议

### 4.1 按总设计文档推进 A2

```
A2: soft edge target + Multi-scale Deep Supervision (HED-style side outputs)
目标: 提升全局海岸线语义识别，降低非海岸线纹理误检
```

A2 进入条件已满足（诊断计划 Section 13）：

| 条件 | 状态 |
|---|---|
| heatmap 有强响应 | ✅ p_max=0.91 |
| pred_fg 与 GT_fg 口径一致 | ✅ 0.69% vs 0.76% |
| 无明显系统性偏移需修复 | ✅ group-wise 不一致，非全局平移 |
| 模型已学会边缘语义但需全局上下文 | ✅ tolerance curve 证明 |

### 4.2 A2 关键设计要点

1. **Soft edge target**: `exp(-distance²/2σ²)` 替代 hard binary，σ=1.0, radius=3px
2. **Multi-scale side outputs**: P1→P4 各出一个 side logit, upsample 到 224 后算 loss
3. **Deep supervision weights**: fused=1.0, side1=0.5, side2=0.3, side3=0.2, side4=0.1
4. **Loss**: 每个 sub-loss = Focal(α=0.75, γ=2.0) + Dice
5. **P3/P4 层级强制学习全局海陆语义** — ViT g_grid 已注入 P3

### 4.3 不建议做的事情

- ❌ 继续 A1 训练到 40 epoch（不会改善 non-coastline edge 误检）
- ❌ 修 GT pipeline（已确认不是坐标系统 bug）
- ❌ 加大 focal_alpha/gamma（不会区分海岸线 vs 其他边缘）
- ❌ 用后处理掩盖问题（skeleton graph 无法修复语义错误）

---

## 5. Checkpoint 位置

| 阶段 | 路径 |
|---|---|
| A0 epoch 10 | `outputs/poc3_edge/a0_bce_dice_closure/checkpoints/epoch_010.pt` |
| A0 epoch 10 | `outputs/poc3_edge/a0_bce_dice_closure/checkpoints/epoch_005.pt` |
| A1 epoch 10 | `outputs/poc3_edge/a1_focal_dice_sweep/checkpoints/epoch_010.pt` |
| A1 epoch 5  | `outputs/poc3_edge/a1_focal_dice_sweep/checkpoints/epoch_005.pt` |

视觉输出:
- A0 overlay: `outputs/poc3_edge/a0_bce_dice_closure/vis/val_epoch_005/`
- A1 overlay: `outputs/poc3_edge/a1_focal_dice_sweep/vis/val_epoch_005/`
- A1 shifted overlay: `outputs/poc3_edge/a1_focal_dice_sweep/shifted_overlay/`
- Pre-flight GT overlay: `outputs/poc3_edge/preflight/gt_overlay/`

---

## 6. 执行命令速查

```bash
# 环境
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd /home/ma-user/work/CoastGPT

# A0 训练
python scripts/poc_stage_edge.py --config configs/poc3_edge_a0_closure.yaml --device npu

# A1 训练 (warm-start from A0)
python scripts/poc_stage_edge.py --config configs/poc3_edge_a1_focal_dice.yaml --device npu \
  --a0-checkpoint outputs/poc3_edge/a0_bce_dice_closure/checkpoints/epoch_010.pt

# 仅构建 manifest
python scripts/poc_stage_edge.py --config configs/poc3_edge_a0_closure.yaml --build-manifest-only

# 单元测试
python utils/edge_losses.py
python Models/edge_head.py
python utils/edge_postprocess.py
python utils/coastline_metrics.py
python Dataset/coastline_dataset.py
```

---

## 7. Git 信息

- Branch: `worktree-poc3-edge-head`
- Worktree 路径: `/home/ma-user/work/CoastGPT/.claude/worktrees/poc3-edge-head`
- 共 11 次 commit，基于 main 分支的 `CoastGPT_dual`

---

## 8. 一句话总结

**PoC-3 A0 闭环验证通过，A1 证明 Focal+Dice 有效且模型能学到边缘，但 224px tile 上下文不足以让 ViT-FPN 区分海岸线和其他纹理边缘——下一步应通过 soft edge target + 多尺度深监督强制学习全局海陆语义。**

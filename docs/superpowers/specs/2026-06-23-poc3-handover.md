# PoC-3 海岸线 Edge Detection — 交接文档

> 日期: 2026-06-23
> 状态: Pipeline 跑通，A1-relabel F1@1px=0.306，F1@3px=0.476
> 核心发现: 标签不一致是全部瓶颈，不是模型架构

---

## 1. 实验历程与关键结果

### 时间线

```
A0: BCE+Dice, single-scale           → 闭环通过，F1=0.004
A1: Focal+Dice, single-scale         → F1=0.017, 诊断: 模型检测边缘但混淆纹理
A2: Soft target + Multi-scale DS     → F1=0.039, FP median=51px, oracle ceiling=0.048
A2b: A2 放大到 448                   → F1=0.023, FP median=137px, 分辨率不是瓶颈
A3: GDF vector field                 → F1=0.024, FP distance 微弱改善, Gate 3 未通过
                                   
★ P3-L: 标签统一 (GeoJSON single source)
  ├── 审计 GeoJSON vs Binary TIF:    pixel F1 仅 0.006, buf3 仅 0.040
  ├── 标签工厂:                      954 tiles, edge_center/edge_soft/GDF/sealand
  ├── A1-relabel (unified labels):  F1@1px=0.316, F1@3px=0.476
  └── A2-relabel (unified labels):  F1@1px=0.295, multi-scale 无增益

P3-B1 Oracle: sea/land flood-fill     → 失败 (GeoJSON 不闭合)
终态: A1-relabel + GeoJSON pipeline   → 253 tiles, 3.8 features/tile
```

### 所有实验对比

| 实验 | 标签 | 架构 | F1@1px | F1@3px | FP med | 结论 |
|------|------|------|--------|--------|--------|------|
| A1 | 混用 | single-scale | 0.017 | — | — | 基线 |
| A2 | 混用 | multi-scale DS | 0.039 | 0.063 | 51.5px | 标签天花板 |
| A2b | 混用 | A2@448 | 0.023 | — | 137px | 分辨率不是瓶颈 |
| A3 | 混用 | GDF field | 0.024 | — | 48.3px | 边际改善 |
| **A1-relabel** | **统一** | **single-scale** | **0.306** | **0.476** | — | **★ 最优** |
| A2-relabel | 统一 | multi-scale DS | 0.295 | — | — | 简单更好 |

---

## 2. 核心发现: 标签不一致是全部瓶颈

### GeoJSON vs Binary TIF 一致性

```
ORIGINAL 128×128: pixel F1=0.006, buf3 F1=0.040
224 NEAREST:      pixel F1=0.004, buf3 F1=0.018
```

旧 Binary TIF 与 GeoJSON 在像素级几乎不是同一条线。所有 A0-A3 的低 F1 都是标签天花板造成的，不是模型问题。

### 统一标签后的真实天花板

| 指标 | 旧标签 (mixed) | 新标签 (unified) | 提升 |
|------|---------------|-----------------|------|
| F1@1px (pixel) | 0.039 | 0.269 | 6.9× |
| Oracle ceiling | 0.048 | 0.289 | 6.0× |
| FP median dist | 51.5 px | 25.2 px | -51% |
| FP within 3px | 3.0% | 25.5% | +22.5pp |
| Best threshold | 0.05 | 0.25 | 正常化 |

---

## 3. 代码产出

### 新增文件

```
utils/
  coastline_label_factory.py   # GeoJSON→edge/soft/GDF/sealand 统一标签工厂
  gdf_target.py                # GDF target 生成与验证 (A3)
  gdf_losses.py                # GDF loss (A3)
  gdf_postprocess.py           # Endpoint voting (A3)

Models/
  gdf_head.py                  # SingleScaleGDFHead, MultiScaleGDFHead (A3)

scripts/
  poc3_b1_oracle_sea_land.py   # P3-B1 Oracle 实验
  poc3_edge_infer.py           # 单张推理: TIF → heatmap → GeoJSON
  poc3_batch_infer.py          # 批量推理 + 评估
  poc_stage_gdf.py             # A3 GDF 训练入口
  diag_a2_attribution.py       # A2 归因诊断脚本
  poc_stage_edge_ddp.py        # 多卡 DDP 训练 (A2b/A2)
  poc_stage_edge_ds.py         # DeepSpeed 训练 (废弃)
  run_a2_8npu.sh               # A2 启动脚本
  run_a2b_448_8npu.sh          # A2b 启动脚本
  run_a2_8npu_ddp.sh           # DDP 启动脚本

configs/
  poc3_edge_a2_soft_multiscale.yaml   # A2 224 multi-scale
  poc3_edge_a2b_448.yaml             # A2b 448
  poc3_edge_a2b_448_v2.yaml          # A2b native 28×28
  poc3_edge_a2_relabel.yaml          # A2-relabel unified
  poc3_a3_0_gdf_scratch.yaml         # A3 from scratch
  poc3_a3_1_gdf_warmstart.yaml       # A3 warm-start
  poc3_a3_1_gdf_warmstart.yaml       # A3 warm-start

Dataset/
  coastline_dataset.py        # 添加 unified_label_dir 支持

docs/superpowers/specs/
  2026-06-18-poc3-a1-handover.md                  # A1 交接
  2026-06-22-poc3-b-semantic-prior-design.md      # P3-B 设计
  2026-06-23-poc3-handover.md                     # 本文档

docs/superpowers/temp/poc3/
  poc3_a3_gravitational_distance_field_execution_plan.md  # A3 执行方案
  poc3_label_unification_execution_plan.md                # P3-L 执行方案
```

### 修改文件

```
Dataset/coastline_dataset.py     # +unified_label_dir, IMAGE_SIZE→self.image_size
Models/dual_vision_encoder.py    # +pos_embed interpolation, img_size override, semantic_grid_size None handling
scripts/poc_stage_edge.py        # +MultiScaleEdgeHead, +DeepSupervisedEdgeLoss, +unified-labels flag
utils/edge_losses.py             # +DeepSupervisedEdgeLoss (A2)
Models/edge_head.py              # +MultiScaleEdgeHead (A2)
```

---

## 4. 资产位置

| 资产 | 路径 |
|------|------|
| Coastline manifest | `outputs/poc3_edge/coastline_manifest.json` (954 tiles) |
| 统一标签 (224) | `outputs/poc3_labels_unified/labels_224/` |
|  | `├── edge_center/` (954 .npy) |
|  | `├── edge_soft/` (954 .npy) |
|  | `└── gdf_R32/` (954 .npy) |
| A1-relabel 最佳 ckpt | `outputs/poc3_edge/a1_relabel_unified/checkpoints/epoch_010.pt` (F1=0.316) |
| A2-relabel 最佳 ckpt | `outputs/poc3_edge/a2_relabel_unified/checkpoints/epoch_015.pt` (F1=0.295) |
| 旧 A2 最佳 ckpt | `outputs/poc3_edge/a2_8npu_ddp/checkpoints/epoch_000.pt` |
| 旧 A3 best ckpt | `outputs/poc3_gdf/a3_1_warmstart/checkpoints/epoch_005.pt` |
| 批量 GeoJSON 输出 | `outputs/poc3_edge/batch_infer_full/geojson/` (253 files) |

---

## 5. 最终 Pipeline

```
4-band TIF (128×128, R/G/B/NIR)
  → PIL resize 224×224 (BILINEAR)
  → A1-relabel SingleScaleEdgeHead (2.66M params)
  → sigmoid → heatmap [224, 224]
  → threshold=0.25 → binary
  → skeletonize → path extraction
  → Douglas-Peucker simplify (ε=0.1px)
  → pixel→WGS84 (georef model_transform from GeoJSON bounds)
  → GeoJSON FeatureCollection {LineString/MultiLineString}
```

### 推理命令

```bash
# 单张
python scripts/poc3_edge_infer.py \
    --image <4-band TIF> \
    --geojson <label.geojson for georef> \
    -o output.geojson

# 批量
python scripts/poc3_batch_infer.py \
    --output outputs/poc3_edge/batch_infer
```

---

## 6. 下一步建议

### 提精度路径（按优先级）

1. **语义分割替代 edge detection** — 用 landcover 21 类 → sea/land 二分类 → morphological boundary 提取 coastline。P3-B 设计已完成，unified labels 可直接复用。

2. **数据增强/更多 coastline 数据** — 当前 954 tiles 偏少。

3. **更大 backbone** — ConvNeXt-Large 替代 Base，或上 SAM/Mask2Former 等预训练分割模型。

4. **Topology 后处理** — 用 coastline 的拓扑约束（必须是闭合或连接边界的曲线）过滤伪路径。

### 不建议的方向

- ❌ 继续调 edge loss (BCE/Focal/Dice/GDF) — unified labels 下简单方案已经最好
- ❌ 单纯放大分辨率 (448/512) — A2b 已证明不是瓶颈
- ❌ 在旧 Binary TIF 标签上做任何实验

---

## 7. 训练命令速查

```bash
# 环境
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd /home/ma-user/work/CoastGPT/.claude/worktrees/poc3-edge-head

# A1-relabel (single-scale, unified labels)
python scripts/poc_stage_edge.py \
    -c configs/poc3_edge_a1_focal_dice.yaml \
    --device npu --epochs 10 \
    --unified-labels outputs/poc3_labels_unified/labels_224 \
    --output outputs/poc3_edge/a1_relabel

# A2-relabel (multi-scale DS, unified labels)
python scripts/poc_stage_edge.py \
    -c configs/poc3_edge_a2_relabel.yaml \
    --device npu --epochs 20 \
    --unified-labels outputs/poc3_labels_unified/labels_224 \
    --output outputs/poc3_edge/a2_relabel

# 重建统一标签
python scripts/poc_stage_edge.py \
    -c configs/poc3_edge_a1_focal_dice.yaml \
    --build-manifest-only

# 推理
python scripts/poc3_edge_infer.py \
    --image <TIF> --geojson <geojson> -o out.geojson

# 批量推理 + 评估
python scripts/poc3_batch_infer.py \
    --output outputs/poc3_edge/batch_eval
```

---

## 8. Git 信息

- Branch: `worktree-poc3-edge-head`
- Worktree: `/home/ma-user/work/CoastGPT/.claude/worktrees/poc3-edge-head`
- 基于 `CoastGPT_dual` 分支

# PoC-3 P3-L 标签统一执行方案：GeoJSON 作为唯一权威源

> 日期：2026-06-18  
> 版本：v1.0  
> 状态：可执行计划  
> 阶段：PoC-3 / P3-L  
> 主题：修复 coastline GeoJSON 与 Binary TIF 标签不一致问题，重建 A0–A3 训练与评估基线  
> 结论：当前 A0–A3 的主要瓶颈不是模型、loss、分辨率、GDF 或语义先验，而是 **GeoJSON vector label 与 Binary TIF raster label 在像素级严重不一致**。必须先统一标签源，再继续评估模型路线。

---

## 0. 决策结论

当前必须新增前置阶段：

```text
P3-L: Coastline Label Unification
```

核心决策：

```text
1. 以 coastline GeoJSON 作为唯一权威标签源。
2. 完全停止使用 Binary TIF 作为 edge 训练或评估主标签。
3. Binary TIF 只保留为 legacy QA / format comparison，不进入训练 loss，不进入正式 eval GT。
4. 从 GeoJSON 统一生成：
   - edge centerline
   - soft edge target
   - GDF target
   - sea/land oracle mask
   - semantic prior 的训练 / 验证辅助标签
5. 重跑 A1/A2，重新判断模型真实上限。
```

一句话结论：

> A2 的 F1≈0.039 已经接近 GeoJSON-vs-Binary 的标签一致性天花板 buf3≈0.040。继续调模型没有意义，必须先统一标签。

---

## 1. 关键发现

### 1.1 GeoJSON vs Binary TIF 一致性测试

基于 30 samples 的 GeoJSON 与 Binary TIF 标签一致性：

| Resolution | pixel F1 | buf1 F1 | buf3 F1 |
|---|---:|---:|---:|
| ORIGINAL 128 | 0.006 | 0.019 | 0.040 |
| 224 NEAREST | 0.004 | 0.010 | 0.018 |

解释：

```text
即使在原始 128×128 分辨率下，
GeoJSON rasterized label 与 Binary TIF label 的 pixel F1 也只有 0.006，
buf3 F1 也只有 0.040。
```

这说明：

```text
GeoJSON vector label
和
Binary TIF raster label

在像素级几乎不是同一条线。
```

### 1.2 为什么这是根本问题

之前模型训练 / 评估中可能混用了：

```text
训练：Binary TIF 或 Binary TIF resize 后标签
评估：GeoJSON-derived 或不同 rasterization 规则标签
```

或者反过来。

这种情况下，即使模型完美拟合训练标签，也会在评估标签上得到极低 F1。

---

## 2. 对 A0–A3 诊断结论的重写

### 2.1 原先认为的瓶颈 vs 当前真实瓶颈

| 原先认为的瓶颈 | 当前更准确的解释 |
|---|---|
| 模型架构不好 | 标签源不一致，模型被错误 ceiling 限制 |
| BCE/Focal/Dice 不行 | loss 不是主因，监督目标本身不一致 |
| 需要 soft target | soft target 只是在缓解标签错位，不是根治 |
| 需要更大 input size | 分辨率不是第一瓶颈，448 会放大标签不一致 |
| 需要 GDF | GDF 也被不一致标签限制 |
| 需要 semantic prior | 方向仍正确，但前提是先统一标签 |
| A2/A3 失败 | 当前标签体系不足以判断 A2/A3 是否真正失败 |

### 2.2 当前 A2 结果的新解释

```text
A2 full-val F1@1px ≈ 0.039
GeoJSON-vs-Binary buf3 ceiling ≈ 0.040
```

这说明：

```text
A2 很可能已经接近当前 noisy-label / mixed-label 体系允许的上限。
```

因此，不能再用当前标签体系下的 F1 判断模型结构优劣。

---

## 3. 为什么选择 GeoJSON 作为唯一权威源

### 3.1 任务输出决定权威标签形式

PoC-3 最终输出是：

```text
GeoJSON LineString / MultiLineString
```

因此最合理的权威标签应该是：

```text
coastline GeoJSON vector geometry
```

而不是历史 Binary TIF。

### 3.2 Binary TIF 的问题

Binary TIF 存在以下不可控因素：

```text
1. 栅格化工具未知；
2. pixel-center / pixel-corner 约定未知；
3. 原始分辨率不同；
4. resize 到 224 时 NEAREST 对 1px 线极不稳定；
5. 可能与 GeoJSON 使用不同标注版本；
6. 与最终 GeoJSON 输出格式不一致；
7. 无法稳定生成 soft target / GDF / sea-land mask。
```

### 3.3 GeoJSON 的优势

GeoJSON 可以统一生成所有任务标签：

```text
1. edge centerline
2. anti-aliased edge map
3. soft edge target
4. distance transform target
5. GDF target
6. sea/land oracle mask
7. LineString / MultiLineString evaluation target
```

并且可在任意 resolution 下重新 rasterize：

```text
128 / 224 / 448 / 512
```

---

## 4. P3-L 总目标

P3-L 的目标是建立统一标签工厂：

```text
coastline GeoJSON
  → unified pixel geometry
  → edge centerline
  → soft edge target
  → GDF target
  → sea/land oracle mask
  → evaluation geometry
```

所有后续实验必须只使用这套标签。

---

## 5. 标签统一原则

### 5.1 单一源原则

正式训练和评估只使用：

```text
label_geojson_path
```

不再使用：

```text
binary_label_path
```

Dataset 逻辑应从：

```python
if binary_label_path:
    use_binary_tif()
elif label_geojson_path:
    use_geojson()
```

改成：

```python
if label_geojson_path:
    use_geojson_derived_label()
else:
    mark_invalid_or_ignore()
```

### 5.2 坐标约定

统一约定：

```text
row = y
col = x

pixel coordinate = pixel center

GeoJSON coordinate:
  [lon, lat]

pixel coordinate:
  (col, row)

edge raster index:
  edge_map[row, col]

GDF:
  dx = target_col - current_col
  dy = target_row - current_row
```

禁止在不同模块中混用：

```text
(row, col)
(col, row)
(x, y)
(y, x)
```

除非函数名和文档明确说明。

### 5.3 分辨率约定

每个样本保存：

```text
original_size
model_input_size
original_transform
model_transform
source_crs
```

统一生成标签时，使用目标分辨率直接 rasterize：

```text
GeoJSON → target_size pixel space → rasterize
```

不要：

```text
GeoJSON → original binary → NEAREST resize → target
```

### 5.4 Rasterization 约定

推荐：

```text
1. GeoJSON LineString / MultiLineString
2. WGS84 → source CRS
3. source CRS → model pixel coordinate
4. clip to tile bounds
5. densify max_step <= 0.5px
6. anti-aliased rasterization
7. centerline width=1
```

---

## 6. 需要生成的标签资产

### 6.1 Edge centerline

用途：

```text
evaluation
skeleton target
GDF distance source
```

文件：

```text
sample_id_edge_center_1px.npy
```

格式：

```text
[H, W] uint8 / float32
1 = coastline centerline
0 = background
```

### 6.2 Soft edge target

用途：

```text
A1/A2 edge training
```

生成方式：

```text
distance = distance_transform_to_centerline
soft = exp(-(distance^2) / (2 * sigma^2))
soft[distance > radius] = 0
```

推荐：

```yaml
soft_edge:
  sigma_px: 1.0
  radius_px: 3.0
```

文件：

```text
sample_id_edge_soft_sigma1_radius3.npy
```

### 6.3 GDF target

用途：

```text
A3 GDF training
```

生成：

```text
distance_transform_edt(1 - edge_center, return_indices=True)
```

输出：

```text
dx_norm
dy_norm
log_dist_norm
valid_mask
```

文件：

```text
sample_id_gdf_R32.npy
```

### 6.4 Sea/Land oracle mask

用途：

```text
P3-B0 oracle semantic prior
P3-B1/B2 sanity
```

生成：

```text
coastline GeoJSON → flood-fill / side classification → sea/land mask
```

文件：

```text
sample_id_oracle_sealand_mask.npy
```

注意：

```text
该 mask 是 oracle-style，不能作为最终泛化方案，只用于验证 semantic prior 上限。
```

### 6.5 Metadata

每个样本保存：

```text
sample_id_georef.json
```

内容：

```json
{
  "sample_id": "...",
  "source_crs": "EPSG:xxxx",
  "original_size": [128, 128],
  "model_input_size": [224, 224],
  "original_transform": [...],
  "model_transform": [...],
  "tile_bounds_wgs84": [...]
}
```

---

## 7. 推荐输出目录

```text
outputs/poc3_labels_unified/
  manifest_unified.json

  labels_224/
    edge_center/
      sample_id.npy
    edge_soft/
      sample_id.npy
    gdf_R32/
      sample_id.npy
    sealand_oracle/
      sample_id.npy
    georef/
      sample_id.json
    qa_overlay/
      sample_id_overlay.png

  labels_448/
    edge_center/
    edge_soft/
    gdf_R64/
    sealand_oracle/
    georef/
    qa_overlay/

  audits/
    geojson_vs_binary/
      agreement_metrics.csv
      overlays/
    same_source_resolution/
      consistency_metrics.csv
      overlays/
```

---

## 8. 同源跨分辨率一致性 Gate

### 8.1 目标

确认新的 GeoJSON-derived label pipeline 在不同分辨率下一致。

### 8.2 测试

对同一个 GeoJSON：

```text
A = GeoJSON directly rasterized @224
B = GeoJSON rasterized @original, then transformed/resampled consistently to 224
```

计算：

```text
pixel F1
buffered-F1@1px
buffered-F1@3px
Chamfer distance
```

### 8.3 Gate L0 标准

至少满足：

```text
buf1 F1 > 0.70
buf3 F1 > 0.90
```

如果不满足，说明：

```text
GeoJSON → pixel → rasterize 的 pipeline 仍有 bug。
```

不得进入训练。

---

## 9. 视觉 QA Gate

### 9.1 抽样

人工检查：

```text
30–50 个样本
覆盖不同 size / category / shoreline type / source image
```

### 9.2 每个样本可视化

```text
image
old Binary TIF
GeoJSON-derived edge center
GeoJSON-derived soft target
GeoJSON vs Binary overlay
GeoJSON-derived edge on image
```

### 9.3 Gate L1 标准

必须确认：

```text
1. GeoJSON-derived edge center 贴真实海岸线；
2. 没有明显 row/col 反转；
3. 没有系统性平移；
4. 没有 resize scale 错误；
5. tile bounds clipping 正确；
6. 空样本 / 多段样本处理正确。
```

---

## 10. Binary TIF 的新角色

Binary TIF 不再作为训练或评估标签。

保留用途：

```text
1. legacy reference
2. 标注格式一致性审计
3. 栅格化版本差异分析
4. 排查历史实验为何低 F1
```

报告中应明确写：

```text
Binary TIF is not used as ground truth in P3-L and later experiments.
```

---

## 11. Dataset 需要修改的地方

### 11.1 旧逻辑

```python
if tile.get("binary_label_path"):
    target = generate_edge_target_from_binary_tif(...)
elif tile.get("label_geojson_path"):
    target = generate_edge_target_from_geojson(...)
else:
    target = zeros
```

### 11.2 新逻辑

```python
if tile.get("label_geojson_path"):
    target = generate_edge_target_from_geojson(...)
else:
    target = ignore_sample_or_empty_if_verified
```

### 11.3 更推荐的离线标签模式

训练时不要动态生成标签，而是从统一标签目录读取：

```python
target = np.load(unified_label_path)
```

优点：

```text
1. 可复现；
2. 训练更快；
3. 每次实验使用完全相同标签；
4. 方便 QA；
5. 避免 data loader 中坐标转换 bug 难以追踪。
```

---

## 12. 代码落点

新增：

```text
scripts/
  build_unified_coastline_labels.py
    - 读取 manifest
    - 读取 GeoJSON
    - 生成 edge_center / edge_soft / GDF / sealand oracle
    - 保存 .npy / .json / overlay

  audit_label_consistency.py
    - GeoJSON-derived vs Binary TIF
    - same-source cross-resolution consistency
    - 输出 metrics 和 overlays

utils/
  coastline_label_factory.py
    - geojson_to_pixel_lines
    - densify_polyline
    - rasterize_centerline
    - build_soft_edge
    - build_gdf_target
    - build_sealand_oracle_mask

Dataset/
  coastline_dataset.py
    - 修改为优先读取 unified labels
    - 移除 Binary TIF 优先逻辑
```

---

## 13. 命令级执行计划

### 13.1 审计历史标签一致性

```bash
python scripts/audit_label_consistency.py \
  --manifest outputs/poc3_edge/coastline_manifest.json \
  --compare geojson_vs_binary \
  --num-samples 200 \
  --output outputs/poc3_labels_unified/audits/geojson_vs_binary
```

输出：

```text
agreement_metrics.csv
summary.json
overlays/
```

### 13.2 构建 224 统一标签

```bash
python scripts/build_unified_coastline_labels.py \
  --manifest outputs/poc3_edge/coastline_manifest.json \
  --output outputs/poc3_labels_unified/labels_224 \
  --target-size 224 \
  --soft-sigma 1.0 \
  --soft-radius 3.0 \
  --gdf-radius 32 \
  --source geojson
```

### 13.3 构建 448 统一标签

```bash
python scripts/build_unified_coastline_labels.py \
  --manifest outputs/poc3_edge/coastline_manifest.json \
  --output outputs/poc3_labels_unified/labels_448 \
  --target-size 448 \
  --soft-sigma 2.0 \
  --soft-radius 6.0 \
  --gdf-radius 64 \
  --source geojson
```

### 13.4 同源跨分辨率一致性审计

```bash
python scripts/audit_label_consistency.py \
  --manifest outputs/poc3_edge/coastline_manifest.json \
  --compare same_source_resolution \
  --source-a outputs/poc3_labels_unified/labels_224 \
  --source-b outputs/poc3_labels_unified/labels_448 \
  --output outputs/poc3_labels_unified/audits/same_source_resolution
```

### 13.5 人工 QA overlay

```bash
python scripts/build_unified_coastline_labels.py \
  --manifest outputs/poc3_edge/coastline_manifest.json \
  --output outputs/poc3_labels_unified/labels_224 \
  --target-size 224 \
  --export-overlay \
  --num-overlay 50
```

---

## 14. 重跑实验顺序

统一标签通过 Gate L0/L1 后，重跑最小实验集。

### 14.1 A1-relabel

```text
single-scale Edge Head
Focal + Dice
GeoJSON-derived soft edge
10 epoch
```

目的：

```text
验证是否突破旧标签 ceiling≈0.04。
```

通过标准：

```text
F1@1px / F1@3px 明显高于旧 A1/A2
```

### 14.2 A2-relabel

```text
multi-scale deep supervision
soft target
GeoJSON-derived labels
20–30 epoch
```

目的：

```text
验证 A2 在干净标签下是否仍然有收益。
```

### 14.3 A3-relabel

仅在 A2-relabel 仍受限时运行：

```text
GDF target from GeoJSON-derived centerline
endpoint voting
```

### 14.4 P3-B

等 edge/GDF 统一标签稳定后再做：

```text
GeoJSON-derived sea/land oracle mask
landcover semantic prior
semantic boundary extraction
semantic prior + edge/GDF refinement
```

---

## 15. 新的执行 DAG

```text
P3-L0: audit GeoJSON vs Binary TIF
  ↓
P3-L1: build GeoJSON-derived unified labels
  ↓
Gate L0: same-source cross-resolution consistency
  ↓
Gate L1: visual overlay QA
  ↓
A1-relabel
  ↓
A2-relabel
  ↓
A3-relabel if needed
  ↓
P3-B semantic prior
```

---

## 16. Gate 设计

### Gate L0：同源一致性

必须满足：

```text
GeoJSON-derived @ different resolution consistency:
  buf1 F1 > 0.70
  buf3 F1 > 0.90
```

### Gate L1：人工 overlay QA

必须满足：

```text
30–50 samples 中无系统性错位。
```

### Gate L2：Binary TIF 降级为 QA

必须确保：

```text
训练 / 正式评估 pipeline 不再读取 Binary TIF 作为 GT。
```

### Gate L3：A1-relabel 是否突破旧 ceiling

如果 A1-relabel 仍然接近旧 ceiling：

```text
F1 ≈ 0.04
```

说明仍有坐标 / rasterization / eval bug。

如果 A1/A2-relabel 明显突破：

```text
证明此前主要瓶颈是标签不一致。
```

---

## 17. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| GeoJSON 本身也不贴图像 | overlay 仍错 | 回查 source CRS / tile bounds |
| 同源跨分辨率一致性低 | Gate L0 失败 | 修 rasterization / resize / affine |
| flood-fill sea/land mask 歧义 | 岛屿/河口错误 | 先作为 oracle，不进正式训练 |
| GeoJSON 缺失样本多 | 数据量下降 | 标记 invalid，后续补数据 |
| Binary TIF 与 GeoJSON 差异大 | QA 指标低 | 记录但不用于训练 |
| A1-relabel 仍低 | 模型或 eval 仍有问题 | 做 2-sample overfit + overlay |
| 标签生成动态不稳定 | 每次结果不同 | 离线保存 .npy artifact |

---

## 18. 当前项目状态记录

建议把当前阶段记录为：

```text
P3-A0/A1/A2/A2b/A3:
  status: inconclusive under mixed label sources
  root_cause:
    GeoJSON vector label and Binary TIF raster label are inconsistent
  evidence:
    GeoJSON-vs-Binary ORIGINAL 128 buf3 F1 ≈ 0.040
    A2 full-val F1@1px ≈ 0.039
  interpretation:
    A2 reached noisy-label ceiling
  next_action:
    P3-L label unification using GeoJSON as single source of truth
```

---

## 19. 最终建议

立即执行：

```text
1. 冻结当前 A0–A3 结论；
2. 新增 P3-L label unification；
3. 用 GeoJSON 作为唯一权威源；
4. 重新生成 edge / soft / GDF / sea-land 标签；
5. 移除 Binary TIF 训练路径；
6. 通过同源跨分辨率一致性与人工 overlay QA；
7. 重跑 A1/A2；
8. 再决定是否继续 A3/P3-B。
```

最终一句话：

> 当前真正瓶颈是标签分辨率与标注格式不一致。必须先用 coastline GeoJSON 统一生成所有训练和评估标签，完全替代 Binary TIF，再重新评估 A1/A2/A3 与 P3-B。否则所有模型实验都会被错误的标签 ceiling 限制。

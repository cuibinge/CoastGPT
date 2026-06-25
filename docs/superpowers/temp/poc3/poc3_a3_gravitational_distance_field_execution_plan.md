# PoC-3 A3 执行方案：Truncated Gravitational Distance Field Coastline Detection

> 日期：2026-06-18  
> 版本：v1.0  
> 状态：可执行计划  
> 阶段：PoC-3 / A3  
> 名称：Truncated Gravitational Distance Field for Coastline Detection  
> 目标：用密集几何回归 + endpoint voting 替代 edge-only binary classification，缓解当前海岸线检测中的远距离 FP 与语义混淆问题。

---

## 0. 背景与决策结论

A0/A1/A2/A2b 的诊断已经明确：

```text
1. BCE/Dice 与 Focal/Dice 能让模型产生边缘响应，但 F1 极低。
2. Soft target + multi-scale deep supervision 有相对收益，但无法解决远距离 FP。
3. 224 和 448 的 edge-only 路线均没有学会区分“真实海岸线”与田埂、道路、建筑等其他线状纹理。
4. A2b-448 v2 证明：单纯扩大输入分辨率不是充分解。
5. 当前瓶颈不是 threshold、不是训练轮数、不是单纯定位误差，而是 edge-only supervision 缺少几何/语义约束。
```

因此 A3 不再直接预测：

```text
P(edge) ∈ [0, 1]
```

而是预测每个像素到最近海岸线的局部引力场：

```text
dx, dy, log_dist, valid_logit
```

然后通过 endpoint voting 重建海岸线。

一句话目标：

> 把“这个像素是不是海岸线”的极稀疏分类问题，改成“附近海岸线在哪里”的密集几何回归问题。

---

## 1. A3 核心假设

### 1.1 当前 edge-only 的失败模式

当前模型容易学习到：

```text
强线状纹理 = candidate coastline
```

因此田埂、道路、建筑边缘、河道边界都会产生高响应。

### 1.2 引力场方案的核心优势

Gravitational Distance Field 不要求模型只在海岸线像素上输出 1，而是让海岸线周围一片区域都学习：

```text
这个像素应该指向最近海岸线的哪个方向？
距离多远？
这个像素是否处在海岸线影响范围内？
```

优势：

```text
1. GT 从极稀疏 1px edge 变为密集监督；
2. 局部纹理边缘必须形成一致的向量收敛，才会在 voting map 中变强；
3. 远离 GT 的错误纹理更容易被 valid/conf 或 voting consistency 抑制；
4. 后处理不再依赖单点 threshold，而依赖区域投票聚集。
```

---

## 2. A3 总体路线

推荐主线：

```text
A3-0: Truncated field baseline
  → A3-1: A2 warm-start field head
  → A3-2: Multi-scale field head
  → A3-3: Semantic prior + field head
```

首轮只做 A3-0 / A3-1，不要一次性引入语义分割先验。

---

## 3. A3-0 最小可执行目标

### 3.1 输入

```text
image tile: [B, 3, 224, 224]
edge GT:    [B, 1, 224, 224] binary centerline or width=1 edge map
```

### 3.2 输出

Field head 输出：

```text
field_pred: [B, 4, H, W]
  channel 0: dx_norm
  channel 1: dy_norm
  channel 2: log_dist_norm
  channel 3: valid_logit
```

其中：

```text
dx_norm = dx / R
dy_norm = dy / R
log_dist_norm = log1p(dist) / log1p(R)
valid_logit = 是否处于海岸线影响范围
```

### 3.3 GT

从 binary edge map 生成：

```text
distance_transform_edt(1 - edge_map, return_indices=True)
```

得到每个像素到最近海岸线点的：

```text
nearest_edge_row
nearest_edge_col
distance
```

然后：

```text
dx = nearest_edge_col - col
dy = nearest_edge_row - row
dist = sqrt(dx² + dy²)
valid = dist <= R
```

---

## 4. 为什么必须使用 Truncated Field

不建议全图每个像素都回归最近海岸线。

错误版本：

```text
所有像素都预测到全图最近海岸线的方向与距离
```

问题：

```text
1. 离海岸线很远的农田/建筑/道路区域也被强制预测长向量；
2. 远距离向量噪声大；
3. 任务重新退化成全局语义推理；
4. 会让模型在背景区域学习不稳定的长程场。
```

推荐版本：

```text
只监督距离海岸线 R 像素以内的局部场；
R 之外只作为 valid=0 或 ignore。
```

初始建议：

```yaml
field:
  max_radius_px: 32
  ignore_vector_outside_radius: true
  supervise_valid_outside_radius: true
```

解释：

```text
dist <= R:
  监督 dx/dy/log_dist + valid=1

dist > R:
  不监督 dx/dy/log_dist
  valid=0
```

---

## 5. GT 生成详细设计

### 5.1 输入 edge map

使用现有 coastline GT：

```text
edge_center_1px: [H, W] binary
```

如果已有 width=3 train target，也应重新生成 width=1 centerline 作为 field 源。  
原因：distance transform 应以真实中心线为源，不应以厚边 band 为源。

### 5.2 distance transform

伪代码：

```python
import numpy as np
from scipy.ndimage import distance_transform_edt

def build_gdf_target(edge_center: np.ndarray, max_radius: int = 32):
    H, W = edge_center.shape

    non_edge = 1 - edge_center.astype(np.uint8)
    dist, indices = distance_transform_edt(
        non_edge,
        return_indices=True
    )

    nearest_r = indices[0]
    nearest_c = indices[1]

    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    dy = nearest_r.astype(np.float32) - rr.astype(np.float32)
    dx = nearest_c.astype(np.float32) - cc.astype(np.float32)

    valid = dist <= max_radius

    dx_norm = np.clip(dx / max_radius, -1.0, 1.0)
    dy_norm = np.clip(dy / max_radius, -1.0, 1.0)
    log_dist_norm = np.log1p(np.minimum(dist, max_radius)) / np.log1p(max_radius)

    target = np.stack([
        dx_norm,
        dy_norm,
        log_dist_norm.astype(np.float32),
        valid.astype(np.float32),
    ], axis=0)

    loss_mask = valid.astype(np.float32)[None, :, :]
    return target.astype(np.float32), loss_mask.astype(np.float32)
```

### 5.3 坐标约定

必须固定：

```text
row = y
col = x

dx = target_col - current_col
dy = target_row - current_row
```

不要和 georef 中的 `(col,row)` 混用。

### 5.4 Edge GT 为空时

若某 tile 没有海岸线：

```text
edge_center.sum() == 0
```

则：

```text
dx = 0
dy = 0
log_dist = 1
valid = 0
loss_mask = 0
```

只计算 valid loss，不计算 vector/distance loss。

---

## 6. 模型设计

### 6.1 A3-0：Single-scale Field Head

复用 ViT-FPN：

```text
P1/P2/P3/P4
  → upsample to P1 size
  → concat
  → Conv decoder
  → 4-channel field output
  → upsample to 224×224
```

结构：

```text
P1 [B,256,56,56]
P2 [B,256,28,28] → 56
P3 [B,256,14,14] → 56
P4 [B,256,7,7]   → 56

concat → [B,1024,56,56]
3×3 Conv 1024→256 + BN + ReLU
3×3 Conv 256→128 + BN + ReLU
1×1 Conv 128→4
upsample → [B,4,224,224]
```

输出解释：

```text
out[:,0] = dx_norm
out[:,1] = dy_norm
out[:,2] = log_dist_norm
out[:,3] = valid_logit
```

### 6.2 A3-1：A2 Warm-start Field Head

推荐主线：

```text
load A2 FPN weights
replace edge head with FieldHead
train FPN + FieldHead
```

原因：

```text
A2 FPN 已经学到一定边缘/纹理基础；
A3 只替换监督形式；
这样比从 scratch 更稳定。
```

### 6.3 A3-2：Multi-scale Field Head

类似 A2 multi-scale DS，但每个 side 输出 4 channel：

```text
P1 → side1_field [B,4,224,224]
P2 → side2_field [B,4,224,224]
P3 → side3_field [B,4,224,224]
P4 → side4_field [B,4,224,224]

fused_field = fusion(side1, side2, side3, side4)
```

首轮不建议做 A3-2。  
先验证 A3-0/A3-1 是否改善 FP distance。

---

## 7. Loss 设计

### 7.1 总 Loss

```text
L_total =
  λ_vec   * L_vec
+ λ_dist  * L_dist
+ λ_valid * L_valid
+ λ_cons  * L_consistency
```

首轮：

```yaml
loss:
  lambda_vec: 1.0
  lambda_dist: 0.5
  lambda_valid: 0.5
  lambda_consistency: 0.0
```

### 7.2 Vector loss

只在 valid 区域计算：

```python
L_vec = SmoothL1(pred_dxdy, gt_dxdy, mask=valid_mask)
```

推荐使用 SmoothL1，而不是纯 L2：

```text
SmoothL1 对局部异常 GT 更稳；
不容易被少量错误距离场拖坏。
```

### 7.3 Distance loss

```python
L_dist = SmoothL1(pred_log_dist, gt_log_dist, mask=valid_mask)
```

### 7.4 Valid loss

valid mask 是 dense binary：

```text
valid = 1: 距离海岸线 <= R
valid = 0: 距离海岸线 > R
```

Loss：

```python
L_valid = BCEWithLogits(valid_logit, valid_mask)
```

注意：valid 正负样本可能不平衡，建议使用 focal 或 pos_weight 作为可选项。

初始版本：

```yaml
valid_loss:
  type: bce
  pos_weight: auto
```

### 7.5 Consistency loss（可选）

暂不启用。

后续可加入：

```text
endpoint = pixel + pred_vector
endpoint 应靠近 edge manifold
```

或：

```text
pred_dist ≈ ||pred_vector||
```

简单一致性：

```python
pred_vec_norm = sqrt(pred_dx**2 + pred_dy**2)
L_cons = SmoothL1(pred_vec_norm, pred_dist)
```

首轮不加，避免变量过多。

---

## 8. 后处理：Endpoint Voting Reconstruction

### 8.1 不建议直接找零点

理论：

```text
dx=0, dy=0 处就是边缘
```

但实际模型很难精确输出零向量，因此不推荐作为首版后处理。

### 8.2 推荐 voting

对每个像素：

```text
current pixel p = (row, col)
pred vector v = (dy, dx)
endpoint e = p + v
```

所有 valid/conf 高的像素向 endpoint 投票。

### 8.3 Voting pipeline

```text
field_pred
  → decode dx/dy/log_dist/valid_prob
  → select pixels with valid_prob > threshold
  → endpoint = pixel + vector * R
  → bilinear scatter vote into vote_map
  → normalize / smooth
  → extract ridge
  → threshold / NMS / skeleton
  → LineString / MultiLineString
```

### 8.4 Voting 权重

建议：

```text
vote_weight = valid_prob * exp(-pred_dist / tau)
```

或者首版简单使用：

```text
vote_weight = valid_prob
```

推荐初始：

```yaml
voting:
  valid_threshold: 0.5
  tau: 16
  smooth_sigma: 1.0
```

### 8.5 Bilinear scatter

endpoint 是浮点坐标，不能直接 round。  
应把 vote 分配到四邻域：

```text
floor(row), floor(col)
floor(row)+1, floor(col)
floor(row), floor(col)+1
floor(row)+1, floor(col)+1
```

按双线性权重累加。

### 8.6 Vote map 转 coastline

```text
vote_map
  → normalize
  → threshold sweep
  → remove small components
  → skeletonize
  → path extraction
  → Douglas-Peucker
  → GeoJSON
```

A3 的输出可以复用 A2/A4 的 `edge_postprocess.py`，只需把输入 heatmap 换成 `vote_map`。

---

## 9. A3 评估指标

不要只看 F1@1px。  
A3 首要目标是降低远距离 FP，提高正确海岸线的聚集度。

### 9.1 主诊断指标

| 指标 | 当前问题 | A3 目标 |
|---|---|---|
| FP median distance | A2: 51.5px, A2b: 137px | 显著下降 |
| FP beyond 20px | A2: 79.4%, A2b: 92.3% | 显著下降 |
| TP/FP vote intensity ratio | A2≈1.0 | >1.2 |
| Oracle ceiling | A2≈0.048 | >0.10 |
| pred_fg / GT_fg | A2≈4.01 | 1–2 |
| vote map mass near GT | 低 | 明显提升 |

### 9.2 常规指标

```text
buffered-F1@1px
buffered-F1@3px
buffered-F1@5px
Chamfer distance
Hausdorff distance
component count
length ratio
GeoJSON validity
```

### 9.3 A3 成功判定

A3-0/A3-1 不要求立刻达到 PoC-3 最终目标。  
只要满足：

```text
1. FP median distance 明显低于 A2；
2. FP beyond 20px 明显低于 A2；
3. TP/FP vote intensity ratio > 1.2；
4. oracle ceiling > 0.10；
5. vote map 可视化比 edge heatmap 更集中到 GT 附近；
```

即可判定 A3 方向有效。

---

## 10. 实验矩阵

### 10.1 A3-0：Field from scratch baseline

```text
name: poc3_a3_0_gdf_from_scratch
input_size: 224
init: random FPN + FieldHead
loss: vector + distance + valid
postprocess: endpoint voting
```

目的：验证 field formulation 是否能跑通。

### 10.2 A3-1：A2 warm-start

```text
name: poc3_a3_1_gdf_warmstart_a2
input_size: 224
init:
  FPN: load A2 checkpoint
  FieldHead: random init
train:
  FPN + FieldHead
```

目的：主实验。验证在已有边缘特征基础上，field supervision 是否降低远距离 FP。

### 10.3 A3-1b：Freeze FPN, train FieldHead only

```text
name: poc3_a3_1b_gdf_head_only
init:
  FPN: load A2 checkpoint, freeze
  FieldHead: train
```

目的：判断 field head 本身是否能利用 A2 feature。

### 10.4 A3-2：Multi-scale GDF

```text
name: poc3_a3_2_multiscale_gdf
side outputs:
  P1/P2/P3/P4 each predicts field
```

目的：如果 A3-1 有效，再做多尺度 field deep supervision。

### 10.5 A3-3：Semantic prior + GDF

```text
name: poc3_a3_3_semantic_prior_gdf
input:
  FPN features + sea/land semantic prior
```

目的：最终增强路线，与 P3-B semantic prior 融合。

---

## 11. 推荐执行顺序

```text
Step 1: 实现 GDF target 生成
Step 2: 可视化 dx/dy/log_dist/valid target
Step 3: 实现 FieldHead
Step 4: 实现 GDF loss
Step 5: 实现 endpoint voting
Step 6: 跑 synthetic unit test
Step 7: 跑 2-sample overfit
Step 8: 跑 A3-1 warm-start 10 epoch
Step 9: 诊断 FP distance / vote map / oracle ceiling
Step 10: 决定是否进入 A3-2 或 P3-B semantic prior
```

---

## 12. 代码落点

建议新增文件：

```text
utils/
  gdf_target.py
    - build_gdf_target
    - visualize_gdf_target
    - validate_gdf_target

Models/
  gdf_head.py
    - SingleScaleGDFHead
    - MultiScaleGDFHead

utils/
  gdf_losses.py
    - masked_smooth_l1
    - gdf_loss
    - optional consistency_loss

utils/
  gdf_postprocess.py
    - decode_field
    - endpoint_voting
    - vote_map_to_geojson

scripts/
  poc_stage_gdf.py
    - train/eval entrypoint

configs/
  poc3_a3_0_gdf_from_scratch.yaml
  poc3_a3_1_gdf_warmstart_a2.yaml
  poc3_a3_1b_gdf_head_only.yaml
```

可复用文件：

```text
Dataset/coastline_dataset.py
utils/edge_postprocess.py
utils/coastline_metrics.py
Models/fpn_neck.py
Models/dual_vision_encoder.py
```

---

## 13. 配置模板

### 13.1 A3-1 warm-start config

```yaml
experiment:
  name: poc3_a3_1_gdf_warmstart_a2
  output_dir: outputs/poc3_gdf/a3_1_warmstart_a2
  stage: P3-A3
  seed: 42

data:
  roots:
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线二级/Patches
    - /home/ma-user/work/Stage3Data/海岸线/RS-海岸线一级/Patches
  manifest_path: outputs/poc3_edge/coastline_manifest.json
  image_size: 224
  line_width_eval: 1
  num_workers: 2

field:
  max_radius_px: 32
  ignore_vector_outside_radius: true
  supervise_valid_outside_radius: true
  output_channels: 4

model:
  backbone: dual_vision_encoder
  fpn: vit_fpn
  vit_fusion: true
  input_size: [224, 224]

  init:
    load_a2_checkpoint: outputs/poc3_edge/a2_multiscale_deepsup/checkpoints/best.pt
    load_fpn: true
    load_edge_head: false

  gdf_head:
    type: single_scale
    in_channels: 256
    decoder_channels: [256, 128]
    output_size: [224, 224]

train:
  device: npu
  epochs: 20
  batch_size: 2
  accum_steps: 8
  lr_fpn: 2.0e-5
  lr_gdf_head: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  freeze_vision: true
  precision: bf16
  log_interval: 20
  val_interval: 5
  save_interval: 5

loss:
  lambda_vec: 1.0
  lambda_dist: 0.5
  lambda_valid: 0.5
  lambda_consistency: 0.0
  valid_loss:
    type: bce
    pos_weight: auto

voting:
  valid_threshold: 0.5
  use_dist_weight: true
  tau: 16.0
  smooth_sigma: 1.0
  threshold_sweep: true
  threshold_values: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

postprocess:
  min_component_area: 8
  min_line_length_px: 10
  max_components: 5
  douglas_peucker_epsilon_px: 1.0

eval:
  metrics:
    - buffered_f1_1px
    - buffered_f1_3px
    - buffered_f1_5px
    - fp_median_distance
    - fp_beyond_20px
    - tp_fp_intensity_ratio
    - oracle_ceiling
    - pred_fg_gt_fg_ratio
    - vote_mass_near_gt
  export_overlay: true
  export_vote_map: true
  export_geojson: true
  max_overlay_samples: 16
```

---

## 14. 单元测试与 Smoke Test

### 14.1 GDF target synthetic test

构造一条水平线：

```text
edge row = 100
```

期望：

```text
像素 (row=90, col=50):
  dy = +10
  dx = 0

像素 (row=110, col=50):
  dy = -10
  dx = 0

edge 上:
  dx = 0
  dy = 0
  dist = 0
```

### 14.2 Voting synthetic test

使用 GT field 直接 voting：

```text
输入 perfect GT field
输出 vote_map 应在原 edge 处聚集
buffered-F1 应接近 1
```

如果 perfect GT field voting 都失败，说明 voting 实现有 bug。

### 14.3 2-sample overfit

```text
num_samples: 2
iterations: 300
batch_size: 2
```

通过标准：

```text
1. L_vec / L_dist / L_valid 均下降；
2. valid_prob 在海岸线附近升高；
3. endpoint voting map 能贴近 GT；
4. FP median distance 明显低。
```

---

## 15. 训练与评估命令建议

### 15.1 Target 可视化

```bash
python scripts/visualize_gdf_target.py   --config configs/poc3_a3_1_gdf_warmstart_a2.yaml   --num-samples 16
```

### 15.2 Synthetic voting test

```bash
python utils/gdf_postprocess.py --test
```

### 15.3 2-sample overfit

```bash
python scripts/poc_stage_gdf.py   --config configs/poc3_a3_1_gdf_warmstart_a2.yaml   --overfit-samples 2   --epochs 1   --max-iters 300
```

### 15.4 A3-1 warm-start training

```bash
python scripts/poc_stage_gdf.py   --config configs/poc3_a3_1_gdf_warmstart_a2.yaml
```

---

## 16. Gate 设计

### Gate 0：Target generation

必须满足：

```text
1. dx/dy 方向正确；
2. edge 上 dist=0；
3. valid mask 是海岸线周围 R 像素带；
4. 空 GT tile 不崩；
5. target 可视化合理。
```

### Gate 1：Perfect field voting

必须满足：

```text
1. 用 GT field voting 能恢复 edge；
2. vote_map ridge 在 GT 上；
3. F1@1/3px 接近 1；
4. 无 row/col 反向。
```

### Gate 2：Overfit

必须满足：

```text
1. 2-sample loss 明显下降；
2. predicted field 可视化合理；
3. vote_map 与 GT 重合；
4. endpoint 分布收敛到海岸线。
```

### Gate 3：A3-1 validation

必须满足至少两个：

```text
1. FP median distance < A2 baseline；
2. FP beyond 20px < A2 baseline；
3. TP/FP vote intensity ratio > 1.2；
4. oracle ceiling > A2 baseline；
5. buffered-F1@3/5px 提升。
```

如果 Gate 3 全部不满足，A3 几何回归路线失败，应转 P3-B semantic prior。

---

## 17. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| valid 区域太大 | FP 多，背景也投票 | 降 R: 32→24 |
| valid 区域太小 | 监督仍稀疏 | 升 R: 32→48 |
| valid loss 主导 | 模型只学 valid，不学方向 | 降 λ_valid |
| vector loss 不收敛 | dx/dy 噪声大 | 用 SmoothL1，clip dx/dy |
| vote map 很散 | 向量方向不一致 | 加 consistency 或提高 valid threshold |
| vote map FP 远离 GT | 仍缺语义 | 转 P3-B semantic prior |
| perfect field voting 失败 | 后处理 bug | 修 bilinear scatter / row-col |
| overfit 失败 | head/loss/target bug | 先不要跑 full train |

---

## 18. 预期结果与判断

### 如果 A3 成功

期望：

```text
FP median distance: 明显下降
FP beyond 20px: 明显下降
TP/FP intensity ratio: >1.2
oracle ceiling: >0.10
vote map 比 edge heatmap 更贴 GT
```

则进入：

```text
A3-2 multi-scale GDF
或
A4 topology postprocess
```

### 如果 A3 部分成功

例如 FP distance 下降，但 F1 仍低：

```text
说明 field 有几何收益，但还需要语义先验。
```

进入：

```text
A3-3 semantic prior + GDF
```

### 如果 A3 失败

例如：

```text
FP distance 不降；
oracle ceiling 不升；
TP/FP ratio 仍接近 1；
vote map 仍落在田埂/道路/建筑边缘。
```

说明：

```text
几何回归无法替代语义判别；
必须转 P3-B sea/land semantic prior。
```

---

## 19. 最终建议

A3 是值得执行的低成本路线，因为：

```text
1. 不需要额外 landcover label；
2. 可以复用现有 coastline edge GT；
3. 改动集中在 target / head / loss / voting；
4. 能快速验证“分类 → 回归”是否缓解远距离 FP；
5. 若成功，可与 semantic prior 进一步融合。
```

推荐立即执行：

```text
A3-0 target + voting unit test
→ A3-1 A2 warm-start GDF
→ 根据 FP distance / oracle ceiling 决定是否进入 A3-2 或 P3-B。
```

最终一句话：

> A3 不再把海岸线当作稀疏二值边缘分类，而是把海岸线作为一个局部几何吸引子，通过 truncated distance field 和 endpoint voting 重建；它是当前 edge-only 路线失败后的合理低成本转向。

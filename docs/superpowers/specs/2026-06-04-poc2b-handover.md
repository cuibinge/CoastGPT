# PoC-2b 土地覆盖 Semantic Head — 工作交接

> 日期: 2026-06-04 (最后更新: 2026-06-06)
> 关联: `2026-06-01-poc2-landcover-semantic-head-design.md`, `poc2_full_eval_findings_and_next_steps.md`

---

## 一、PoC-2b 是什么

CoastGPT v2.1 的 PoC-2 阶段：在冻结的 DualVisionEncoder 上接入 FPN + LandcoverSemanticHead，做 25 类（含 background）土地覆盖语义分割。

完整 pipeline：
```
DualVisionEncoder (frozen)
  → FPNNeck (trainable)
  → LandcoverSemanticHead (trainable)
  → per-pixel softmax → argmax → connected components → polygonize → MultiPolygon GeoJSON
```

- 训练脚本: `scripts/poc_stage_semantic.py`
- 评估脚本: `scripts/eval_semantic_full.py`
- 配置文件: `configs/poc2_landcover_semantic.yaml`

---

## 二、数据

| 项目 | 值 |
|------|-----|
| 源图数 | 11 张 GF-2 卫星影像 |
| 总 tile | 26,816 |
| 训练 | 22,440 |
| 验证 | 4,376 (按 source_image 划分，非 stratified) |
| 类别数 | 24 active DLMC + 1 background (IGNORE_INDEX=255) |
| 尺寸分布 | 128×128 为主，少量 256×256 和 512×512 |
| labeled_ratio | mean 81%, median ~92% |

**Val split 问题**: 原始 random split 中 512×512 tile 为 0（所有 512 tile 来自同一张源图，被整张划入训练）。后续构建了 `val_stratified_v2` 解决（见 §七）。

---

## 三、已完成的实验

### 3.1 Baseline（无 background prior）

| 指标 | 值 |
|------|-----|
| 训练 epoch | 1 |
| Full eval mIoU | **0.5324** (4,376 val samples) |
| pred_fg_ratio | **100%** |
| 沿海滩涂 IoU | 0.80 |
| 盐田 IoU | 0.71 |
| 养殖坑塘 IoU | 0.60 |
| 12 个类别 IoU≈0 | 含公路用地/农村道路等 thin classes |

**结论**: 架构闭环通过，主类有效，但 model 完全不会预测 background。

### 3.2 Background Prior 消融实验（共 8 次）

三种方案，共 8 次实验：

| # | 方案 | λ | 关键参数 | pred_fg | mIoU | 结论 |
|---|------|----|----------|---------|------|------|
| 1 | penalty | 0.05 | min_labeled=0.5 | ~100% | - | 梯度太弱，无效 |
| 2 | penalty | 0.2 | min_labeled=0.5 | ~100% | - | 同上 |
| 3 | sampled BCE | 1.0 | max_ignore=0.25 | 72.9% | 公路用地崩 (0.35→0.01) | 过强 |
| 4 | sampled BCE | 0.05 | max_ignore=0.1 | 98.1% | 0.6174 | 过弱 |
| **5** | **sampled BCE** | **0.10** | max_ignore=0.1 | **94.6%** | **0.6080** | 压制不足 |
| **6** | **sampled BCE** | **0.30** | max_ignore=0.1 | **83.2%** | **0.5632** | **首次有效压制 bg，但 thin class 崩塌** |
| 7 | penalty | 0.05 | min_labeled=0.5 (复现) | ~100% | - | 复现确认无效 |
| 8 | penalty | 0.20 | min_labeled=0.5 (复现) | ~100% | - | 复现确认无效 |

### 3.3 方案细节

**Penalty 方案** (`_background_prior_loss`):
```python
p_fg = 1 - softmax(logits, dim=1)[:, 0]
loss = λ * p_fg[ignore_mask].mean()
```
问题：λ 被 |ignore_mask| 稀释，单像素梯度 ~0.002，比 CE loss 弱 20-40 倍。

**Sampled BCE 方案** (`_sampled_background_bce_loss`):
```python
log_p_bg = log_softmax(logits, dim=1)[:, 0]
sample = random_subset(ignore_pixels, max_ignore_ratio * n_labeled)
loss = λ * (-log_p_bg[sample].mean())
```
问题：对所有 ignore 像素无差别施压，无法区分"真是背景"和"只是没标注"。

### 3.4 核心矛盾

```
模型在 partial-label 数据上训练：
  - 标注像素 → CE + Dice (明确监督)
  - ignore 像素 → 无监督 (IGNORE_INDEX=255)

需要模型做到：
  - 标注为"水田"的像素 → 预测为"水田" ✓
  - 未标注但实际是"水田"的像素 → 预测为"水田" (不要压)
  - 未标注且真的是背景的像素 → 预测为"background" (需要压)

但 BCE prior 无法区分后两种情况。
```

### 3.5 中间 λ 搜索：per-class IoU 对比 (2026-06-06 新增)

BCE λ=0.05 / 0.10 / 0.30 三个值在 epoch 5 的逐类对比：

| 类别 | λ=0.05 | λ=0.10 | λ=0.30 | 趋势 |
|------|:------:|:------:|:------:|------|
| 沿海滩涂 | 0.842 | 0.836 | **0.814** | 微降 (~3%) |
| 养殖坑塘 | 0.761 | 0.750 | **0.695** | ↓ 8.7% |
| 盐田 | 0.706 | 0.708 | 0.708 | **稳定** |
| 农村宅基地 | 0.453 | 0.432 | 0.365 | ↓ 19% |
| 工业用地 | 0.429 | 0.423 | 0.376 | ↓ 12% |
| 公路用地 | 0.351 | 0.308 | **0.137** | **崩塌** (−61%) |
| 水田 | 0.350 | 0.359 | 0.353 | **稳定** |
| 河流水面 | 0.298 | 0.286 | 0.227 | ↓ 24% |
| 其他草地 | 0.225 | 0.215 | 0.190 | ↓ 16% |
| 旱地 | 0.108 | 0.098 | **0.046** | **崩塌** (−57%) |
| 水浇地 | 0.080 | 0.092 | 0.044 | ↓ 45% |
| 沟渠 | 0.089 | 0.076 | 0.039 | ↓ 56% |
| 其他林地 | 0.043 | 0.043 | 0.042 | 稳定 |
| 港口码头用地 | 0.026 | 0.026 | 0.025 | 稳定 |
| 水工建筑用地 | 0.025 | 0.019 | 0.006 | ↓ 76% |
| 坑塘水面 | 0.018 | 0.019 | 0.017 | 稳定 |
| 设施农用地 | 0.019 | 0.013 | 0.012 | ↓ 37% |
| 内陆滩涂 | 0.011 | 0.008 | 0.001 | 崩塌 |
| 农村道路 | 0.0 | 0.0 | 0.0 | 始终 0 |
| 其他园地 | 0.0 | 0.0 | 0.0 | 始终 0 |
| 城镇村道路用地 | 0.0 | 0.0 | 0.0 | 始终 0 |

**各 λ 的 bg suppression 与 mIoU 对比：**

| 指标 | λ=0.05 | λ=0.10 | λ=0.30 |
|------|:------:|:------:|:------:|
| **observed mIoU** | 0.6174 | 0.6080 | 0.5632 |
| pixel_accuracy | 0.6846 | 0.6735 | 0.6225 |
| pred_fg_ratio | 98.1% | 94.6% | **83.2%** |
| ignore_region_fg_rate | 89.7% | 71.6% | **11.6%** |
| pred/labeled area ratio | 1.21 | 1.17 | 1.03 |
| fg_bg_confusion_rate | 0.449 | 0.464 | 0.526 |

**结论**: λ=0.30 是第一个有效压制 background 的值（ignore_region_fg 从 90% → 12%），但代价是 thin/rare class 大范围崩塌。λ=0.10 处于中间地带，bg 压制不够（ignore_region_fg 仍 ~72%）。

---

## 四、关键代码文件

### 新增

| 文件 | 用途 |
|------|------|
| `Models/fpn_neck.py` | FPN 特征金字塔 |
| `Models/semantic_head.py` | LandcoverSemanticHead, 25-class per-pixel classification |
| `Dataset/landcover_dataset.py` | LandcoverSemanticDataset, 含 target cache |
| `Dataset/landcover_tile_grouping.py` | 扫描+合并 tile → spatial groups |
| `Dataset/landcover_label_map.py` | DLMC 标签映射 (name↔train_id↔dlmc_code) |
| `scripts/poc_stage_semantic.py` | 训练主脚本 |
| `scripts/eval_semantic_full.py` | 全量验证评估脚本 |
| `scripts/build_stratified_val.py` | 构建 size-stratified val split |
| `scripts/precache_landcover_targets.py` | 预缓存 GT target tensor |
| `configs/poc2_landcover_semantic.yaml` | 训练配置 |

### 修改

| 文件 | 改动 |
|------|------|
| `Models/dual_vision_encoder.py` | `encode_with_spatial()` 返回 pyramid features c4/c8/c16/c32 |
| `Dataset/__init__.py` | 注册 landcover 模块 |
| `Dataset/landcover_tile_grouping.py` | merge 输出新增 `grid` 字段 |

---

## 五、Checkpoint 和输出

### Baseline

| 文件 | 说明 |
|------|------|
| `outputs/poc2_landcover_semantic/checkpoints/epoch_001.pt` | baseline (无 prior), mIoU=0.7135 (subset) |
| `outputs/poc2_landcover_semantic/checkpoints/best_miou.pt` | 同 epoch_001 |
| `outputs/poc2_landcover_semantic/metrics/eval_full.json` | baseline full eval (mIoU=0.5324) |
| `outputs/poc2_landcover_semantic/metrics/eval_stratified_v2_baseline.json` | stratified v2 baseline eval |

### BCE λ=0.05

| 文件 | 说明 |
|------|------|
| `outputs/poc2b_bce_005/checkpoints/epoch_002.pt` ~ `epoch_005.pt` | BCE λ=0.05 训练 |
| `outputs/poc2b_bce_005/metrics/metrics_epoch_005.json` | mIoU=0.6174, pred_fg=98.1% |

### BCE λ=0.10 (2026-06-06 新增)

| 文件 | 说明 |
|------|------|
| `outputs/poc2b_bce_010/checkpoints/epoch_002.pt` ~ `epoch_005.pt` | BCE λ=0.10 训练 |
| `outputs/poc2b_bce_010/metrics/metrics_epoch_005.json` | mIoU=0.6080, pred_fg=94.6% |

### BCE λ=0.30 (2026-06-06 新增)

| 文件 | 说明 |
|------|------|
| `outputs/poc2b_bce_030/checkpoints/epoch_002.pt` ~ `epoch_005.pt` | BCE λ=0.30 训练 |
| `outputs/poc2b_bce_030/metrics/metrics_epoch_005.json` | mIoU=0.5632, pred_fg=83.2% |

### 之前已清理的实验

- `outputs/poc2b_bg_prior_005/` — penalty λ=0.05 (已删除)
- `outputs/poc2b_bg_prior_020/` — penalty λ=0.2 (已删除)
- `outputs/poc2b_bce_sampled/` — BCE λ=1.0 (已删除)
- `outputs/poc2b_bce_050/` — BCE λ=0.50 (空目录，未实际训练)

### Stratified Val

| 文件 | 说明 |
|------|------|
| `outputs/poc2_landcover_semantic/val_stratified_v2.json` | 119 个 512 tiles 的 val split |

---

## 六、已解决的工程问题

1. **eval_semantic_full.py 多次 bug fix**: alignment_dim 缺失 → final_proj shape 不匹配、FPN/semantic_head 参数名错误、argmax dim 错误、mIoU 计算用 pred∪GT 而非 GT-only
2. **GT cache 预缓存**: dataset 初始化时预计算 target tensor，避免重复 rasterize
3. **stratified val split**: 512 tile 全部来自 1 张源图 → grid parity split 解决
4. **训练输出 unbuffered**: `sys.stdout.reconfigure(line_buffering=True)`
5. **source-image 级防泄漏**: 同一卫星影像的 tile 不跨 train/val

---

## 七、核心问题进展 (2026-06-06 更新)

### Background prior：λ=0.30 首次有效，但 thin class 代价大

| λ=0.05 | 压不动，pred_fg=98.1% |
| λ=0.10 | 压制不足，ignore_region_fg 仍 71.6% |
| λ=0.30 | **ignore_region_fg 降至 11.6%**，但公路用地 IoU 0.351→0.137、旱地 0.108→0.046、内陆滩涂 0.011→0.001 |

λ=0.30 的 pred_fg=83.2% 已经接近 labeled_ratio=81.0%，说明背景压制在数值上有效。但 BCE loss 对所有 ignore 像素无差别施压→thin/rare class 的少量未标注前景像素被当成背景惩罚→类别崩塌。

### 7 个类别始终 IoU≈0

农村道路、其他园地、城镇村道路用地 三项始终为 0（全部 λ 都是 0），加上内陆滩涂在 λ=0.30 崩塌。其中部分是 thin class（道路 1-2px），部分可能 train/val 都 low support。

### 可能的替代方向（按优先级）

1. **Explicit background tiles**（未标注区域 ≥ 90% 的 tile 混入训练）：在这些 tile 上 BCE prior 是无害的（几乎全是真背景），避免在 partial-label tile 的无差别压制
2. **Per-class frequency weighting**：对 rare/thin class 像素 BCE prior 降权，保护它们不被压死
3. **Label smoothing**：替 BCE prior，将 bg 目标从 hard 0/1 改为 0.1
4. **UNK class**：不压成 bg，压成"不确定"类，推理时 argmax 跳过

---

## 八、决策树（2026-06-06 更新）

```
PoC-2b 成功？
├─ BCE λ 中间搜索完成 (0.05/0.10/0.30)
│   ├─ λ=0.05: 压制无效
│   ├─ λ=0.10: 压制不足
│   └─ λ=0.30: bg 有效压制，但 thin class 崩塌 → 需要新方案
│
├─ 下一步选择
│   ├─ 尝试 explicit bg tiles 混入训练
│   ├─ 尝试 per-class weighting 保护 thin class
│   ├─ 尝试 label smoothing / UNK class
│   └─ 如果 thin class 本质是 224px 不可见 → 不应在 224 继续优化
│
└─ 如果所有方案都无效
    → 判定 partial-label semantic segmentation 在当前设置下不可行
    → 考虑换监督范式（weakly-supervised / point supervision / MIL）
```

当前状态：**BCE λ 搜索已完成（三个 λ 点），确认了 trade-off 的存在。下一步应尝试能区分"未标注前景"和"真背景"的方案，而非继续微调 λ。**

---

## 九、下一步建议 (2026-06-06 更新)

~~1. 512 stratified eval~~ — 已完成
~~2. 中间 λ 搜索~~ — **已完成**: λ ∈ {0.05, 0.10, 0.30} 三点搜索，确认 BCE prior 存在 trade-off（λ=0.30 有效压制 bg 但 thin class 崩塌）

**后续方向（按优先级）：**

1. **Explicit background tiles**: 从训练集筛选未标注区域 ≥ 90% 的 tile，混入训练。在这些 tile 上 BCE prior 几乎全是真背景，不会误伤 thin class
2. **Per-class weighting**: 对 rare/thin class 像素施加保护权重，让 BCE prior 对它们降权
3. **Label smoothing / UNK class**: 切换 punish 机制，避免 hard 0/1 惩罚
4. **PoC-2 pass/fail 决策**: 如果 thin classes 本质是"224 分辨率下不可见"，则不应在 224 下继续优化，应在动态分辨率改造后重新评估

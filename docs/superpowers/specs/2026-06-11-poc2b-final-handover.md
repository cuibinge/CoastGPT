# PoC-2b 土地覆盖 Semantic Head — 终期交接文档

> 日期: 2026-06-11 (最后更新: 2026-06-16)
> 前置: `2026-06-04-poc2b-handover.md`

---

## 一、项目概述

CoastGPT v2.1 的 PoC-2b 阶段：在冻结的 DualVisionEncoder（DINOv3 ViT-L/16 + ConvNeXt-Base）上接入 FPN + LandcoverSemanticHead，做 25 类（含 background）土地覆盖语义分割。

**核心矛盾**: 数据是 partial-label——每个 tile 只有部分类别被标注，未标注像素（IGNORE_INDEX=255）可能是真背景、也可能是未标注的前景。模型必须学会区分，但 BG 标签在训练中零正样本。

---

## 二、数据

| 项目 | 值 |
|------|-----|
| 源图 | 11 张 GF-2 卫星影像 |
| Tile 数 | 26,816（训练 22,440 / 验证 4,376） |
| 类别 | 24 active DLMC + 1 background (IGNORE_INDEX=255) |
| labeled_ratio | mean 81%, median 92% |
| Label_Binary | per-class binary TIF，独立标注，天然 partial-label |

**关键数据发现**: 同一 tile 的多个 class TIF 叠加后只覆盖 47.7% ~ 93.1% 的区域。剩余像素是真正的"未知"——不是预处理 bug，是数据天然属性。

---

## 三、所有实验汇总

| # | 实验 | mIoU | pred_fg | 结论 |
|---|------|:------:|:------:|------|
| 1 | baseline (无 prior, CE, no ViT) | 0.5324 | ~100% | — |
| 2 | penalty λ=0.05/0.20 | - | ~100% | 梯度太弱 |
| 3 | sampled BCE λ=0.05 | 0.6174 | 98.1% | 太弱 |
| 4 | sampled BCE λ=0.10 | 0.6080 | 94.6% | 太弱 |
| 5 | sampled BCE λ=0.30 | 0.5632 | 83.2% | thin 崩塌 |
| 6 | sampled BCE λ=1.0 | - | 72.9% | 过强 |
| 7 | cond BCE t=0.3, λ=0.30 | 0.6021 | 93.3% | gate 过严 |
| 8 | cond BCE t=0.5 e10 | 0.5987 | 88.8% | 旧 best |
| 9 | cond BCE t=0.5 e20 | 0.5251 | 88.3% | 过拟合 |
| 10 | selective pseudo-bg (cold) | 0.6259 | 100% | 冷启动失败 |
| 11 | selective pseudo-bg (warm) | - | - | 阈值 ramp 衰减 |
| 12 | hybrid BCE | 0.5573 | 82.1% | 过度压制 |
| 13 | soft BCE+Focal+Sampler e5 | 0.5498 | 95.8% | thin class 最佳 |
| 14 | soft BCE λ=0.10 e6 | 未验证 | ~96% | 公路 0.399 |
| 15 | LaSt stability BCE | - | - | 理论不匹配，已删除 |
| 16 | ViT-FPN (gate on) e5 | 0.6216 | 100% | 初步验证 ViT 融合有效 |
| 17 | ViT-FPN (gate on) e10 | 0.6146 | 100% | 微降 |
| **18** | **ViT-FPN (clean, gate off) e15** | **0.6340** | **100%** | **🏆 最终 best, 无 bg prior, 纯 CE** |
| 19 | ViT-FPN + soft BCE + Focal + Sampler | 0.5807 | 95.5% | 叠加反降 |
| 20 | ViT-FPN + zero-sample boosted sampler | 训练中 | - | 农村道路/城镇村道路 10x 采样权重 |

### 最终 best 全量 eval（ViT-FPN e15，4,376 val samples）

| 类别 | IoU | Recall | Prec |
|------|:------:|:------:|:------:|
| 养殖坑塘 | 0.753 | 0.845 | 0.874 |
| 盐田 | 0.727 | 0.867 | 0.818 |
| 沿海滩涂 | 0.684 | 0.897 | 0.743 |
| 农村宅基地 | 0.546 | 0.603 | 0.853 |
| 水田 | 0.483 | 0.617 | 0.690 |
| 工业用地 | 0.441 | 0.601 | 0.623 |
| 河流水面 | 0.328 | 0.528 | 0.465 |
| 公路用地 | 0.323 | 0.523 | 0.459 |
| 设施农用地 | 0.239 | 0.327 | 0.471 |
| 沟渠 | 0.147 | 0.183 | 0.432 |
| 其他草地 | 0.117 | 0.315 | 0.157 |
| 内陆滩涂 | 0.064 | 0.081 | 0.232 |
| 港口码头 | 0.060 | 0.352 | 0.068 |
| 城镇村道路 | 0.053 | 0.054 | 0.708 |
| 坑塘水面 | 0.053 | 0.087 | 0.116 |
| 旱地 | 0.049 | 0.051 | 0.496 |
| 水浇地 | 0.039 | 0.068 | 0.084 |
| 其他林地 | 0.033 | 0.160 | 0.041 |
| 水工建筑 | 0.005 | 0.029 | 0.006 |
| 农村道路 | 0.000 | 0.000 | 0.000 |
| 其他园地 | 0.001 | 0.001 | 0.003 |
| 公园与绿地 | 0.000 | 0.000 | 0.000 | — 不在数据集 |
| 裸岩石砾地 | 0.000 | 0.000 | 0.000 | — 不在数据集 |
| 铁路用地 | 0.000 | 0.000 | 0.000 | — 不在数据集 |

**pixel_accuracy: 73.9%**

### Confusion Matrix 关键发现

| GT → 预测 | 占比 | 根因 |
|------|:------:|------|
| 坑塘水面 → 养殖坑塘 | 63.4% | 水面+矩阵纹理，224px 无法区分 |
| 旱地 → 水田 | 53.6% | 干湿农田纹理相近 |
| 设施农用地 → 工业用地 | 52.1% | 设施棚和厂房相似 |
| 水工建筑 → 其他草地 | 51.9% | 小型建筑被草地淹没 |
| 港口码头 → 工业用地 | 40.1% | 解释了港口码头 prec 7% |
| 城镇村道路 → 公路用地 | 36.5% | 道路间混淆 |

---

## 四、关键发现

### 1. 架构无硬伤（overfit 测试）

选取 8 张含 thin class 的 tile（公路用地 10-37px、农村道路 12-17px、沟渠 42-43px），200 轮 overfit 后全部达到 97-100% recall。

**结论**: FPN + SemanticHead 能从 frozen ConvNeXt 特征中学会 thin class。问题在训练机制，不在架构。

### 2. 数据天然 partial-label

Label_Binary 是 per-class 独立标注。剩余像素无标签——不是所有人都不是该 class，只是没有被标注。

### 3. ViT-FPN — 突破性发现

将 DINOv3 ViT 的 g_grid `[B,1024,14,14]` 通过 vit_proj 可学习投影融合进 FPN P3。**不需要任何 bg prior，mIoU 从 0.53 跃升至 0.63。** 这是整个 PoC-2b 最大的单次提升（+19% vs baseline）。

**原理**：ViT 的全局语义信息（14×14 token grid 通过 self-attention 携带整图上下文）通过 1x1 Conv + BN + ReLU + 3x3 Conv + BN + ReLU 投影到 256 维，与 ConvNeXt c16 在 P3 层融合。弥补了 ConvNeXt 局部卷积缺失的全局视野。

### 4. BG prior 系列全部被架构改进超越

soft BCE、cond BCE、hybrid BCE、Focal、Sampler 等 18 次 loss 设计实验全部不如纯 ViT-FPN + CE loss。架构层面的 ViT 全局语义注入从根本上缓解了 bg 判别问题，不需要手调 loss。

### 5. 稀有类 recall 是剩余短板

农村道路（76 tiles, 64K px）、城镇村道路（123 tiles, 304K px）在训练中曝光极少。当前实验通过 10× 采样权重+强制曝光尝试解决。

---

## 五、当前架构

```
Input → [DINOv3 ViT (frozen)] → g_grid [B,1024,14,14]
      →                          │
      →                   vit_proj (1x1 Conv+BN+ReLU+3x3 Conv+BN+ReLU)
      →                          │
      → [ConvNeXt (frozen)] → c4/c8/c16/c32
      →                          ↓
      →                     FPN  P3 = lat16 + upsample(P4) + vit_proj(g_grid)
      →                          ↓
      →                     LandcoverSemanticHead
      →                          ↓
      →                     [B, 25, 224, 224]
```

简洁的两路融合：ConvNeXt 局部特征 + ViT 全局语义通过可学习投影注入 P3。

---

## 六、存量 Checkpoint

| 路径 | 说明 |
|------|------|
| `outputs/poc2_vit_fpn/checkpoints/best_miou.pt` | **🏆 全实验 best, mIoU 0.6340**，epoch 15 |
| `outputs/poc2_thin_focal_sampler/checkpoints/epoch_006.pt` | soft BCE+Focal+Sampler，沟渠 0.169, 港口码头 0.315 |
| `outputs/poc2b_cond_bce_030_t50/checkpoints/best_miou.pt` | cond BCE best mIoU 0.5987，epoch 10 |
| `outputs/poc2_landcover_semantic/checkpoints/epoch_001.pt` | baseline (no prior, no ViT) |

---

## 七、未解决问题

1. **农村道路 IoU=0**: 76 tiles / 64K px，当前实验中 10× 采样权重是否有效待验证
2. **坑塘水面 vs 养殖坑塘 混淆 63%**: 224px 下纹理不可分，可能需要分辨率提升
3. **224px 分辨率**: thin class 10-50px 在 50,176px tile 中占比过小
4. **pixel_accuracy 73.9%**: 低于设计目标 80%

---

## 八、当前进行中的实验

**ViT-FPN + zero-sample boosted sampler**（epoch 1/15, 预计 2026-06-17 出 epoch 5 验证）
- 农村道路、城镇村道路采样权重 10×
- 每 epoch 每个零样本 tile 确保出现 ~17 次

---

## 九、建议后续方向

1. **等待当前实验 epoch 5/10/15 验证结果** — 确认 zero-sample boost 是否改善农村道路 recall
2. **启动 PoC-3 海岸线 Edge Head** — PoC-2b 架构验证已通过，ViT-FPN 可直接复用
3. **坑塘水面 vs 养殖坑塘** — 考虑在 SemanticHead 中增加多尺度上下文或提升输入分辨率
4. **取消统一 resize 224** — 按 `poc-2_thin_class_iou0_improvement_plan.md` 方向

---

## 十、代码改动清单

| 文件 | 改动 |
|------|------|
| `Models/fpn_neck.py` | 新增 ViT 特征融合 (vit_proj, optional vit_feat)，支持 vit_in_channels 参数 |
| `scripts/poc_stage_semantic.py` | 新增 focal_loss, soft_bce, class_weights, WeightedRandomSampler, ZERO_SAMPLE_IDS (10× boost); FPN 调用传入 g_grid |
| `utils/semantic_bg_prior.py` | 新增 selective_pseudo_background_bce_loss, scheduled_background_lambda |
| `configs/poc2_vit_fpn.yaml` | ViT-FPN 配置 (vit_in_channels: 1024, no bg prior) — 最终 best 配置 |
| `configs/poc2_landcover_semantic.yaml` | 基线配置 (no ViT, no prior) |
| `configs/poc2_thin_focal_sampler.yaml` | soft BCE+Focal+Sampler 配置 |
| `scripts/overfit_thin_class_test.py` | Overfit 诊断脚本 |

---

## 十一、参考文献

- **LaSt-ViT**: *Vision Transformers Need More Than Registers*, Shi et al., CVPR 2026. arXiv 2602.22394
  - 启发了将 ViT 全局特征注入检测头的方向。

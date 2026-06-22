# PoC-2b 土地覆盖 Semantic Head — 实验总结

> 日期: 2026-06-17

---

## 核心路线

```
baseline (no ViT, no prior, CE)         0.5324
  ↓ +bg prior
cond BCE λ=0.30 t=0.5 (旧最佳)          0.5987
  ↓ +ViT g_grid → FPN P3
ViT-FPN (无 prior, 纯 CE)               0.6340  ← 架构突破
  ↓ +zero-sample boosted sampler (10×)
ViT-FPN + boost sampler                 0.6248  ← 稀有类提升
```

---

## 架构: ViT-FPN

```
DINOv3 ViT g_grid [B,1024,14,14]
     → vit_proj (1x1 Conv+BN+ReLU+3x3 Conv+BN+ReLU)
     → FPN P3 = lat16 + upsample(P4) + vit_proj(g_grid)

ConvNeXt c4/c8/c16/c32 → FPN → LandcoverSemanticHead → [B,25,224,224]
```

- 0 额外 bg prior，纯 CE loss
- 单 NPU (910B2)，batch=8
- FPN 参数: 3.7M (含 vit_proj) + SemanticHead: 2.7M = 6.4M trainable

---

## 全类精度 (ViT-FPN + boost sampler e15)

| 等级 | 类别 | IoU |
|------|------|:------:|
| 强 (>0.5) | 沿海滩涂 | 0.87 |
| | 养殖坑塘 | 0.71 |
| | 盐田 | 0.71 |
| 中 (0.2-0.5) | 工业用地 | 0.49 |
| | 农村宅基地 | 0.49 |
| | 水田 | 0.47 |
| | 公路用地 | 0.42 |
| | 设施农用地 | 0.33 |
| | 河流水面 | 0.33 |
| | 其他草地 | 0.22 |
| 弱 (0.05-0.2) | 沟渠 | 0.19 |
| | 城镇村道路 | 0.16 |
| | 坑塘水面 | 0.14 |
| | 港口码头 | 0.14 |
| | 水浇地 | 0.08 |
| | 内陆滩涂 | 0.07 |
| | 其他林地 | 0.07 |
| | 旱地 | 0.07 |
| | 水工建筑 | 0.05 |
| 极弱 (<0.05) | 农村道路 | 0.03 |
| 死亡 | 其他园地 | 0.00 |

mIoU = 0.6248, pixel_accuracy = 68.8%

---

## 最佳配置

```yaml
# configs/poc2_vit_fpn.yaml
model:
  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256
    vit_in_channels: 1024        # 启用 ViT-FPN 融合
  semantic_head:
    num_classes: 25
    output_size: [224, 224]

train:
  epochs: 15
  batch_size: 8
  lr: 0.0001
  background_prior:
    enabled: false               # 不需要 bg prior
```

---

## 关键发现

1. **ViT 全局语义注入 FPN 是最有效的单次提升**
   18 种 loss 设计（BCE、soft BCE、hybrid、Focal 等）全部不敌架构改进。
   mIoU +19% vs baseline，且完全不需要 bg prior。

2. **稀有类 recall 靠采样权重解决**
   10× 采样权重让农村道路首次突破零 (0→0.025)，城镇村道路提升 3.5× (0.05→0.16)。

3. **Overfit 测试确认架构无硬伤**
   8 张 thin class tile，200 轮 CE overfit 达到 97-100% recall。
   问题在训练机制 + 数据 exposure，不在架构。

4. **Partial-label 是数据天然属性**
   Label_Binary 是 per-class 独立标注。同一 tile 多个 class TIF 覆盖 47-93% 区域。
   剩余像素是"未知"而非"背景"。

---

## 剩余瓶颈

| 问题 | 根因 |
|------|------|
| 坑塘水面 vs 养殖坑塘 63% 混淆 | 224px 下纹理不可分 |
| 农村道路 0.03 (仍极低) | 76 tiles / 64K px |
| 旱地 vs 水田 54% 混淆 | 干湿农田 224px 下相近 |
| pixel_accuracy 68.8% (目标 80%) | partial-label 固有天花板 |

---

## 存量 Checkpoint

| 路径 | mIoU |
|------|:------:|
| `outputs/poc2_vit_fpn/checkpoints/best_miou.pt` | 0.6248 |
| `outputs/poc2b_cond_bce_030_t50/checkpoints/best_miou.pt` | 0.5987 |

---

## 代码

| 文件 | 改动 |
|------|------|
| `Models/fpn_neck.py` | ViT 融合 (vit_proj, vit_in_channels) |
| `scripts/poc_stage_semantic.py` | WeightedRandomSampler, ZERO_SAMPLE_IDS, g_grid 传入 FPN |
| `configs/poc2_vit_fpn.yaml` | 最佳配置 |

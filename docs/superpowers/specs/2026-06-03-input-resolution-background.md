# 输入分辨率提升 — 问题背景

> 日期: 2026-06-03
> 状态: 设计讨论中
> 关联: `总设计.md` v2.1, PoC-2a GF6 RPN Recall 修复

---

## 1. 现状

CoastGPT 全链路硬编码 224×224 输入分辨率：

| 层级 | 硬编码位置 |
|------|-----------|
| YAML 配置 | `rgb_vision.input_size: [224, 224]` (全部 6 个 config) |
| 训练 transform | `crop_pct = 224 / 256` → CenterCrop(224) |
| 推理 transform | 同上，始终输出 224×224 |
| ConvNeXt local pyramid | 用 `self.input_size` (固定 224) 计算多尺度，非实际 tensor shape |
| 检测头 | `min_size=224, max_size=224` |
| GT 栅格化 | `model_size = (224, 224)` |
| 土地覆盖数据集 | `image_size: int = 224`, shape 断言 `(B, 3, 224, 224)` |
| DINOv3 ViT | timm arch 后缀 `_224`，未传 `dynamic_img_size=True` |

唯一例外：`_patch_vit_pos_embed()` / `_interpolate_pos_embed()` 已实现 pos_embed 插值，多尺寸前向验证通过（224/256/288/320/384/448），但从未在正常 pipeline 中触发。

## 2. 动机

DINOv3 的 `dynamic_img_size` 天然支持任意尺寸前向。提高分辨率 → 更多 ViT token → 更细粒度空间特征，对以下任务有直接收益：

- **养殖区实例分割**: 小网箱边界在 224 下只有几个像素
- **土地覆盖语义分割**: 线性特征（公路、河流）在 224 下 1-2px 宽
- **海岸线边缘检测**: 亚像素精度对 coastline 偏移量敏感

## 3. 已知约束

### 3.1 NPU 显存
224 时 Stage 3 已是 batch=2/GPU, accum=8。ViT self-attention 是 O(n²)，提高分辨率会显著增加显存，可能需进一步缩小 batch size。

### 3.2 GF6 PoC-2a 仍在进行
当前 GF6 RPN recall 仅 0.695 (vs non-GF6 0.888)。根因分析已排除 anchor scale / aspect ratio / 输入分辨率——missed 目标与 covered 目标尺寸相同。更可能是特征冻结导致的纹理判别力不足（Stage 3.5 partial unfreeze ConvNeXt）。分辨率提升与 partial unfreeze 是两个独立维度。

### 3.3 多尺寸直接推理已崩
224→448 零训练直接推理，RPN recall 从 0.70 跌至 0.16。pos_embed 能插值，但检测头在 224 特征空间学到的权重不兼容高分辨率特征分布。**不能只改推理分辨率，必须用目标分辨率训练。**

### 3.4 GT 数据全链路 224
土地覆盖 26,816 tiles 的 binary TIF 全部 NEAREST resize 到 224。改分辨率需重新生成 GT。

## 4. 训练代价

| 方案 | 重训范围 | 代价 |
|------|---------|------|
| 乐观 | Stage 3 + Stage 4 | 中等 |
| 悲观 | Stage 1 → Stage 2 → Stage 3 → Stage 4 | 高 |

Stage 1/2 的 Projector + LoRA 理论上对 token 数量不敏感（per-token 线性映射），但特征统计分布偏移可能导致 CAP/VQA 退化。需在 Stage 3 训完后验证 CAP/VQA loss 再决定是否回补。

## 5. 候选分辨率

DINOv3 ViT-L/16 patch_size=14，输入需为 14 的倍数：

| 分辨率 | ViT token grid | token 增量 | 显存压力 |
|--------|---------------|-----------|---------|
| 224 (当前) | 14×14 = 196 | 1× | 基准 |
| 336 | 24×24 = 576 | 2.9× | 中等 |
| 392 | 28×28 = 784 | 4× | 较高 |
| 448 | 32×32 = 1024 | 5.2× | 高 |

## 6. 待决策

1. 目标分辨率
2. 单一固定分辨率 vs 真正多分辨率动态支持
3. 训练范围（乐观 vs 悲观路径）
4. 与 Stage 3.5 partial unfreeze 的执行顺序

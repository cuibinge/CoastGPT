# PoC-2 土地覆盖 Semantic Head 设计 Spec

> 日期: 2026-06-01
> 状态: 已定稿，待实现
> 基于: v2.1 检测头混合架构设计 + final_confirmation 文档
> 前置: PoC-1 养殖区 Instance Head（已完成）

---

## 1. 目标

独立验证在 frozen DualVisionEncoder + FPNNeck + LandcoverSemanticHead 架构下，
土地覆盖 partial-label semantic segmentation 闭环：

- 空间重叠 tile 合并 → partial target mask → ignore-mask loss
- 25 类 semantic head → mask → polygonize → GeoJSON 输出
- observed-pixel mIoU / per-class IoU 评估
- size-stratified eval + high-res ablation

---

## 2. 架构

```
Input [B, 3, 224, 224]
  ↓
Frozen DualVisionEncoder
  ↓
ConvNeXt: c4 [B,128,56,56], c8 [B,256,28,28], c16 [B,512,14,14], c32 [B,1024,7,7]
  ↓
FPNNeck (from scratch)
  ↓
P1 [B,256,56,56], P2 [B,256,28,28], P3 [B,256,14,14], P4 [B,256,7,7]
  ↓
LandcoverSemanticHead (from scratch)
  P2/P3/P4 upsample to 56×56 → concat [B,1024,56,56]
  → 3×3 Conv 256 + BN + ReLU
  → 3×3 Conv 128 + BN + ReLU
  → 1×1 Conv 25
  → bilinear upsample to [B, 25, 224, 224]
  ↓
logits [B, 25, 224, 224]
```

模块状态:

| 模块 | 状态 | 说明 |
|------|------|------|
| DualVisionEncoder | frozen | PoC-1 checkpoint 加载 |
| FPNNeck | train from scratch | 从 det_head.py 拆分出，共用 |
| LandcoverSemanticHead | train from scratch | 新建 |

---

## 3. 类别空间

num_classes = 25 (0=background, 1-24=24 active DLMC, 255=ignore_index)

24 active DLMC classes（按数据实际存在）:

其他园地、其他林地、其他草地、农村宅基地、农村道路、内陆滩涂、
公园与绿地、公路用地、养殖坑塘、坑塘水面、工业用地、旱地、
水浇地、水工建筑用地、水田、沟渠、河流水面、沿海滩涂、
港口码头用地、盐田、设施农用地、裸岩石砾地、铁路用地、城镇村道路用地

label_map 两层映射: DLMC → canonical_id → semantic_train_id (1-24)

---

## 4. 数据管线

### 4.1 数据源
- 路径: `/home/ma-user/work/Stage3Data/土地分类/Patches/`
- 24 类目录，每目录含 Size_128/Size_256/Size_512
- 总样本: ~48,930 (128: 39,274 / 256: 9,142 / 512: 514)
- GF1 传感器，RGB jpg + GeoJSON label

### 4.2 空间重叠 tile 合并
- group key: `(source_image_id, quantized_tile_bounds, quantized_transform, model_input_size)`
- 同 key 的多个类目录 GeoJSON 合并为多类 label

### 4.3 Target mask 构造
```python
target = np.full((224, 224), IGNORE_INDEX=255, dtype=np.uint8)
# polygon 内写入 class_id, polygon 外保持 ignore
for feature in merged_features:
    class_id = dlmc_to_train_id[feature["DLMC"]]
    polygon_mask = rasterize_polygon(feature, georef, size=(224,224))
    target[polygon_mask == 1] = class_id
```

关键规则:
- polygon 外 NOT background — 保持 ignore
- GeoJSON 直接 rasterize 到 224×224 model pixel space（不先 rasterize 再 resize）
- image resize: bilinear; label mask resize（如存在）: NEAREST only

### 4.4 验证集划分
- 按 source_image_id 划分 train/val (8:2)
- 避免空间泄漏

---

## 5. Loss

```text
L_sem = CE(ignore_index=255) + Dice(observed_classes_only)
```

- CE: `F.cross_entropy(logits, target.long(), ignore_index=255)`
- Dice: 只对当前 tile 中实际出现的 foreground classes 计算
- background 不参与 Dice（无可靠 bg 标注）
- 全 ignore tile → loss=0

---

## 6. 文件结构

```
新建:
  Models/fpn_neck.py              ← FPNNeck 从 Models/det_head.py 拆分
  Models/semantic_head.py         ← LandcoverSemanticHead
  Dataset/landcover_label_map.py  ← DLMC → train_id 映射
  Dataset/landcover_tile_grouping.py ← 空间重叠合并
  Dataset/rasterize_geojson.py    ← GeoJSON → pixel mask rasterize
  Dataset/landcover_dataset.py    ← PyTorch Dataset
  scripts/poc_stage_semantic.py   ← 训练脚本
  configs/poc2_landcover_semantic.yaml ← 配置

修改:
  Models/det_head.py              ← FPNNeck 兼容导入 (from Models.fpn_neck import FPNNeck)
```

---

## 7. 训练配置

| 参数 | 值 |
|------|-----|
| epochs | 50-100（按 observed-pixel mIoU early stop） |
| batch_size | 4 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| optimizer | AdamW |
| device | NPU (单卡) |
| image_size | 224 |

---

## 8. 评估

### 8.1 主指标
- observed-pixel mIoU（仅 target != 255 像素）
- per-class IoU（24 foreground classes）
- per-class recall（重点 rare classes）
- foreground vs background confusion rate

### 8.2 size-stratified eval
per-size (128/256/512) 拆分评估，输出以上指标

### 8.3 high-res ablation
512 tile 对比 224×224 vs 原始分辨率 rasterize 的 per-class IoU，
量化细窄地物（沟渠、农村道路、城镇村道路用地）的精度损失

### 8.4 可视化
- image / GT / pred / GT+pred overlay
- polygonize → GeoJSON → geometry validity check

---

## 9. Checkpoint 格式

```python
{
    "fpn": fpn.state_dict(),
    "sem_head": sem_head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "num_classes": 25,
    "ignore_index": 255,
    "label_map": label_map,
    "config": config,
}
```

---

## 10. 实现顺序

| Step | 内容 |
|------|------|
| 1 | 拆分 FPNNeck 到 Models/fpn_neck.py，det_head.py 兼容导入 |
| 2 | 实现 Dataset/landcover_label_map.py |
| 3 | 实现 Dataset/landcover_tile_grouping.py |
| 4 | 实现 Dataset/rasterize_geojson.py |
| 5 | 实现 Dataset/landcover_dataset.py |
| 6 | 实现 Models/semantic_head.py |
| 7 | 实现 configs/poc2_landcover_semantic.yaml |
| 8 | 实现 scripts/poc_stage_semantic.py |
| 9 | Sanity check: label map / target mask / rasterize / loss / output |
| 10 | 5-10 tile overfit test |
| 11 | 小规模 train/val |
| 12 | Overlay + GeoJSON 输出 |
| 13 | 完整 PoC-2 baseline 训练 |
| 14 | Size-stratified eval + high-res ablation |

---

## 11. Sanity Check 清单

- [ ] 24 DLMC 全部有 train_id，连续 1-24
- [ ] target shape = [224,224], dtype=uint8, values ∈ {1-24, 255, optionally 0}
- [ ] polygon 外保持 255，无 bilinear 非法值
- [ ] GeoJSON rasterize 与原图视觉对齐
- [ ] CE 使用 ignore_index=255
- [ ] Dice 只计算 observed classes
- [ ] 全 ignore tile 正确处理
- [ ] logits shape = [B,25,224,224]

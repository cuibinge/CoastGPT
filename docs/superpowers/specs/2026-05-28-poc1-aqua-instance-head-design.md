# PoC-1: 养殖区 Instance Head 闭环设计文档

> 日期: 2026-05-28
> 状态: 设计定稿，待进入实现
> 基于: `2026-05-25-detection-head-hybrid-design.md` v2.1 + brainstorming 讨论 + `poc1_integration_recommendation.md` + `poc1_module_boundary_design.md` + `poc1_data_flow_decisions.md`

---

## 1. 目标与范围

在冻结的 DualVisionEncoder 特征之上，实现 FPN Neck + Aquaculture Instance Head (torchvision Mask R-CNN)，完成：

```
GeoJSON answer (WGS84)
  → WGS84 → pixel GT (bbox + mask)
  → FPN + Mask R-CNN 训练
  → pixel polygon 推理
  → pixel → WGS84 → GeoJSON Builder
  → Geometry Validation
  → Overlay 可视化
```

### 不做

- 不接 Land-cover Semantic Head
- 不接 Coastline Edge Head
- 不接 LLM fallback
- 不做 Stage 4 联合训练
- 不接 DeepSpeed / 分布式训练
- 不接 EpochBasedTrainer
- 不修改 CoastGPT 主训练路径

---

## 2. 文件结构与模块边界

```
CoastGPT/
  Models/
    det_head.py                # FPNNeck + DualVisionFPNBackboneAdapter + build_aqua_maskrcnn()
  Dataset/
    aqua_poc_dataset.py        # AquaPoCDataset + GeoJSON → pixel GT 转换
  utils/
    georef_transform.py        # resize_georef / wgs84_to_pixel / pixel_to_wgs84 / round_trip_check
    mask_utils.py              # rasterize_polygon / polygon_to_bbox / mask_to_polygon / clip
    geojson_builder.py         # polygon_pixel→GeoJSON / validate_geojson / repair
    vis_overlay.py             # GT/Pred pixel overlay 可视化
  scripts/
    build_poc_aqua_data.py     # 从 GF tiles 构建 PoC manifest
    poc_stage_one_det.py       # 主训练脚本 (单 NPU)
  configs/
    poc_aqua_instance.yaml     # PoC 专用配置
```

### 依赖原则

```
model 不依赖 dataset / geojson_builder / vis
dataset 不依赖 model / training script
geojson_builder 不依赖 model / training script
vis_overlay 不依赖 model
poc_stage_one_det.py 组合所有模块
```

### 模块关系图

```
poc_stage_one_det.py
  ├── Models/det_head.py
  │     ├── FPNNeck
  │     ├── DualVisionFPNBackboneAdapter  (wraps frozen DualVisionEncoder)
  │     └── build_aqua_maskrcnn()
  ├── Dataset/aqua_poc_dataset.py
  │     ├── utils/georef_transform.py
  │     └── utils/mask_utils.py
  ├── utils/geojson_builder.py
  │     ├── utils/georef_transform.py
  │     └── utils/mask_utils.py
  └── utils/vis_overlay.py
```

---

## 3. FPN + Mask R-CNN 架构

### 3.1 FPN Neck

```
输入: ConvNeXt pyramid_raw = (c4, c8, c16, c32)
  c4:  [B, 128, 56, 56]
  c8:  [B, 256, 28, 28]
  c16: [B, 512, 14, 14]
  c32: [B, 1024, 7, 7]

处理:
  1×1 Conv 统一通道到 256
  top-down 上采样 + 横向连接

输出:
  P1: [B, 256, 56, 56]
  P2: [B, 256, 28, 28]
  P3: [B, 256, 14, 14]
  P4: [B, 256, 7, 7]
```

### 3.2 DualVisionFPNBackboneAdapter

```python
from collections import OrderedDict
import torch
import torch.nn as nn


class DualVisionFPNBackboneAdapter(nn.Module):
    """
    Wrap frozen DualVisionEncoder + FPNNeck as torchvision Mask R-CNN backbone.

    torchvision GeneralizedRCNNTransform 内部将 List[Tensor] batch 为
    ImageList，传给 backbone 的是 batched tensor。

    Input:  Tensor[B, 3, 224, 224]
    Output: OrderedDict[str, Tensor]
            {"0": P2, "1": P3, "2": P4}
    """

    def __init__(self, vision_encoder, fpn_neck):
        super().__init__()
        self.vision = vision_encoder
        self.fpn = fpn_neck
        self.out_channels = 256

        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> OrderedDict:
        """
        Args:
            images: Tensor[B, 3, 224, 224]  (不是 List[Tensor])
        """
        with torch.no_grad():
            image_seq, g_grid, pyramid_raw = self.vision.encode_with_spatial(images)

        c4, c8, c16, c32 = pyramid_raw
        p1, p2, p3, p4 = self.fpn(c4, c8, c16, c32)

        return OrderedDict({
            "0": p2,  # [B, 256, 28, 28]
            "1": p3,  # [B, 256, 14, 14]
            "2": p4,  # [B, 256, 7, 7]
        })
```

**禁止在 forward() 中逐张 image 循环，否则 batch 维度被破坏，RPN/RoIHeads 特征与 targets 对不上。**

### 3.3 Mask R-CNN 配置

```python
num_classes = 2  # 0=background, 1=海水养殖区

# Anchor: 3 levels, 3 aspect ratios
anchor_sizes = ((16, 32), (32, 64), (64, 128, 192))
aspect_ratios = ((0.5, 1.0, 2.0, 3.0),) * 3

# RoIAlign: 对应 P2/P3/P4
featmap_names = ["0", "1", "2"]
box_roi_pool = MultiScaleRoIAlign(featmap_names, output_size=7, sampling_ratio=2)
mask_roi_pool = MultiScaleRoIAlign(featmap_names, output_size=14, sampling_ratio=2)
```

### 3.4 训练/冻结状态

| 模块 | 状态 |
|------|------|
| DINOv3 ViT-L16 | 冻结 |
| ConvNeXt Base | 冻结 |
| FPN Neck | 训练 |
| Mask R-CNN (RPN+ROI+Mask heads) | 训练 |

---

## 4. 数据管线

### 4.1 数据来源

| Sensor | Size | 可用 tile 数 | Phase |
|--------|------|-------------|-------|
| GF2 | 256×256 | 39 | Phase A (smoke test) |
| GF1 | 128×128 | 720 | Phase B (凑足 50-100) |

### 4.2 离线阶段: build_poc_aqua_data.py

1. 扫描 `GF2/Size_256/Image_TrueColor/*.jpg`，匹配 `Label_GeoJSON/*.geojson`
2. 校验 GeoJSON parse
3. 获取 tile-level georef（tile bounds WGS84）
4. 统计 features 数量，标记 `has_object`
5. 按 `source_image_id` + `sensor` 分层划分 train/val (8:2)
6. 写出 manifest JSON

### 4.3 tile_transform 来源 (关键)

**禁止从 object polygon 的 min/max 推导 tile_transform。** 必须使用整张 tile 的空间范围。

GF2/256 tile 的 `original_transform` 推导：

```python
# tile_bounds_wgs84 = [min_lon, min_lat, max_lon, max_lat]
x_res = (max_lon - min_lon) / 256
y_res = (max_lat - min_lat) / 256

original_transform = [
    x_res, 0.0, min_lon,
    0.0, -y_res, max_lat,
]
source_crs = "EPSG:4326"
```

**限制条件**: `EPSG:4326` 仅当 tile_bounds_wgs84 与 GeoJSON 坐标均为 WGS84 lon/lat 时适用。如果后续从 GeoTIFF / UTM / 投影坐标获取 tile transform，source_crs 必须改为真实 CRS，不能继续写死。错误 CRS 会导致 WGS84↔pixel 变换完全失效。

### 4.4 Manifest 字段

```json
{
  "sample_id": "GF2_xxx_tile_001",
  "sensor": "GF2",
  "source_image_id": "GF2_xxx",
  "image_path": "GF2/Size_256/Image_TrueColor/xxx.jpg",
  "label_path": "GF2/Size_256/Label_GeoJSON/xxx.geojson",
  "task": "DET",
  "branch": "instance",
  "known_classes": ["海水养殖区"],
  "source_crs": "EPSG:4326",
  "original_size": [256, 256],
  "model_input_size": [224, 224],
  "original_transform": [0.00001, 0.0, 120.0, 0.0, -0.00001, 30.0],
  "tile_bounds_wgs84": [120.0, 29.99744, 120.00256, 30.0],
  "has_object": true,
  "num_features": 3,
  "split_group": "GF2_xxx",
  "label_source": "geojson_answer"
}
```

`model_transform` 由 Dataset 在线计算（根据 resize）。

### 4.5 在线阶段: AquaPoCDataset

```text
读取 manifest item
  → 加载 image (256×256 或 128×128)
  → resize → 224×224 (bilinear)
  → resize_georef: model_transform = original_transform * Affine.scale(sx, sy)
  → 读取 GeoJSON FeatureCollection
  → parse Polygon / MultiPolygon
  → wgs84_to_pixel() → model pixel space
  → clip to [0, 224) bounds
  → rasterize mask (224×224)
  → bbox from mask (避免 polygon clip 后 outer ring 与 mask 不一致)
  → labels = [1] * N
  → 返回 {image, target, meta}
```

bbox 必须从 mask 生成（非从 polygon outer ring 计算）：

```python
def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return [x1, y1, x2, y2]
```

mask.sum() == 0 的 instance 剔除；features 非空但全部 rasterize 后为空 = bad sample，不直接当 negative。

### 4.6 Dataset 输出 Contract

```python
sample = {
    "image": Tensor[3, 224, 224],
    "target": {
        "boxes": FloatTensor[N, 4],       # xyxy, model pixel
        "labels": LongTensor[N],           # all 1
        "masks": UInt8Tensor[N, 224, 224],
        "image_id": LongTensor[1],
        "area": FloatTensor[N],
        "iscrowd": LongTensor[N],          # all 0
    },
    "meta": {
        "image_path": str,
        "source_crs": "EPSG:4326",
        "original_transform": list[float],
        "model_transform": list[float],
        "original_size": [int, int],
        "model_input_size": [224, 224],
        "resize_scale": [float, float],
        "sample_id": str,
    }
}
```

collate_fn 输出: `(images: List[Tensor], targets: List[dict], metas: List[dict])`

### 4.7 MultiPolygon 处理

默认拆成多个 instance（各自有独立的 mask/bbox/label=1）。除非标注规范明确说明同一 MultiPolygon 表示同一实例。

### 4.8 空标注 tile

保留为 negative sample，target 使用零长度 tensor：

```python
target = {
    "boxes": torch.zeros((0, 4), dtype=torch.float32),
    "labels": torch.zeros((0,), dtype=torch.int64),
    "masks": torch.zeros((0, 224, 224), dtype=torch.uint8),
    "image_id": torch.tensor([image_id], dtype=torch.int64),
    "area": torch.zeros((0,), dtype=torch.float32),
    "iscrowd": torch.zeros((0,), dtype=torch.int64),
}
```

控制比例 `positive : negative >= 2 : 1`。

以下样本不能当 negative，写入 `bad_samples.json` 剔除：
- GeoJSON parse 失败
- georef 缺失或不可信
- polygon 全部越界
- image/label 无法匹配

---

## 5. 坐标转换

### 5.1 工具函数 (georef_transform.py)

```python
resize_georef(original_size, target_size, original_transform)
    → (model_transform, resize_scale)

wgs84_to_pixel(coords_wgs84, georef)
    → coords_model_pixel  # 224×224 space

pixel_to_wgs84(coords_pixel, georef)
    → coords_wgs84  # EPSG:4326

round_trip_check(coords_wgs84, georef)
    → error_degrees  # 目标 < 1e-6
```

### 5.2 Resize Affine 公式

```python
sx = original_width / 224
sy = original_height / 224
model_affine = Affine(*original_transform) * Affine.scale(sx, sy)
```

### 5.3 像素坐标约定

- Polygon 顶点: pixel corner 坐标 (0-indexed)
- Mask rasterization: 整像素 grid
- 所有检测头输出: model input pixel space 224×224

---

## 6. GeoJSON Builder 与 Validation

### 6.1 推理后处理管线

```text
Mask R-CNN outputs (boxes, scores, masks)
  → score threshold (default 0.5)
  → mask threshold (0.5) → binary mask
  → mask_to_polygon() → pixel polygon
  → pixel_to_wgs84() → WGS84 polygon
  → polygon_pixel_to_geojson_feature()
  → build_feature_collection()
  → validate_geojson()
```

### 6.2 Validation 步骤

1. JSON parse 成功
2. GeoJSON schema: type=FeatureCollection, features 数组
3. Per-feature: type=Feature, geometry, properties.class/confidence
4. Geometry validity (shapely):
   - Polygon: closed, non-self-intersecting, correct ring direction
   - 坐标在 tile_bounds_wgs84 内
5. Empty output: features=[] 正确处理

### 6.3 Repair 策略

```
Polygon not closed → auto-close
Ring direction wrong → orient(ccw=True)
Self-intersection → buffer(0)
Tiny geometry → filter (min_area)
```

---

## 7. 可视化

### 7.1 优先 pixel space overlay

直接在 model pixel space 224×224 上 overlay，避免 WGS84 投影来回转换掩盖 GT 生成阶段问题。

```text
outputs/poc_aqua_instance/vis/
  xxx_gt.png         # 原图 + GT mask/bbox overlay
  xxx_pred.png       # 原图 + Pred mask/bbox overlay
  xxx_gt_pred.png    # 原图 + GT(绿) + Pred(红) 对齐检查
```

---

## 8. 训练配置

### 8.1 模型加载

直接构造 `DualVisionEncoder(config)`，从 checkpoint 中提取 `vision_ckpt` 部分加载权重。不实例化完整 CoastGPT（避免加载 LLaMA-2-7B）。

### 8.2 NPU Compatibility Smoke Test

训练开始前，先验证 torchvision ops 在 NPU 上的可用性。

**第一阶段：算子级测试**

```python
# 必须通过
torchvision.ops.nms(boxes, scores, iou_threshold)
torchvision.ops.batched_nms(boxes, scores, idxs, iou_threshold)
torchvision.ops.MultiScaleRoIAlign(...)(features, proposals, image_shapes)
torchvision.ops.roi_align(...)
```

**第二阶段：完整 Mask R-CNN forward/backward 测试**

```python
def smoke_test_maskrcnn_forward_backward(model, device):
    """
    验证完整 Mask R-CNN 训练链路在 NPU 上可用。
    检查所有 loss finite + 梯度状态正确。
    """
    model.train()

    images = [torch.rand(3, 224, 224, device=device)]
    mask = torch.zeros((1, 224, 224), dtype=torch.uint8, device=device)
    mask[0, 50:180, 40:160] = 1

    targets = [{
        "boxes": torch.tensor([[40, 50, 160, 180]], dtype=torch.float32, device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
        "masks": mask,
        "image_id": torch.tensor([0], dtype=torch.int64, device=device),
        "area": torch.tensor([120 * 130], dtype=torch.float32, device=device),
        "iscrowd": torch.tensor([0], dtype=torch.int64, device=device),
    }]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    # Check all losses
    required_losses = [
        "loss_objectness", "loss_rpn_box_reg",
        "loss_classifier", "loss_box_reg", "loss_mask"
    ]
    for name in required_losses:
        assert name in loss_dict, f"Missing loss: {name}"
        assert torch.isfinite(loss_dict[name]), f"Non-finite loss: {name}"

    loss.backward()

    # Check gradients
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"Frozen param {name} should have no grad"
        else:
            assert p.grad is not None, f"Trainable param {name} should have grad"

    return loss_dict
```

如果算子级测试失败，降级到极简 dense detector（方案 3）。如果只有第二阶段失败，排查模型组装问题。如果全部通过，可进入训练。

### 8.3 训练参数

```yaml
train:
  device: npu
  epochs: 20
  batch_size: 2
  lr: 0.0001
  weight_decay: 0.0001
  num_workers: 2
  log_interval: 10
  val_interval: 1
  save_interval: 1

model:
  fpn:
    in_channels: [128, 256, 512, 1024]
    out_channels: 256
  mask_rcnn:
    num_classes: 2
    min_size: 224
    max_size: 224
    rpn_nms_thresh: 0.7
    box_score_thresh: 0.05
    box_nms_thresh: 0.5
    box_detections_per_img: 50
```

---

## 9. 执行顺序

### Step 1: 检查 georef 来源
确认每张 GF tile 有可靠 tile-level bounds（非 object polygon 推导）。禁止从 object polygon min/max 推导 tile_transform。

### Step 2: 构建 GF2/256 manifest
```bash
python scripts/build_poc_aqua_data.py --sensor GF2 --split 0.8
```
生成 train.json / val.json / bad_samples.json

### Step 3: Dataset dry run
随机抽样检查: image resize, model_transform, GeoJSON parse, WGS84→pixel, bbox, mask, GT overlay, round-trip error。导出 `vis/dry_run/xxx_gt.png`。

### Step 4: NPU ops smoke test
单独测试 `nms` / `batched_nms` / `roi_align` / `MultiScaleRoIAlign` 在 NPU 上可用。

### Step 5: 完整 Mask R-CNN forward/backward smoke test
用合成 tensor 测试完整训练链路: 全部 loss finite，冻结参数无梯度，可训练参数有梯度。

### Step 6: GF2 smoke training
只用 GF2/256 (39 tiles)，单 NPU，确认 loss finite 且下降。

### Step 7: 加入 GF1/128
```bash
python scripts/build_poc_aqua_data.py --sensor GF2,GF1 --max-samples 100 --split 0.8
```
补足 50-100 tiles，按 source_image_id + sensor 分层划分。

### Step 8: PoC-1 正式训练
训练 FPN + Mask R-CNN，导出 GeoJSON + overlay，记录 metrics。

---

## 10. 通过标准

```
[ ] 50-100 张 tile 可正常读取
[ ] GeoJSON answer → pixel GT 成功率 ~100%
[ ] round-trip 坐标误差 < 1e-6 degree
[ ] GT mask/bbox overlay 与原图对齐
[ ] 单 NPU loss 正常下降
[ ] NPU ops smoke test 通过
[ ] 完整 Mask R-CNN forward/backward smoke test 通过
[ ] 推理输出 polygon_pixel
[ ] pixel → WGS84 → 合法 GeoJSON FeatureCollection
[ ] GeoJSON parse success = 100%
[ ] geometry valid rate >= 90%
[ ] 导出 GT/Pred overlay 图人工检查通过
```

---

## 11. 后续迁移路径

PoC-1 通过后：

```
PoC-1 模块 → 正式集成:
  Models/det_head.py        → train_stage_three_det.py (复用)
  utils/georef_transform.py  → 直接迁移
  utils/mask_utils.py        → 直接迁移
  utils/geojson_builder.py   → 直接迁移
  Dataset aqua_poc_dataset   → GT converter 逻辑迁移
  poc_stage_one_det.py       → 参考实现，不直接迁移

PoC-2: Land-cover Semantic Head
PoC-3: Coastline Edge Head
PoC-4: LLM fallback + 融合
PoC-5: Stage 4 联合训练
```

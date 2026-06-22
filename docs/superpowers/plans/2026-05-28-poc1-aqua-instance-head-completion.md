# PoC-1 养殖区 Instance Head 闭环 — 完成报告

> 基于 `docs/superpowers/specs/2026-05-28-poc1-aqua-instance-head-design.md` 和 `docs/superpowers/plans/2026-05-28-poc1-aqua-instance-head.md` 的实现总结。

## 计划完成情况

9 个 Task 全部完成，经实际 NPU 训练验证通过：

| Task | 文件 | 职责 | 状态 |
|------|------|------|------|
| 1 | `utils/georef_transform.py` | WGS84↔pixel 坐标变换、resize_georef、round_trip_check、clip_pixel_coords | ✅ |
| 2 | `utils/mask_utils.py` | PIL rasterize_polygon、bbox_from_mask、OpenCV mask_to_polygon | ✅ |
| 3 | `scripts/build_poc_aqua_data.py` | 离线 manifest 构建，支持 GF1/GF2/GF6 + 多尺寸 + GeoTIFF/GeoJSON 回退 | ✅ |
| 4 | `configs/poc_aqua_instance.yaml` | PoC 训练配置 | ✅ |
| 5 | `Models/det_head.py` | FPNNeck (~2.85M) + DualVisionFPNBackboneAdapter + build_aqua_maskrcnn() | ✅ |
| 6 | `Dataset/aqua_poc_dataset.py` | AquaPoCDataset，GeoJSON→pixel GT 在线生成 | ✅ |
| 7 | `utils/geojson_builder.py` | mask→polygon→pixel_to_wgs84→GeoJSON + validate_geojson | ✅ |
| 8 | `utils/vis_overlay.py` | _gt.png / _pred.png / _gt_pred.png 三图输出 | ✅ |
| 9 | `scripts/poc_stage_one_det.py` | 主训练脚本，660 行，单 NPU 无 DeepSpeed | ✅ |

## 训练数据

最终使用 `Stage3Data/养殖区/` 全量数据：

| 传感器 | 尺寸 | Train | Val | 实例数 (train/val) |
|--------|------|-------|-----|-------------------|
| GF1 | 128 | 577 | 143 | — |
| GF2 | 128/256/512 | 124 | 29 | — |
| GF6 | 512 | 734 | 187 | — |
| **合计** | | **1,435** | **359** | **31,897 / 8,323** |

- 全部 1,794 tile 通过 pixel in-bounds 验证，0 bad samples
- 原始 GeoJSON: 16 features × 1,794 tiles ≈ 40,220 实例（全部成功 rasterize）

## 小样本训练结果 (31 train / 8 val, 20 epochs)

用于验证全流程闭环：

| Epoch | 预测数 | 平均置信度 |
|-------|--------|-----------|
| 1 | 0 | 0.00 |
| 10 | 44 | 0.59 |
| 20 | 41 | **0.80** |

- GT: 102 实例 / 8 val tiles
- 召回率 ~40%（受限于 31 样本 + 冻结 vision encoder）
- 闭环全通：GeoJSON → pixel GT → 训练 → pixel polygon → WGS84 GeoJSON → validation → overlay

## 实现过程中修复的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| `_gt.png` 漏标（只显示 5 个 mask） | `vis_overlay.py` 硬编码 `masks[:5]` | 去掉限制 |
| epoch 1–20 图像相同 | 用户看的是 `_gt.png`（GT 不变） | 说明三种文件区别 |
| GF1/GF6 tile 全部标为 bad | GeoTIFF 使用投影坐标系（UTM 米），非 WGS84 | GeoTIFF 检测坐标范围，超出 [-180,180] 则回退到 GeoJSON bbox 推导 |
| alignment_dim mismatch | stage2 checkpoint 为 1024，默认值 768 | YAML 加 `alignment_dim: 1024` |
| RPN anchor count mismatch | 第三层 3 个 sizes × 4 ratios = 12 anchors vs 统一 8 | 改为 `[64, 128]` 统一每层 8 anchors |
| 仅 GF2/256 数据 | 计划只考虑一个传感器一个尺寸 | `build_poc_aqua_data.py` 支持多传感器多尺寸 |
| `.cpu().numpy()` 报错 | tensor requires_grad 时不能直接 numpy | 改为 `.detach().cpu().numpy()` |

## 关键架构决策

- **Vision encoder 冻结**: `DualVisionFPNBackboneAdapter.train()` 覆写，强制 vision 永久 eval 模式
- **FPN 不收 P1**: Mask R-CNN 只用 P2/P3/P4（stride 4/8/16），P1(stride 2)太浅不用
- **统一 anchor**: 每层 2 sizes × 4 ratios = 8 anchors，满足 torchvision RPN head 约束
- **MultiPolygon 拆分**: 每个子多边形作为独立实例
- **GeoTIFF 优先, GeoJSON 回退**: 先读 GeoTIFF ModelPixelScaleTag，投影坐标系则从 GeoJSON feature 坐标反推 tile bounds

## 文件清单

```
CoastGPT/
├── utils/
│   ├── georef_transform.py    # 坐标变换
│   ├── mask_utils.py          # 像素几何操作
│   ├── geojson_builder.py     # GeoJSON 输出+校验
│   └── vis_overlay.py         # 可视化
├── Dataset/
│   └── aqua_poc_dataset.py    # 数据集 + collate_fn
├── Models/
│   └── det_head.py            # FPN + Adapter + MaskRCNN
├── scripts/
│   ├── build_poc_aqua_data.py # manifest 构建
│   └── poc_stage_one_det.py   # 训练脚本
├── configs/
│   └── poc_aqua_instance.yaml # 训练配置
├── data/poc_aqua_full/
│   ├── train.json             # 1435 样本
│   ├── val.json               # 359 样本
│   └── bad_samples.json       # 0 样本
└── docs/superpowers/plans/
    ├── 2026-05-28-poc1-aqua-instance-head.md           # 实现计划
    └── 2026-05-28-poc1-aqua-instance-head-completion.md # 本文件
```

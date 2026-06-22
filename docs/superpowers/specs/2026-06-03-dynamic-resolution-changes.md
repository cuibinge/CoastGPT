# CoastGPT 动态分辨率改造 — 改动文档

> 日期: 2026-06-03
> 基础设计: `docs/superpowers/temp/Final/coastgpt_dynamic_resolution_design.md`
> 实施计划: `docs/superpowers/plans/2026-06-03-dynamic-resolution.md`
> 背景文档: `docs/superpowers/specs/2026-06-03-input-resolution-background.md`

---

## 改动总览

| 类别 | 修改 | 新增 |
|------|------|------|
| YAML 配置 | 7 | 0 |
| 核心模型 | 3 | 0 |
| 数据/Transform | 1 | 0 |
| 工具/Utils | 3 | 0 |
| 训练脚本 | 2 | 0 |
| 数据集 | 1 | 0 |
| 新增模块 | 0 | 2 |
| 脚本 | 1 | 1 |
| **合计** | **18** | **3** |

---

## 一、YAML 配置 (7 文件)

### 1.1 Key 重命名

所有 7 个 YAML 配置文件中的 `input_size` 全部改名为 `default_input_size`，值保持 `[224, 224]` 不变。

**影响文件：**

| 文件 | rgb_vision.default_input_size | transform.default_input_size |
|------|------|------|
| `Configs/train_dual.yaml` | line 45 | line 99 |
| `Configs/step2_dual.yaml` | line 46 | line 100 |
| `Configs/step3_dual.yaml` | line 55 | line 108 |
| `Configs/step3_dual_fallback.yaml` | line 52 | line 105 |
| `Configs/step3_dual_v3_success.yaml` | line 55 | line 108 |
| `Configs/seg_train.yaml` | line 50 | line 104 |
| `Configs/inference.yaml` | line 28 | line 80 |

**未改:** `sar_vision.input_size: [192, 192]` 保持不变。

### 1.2 新增配置块 (仅 step3_dual.yaml)

`Configs/step3_dual.yaml` 末尾新增三个配置块：

```yaml
multi_scale:
  enabled: false          # Phase 2 启动前改为 true
  sizes:
    - [224, 224]
    - [280, 280]
    - [336, 336]
    - [392, 392]
  batch_same_size: true
  sampling: weighted
  weights: [0.35, 0.20, 0.30, 0.15]

anchor_generator:
  mode: relative
  base_on: min_side
  scales:
    - [0.0714, 0.1429]          # P2 level: 16/224, 32/224
    - [0.1429, 0.2857]          # P3 level: 32/224, 64/224
    - [0.2857, 0.4286, 0.5714]  # P4 level: 64/224, 96/224, 128/224
  aspect_ratios: [0.5, 1.0, 2.0, 3.0]

inference_transform:
  resize_policy: square_resize
  target_size: null
  ensure_multiple_of: 14
```

---

## 二、核心模型 (3 文件)

### 2.1 `Models/dual_vision_encoder.py` — 3 处改动

**改动 1 (line 174-177):** config key 支持新旧兼容

```python
# 旧
self.input_size = tuple(rgb_cfg.get("input_size", [224, 224]))

# 新
self.input_size = tuple(
    rgb_cfg.get("default_input_size")
    or rgb_cfg.get("input_size", [224, 224])
)
```

**改动 2 (line 842):** `_get_local_pyramid()` — tensor shape 替代 self.input_size

```python
# 旧
h, w = self.input_size

# 新
_, _, h, w = pixel_values.shape
```

**改动 3 (line 872):** `_get_local_pyramid_raw()` — 同上

**未改:** `_map_global_timm_arch()` 的 `_224` suffix 保持不变（pos_embed 插值处理运行时分辨率）。

### 2.2 `Models/semantic_head.py` — 2 处改动

**改动 1 (line 36):** output_size 改为 Optional

```python
# 旧
output_size: Tuple[int, int] = (224, 224),

# 新
output_size: Optional[Tuple[int, int]] = None,
```

**改动 2 (lines 102-107):** forward() 动态推断 upsample 尺寸

```python
# 旧
logits = F.interpolate(x, size=self.output_size, ...)

# 新
if self.output_size is not None:
    upsample_size = self.output_size
else:
    upsample_size = (p1.shape[2] * 4, p1.shape[3] * 4)  # P1 = H/4
logits = F.interpolate(x, size=upsample_size, ...)
```

### 2.3 `Models/det_head.py` — 3 处改动

**改动 1 (lines 95-96):** min_size/max_size 改为 Optional

```python
# 旧
min_size: int = 224,
max_size: int = 224,

# 新
min_size: Optional[int] = None,
max_size: Optional[int] = None,
```

新增 fallback: `None` → 224。

**改动 2 (lines 80-98):** 新增 `_build_anchor_sizes()` 辅助函数

```python
def _build_anchor_sizes(relative_scales, input_h, input_w):
    base = min(input_h, input_w)
    return tuple(
        tuple(max(1, round(s * base)) for s in scales)
        for scales in relative_scales
    )
```

使用 `round()` 而非 `int()` 避免截断偏置（如 `int(0.0714 * 336) = 23` vs `round = 24`）。

**改动 3 (line 105, 107, 137-141):** build_aqua_maskrcnn() 支持 anchor_relative_scales

```python
# 签名新增
anchor_relative_scales: Optional[Tuple[Tuple[float,...],...]] = None,
anchor_sizes: Optional[Tuple[Tuple[int,...],...]] = None,  # 改为 Optional

# 解析逻辑
if anchor_relative_scales is not None:
    anchor_sizes = _build_anchor_sizes(anchor_relative_scales, min_size, min_size)
elif anchor_sizes is None:
    anchor_sizes = ((16, 32), (32, 64), (64, 96, 128))  # legacy
```

---

## 三、Transform (1 文件)

### 3.1 `Dataset/build_transform.py`

**build_cls_transform 和 build_vlp_transform 均做相同改动：**

**训练路径 (is_train=True):**

```python
# 旧
input_size=config.transform.input_size,

# 新
input_size = getattr(config.transform, 'default_input_size', None) \
          or getattr(config.transform, 'input_size', [224, 224])
```

**评估路径 (is_train=False):**

```python
# 旧
crop_pct = 224 / 256
size = int(config.transform.input_size[0] / crop_pct)
# Resize(size) → CenterCrop(input_size)

# 新
input_size = getattr(config.transform, 'default_input_size', None) \
          or getattr(config.transform, 'input_size', [224, 224])
crop_pct = getattr(config.transform, 'crop_pct', None)
if crop_pct is not None:
    resize_size = int(input_size[0] / crop_pct)
else:
    resize_size = input_size[0]
# Resize(resize_size) → CenterCrop(input_size) 仅当 crop_pct is not None
```

---

## 四、工具/Utils (3 文件)

### 4.1 `Dataset/rasterize_geojson.py`

`compute_model_transform_from_bounds()` 和 `rasterize_features_to_target()`:

```python
# 旧
model_size: Tuple[int, int] = (224, 224),

# 新
model_size: Optional[Tuple[int, int]] = None,
# 函数体开头: if model_size is None: model_size = (224, 224)
```

### 4.2 `utils/mask_utils.py`

`rasterize_polygon()` 和 `rasterize_multipolygon()`:

```python
# 旧
width: int = 224,
height: int = 224,

# 新
width: Optional[int] = None,
height: Optional[int] = None,
# 函数体开头: if width is None: width = 224
```

### 4.3 `utils/georef_transform.py`

`clip_pixel_coords()`:

```python
# 旧
width: int = 224,
height: int = 224,

# 新
width: Optional[int] = None,
height: Optional[int] = None,
```

---

## 五、训练脚本配置读取 (2 文件)

### 5.1 `Trainer/Data/data.py`

```python
# 旧
input_size: Union[List, Tuple] = [224, 224]
crop_size: Union[List, Tuple] = [224, 224]

# 新
default_input_size: Union[List, Tuple] = [224, 224]
default_crop_size: Union[List, Tuple] = [224, 224]
```

### 5.2 `Tools/model_evaluate/seg_train.py` (line 357)

```python
# 旧
input_size = tuple(cfg.get("rgb_vision", {}).get("input_size", [224, 224]))

# 新
input_size = tuple(
    cfg.get("rgb_vision", {}).get("default_input_size")
    or cfg.get("rgb_vision", {}).get("input_size", [224, 224])
)
```

---

## 六、数据集断言 (1 文件)

### 6.1 `Dataset/landcover_dataset.py` (`__main__` 块, lines 281-285)

```python
# 旧
assert imgs.shape == (4, 3, 224, 224)
assert tgts.shape == (4, 224, 224)

# 新
assert imgs.shape[0] == 4 and imgs.shape[1] == 3
assert imgs.shape[2] == imgs.shape[3]  # 方形检查
assert tgts.shape[0] == 4
assert tgts.shape[1] == imgs.shape[2]
assert tgts.shape[2] == imgs.shape[3]
```

---

## 七、新增模块 (2 文件)

### 7.1 `Dataset/multiscale_sampler.py`

MultiScaleBatchSampler 类和 build_multiscale_sampler() 工厂函数。

**核心功能:**
- 每 step 随机采样一个分辨率（支持加权采样）
- DDP 下 rank 0 采样 → broadcast 到所有 rank
- `enabled=False` 时回退固定尺寸模式
- `build_multiscale_sampler(cfg)` 从 YAML `multi_scale` 块自动构建

### 7.2 `docs/superpowers/plans/multiscale_integration_notes.md`

train_stage_three.py 集成说明，包含 per-batch 和 per-epoch 两种切换策略的代码片段。

---

## 八、脚本 (2 文件)

### 8.1 `scripts/run_stage3_multiscale.sh` (新增)

Stage 3 multi-scale fine-tune 启动脚本，支持环境变量覆盖关键参数。

### 8.2 `scripts/precache_landcover_targets.py` (修改)

- 移除硬编码 `IMAGE_SIZE = 224`
- 新增 `--image-size` CLI 参数 (default=224)

---

## 九、向后兼容性

| 使用方式 | 行为 |
|----------|------|
| 新 YAML key `default_input_size: [224, 224]` | 正常，优先读取 |
| 旧 YAML key `input_size: [224, 224]` | 通过 `getattr` fallback 兼容 |
| 两个 key 均不设置 | 回退到 `[224, 224]` |
| 检测头不传 min_size | 回退到 224 |
| anchor 不传 relative_scales | 回退到 legacy absolute sizes |
| utils 不传 width/height | 回退到 224 |

---

## 十、多尺寸验证矩阵

7 项回归测试全部通过：

| # | 测试 | 224 | 280 | 336 | 392 |
|---|------|:---:|:---:|:---:|:---:|
| 1 | YAML 配置 | ✅ | — | — | — |
| 2 | Transform | ✅ | — | — | — |
| 3 | FPN 空间尺寸 | ✅ | ✅ | ✅ | ✅ |
| 4 | Semantic head | ✅ | — | ✅ | — |
| 5 | Anchor generation | ✅ | ✅ | ✅ | ✅ |
| 6 | Utils 兼容 | ✅ | — | — | — |
| 7 | Import 完整性 | ✅ | — | — | — |

---

## 十一、待 NPU 环境执行

```bash
# 1. 生成多尺寸 GT cache (可选，LandcoverSemanticDataset 已支持在线 resize)
python scripts/precache_landcover_targets.py --image-size 280
python scripts/precache_landcover_targets.py --image-size 336
python scripts/precache_landcover_targets.py --image-size 392

# 2. 启用 multi-scale
# 编辑 Configs/step3_dual.yaml → multi_scale.enabled: true

# 3. 启动训练
bash scripts/run_stage3_multiscale.sh

# 4. 跨分辨率评估
for size in 224 280 336 392; do
    python scripts/eval_poc1.py \
        --checkpoint ./output/stage3/multiscale_v1/FINAL.pt \
        --image-size $size \
        --output ./output/stage3/multiscale_v1/eval_${size}.json
done
```

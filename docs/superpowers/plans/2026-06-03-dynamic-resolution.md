# CoastGPT 动态分辨率改造 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除全链路 224×224 硬编码，使模型支持任意 14 的倍数输入分辨率，默认 224 行为不变。

**Architecture:** 将绝对像素量改为配置驱动 + tensor shape 驱动 + 相对比例。YAML `input_size` 改名为 `default_input_size`，anchor/scales 改用相对值，检测头 upsample 用动态尺寸。Phase 1 完成 resolution-agnostic refactor 并通过 224 regression test，Phase 2 从 224 checkpoint 启动 multi-scale fine-tune。

**Tech Stack:** PyTorch 2.1.2, torchvision, timm, DeepSpeed, Ascend NPU, Python 3.10

---

## 文件结构

| 文件 | 职责 | 改动类型 |
|------|------|----------|
| `Configs/*.yaml` (7 files) | 配置入口 | 改名 + 新增 multi_scale 块 |
| `Dataset/build_transform.py` | 图像 transform | 去 crop_pct 硬编码 |
| `Models/dual_vision_encoder.py` | DINOv3 ViT + ConvNeXt | 去 `_224` timm suffix + pyramid 动态化 |
| `Models/det_head.py` | Mask R-CNN builder | min/max_size 动态 + anchor 相对化 |
| `Models/semantic_head.py` | 语义分割头 | output_size 动态 |
| `Models/fpn_neck.py` | FPN neck | 代码已动态，不改 |
| `Dataset/rasterize_geojson.py` | GT 栅格化 | model_size 默认值 |
| `Dataset/landcover_dataset.py` | 土地覆盖数据集 | image_size 默认值 + shape 断言 |
| `Dataset/aqua_poc_dataset.py` | 养殖区数据集 | image_size 默认值 |
| `Dataset/cap_dataset.py` | CAP 数据集 | crop_size 默认值 |
| `utils/georef_transform.py` | 坐标转换 | width/height 默认值 |
| `utils/mask_utils.py` | mask 工具 | width/height 默认值 |
| `Models/edge_head.py` | 海岸线边缘头 | output_size 动态（如有） |
| `Trainer/hook/param_flops_hook.py` | FLOPS 统计 | img_size 默认值 |
| `Trainer/Data/data.py` | 数据配置 | input_size/crop_size 默认值 |

---

## Phase 1: Resolution-Agnostic Refactor

Phase 1 目标: 代码支持任意分辨率，但默认行为与改前完全一致（224×224）。

### Task 1: 改造 YAML 配置

**Files:**
- Modify: `Configs/train_dual.yaml:45,99`
- Modify: `Configs/step2_dual.yaml:46,100`
- Modify: `Configs/step3_dual.yaml:55,108`
- Modify: `Configs/step3_dual_fallback.yaml:52,105`
- Modify: `Configs/step3_dual_v3_success.yaml:55,108`
- Modify: `Configs/seg_train.yaml:50,104`
- Modify: `Configs/inference.yaml:28,80`

- [ ] **Step 1: 将 `input_size` 改名为 `default_input_size`**

在每个 YAML 的 `rgb_vision` 块中:

```yaml
# 原来
rgb_vision:
  input_size: [224, 224]

# 改为
rgb_vision:
  default_input_size: [224, 224]
```

在每个 YAML 的 `transform` 块中:

```yaml
# 原来
transform:
  input_size: [224, 224]

# 改为
transform:
  default_input_size: [224, 224]
```

- [ ] **Step 2: 新增 multi_scale + anchor 配置块**

在 `step3_dual.yaml` 末尾追加:

```yaml
multi_scale:
  enabled: false
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
    - [0.0714, 0.1429]
    - [0.1429, 0.2857]
    - [0.2857, 0.4286, 0.5714]
  aspect_ratios: [0.5, 1.0, 2.0, 3.0]

inference_transform:
  resize_policy: square_resize
  target_size: null
  ensure_multiple_of: 14
```

同步更新其余 6 个 config 的 `input_size` → `default_input_size` 重命名。

- [ ] **Step 3: 验证 YAML 加载**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import ml_collections
from ml_collections import ConfigDict
cfg = ConfigDict()
with open('Configs/step3_dual.yaml') as f:
    # just verify parse
    import yaml
    d = yaml.safe_load(f)
    print('rgb_vision.default_input_size:', d['rgb_vision']['default_input_size'])
    print('transform.default_input_size:', d['transform']['default_input_size'])
    print('OK')
"
```

Expected: 打印 `[224, 224]` 两次，无报错。

---

### Task 2: Refactor build_transform.py — 去除 crop_pct 硬编码

**Files:**
- Modify: `Dataset/build_transform.py:55,105`

- [ ] **Step 1: 参数化 crop_pct 计算**

将 `Dataset/build_transform.py` 中 `build_cls_transform` 和 `build_vlp_transform` 的 eval 分支从硬编码改为配置驱动。

修改 `build_vlp_transform` (line 105-116):

```python
# 原来 (line 104-116)
t = []
crop_pct = 224 / 256
size = int(config.transform.input_size[0] / crop_pct)
t.append(
    transforms.Resize(
        size, interpolation=PIL.Image.BICUBIC
    ),  # to maintain same ratio w.r.t. 224 images
)
t.append(transforms.CenterCrop(config.transform.input_size))

t.append(transforms.ToTensor())
t.append(transforms.Normalize(mean, std))
return transforms.Compose(t)

# 改为
t = []
input_size = getattr(config.transform, 'default_input_size', [224, 224])

# Derive resize size: if crop_pct configured, use it; otherwise resize directly to target
crop_pct = getattr(config.transform, 'crop_pct', None)
if crop_pct is not None:
    resize_size = int(input_size[0] / crop_pct)
else:
    resize_size = input_size[0]

t.append(transforms.Resize(resize_size, interpolation=PIL.Image.BICUBIC))

if crop_pct is not None:
    t.append(transforms.CenterCrop(input_size))

t.append(transforms.ToTensor())
t.append(transforms.Normalize(mean, std))
return transforms.Compose(t)
```

同样修改 `build_cls_transform` (line 55-66) 中的对应逻辑。

- [ ] **Step 2: 验证**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import ml_collections
from Dataset.build_transform import build_vlp_transform

config = ml_collections.ConfigDict()
config.rgb_vision = ml_collections.ConfigDict()
config.rgb_vision.arch = 'convnext'
config.transform = ml_collections.ConfigDict()
config.transform.default_input_size = [224, 224]

t = build_vlp_transform(config, is_train=False)
print('Transform:', t)

import torch
dummy = torch.randn(3, 256, 256)
out = t(dummy)
print(f'Output shape: {out.shape}')  # Expected: [3, 224, 224]
"
```

Expected: `Output shape: torch.Size([3, 224, 224])`

---

### Task 3: 改造 DualVisionEncoder — pyramid 动态化 + ViT timm arch

**Files:**
- Modify: `Models/dual_vision_encoder.py:174,385-410,836-880`

- [ ] **Step 1: 修改 `input_size` 读取 key**

`Models/dual_vision_encoder.py:174`:

```python
# 原来
self.input_size = tuple(rgb_cfg.get("input_size", [224, 224]))

# 改为 (兼容新旧 key)
self.input_size = tuple(
    rgb_cfg.get("default_input_size")
    or rgb_cfg.get("input_size", [224, 224])
)
```

- [ ] **Step 2: 修改 `_map_global_timm_arch` — 支持动态分辨率 arch**

`Models/dual_vision_encoder.py:385-410`:

```python
def _map_global_timm_arch(self, global_name: str, ckpt_path: str) -> str:
    name_lower = (global_name or '').lower()
    ckpt_lower = (ckpt_path or '').lower()
    patch = 14 if ('14' in name_lower or 'patch14' in name_lower or '14' in ckpt_lower) else 16

    if 'vitg' in name_lower or 'giant' in name_lower or '7b' in name_lower or 'vit7b' in ckpt_lower:
        base = 'vit_giant'
    elif 'vith' in name_lower or 'huge' in name_lower:
        base = 'vit_huge'
    elif 'vitl' in name_lower or 'large' in name_lower:
        base = 'vit_large'
    else:
        base = 'vit_large'

    # Always use _224 suffix for timm model registry (pretrained checkpoint resolution).
    # Position embeddings are dynamically interpolated at forward time.
    candidate = f"{base}_patch{patch}_224"
    try:
        import timm
        if candidate not in timm.list_models('*vit*') and base != 'vit_large':
            import warnings
            warnings.warn(f"timm arch '{candidate}' not found; falling back to vit_large_patch{patch}_224")
            candidate = f"vit_large_patch{patch}_224"
    except Exception:
        pass
    return candidate
```

关键点: timm arch 后缀 `_224` **保持不变**，它只是 pretrained checkpoint 的分辨率标记，不是运行时分辨率。运行时由 pos_embed 插值决定。不需要真正支持 `dynamic_img_size=True`。

- [ ] **Step 3: 修改 `_get_local_pyramid_raw` — 用 tensor shape 替代 self.input_size**

`Models/dual_vision_encoder.py:866-881`:

```python
# 原来 (_get_local_pyramid_raw, line 866-881)
def _get_local_pyramid_raw(self, pixel_values: torch.Tensor):
    if self.local_source == "openclip":
        h, w = self.input_size  # ← 硬编码
        sizes = [(h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16)]
        ...

# 改为
def _get_local_pyramid_raw(self, pixel_values: torch.Tensor):
    if self.local_source == "openclip":
        _, _, h, w = pixel_values.shape  # ← 从 tensor 读取
        sizes = [(h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16)]
        ...
```

同样修改 `_get_local_pyramid` (line 836-840) 中的对应逻辑。

- [ ] **Step 4: 验证 ViT 多尺寸前向**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import torch
import ml_collections
from Models.dual_vision_encoder import DualVisionEncoder

config = ml_collections.ConfigDict()
config.alignment_dim = 1024
config.rgb_vision = ml_collections.ConfigDict()
config.rgb_vision.default_input_size = [224, 224]
config.rgb_vision.global_encoder_name = 'dinov3_vitl16'
config.rgb_vision.local_source = 'dino'
config.rgb_vision.local_encoder_name = 'convnext_base'
config.rgb_vision.local_ckpt_path = './dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'
config.rgb_vision.freeze_global = True
config.rgb_vision.freeze_local = True
config.rgb_vision.global_ckpt_path = './dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
config.rgb_vision.input_patchnorm = False
config.rgb_vision.patch_dropout = 0.0
config.rgb_vision.tune_pooler = True
config.rgb_vision.attn_pooler = ml_collections.ConfigDict()
config.rgb_vision.attn_pooler.num_query = 144
config.rgb_vision.attn_pooler.num_attn_heads = 16
config.rgb_vision.attn_pooler.num_layers = 6
config.rgb_vision.physical_prompt_dim = 4096

encoder = DualVisionEncoder(config)
encoder.eval()

for size in [224, 280, 336, 392]:
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        seq, g_grid, pyramid = encoder.encode_with_spatial(x)
    c4, c8, c16, c32 = pyramid
    print(f'Size {size}:')
    print(f'  c4={list(c4.shape)}, c8={list(c8.shape)}, c16={list(c16.shape)}, c32={list(c32.shape)}')
    print(f'  g_grid={list(g_grid.shape)}')
    # Verify spatial ratios
    _, _, h, w = x.shape
    assert c4.shape[2] == h // 4, f'c4 H mismatch: {c4.shape[2]} vs {h//4}'
    assert c8.shape[2] == h // 8
    assert c16.shape[2] == h // 16
    assert c32.shape[2] == h // 32
    print('  OK')
"
```

Expected: All four sizes pass assertions.

---

### Task 4: 改造 FPN + 检测头 — 动态尺寸

**Files:**
- Modify: `Models/semantic_head.py:36,103-106`
- Create/Modify: `Models/edge_head.py` (if exists)
- Modify: `Models/det_head.py:95-96` (min/max_size 默认值)

- [ ] **Step 1: LandcoverSemanticHead — output_size 改为 None（动态）**

`Models/semantic_head.py:32-41`:

```python
def __init__(
    self,
    in_channels: int = 256,
    num_classes: int = 25,
    output_size: Optional[Tuple[int, int]] = None,  # None = use input spatial size
):
    super().__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.output_size = output_size
    ...
```

`Models/semantic_head.py:102-106`:

```python
# 原来
logits = F.interpolate(
    x, size=self.output_size, mode='bilinear', align_corners=False
)

# 改为
if self.output_size is not None:
    logits = F.interpolate(
        x, size=self.output_size, mode='bilinear', align_corners=False
    )
else:
    # Use input image spatial size (passed via keyword or derived from p1)
    logits = x  # keep at fused resolution (56×56 for 224); caller can upsample
```

**注意:** 语义头输出 56×56（= input_size/4），caller 负责最终 upsample 到 input_size。这是设计 doc 的意图——logits 在 P1 空间，由调用方或后处理 upsample。如果当前约定是 logits 必须已 upsample 到 input_size，则改成:

```python
# 改为: 动态从 p1 尺寸推断 input_size
if self.output_size is not None:
    upsample_size = self.output_size
else:
    upsample_size = (p1.shape[2] * 4, p1.shape[3] * 4)  # P1 = H/4

logits = F.interpolate(
    x, size=upsample_size, mode='bilinear', align_corners=False
)
```

- [ ] **Step 2: 验证 semantic_head 多尺寸**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import torch
from Models.semantic_head import LandcoverSemanticHead

model = LandcoverSemanticHead(num_classes=25, output_size=None)

for size in [224, 336]:
    h, w = size // 4, size // 4  # P1 spatial size
    p1 = torch.randn(2, 256, h, w)
    p2 = torch.randn(2, 256, h//2, w//2)
    p3 = torch.randn(2, 256, h//4, w//4)
    p4 = torch.randn(2, 256, h//8, w//8)
    logits = model(p1, p2, p3, p4)
    print(f'Input {size} -> logits {list(logits.shape)}')
"
```

- [ ] **Step 3: det_head.py — min_size/max_size 移除硬编码**

`Models/det_head.py:95-96`:

```python
# 原来
min_size: int = 224,
max_size: int = 224,

# 改为: None 表示自动从 backbone 输出推导
min_size: Optional[int] = None,
max_size: Optional[int] = None,
```

并在函数体开头添加:

```python
if min_size is None:
    min_size = 224  # legacy default, caller should override
if max_size is None:
    max_size = 224
```

**注:** torchvision MaskRCNN 的 `min_size` 是 shortest-side resize 的目标，应保持 caller 可覆盖。Phase 1 保持默认 224，Phase 2 由训练循环传入实际尺寸。

- [ ] **Step 4: 验证 FPN + det_head 多尺寸前向**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import torch
from Models.fpn_neck import FPNNeck

fpn = FPNNeck()

for size in [224, 336]:
    h, w = size, size
    c4 = torch.randn(1, 128, h//4, w//4)
    c8 = torch.randn(1, 256, h//8, w//8)
    c16 = torch.randn(1, 512, h//16, w//16)
    c32 = torch.randn(1, 1024, h//32, w//32)
    p1, p2, p3, p4 = fpn(c4, c8, c16, c32)
    print(f'Size {size}: P1={list(p1.shape)}, P2={list(p2.shape)}, P3={list(p3.shape)}, P4={list(p4.shape)}')
"
```

---

### Task 5: Anchor Generator 相对化

**Files:**
- Modify: `Models/det_head.py:83-84,107-109`

- [ ] **Step 1: 实现相对 anchor 生成逻辑**

在 `Models/det_head.py` 中新增辅助函数:

```python
def _build_anchor_sizes(
    relative_scales: Tuple[Tuple[float, ...], ...],
    input_h: int,
    input_w: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Convert relative anchor scales to absolute pixel sizes.
    
    Args:
        relative_scales: e.g. ((0.0714, 0.1429), (0.1429, 0.2857), (0.2857, 0.4286, 0.5714))
        input_h, input_w: current input spatial dimensions.
    
    Returns:
        Absolute anchor sizes, e.g. ((16, 32), (32, 64), (64, 96, 128)) for 224 input.
    """
    base = min(input_h, input_w)
    return tuple(
        tuple(max(1, int(s * base)) for s in scales)
        for scales in relative_scales
    )
```

**注:** `max(1, ...)` 防止极小尺寸输入下 anchor 归零。

- [ ] **Step 2: 修改 `build_aqua_maskrcnn` 支持相对 scales**

`Models/det_head.py:80-110`:

```python
def build_aqua_maskrcnn(
    backbone_adapter: DualVisionFPNBackboneAdapter,
    num_classes: int = 2,
    # New relative API (preferred):
    anchor_relative_scales: Optional[Tuple[Tuple[float, ...], ...]] = None,
    # Old absolute API (backward compat):
    anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0, 3.0),) * 3,
    ...
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> MaskRCNN:
    ...
    if min_size is None:
        min_size = 224
    if max_size is None:
        max_size = min_size
    
    if anchor_relative_scales is not None:
        # Will be resolved at forward time; store for later use
        anchor_sizes = _build_anchor_sizes(anchor_relative_scales, min_size, min_size)
    elif anchor_sizes is None:
        # Legacy default
        anchor_sizes = ((16, 32), (32, 64), (64, 96, 128))
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )
    ...
```

- [ ] **Step 3: 验证 anchor 生成**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
from Models.det_head import _build_anchor_sizes

# 224 baseline
scales_224 = _build_anchor_sizes(
    ((0.0714, 0.1429), (0.1429, 0.2857), (0.2857, 0.4286, 0.5714)),
    224, 224
)
print('224 anchor:', scales_224)
# Expected: ((15, 32), (32, 63), (63, 96, 127)) — close to original ((16,32),(32,64),(64,96,128))
assert abs(scales_224[0][0] - 16) <= 1  # rounding diff OK

# 336
scales_336 = _build_anchor_sizes(
    ((0.0714, 0.1429), (0.1429, 0.2857), (0.2857, 0.4286, 0.5714)),
    336, 336
)
print('336 anchor:', scales_336)
assert scales_336[0][0] == 24  # 0.0714 * 336 = 24
print('OK')
"
```

---

### Task 6: GT 数据生成 — 去 224 硬编码

**Files:**
- Modify: `Dataset/rasterize_geojson.py:33,58`
- Modify: `Dataset/landcover_dataset.py:52,281-282`
- Modify: `Dataset/aqua_poc_dataset.py:55`
- Modify: `Dataset/cap_dataset.py:511`
- Modify: `utils/georef_transform.py:113-114`
- Modify: `utils/mask_utils.py:52-53,85-86`

- [ ] **Step 1: rasterize_geojson.py — 默认值改为 None**

```python
# line 31-33
def compute_model_transform_from_bounds(
    tile_bounds_wgs84: Tuple[float, float, float, float],
    model_size: Optional[Tuple[int, int]] = None,
) -> List[float]:
    if model_size is None:
        model_size = (224, 224)
    ...

# line 55-58
def rasterize_features_to_target(
    features: List[dict],
    tile_bounds_wgs84: Tuple[float, float, float, float],
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if target_size is None:
        target_size = (224, 224)
    ...
```

- [ ] **Step 2: utils/mask_utils.py — 默认值改为 None**

```python
# line 50-53
def rasterize_polygon(
    polygon: List[Tuple[float, float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> np.ndarray:
    if width is None:
        width = 224
    if height is None:
        height = 224
    ...

# line 83-86
def rasterize_multipolygon(
    polygons: List[List[Tuple[float, float]]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    if width is None:
        width = 224
    if height is None:
        height = 224
    ...
```

- [ ] **Step 3: utils/georef_transform.py — 默认值改为 None**

```python
# line 111-113
def clip_pixel_coords(
    coords_pixel: List[Tuple[float, float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[Tuple[float, float]]:
    if width is None:
        width = 224
    if height is None:
        height = 224
    ...
```

- [ ] **Step 4: landcover_dataset.py — shape 断言改为相对**

`Dataset/landcover_dataset.py:52`:

```python
# 原来
image_size: int = 224,

# 改为
image_size: int = 224,  # default preserved; caller overrides for non-224
```

`Dataset/landcover_dataset.py:281-282` (`__main__` 块):

```python
# 原来
assert imgs.shape == (4, 3, 224, 224)
assert tgts.shape == (4, 224, 224)

# 改为
assert imgs.shape[0] == 4 and imgs.shape[1] == 3
assert imgs.shape[2] == imgs.shape[3]  # square
assert tgts.shape[0] == 4
assert tgts.shape[1] == imgs.shape[2] and tgts.shape[2] == imgs.shape[3]
```

- [ ] **Step 5: aqua_poc_dataset.py — 默认值保留**

```python
# 原来 line 55
image_size: int = 224,

# 保留不变; 只是默认值，caller 可覆盖
```

- [ ] **Step 6: cap_dataset.py — 默认值保留**

```python
# 原来 line 511
crop_size: int = 224

# 保留不变
```

- [ ] **Step 7: 验证数据集多尺寸**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
from Dataset.rasterize_geojson import compute_model_transform_from_bounds
from utils.mask_utils import rasterize_polygon
from utils.georef_transform import clip_pixel_coords

# Verify defaults still work
t = compute_model_transform_from_bounds((119.0, 34.0, 119.1, 34.1))
print('Default transform:', t)

# Verify explicit sizes
t2 = compute_model_transform_from_bounds((119.0, 34.0, 119.1, 34.1), (336, 336))
print('336 transform:', t2)

m = rasterize_polygon([(0, 0), (100, 0), (100, 100), (0, 100)], width=336, height=336)
print('Mask shape:', m.shape)
assert m.shape == (336, 336)
print('OK')
"
```

---

### Task 7: 更新训练/推理脚本的配置读取

**Files:**
- Modify: `Inference.py` (所有读取 `config.rgb_vision.input_size` / `config.transform.input_size` 的位置)
- Modify: `train_stage_three.py` (同上)
- Modify: `Tools/model_evaluate/seg_train.py:357`
- Modify: `Trainer/Data/data.py:101-102`
- Modify: `Trainer/hook/param_flops_hook.py:13`

- [ ] **Step 1: 全局搜索并替换配置 key 读取**

搜索模式: `"input_size"` → 替换为 `"default_input_size"`（带 fallback 到旧 key）

在 `train_stage_three.py`、`train_stage_two.py`、`train_stage_one.py`、`Inference.py` 中:

```python
# 原来
input_size = config.rgb_vision.input_size

# 改为
input_size = getattr(config.rgb_vision, 'default_input_size', None) \
          or getattr(config.rgb_vision, 'input_size', [224, 224])
```

`Tools/model_evaluate/seg_train.py:357`:

```python
# 原来
input_size = tuple(cfg.get("rgb_vision", {}).get("input_size", [224, 224]))

# 改为
input_size = tuple(
    cfg.get("rgb_vision", {}).get("default_input_size")
    or cfg.get("rgb_vision", {}).get("input_size", [224, 224])
)
```

`Trainer/Data/data.py:101-102`:

```python
# 原来
input_size: Union[List, Tuple] = [224, 224]
crop_size: Union[List, Tuple] = [224, 224]

# 改为 (保留默认值，但改名)
default_input_size: Union[List, Tuple] = [224, 224]
default_crop_size: Union[List, Tuple] = [224, 224]
```

`Trainer/hook/param_flops_hook.py:13`:

```python
# 原来
def __init__(self, img_size: int = 224, ...):

# 改为
def __init__(self, img_size: int = 224, ...):  # 保留默认值
```

---

### Task 8: 224 Regression Test — Phase 1 验收

- [ ] **Step 1: 跑一次 224 全链路前向 + backward**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import torch
import ml_collections
import yaml
from Models.det_head import DualVisionFPNBackboneAdapter, build_aqua_maskrcnn
from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck

# Load config
with open('Configs/step3_dual.yaml') as f:
    d = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(d)

# Build
encoder = DualVisionEncoder(cfg)
fpn = FPNNeck()
adapter = DualVisionFPNBackboneAdapter(encoder, fpn)
model = build_aqua_maskrcnn(adapter, num_classes=2)

model.train()

# Forward: 224 input with 2 dummy instances
images = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
targets = [
    {
        'boxes': torch.tensor([[50., 50., 100., 100.]]),
        'labels': torch.tensor([1]),
        'masks': torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8),
    },
    {
        'boxes': torch.tensor([[80., 80., 150., 150.]]),
        'labels': torch.tensor([1]),
        'masks': torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8),
    },
]

loss_dict = model(images, targets)
total_loss = sum(v for v in loss_dict.values())
print('Loss dict:', {k: round(v.item(), 4) for k, v in loss_dict.items()})
print('Total loss:', total_loss.item())

total_loss.backward()
print('Backward OK')
print('224 regression test PASSED')
"
```

Expected: Loss dict printed, no NaN, backward completes.

- [ ] **Step 2: 验证多尺寸 forward（不训练）**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import torch
import ml_collections
import yaml
from Models.det_head import DualVisionFPNBackboneAdapter, build_aqua_maskrcnn
from Models.dual_vision_encoder import DualVisionEncoder
from Models.fpn_neck import FPNNeck

with open('Configs/step3_dual.yaml') as f:
    d = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(d)

encoder = DualVisionEncoder(cfg)
fpn = FPNNeck()
adapter = DualVisionFPNBackboneAdapter(encoder, fpn)

for size in [224, 280, 336, 392]:
    model = build_aqua_maskrcnn(adapter, num_classes=2, min_size=size, max_size=size)
    model.eval()
    images = [torch.randn(3, size, size)]
    with torch.no_grad():
        preds = model(images)
    print(f'Size {size}: {len(preds)} predictions OK')
print('Multi-size forward test PASSED')
"
```

Expected: All four sizes complete without error.

---

## Phase 2: Multi-Scale Fine-Tune

Phase 2 目标: 从 224 checkpoint 启动 multi-scale training，每 batch 随机采样分辨率。

### Task 9: 生成多尺寸 GT cache

**Files:**
- Create: `scripts/precache_multiscale_targets.py`
- Modify: `scripts/precache_landcover_targets.py` (更新 `IMAGE_SIZE` 常量)

- [ ] **Step 1: 为土地覆盖生成 280/336/392 GT cache**

```bash
cd /home/ma-user/work/CoastGPT && python scripts/precache_landcover_targets.py --image-size 280
cd /home/ma-user/work/CoastGPT && python scripts/precache_landcover_targets.py --image-size 336
cd /home/ma-user/work/CoastGPT && python scripts/precache_landcover_targets.py --image-size 392
```

修改 `scripts/precache_landcover_targets.py:23`:

```python
# 原来
IMAGE_SIZE = 224

# 改为
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image-size', type=int, default=224)
args = parser.parse_args()
IMAGE_SIZE = args.image_size
```

- [ ] **Step 2: 为养殖区生成多尺寸 GT**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Read manifest, resize binary TIF masks for each target size
manifest_path = 'MixedStage3Data_v2/aqua_manifest.json'
sizes = [280, 336, 392]

with open(manifest_path) as f:
    samples = json.load(f)

for sample in samples:
    if not sample.get('binary_label_path'):
        continue
    for sz in sizes:
        out_dir = Path(sample['binary_label_path']).parent.parent / f'Label_Binary_{sz}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(sample['binary_label_path']).name
        if not out_path.exists():
            img = Image.open(sample['binary_label_path'])
            img = img.resize((sz, sz), Image.NEAREST)
            img.save(out_path)
    if samples.index(sample) % 100 == 0:
        print(f'  Processed {samples.index(sample)}/{len(samples)}')

print('Multi-scale GT cache done')
"
```

---

### Task 10: 实现 Multi-Scale Training Loop

**Files:**
- Modify: `train_stage_three.py` (或新增 `trainers/multiscale_sampler.py`)

- [ ] **Step 1: 实现 multi-scale batch sampler**

```python
# 新增 Dataset/multiscale_sampler.py
"""Multi-scale batch sampler for DDP training."""

import torch
import torch.distributed as dist
from typing import List, Tuple


class MultiScaleBatchSampler:
    """Samples a single resolution per global training step, broadcasts in DDP."""

    def __init__(
        self,
        sizes: List[Tuple[int, int]],
        weights: List[float] = None,
        enabled: bool = True,
    ):
        self.sizes = sizes
        self.weights = weights or [1.0 / len(sizes)] * len(sizes)
        self.enabled = enabled
        self._current_size = sizes[0]

    @property
    def current_size(self) -> Tuple[int, int]:
        return self._current_size

    def sample(self) -> Tuple[int, int]:
        """Sample a size for the current global step. Must be called on all ranks."""
        if not self.enabled:
            self._current_size = self.sizes[0]
            return self._current_size

        if dist.is_initialized():
            if dist.get_rank() == 0:
                idx = torch.multinomial(
                    torch.tensor(self.weights, dtype=torch.float),
                    num_samples=1,
                ).item()
                size_tensor = torch.tensor([self.sizes[idx][0], self.sizes[idx][1]])
            else:
                size_tensor = torch.zeros(2, dtype=torch.long)

            dist.broadcast(size_tensor, src=0)
            self._current_size = (int(size_tensor[0]), int(size_tensor[1]))
        else:
            idx = torch.multinomial(
                torch.tensor(self.weights, dtype=torch.float),
                num_samples=1,
            ).item()
            self._current_size = self.sizes[idx]

        return self._current_size
```

- [ ] **Step 2: 集成到 Stage 3 训练循环**

在 `train_stage_three.py` 的训练循环中:

```python
# 初始化
multi_scale_cfg = getattr(config, 'multi_scale', None)
if multi_scale_cfg and multi_scale_cfg.get('enabled', False):
    ms_sampler = MultiScaleBatchSampler(
        sizes=[tuple(s) for s in multi_scale_cfg['sizes']],
        weights=multi_scale_cfg.get('weights'),
    )
else:
    ms_sampler = MultiScaleBatchSampler(
        sizes=[tuple(config.rgb_vision.default_input_size)],
        enabled=False,
    )

# Per-step
for step, batch in enumerate(dataloader):
    current_size = ms_sampler.sample()

    # Resize batch images to current_size
    images = F.interpolate(batch['images'], size=current_size, mode='bilinear')

    # Rebuild Mask R-CNN with current anchor sizes and min_size
    # (or dynamically update anchor sizes in-place)

    loss = model(images, targets)
    ...
```

**简化方案（推荐）:** Phase 2 第一阶段不使用 Mask R-CNN 的 multi-scale training。因为 torchvision MaskRCNN 的 anchor 是在 `__init__` 时固定的，动态切换需要重建模型。简化为：**用 fixed-size collate，每个 epoch 固定一个尺寸**，不同 epoch 切换尺寸。这避免了 per-batch anchor 重建的复杂性。

```python
# 简化方案: epoch-level size switching
def get_epoch_size(epoch: int, cfg) -> Tuple[int, int]:
    sizes = cfg.multi_scale.sizes
    # Round-robin: each epoch uses one size
    idx = epoch % len(sizes)
    return tuple(sizes[idx])
```

- [ ] **Step 3: 更新 det_head 的 min_size 配置**

训练脚本中根据当前尺寸传递 `min_size`:

```python
current_h, current_w = current_size
model = build_aqua_maskrcnn(
    adapter,
    num_classes=2,
    min_size=current_h,
    max_size=current_w,
    anchor_relative_scales=cfg.anchor_generator.scales,
)
```

---

### Task 11: 启动 Multi-Scale Fine-Tune

- [ ] **Step 1: 准备 launch 脚本**

```bash
cd /home/ma-user/work/CoastGPT && cat > scripts/run_stage3_multiscale.sh << 'SHEOF'
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0
export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

deepspeed --num_nodes=1 --num_gpus=2 train_stage_three.py \
  -c Configs/step3_dual.yaml --batch-size 2 --workers 1 \
  --accumulation-steps 8 --epochs 12 \
  --model-path ./output/stage3/multiscale_v1/FINAL.pt \
  --data-path ./MixedStage3Data_v2 \
  --output ./output/stage3/multiscale_v1 \
  --accelerator npu --enable-amp True --use-checkpoint --wandb False \
  --multi-scale
SHEOF
chmod +x scripts/run_stage3_multiscale.sh
```

- [ ] **Step 2: Dry-run 验证 multi-scale 配置加载**

```bash
cd /home/ma-user/work/CoastGPT && python -c "
import ml_collections, yaml
with open('Configs/step3_dual.yaml') as f:
    cfg = ml_collections.ConfigDict(yaml.safe_load(f))

ms = getattr(cfg, 'multi_scale', None)
print('multi_scale:', ms)
anchor = getattr(cfg, 'anchor_generator', None)
print('anchor_generator:', anchor)
"
```

---

## Phase 3: Multi-Resolution Evaluation

### Task 12: 跨分辨率评估矩阵

- [ ] **Step 1: 分别用 224/280/336/392 评估**

```bash
cd /home/ma-user/work/CoastGPT && for size in 224 280 336 392; do
    echo "=== Evaluating at ${size}x${size} ==="
    python scripts/eval_poc1.py \
        --checkpoint ./output/stage3/multiscale_v1/FINAL.pt \
        --image-size $size \
        --output ./output/stage3/multiscale_v1/eval_${size}.json
done
```

- [ ] **Step 2: 收集指标生成矩阵**

追踪指标:
- Overall RPN recall
- GF6 subset RPN recall
- non-GF6 subset RPN recall
- Instance AP@0.5 / AP@0.5:0.95
- Mask AP
- Landcover mIoU (per-size)
- Coastline edge F1 (per-size)
- CAP/VQA loss (如果 LLM 参与)

---

## Regression Test 清单 (Phase 1 完成前必须全部通过)

- [ ] 1. 旧配置下 224 forward 正常
- [ ] 2. 旧配置下 224 loss / backward 正常
- [ ] 3. 224/280/336/392 ViT forward 正常
- [ ] 4. semantic logits 输出尺寸 = input H,W (当 output_size=None 时)
- [ ] 5. RPN anchors 随 min_size 变化
- [ ] 6. FPN pyramid 各 level 尺寸 = input/4, input/8, input/16, input/32
- [ ] 7. ConvNeXt pyramid 使用实际 tensor shape（非 self.input_size）
- [ ] 8. landcover dataset 不再断言固定 224
- [ ] 9. 推理 transform 支持 runtime input_size
- [ ] 10. build_transform 不再硬编码 `crop_pct = 224 / 256`
- [ ] 11. 所有 YAML `input_size` 已改为 `default_input_size`
- [ ] 12. 所有旧 `"input_size"` 读取点有 fallback 兼容

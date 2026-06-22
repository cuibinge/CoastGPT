# WaveletAdapter / WaveletFusion: 基于 DWT 的遥感多波段适配与多源融合方案

## 版本

| 日期 | 版本 | 作者 | 变更 |
|------|------|------|------|
| 2026-06-10 | v0.1 | - | 初稿 |
| 2026-06-10 | v0.2 | - | 收紧 GSD 对齐、LL-only 有损性、多源配准假设；将实施路线改为可归因的消融实验 |

## 1. 背景与动机

### 1.1 现状

CoastGPT 当前图像输入管线已经支持 TIFF fallback，但最终仍会把遥感数据压成 3 通道 RGB：

| 问题 | 当前行为 | 代价 |
|------|----------|------|
| 波段数不统一 | `Dataset/cap_dataset.py::_load_tiff_as_rgb()` 对 `C>=4` 只取 RGB，丢弃 NIR 等额外波段 | NIR、SAR、高光谱等信息无法进入视觉编码器 |
| 分辨率不统一 | 训练 transform 将不同 GSD / 原始尺寸影像变成统一输入尺寸 | 高分辨率影像纹理被压缩，低分辨率影像被插值放大 |
| 传感器域差异 | 光学反射率、SAR 后向散射、热红外等被迫投影到同一 RGB 表示 | 视觉编码器无法显式知道物理来源 |
| 实验归因不清 | 如果直接做“大一统多源融合”，收益来源难以分辨 | 无法判断提升来自 NIR、多波段、DWT、HF，还是多源配准 |

### 1.2 核心假设

DWT 不是万能的遥感配准器；它适合被当成一个**可控的频域下采样与细节分解工具**：

```text
原始图像 [C, H, W]
    |
    v  一级 DWT
+-----------------------------+
| LL [C, H/2, W/2]            | 低频近似：大尺度结构、光谱能量
+-----------------------------+
| LH / HL / HH [C, H/2, W/2]  | 高频细节：边缘、纹理、噪声
+-----------------------------+
```

本方案只依赖以下保守假设：

1. **保留完整子带时 DWT/IDWT 近似可逆**。这只说明算子实现正确，不说明 LL-only 无损。
2. **LL-only 是有损低通表示**。它可能减少跨分辨率差异，但会丢失高频纹理。
3. **DWT 只能按 2 的幂改变有效 GSD**。任意 `source_gsd -> target_gsd` 仍需要残差 resize 或选择近似策略。
4. **多源融合必须建立在同一 AOI / 同一地理网格之上**。随机抽取不同传感器图像再融合，只能作为可视化玩具，不能证明跨传感器融合有效。

### 1.3 设计原则

实验必须拆开变量，而不是一次性引入所有能力：

1. 先验证 **多波段信息是否有用**：4-band TIF 直接适配到 3ch。
2. 再验证 **DWT LL 是否比直接适配更好**。
3. 再验证 **HF 是否补回纹理收益**。
4. 最后在同 AOI / 已配准数据上验证 **多源融合**。

## 2. 设计目标

### 2.1 功能目标

| 目标 | v0.2 实施范围 |
|------|---------------|
| 多波段统一 | 支持单源 `1~N` 波段输入，输出视觉编码器兼容的 `3ch` tensor |
| 分辨率近似对齐 | 用 DWT 做 2 的幂下采样，并显式记录 `effective_gsd` 与残差 resize |
| 多源融合 | 作为后续阶段，只允许同 AOI / 已配准 source group 输入 |
| 即插即用 | 作为 `CoastGPT.forward()` 中 `data["rgb"]` 前置适配器，关闭时完全回退当前路径 |
| 可归因实验 | 每个阶段只新增一个变量，输出统一指标表 |

### 2.2 非目标

- 不做 DN 到地表反射率 / Rrs 的物理校正。
- 不做几何配准；多源融合阶段要求输入已经预配准。
- 不替代 `PhysicsDecoder` 和 `PhysicsGuidedLoss`。
- 第一阶段不做 cross-source attention；先用单源多波段 adapter 建立可靠 baseline。

## 3. 数据模型

### 3.1 SourceImage

```python
@dataclass
class SourceImage:
    tensor: torch.Tensor        # [C, H, W] float32
    gsd: float                  # meters per pixel
    sensor_type: str            # "optical_ms", "sar", "thermal", ...
    wavelengths: Optional[List[Tuple[float, float]]] = None
    path: Optional[str] = None
    transform: Optional[Any] = None    # raster affine transform for future geo alignment
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    valid_mask: Optional[torch.Tensor] = None  # [1, H, W], optional nodata mask
```

### 3.2 Batch Contract

当前主干仍使用 `data["rgb"]`。WaveletAdapter 的集成方式应保持兼容：

```python
if wavelet_adapter.enabled and "sources" in data:
    data["rgb_original"] = data["rgb"]
    data["rgb"] = self.wavelet_adapter(data["sources"])
```

这样现有 `DualVisionEncoder.encode_with_spatial(data["rgb"])` 不需要改签名。物理监督如果仍依赖原图尺寸，应优先使用 `rgb_original` 或显式的 `physics_target_size`，避免 adapter 输出尺寸改变后误伤 `PhysicsDecoder` 对齐逻辑。

## 4. 算法设计

### 4.1 DWT Level 策略

不能使用单纯 `round(log2(target_gsd / source_gsd))`，因为很多 GSD 比值不是 2 的整数幂。

推荐配置：

```yaml
wavelet_adapter:
  target_gsd: 8.0
  level_policy: "ceil"          # "floor" | "ceil" | "nearest"
  residual_resize: true
```

计算规则：

```python
ratio = target_gsd / source_gsd
raw_level = log2(ratio)
levels = clamp(policy(raw_level), min=0, max=max_levels_allowed_by_shape)
effective_gsd = source_gsd * (2 ** levels)
```

策略解释：

| policy | 行为 | 适用场景 |
|--------|------|----------|
| `floor` | 不超过 target_gsd，保留更多细节 | 纹理/实例任务 |
| `ceil` | 达到或超过 target_gsd，更强低通 | 跨分辨率鲁棒性优先 |
| `nearest` | 最小化 GSD 误差 | 默认对照实验 |

对于 GF2 `0.8m -> 8m`：

| policy | levels | effective_gsd |
|--------|--------|---------------|
| floor | 3 | 6.4m |
| ceil | 4 | 12.8m |
| nearest | 3 | 6.4m |

因此文档和实验必须同时记录 `target_gsd` 与 `effective_gsd`，不能声称 DWT 精确对齐到 8m。

### 4.2 Adapter 阶段

#### A. RGB Baseline

当前路径：

```text
TIF/PNG -> load_image_as_rgb() -> transform -> data["rgb"] -> DualVisionEncoder
```

用途：作为所有实验的固定基线。

#### B. MultiBandDirectAdapter

第一阶段只验证“保留多波段是否有收益”：

```text
TIF [C,H,W]
-> per-band robust normalize
-> spectral adapter (1x1 conv / per-pixel MLP) C -> 3
-> resize/normalize to encoder input
```

这是比 DWT 更小的改动。如果它已经明显优于 RGB baseline，说明 NIR/额外波段本身有价值。

#### C. DWTLLAdapter

第二阶段只增加 DWT LL：

```text
TIF [C,H,W]
-> DWT levels by policy
-> LL [C,H',W']
-> spectral adapter C -> 3
-> residual resize to encoder input
```

它回答的问题是：在保留多波段的前提下，LL 低频表示是否提升跨分辨率鲁棒性。

#### D. DWTLLHFAdapter

第三阶段加入 HF，但仍保持单源：

```text
TIF [C,H,W]
-> DWT
-> LL adapter
-> HF summary adapter
-> concat/project -> 3ch
```

HF 不应直接无脑 IDWT。更稳的第一版是把 `LH/HL/HH` 做能量或小卷积摘要，再和 LL 表示融合。只有在有明确收益后再考虑可学习 IDWT 分支。

#### E. CrossSourceWaveletFusion

最后阶段才做多源融合：

```text
List[SourceImage] for same AOI
-> geo grid check / valid mask check
-> per-source DWT adapter
-> per-pixel source attention or gated weighted sum
-> 3ch output
```

约束：

- 所有 source 必须来自同一 AOI，或有明确空间交集。
- 必须有 `transform/crs/bounds` 或由 manifest 提供同源 tile id。
- 未满足配准条件时禁止融合，回退单源 adapter。

### 4.3 为什么第一版不用 Cross-Attention 做所有事情

原 v0.1 中的 `learnable_queries[3]` cross-attention 没有清楚说明空间维度如何恢复到 `[B,3,H,W]`。v0.2 暂不把它作为第一阶段实现。

第一版采用更可控的 per-pixel spectral adapter：

```text
每个像素位置独立地将 C 个波段投影到 3 个通道
```

这相当于共享权重的 `1x1 conv` 或小 MLP，参数少、可测性强，也更容易和 RGB baseline 做公平比较。

## 5. 与现有架构集成

### 5.1 需要新增 / 修改的文件

| 文件 | 变更 |
|------|------|
| `Models/wavelet_adapter.py` | 新增单源 adapter：level 计算、DWT LL、direct adapter、HF 摘要 |
| `tests/test_wavelet_adapter.py` | 新增 CPU 单元测试，覆盖 shape、level policy、DWT/IDWT smoke |
| `scripts/run_wavelet_adapter_ablation.py` | 新增实验脚本，按固定顺序跑 RGB/direct/LL/HF |
| `Dataset/cap_dataset.py` | 后续新增 `load_source_images()`，返回 `List[SourceImage]` |
| `Models/coastgpt.py` | 后续在 `forward()` 中可选插入 `wavelet_adapter` |
| `Configs/step3_dual.yaml` | 后续新增 `wavelet_adapter` 配置段 |

### 5.2 最小集成策略

第一轮实验不改训练主流程，只做离线 adapter 对照与小规模 overfit 入口：

1. 从 `Image_Orig/*.tif` 读取多波段。
2. 生成四种 adapter 输出。
3. 保存可视化和数值指标。
4. 确认 direct / LL / HF 的输出稳定后，再接入 `CoastGPT.forward()`。

这样可以避免在数据读取、模型 forward、NPU 训练三个层面同时引入变量。

## 6. 实验顺序

### 6.1 Phase 0：算子与数据烟测

目的：确认 DWT 和多波段读取没有基础错误。

必须通过：

- `compute_dwt_levels()` 对 GF2/GF1/GF6 给出可解释的 `levels/effective_gsd`。
- DWT/IDWT 在保留完整子带时重建误差 `< 1e-6`。
- 奇数尺寸输入经过 padding/crop 后 shape 正确。
- adapter 输出无 NaN/Inf，范围可控。

### 6.2 Phase 1：RGB Baseline

当前训练/评测路径，不做任何 adapter 改动。

记录：

- 数据集、样本数、输入尺寸。
- 任务指标：mIoU / mAP / F1 / text loss，按具体任务记录。
- wall-clock、显存峰值。

### 6.3 Phase 2：MultiBandDirectAdapter

只引入多波段 `C -> 3`，不使用 DWT。

对比问题：

```text
多波段信息本身是否比当前 RGB 更有用？
```

通过条件：

- 小样本 overfit loss 不劣于 RGB baseline。
- 下游指标至少不下降，或可视化显示 NIR/额外波段被稳定利用。

### 6.4 Phase 3：DWTLLAdapter

在 direct adapter 基础上加入 LL-only。

对比问题：

```text
DWT 低频表示是否带来额外收益？
```

必须同时报告：

- `levels`
- `effective_gsd`
- 是否执行 residual resize
- LL 输出尺寸

### 6.5 Phase 4：DWTLLHFAdapter

加入 HF 摘要。

对比问题：

```text
高频纹理是否补回 LL-only 损失？
```

建议优先在实例分割、岸线提取这类边界敏感任务上验证。

### 6.6 Phase 5：CrossSourceWaveletFusion

只在同 AOI / 已配准样本上做。

通过条件：

- manifest 能证明多个 source 指向同一地理区域。
- 融合前后 valid mask 与 bounds 可追踪。
- 不允许用随机不同传感器图像证明融合收益。

## 7. 指标与产物

每次实验输出一个 JSONL 或 CSV 表：

| 字段 | 含义 |
|------|------|
| `run_id` | 实验编号 |
| `adapter` | `rgb_baseline` / `direct` / `dwt_ll` / `dwt_ll_hf` / `cross_source` |
| `sensor` | GF1 / GF2 / GF6 / mixed |
| `source_gsd` | 原始 GSD |
| `target_gsd` | 目标 GSD |
| `levels` | DWT 级数 |
| `effective_gsd` | DWT 后有效 GSD |
| `input_shape` | 原始 shape |
| `output_shape` | adapter 输出 shape |
| `nan_ratio` | NaN 比例 |
| `latency_ms` | 单样本 adapter 耗时 |
| `metric_name/value` | 下游指标 |

可视化产物：

- RGB baseline
- direct adapter 输出
- LL adapter 输出
- HF energy map
- 差值图

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| DWT level 与 target GSD 不精确匹配 | 实验解释错误 | 显式记录 `effective_gsd`，并设置 residual resize |
| LL-only 丢失边缘 | 实例分割 / 岸线任务下降 | Phase 4 单独验证 HF 摘要 |
| 多源样本不配准 | 融合结果无意义 | Phase 5 之前强制检查 AOI / transform / bounds |
| pseudo-RGB 分布偏离预训练视觉编码器 | 冻结 DINO/ConvNeXt 下效果变差 | adapter 输出使用 ImageNet/SAT 兼容 normalize；必要时微调后几层 |
| attention 过早引入不稳定 | 难以归因，训练震荡 | 第一版使用 1x1 conv / 小 MLP，attention 放到多源阶段 |
| 数据管线一次性改动过大 | debug 困难 | 先离线 adapter ablation，再接入 `CoastGPT.forward()` |

## 9. 当前离线验证状态

已有脚本：

```text
scripts/verify_dwt_fusion.py
```

它已经验证了 DWT/IDWT 的基本可逆性，并能生成可视化。但它的跨传感器融合是随机单图 + 均值融合，只能作为算子演示，不能作为多源融合有效性的证据。

v0.2 后续应新增：

```text
Models/wavelet_adapter.py
tests/test_wavelet_adapter.py
scripts/run_wavelet_adapter_ablation.py
```

先跑 Phase 0 到 Phase 3，再决定是否继续 Phase 4 / Phase 5。

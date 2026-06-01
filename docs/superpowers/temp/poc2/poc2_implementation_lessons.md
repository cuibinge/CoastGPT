# PoC-2 土地覆盖 Semantic Head 实现学习笔记

> 日期: 2026-06-01
> 范围: Step 1-10（代码实现 → overfit sanity check）

---

## 1. FPN 拆分：共用模块的兼容性设计

**做了什么**：将 FPNNeck 从 `Models/det_head.py` 拆到 `Models/fpn_neck.py`。

**行动依据**：PoC-1/2/3 和 Stage 3/4 都依赖同一 FPN 架构。拆分后各 PoC 独立训练各自的 FPN 副本，但 class 定义唯一。

**具体做法**：
- 新建 `Models/fpn_neck.py`，只含 FPNNeck class
- `det_head.py` 中删除 FPNNeck 定义，改为 `from Models.fpn_neck import FPNNeck`
- 顺手清理了不再使用的 `import torch.nn.functional as F`

**为什么这样做而不是让 PoC-2 直接从 det_head.py 导入**：语义上 FPN 是共享基础模块，不属于 Instance Head。拆分后各 head 的代码依赖更清晰。

---

## 2. 土地分类数据的关键发现

**做了什么**：扫描了全部 48,930 个 GeoJSON 标注文件，建立 tile 空间分组。

**发现一：目录名 ≠ DLMC 规范类名**

| 目录名 | DLMC（GeoJSON 内真实类名） |
|--------|---------------------------|
| 养殖池塘 | 养殖坑塘 |
| 农村、城市建筑用地 | 农村宅基地 |
| 工矿仓储用地 | 工业用地 |
| 裸地 | 裸岩石砾地 |
| 林地 | 其他林地 |
| 园地 | 其他园地 |

**行动依据**：必须用 `dir_name_to_dlmc()` 做映射。若直接用目录名作为 class name，会导致与 GeoJSON 内 DLMC 字段不一致，后续数据合并和 label map 都会出错。

**发现二：空间重叠 tile 极少**

48,930 个 GeoJSON → 48,921 个空间 tile，仅 **9 个 tile** 有跨类重叠（最大 2 类）。

**含义**：空间合并策略对数据量帮助很有限，但它仍然是正确的——那 9 个重叠 tile 是仅有的多类标注来源。绝大多数训练依赖 partial-label CE + ignore mask。

**发现三：尺寸分布极端不均衡**

| 尺寸 | 样本数 | 占比 |
|------|--------|------|
| 128×128 | 39,274 | 80.3% |
| 256×256 | 9,142 | 18.7% |
| 512×512 | 514 | 1.0% |

**行动依据**：semantic head 的输出分辨率（224×224）对所有尺寸一致，但 512 的原始像素精度高于 128。size-stratified eval 不受此影响（只按 observed pixel 评估），但 high-res ablation 会量化 512 tile 在 224 rasterize 下的细窄地物精度损失。

---

## 3. 直接 rasterize 到 224×224 model pixel space 的正确性

**做了什么**：`rasterize_geojson.py` 中 GeoJSON → target mask 的实现。

**关键决策**：GeoJSON polygon 的 WGS84 坐标通过 georef 直接映射到 224×224 pixel space 后 rasterize。不先 rasterize 到原始分辨率再 resize。

**原因**：
1. bilinear resize 离散 class mask 会产生非法类别值（例如将 class_id=3 和 255 的边界 mix 成 128）
2. NEAREST resize 可避免非法值，但丢掉亚像素精度
3. WGS84 → 224 pixel 直接 rasterize 精度更高，因为坐标转换在连续空间完成，只在最后一步离散化

**与 PoC-1 的区别**：PoC-1 的养殖区数据自带 binary mask（已对齐 tile pixel space），用 NEAREST resize 即可。PoC-2 的 land cover 数据只有 GeoJSON，所以直接 rasterize 是最优路径。

---

## 4. all-ignore tile 的 NaN 陷阱

**问题**：`F.cross_entropy(logits, target, ignore_index=255)` 在 target 全部为 255 时返回 NaN。

**触发场景**：单类 partial-label tile 中，如果 polygon 太小、rasterize 后全被 clip 或 filter 掉，target 就是全 255。这在 batch_size 较小时尤其可能出现。

**修复**：
```python
valid_mask = target != ignore_index
if valid_mask.sum() == 0:
    return logits.sum() * 0.0, logits.sum() * 0.0, logits.sum() * 0.0
```
使用 `logits.sum() * 0.0` 而非 `torch.tensor(0.0)` 的目的是保持计算图，避免 autograd 报错。

**为什么 NaN 不是 0**：PyTorch CE 的 `ignore_index` 实现会将 ignored 位置从 softmax 分母中排除，当所有位置都被 ignore 时分母为 0，得到 inf → 与 label 交叉熵得到 NaN。

---

## 5. Overfit test 的验证价值

**做了什么**：用 10 个真实 tile 的随机 ConvNeXt 特征（模拟 frozen encoder 输出），训练 FPN + Semantic Head 50 步。

**为什么用随机特征而非真实 encoder**：
- Overfit test 的目标是验证训练框架（loss 通路、梯度流动、DataLoader 链路的正确性），不是评估模型质量
- 加载完整 DualVisionEncoder（DINOv3 ViT-L16 + ConvNeXt-Base）需 ~27GB checkpoint + 大量 NPU 显存
- 用随机 [B, C, H, W] tensor 模拟 ConvNeXt 多尺度输出即可验证 FPN → SemanticHead → loss → backward 全链路

**结果**：loss 7.62 → 3.27（delta 4.35），确认全链路通畅。

---

## 6. 后续步骤的关键风险点

1. **Vision encoder NPU 显存**：frozen encoder 仍需完整加载到 NPU 做前向。batch_size 需根据显存调整。
2. **Size_512 细窄地物**：沟渠、农村道路、城镇村道路用地在 224 rasterize 下可能只有 1-2 像素宽，IoU 会天然偏低。high-res ablation 将量化这个上限。
3. **Dice loss 中 observed_classes 的计算成本**：每个 batch 需要 unique(target) + per-class Dice，O(B*C*H*W) 但 C=25 不大，可接受。

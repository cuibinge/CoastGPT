# PoC-1 原始分辨率训练方案

> 日期: 2026-06-03
> 基于: 动态分辨率改造 Phase 1 完成
> 方案: 主方案 A — bucket by original size

---

## 方案

| 项目 | 设定 |
|------|------|
| 尺寸池 | 128×128, 256×256, 512×512 |
| 训练方式 | bucket by original size（按数据自身分辨率分组） |
| batch 规则 | batch 内同尺寸，DDP 同 step 同尺寸 |
| anchor | relative mode，min anchor = 0.0357×128 ≈ 5px |
| GT | match input size，NEAREST resize from Binary_WFQ.tif |
| 评估 | 按 tile size 分桶报告指标 |
| 备选 | 128 bucket mask/AP 差 → 128→192 |

---

## Anchor 缩放表

```
Relative scales:
  P2 (stride 4): [0.0357, 0.0714, 0.1429]
  P3 (stride 8): [0.1429, 0.2857]
  P4 (stride 16): [0.2857, 0.4286, 0.5714]

Absolute at each size:
         P2 anchors       P3 anchors        P4 anchors
  128:   [5,  9,  18]     [18,  37]          [37,  55,  73]
  256:   [9,  18, 37]     [37,  73]          [73,  110, 146]
  512:   [18, 37, 73]     [73,  146]         [146, 219, 293]
```

---

## 关键约束

- **128 不是 14 的倍数**（128/14=9.14），ViT pos_embed 通过 bicubic 插值处理非整数 grid
- **encoder padding 粒度为 16**（`encode_with_spatial`），128/256 无需 padding，512 整除
- **128 下 P1 = 32×32**，小目标 mask 仅有 ~10-20px，实例分割精度是硬上限
- **128 bucket 采样权重 0.25**，是三个尺寸中占比最低的，兼顾 128 的类别完整性和 256/512 的精度

---

## 数据流

```
Binary_WFQ.tif (原始分辨率: 128/256/512)
  → NEAREST resize → match input (128/256/512)
  → connectedComponentsWithStats → instance masks + boxes
  → 按 original_size 分桶 → batch 内同尺寸
```

GT 在线从 binary TIF NEAREST resize，不做 precache（省磁盘，CPU 成本可接受）。

---

## 评估分桶

```
Per-size metrics:
  Size_128:  RPN recall, AP@0.5, AP@0.5:0.95, GF6 subset, non-GF6 subset
  Size_256:  same
  Size_512:  same

Overall:
  Weighted average by tile count per bucket
```

---

## 执行顺序

```text
1. PoC-1 224 baseline（回归验证）→ 确认未退化
2. PoC-1 原始分辨率（128/256/512 bucket）→ 基准指标
3. 若 128 bucket AP < 阈值 → 试 128→192
4. 对比 224 baseline vs 原始分辨率 → 决策
```

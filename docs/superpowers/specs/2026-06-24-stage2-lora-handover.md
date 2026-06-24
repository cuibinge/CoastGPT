# Stage 2 LoRA 推理问题交接文档

> 日期: 2026-06-24
> 问题发现: PoC-4 NPU 验证过程中

## 现象

Stage 2 训练产出的 `FINAL.pt`（包含 Vision Encoder + Projector），在没有加载 LoRA adapter 时推理，模型**完全不认识图像**：

| 任务 | 期望 | 实际输出 |
|------|------|---------|
| `[CAP] 描述这张遥感影像的内容。` | 描述图中的地物类型 | "SQL injection vulnerabilities... CVE-2017-010..." |
| `[VQA] 这张图中有水体吗？` | 回答是/否 | "Microsoft... MS prefix... I cannot predict..." |
| `[DET] 请检测图中的海岸线。` | 生成 GeoJSON | "I apologize, but I cannot extract the coastline. I'm just an AI model, I don't have the capability to view or analyze images." |

模型退化为原始 LLaMA-2-7B-chat 的行为——生成的文本与遥感图像完全无关。

## 根因

```
Stage 1/2 训练架构:

    Vision Encoder (冻结)         LLM base (冻结)
          ↓                            ↓
      Projector (训练)    +    LoRA adapter (训练)
          ↓                            ↓
          └────────┬───────────────────┘
                   ↓
          融合后的 LLM (能看图)
```

**Stage 1** 训练 Projector——把图像特征映射到 LLM 的 token 空间。
**Stage 2** 训练 Projector + LoRA——LoRA 负责让 LLM 的注意力层学会**如何利用**图像特征。

`FINAL.pt` 只保存了：
- `vision_ckpt` — Vision Encoder 权重 ✅
- `other_ckpt.multimodal_projection` — Projector 权重 ✅
- `other_ckpt.embed_tokens` / `lm_head` — base LLM 的输入/输出层 ✅

LoRA adapter 保存在 `TextLoRA/` 目录（`adapter_model.safetensors`），不在 `FINAL.pt` 内。

**没有 LoRA = LLM 不知道怎么看图。** Projector 把图像特征送进了 LLM 的 embedding 空间，但 LLM 的注意力层（LoRA 作用的地方）没被训练过如何处理这些视觉 token。模型看到一堆"看起来像文字但其实是图像"的 embedding，不知道该怎么处理，于是退回到原始 LLaMA-2 的文本生成行为。

## 加载 LoRA 后的状态

在推理时通过 `PeftModel.from_pretrained()` 加载 LoRA 后：

- 模型加载日志显示 `loading TextLoRA from: output/checkpoints/TextLoRA` ✅
- `unwrap existing TextLoRA adapter before checkpoint load` — 加载流程正确 ✅
- 但 NPU 上 `generate()` 调用**超时或输出为空** ❌

这是因为 `prepare_inputs_for_multimodal` 中处理 `IMAGE_TOKEN_INDEX(-200)` 的逻辑与 PEFT LoRA 包装后的模型在 NPU 上的交互存在兼容性问题。不是 LoRA 权重有问题，是 LoRA + NPU generate 的推理栈问题。

## 修复方案

### 方案 A：推理时将 LoRA merge 到 base 模型（推荐）

不使用 PEFT 的动态 LoRA 推理，而是在加载 checkpoint 后将 LoRA 权重直接**合并**到 base LLM 中，保存为一个新的 `.pt` 文件。合并后的模型就是普通权重，不依赖 PEFT，NPU 可以正常推理。

```python
from peft import PeftModel

# 加载 base + LoRA
model.language.text_encoder = PeftModel.from_pretrained(
    model.language.text_encoder, "TextLoRA"
)
# 合并 LoRA 到 base weights
merged = model.language.text_encoder.merge_and_unload()
model.language.text_encoder = merged
# 保存合并后的完整权重
torch.save(model.state_dict(), "CoastGPT_stage2_merged.pt")
```

合并后保存为单一 `.pt` 文件，推理时直接加载，不需要 PEFT。

### 方案 B：Stage 2 训练时直接全量微调 LLM

如果显存允许，Stage 2 不只用 LoRA，而是全量微调 LLM 的注意力层（或整个 LLM 的最后一层）。这样产出的 checkpoint 本身就是完整权重，不依赖 PEFT。

- 优点：推理简单，兼容性好
- 缺点：训练显存需求增大，checkpoint 体积增大

### 方案 C：修复 NPU + PEFT 交互

排查 `prepare_inputs_for_multimodal` 在 PEFT 包装后的模型上失败的具体原因（可能是 device mapping 或 hook 冲突），打补丁修复。这条路需要深入了解 PEFT 和 NPU 的内部实现。

## 对 Stage 3/4 的影响

- PoC-4 的 FusionPipeline 管线在**无 LoRA** 条件下验证通过（Parser → Gating → Fallback → Dedup 全链路正确）
- Stage 3（检测头训练）不依赖 LoRA——FPN + 三头只用 Vision Encoder 特征，不需要 LLM
- Stage 4（联合训练）需要 LoRA——训练时 PEFT 动态加载没问题（DeepSpeed 原生支持），问题只出现在**推理**侧
- 建议在 Stage 3 期间用方案 A 产出 merged checkpoint，Stage 4 联合训练后再更新

## 相关文件

| 文件 | 说明 |
|------|------|
| `output/checkpoints/FINAL.pt` | Stage 2 产出，3.4GB，含 Vision + Projector |
| `TextLoRA/adapter_model.safetensors` | LoRA adapter 权重，640MB |
| `scripts/verify_poc4_npu.py` | PoC-4 NPU 验证脚本 |
| `Models/fusion_pipeline.py` | FusionPipeline 主流程 |
| `Models/fusion_predictors.py` | LLMPredictor / LLMTextPredictor |

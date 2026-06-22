# CoastGPT Stage2 修复交接文档

生成时间：2026-06-21  
本地工作区：`C:\Users\16606\Documents\Codex\2026-06-10\ssh-ma-user-dev-modelarts-cn`  
远端工作区：`/home/ma-user/work/CoastGPT`  
服务器：`ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:30629`

## 最终验收目标

最终目标不是完成某一次训练，而是恢复并验证 CoastGPT Stage2 的 VQA/VG 能力达到指定指标，然后才能进入后续 Stage3 GeoJSON 输出目标。

硬性验收指标如下：

| 能力 | 数据集 | 必须达到的指标 |
|---|---|---|
| VQA / scene classification | LR-BEN | `label_exact_rate_pct >= 86` |
| VQA / scene classification | HR-BEN | `label_exact_rate_pct >= 92` |
| Visual Grounding | DIOR-RSVG | `Acc@0.25 >= 78.4`, `Acc@0.5 >= 49.7`, `AP@0.5 >= 47.9` |
| Visual Grounding | RSVG | `Acc@0.25 >= 28.4`, `Acc@0.5 >= 14.6`, `AP@0.5 >= 17.6` |

完成定义：

- 必须基于修复后的当前代码重新训练/评测，不能复用旧的失败 summary 作为达标证据。
- 必须有 fresh eval summary，至少覆盖 LR-BEN、HR-BEN、DIOR-RSVG、RSVG。
- VQA 需要报告 `label_exact_rate_pct`、urban/rural 或各类别 recall/confusion。
- VG 需要报告 `Acc@0.25`、`Acc@0.5`、`AP@0.5`。
- 所有目标同时达标后，才可以进入 Stage3 GeoJSON 输出训练。
- 如果任一指标不达标，该轮训练视为失败轮次，按用户规则删除大权重，只保留日志、预测、summary 和脚本。

后续 Stage3 的最终方向：

- 在 Stage2 能力恢复后，再按 `CoastGPT/docs/superpowers/specs/总设计.md` 的方案训练/验证 GeoJSON 格式输出。
- Stage3 目标不能以牺牲 Stage2 基础 VQA/VG 能力为代价。

## 1. 当前目标

继续修复 CoastGPT Stage2，直到达到用户指定指标，或定位到足够具体的根因/阻塞点。

目标指标：

| 任务 | 数据集 | 指标目标 |
|---|---|---|
| VQA/分类 | LR-BEN | `label_exact_rate_pct >= 86` |
| VQA/分类 | HR-BEN | `label_exact_rate_pct >= 92` |
| VG | DIOR-RSVG | `Acc@0.25 >= 78.4`, `Acc@0.5 >= 49.7`, `AP@0.5 >= 47.9` |
| VG | RSVG | `Acc@0.25 >= 28.4`, `Acc@0.5 >= 14.6`, `AP@0.5 >= 17.6` |

当前尚未达到目标。不要在没有新鲜完整评测 summary 的情况下声称达标。

## 2. 连接方式

优先使用本地快捷脚本：

```powershell
.\outputs\connect-modelarts.cmd
```

备用 SSH 命令：

```powershell
ssh -i .\work\modelarts-ssh\CK.pem -p 30629 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.\work\modelarts-ssh\known_hosts ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com
```

远端常用环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0
export HF_HOME=/home/ma-user/work/CoastGPT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. 已完成工作概要

### 3.1 训练和评测基础设施

- 创建并使用了 ModelArts 快速连接脚本。
- 建立了通过 `CoastGPT/.codegraph` 快速检索项目的习惯和辅助脚本。
- 多次监控 Stage3 wavelet/direct 和 Stage2 repair 训练，包括 `train.pid`、`train.log`、NPU 状态、checkpoint 和 eval summary。
- 按用户要求，失败训练轮次评测后删除大权重，只保留日志、summary、预测和脚本。
- 已遵守本地临时目录规则：不再保留 `work\remote-edit`，当前本地检查结果为不存在。

### 3.2 文档和设计

- 审阅并完善过 `CoastGPT/docs/dwt_wavelet_fusion_spec.md`。
- 后续 Stage3 输出 GeoJSON 的方向已转向 `CoastGPT/docs/superpowers/specs/总设计.md` 方案。
- 当前阶段没有直接进入 Stage3 GeoJSON 训练，因为 Stage2 基础能力尚未恢复。

### 3.3 Stage3 wavelet/direct 相关

- 完成过 Stage3 wavelet direct 全量训练监控。
- 处理过 `learnable_direct` 图像空白问题，并进行过 multiband/wavelet adapter 相关 overfit 与 ablation。
- 保存的可视化、临时文件清理和失败权重清理规则已明确。

### 3.4 Stage2 repair 数据

主要训练数据：

```text
/home/ma-user/work/Stage2TargetRepairData_firstonly_bal_20260620_0135
```

该数据由 `/home/ma-user/work/Stage2Data` 派生，包含：

- LR first-turn-only balanced：13200，urban/rural 各 6600
- HR first-turn-only：10658
- RSVG_DIOR：7493
- RSVG：2428
- 总计约 33779

64 样本诊断数据：

```text
/home/ma-user/work/Stage2LR_Overfit64_20260621_160456
```

组成：

- 32 urban
- 32 rural
- first-turn-only 问题：`Is it a rural or an urban area`
- 图像来自 `/home/ma-user/work/Stage2Data/LR_Image` 的 symlink

## 4. 关键代码修复

### 4.1 TextLoRA 双重 PEFT 包装修复

问题：

- `LanguageModel.__init__` 已经通过 `get_peft_model` 创建 LoRA。
- `CoastGPT.custom_load_state_dict` 又对已有 `PeftModel` 调用 `PeftModel.from_pretrained`。
- 这会生成双重前缀 key，例如：

```text
base_model.model.base_model.model...
```

证据：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_dsparams_8npu_20260621_160456
```

该 run 保存出的 adapter key 出现双前缀。大权重已删除，只保留日志和诊断信息。

修复：

```text
/home/ma-user/work/CoastGPT/Models/coastgpt.py
```

`custom_load_state_dict` 在加载 checkpoint TextLoRA 前，会先把已有 `PeftModel` unwrap 回 base text encoder，再加载目标 TextLoRA。

测试：

```text
/home/ma-user/work/CoastGPT/tests/test_stage2_lora_trainable.py
```

### 4.2 DeepSpeed optimizer 显式传入可训练参数

问题：

- Stage2 TextLoRA/多模态参数在 DeepSpeed 初始化中可能没有按预期进入 optimizer。

修复：

```text
/home/ma-user/work/CoastGPT/train_stage_two.py
```

训练时显式将 trainable tensors 传给 DeepSpeed optimizer。

相关验证：

```text
/home/ma-user/work/CoastGPT/tests/test_deepspeed_trainable_parameters.py
```

### 4.3 LLaMA-2 预处理 label 全 mask 修复

问题：

- `Dataset/cap_dataset.py` 中 `preprocess_llama_2` / `preprocess_v1` 使用：

```python
total_len = int(target.ne(tokenizer.pad_token_id).sum())
```

- 训练中 tokenizer 的 `pad_token_id == eos_token_id == 2`。
- 因此真实 EOS 被当成 padding 计数排除，导致 `cur_len != total_len`，最终 `target[:] = IGNORE_INDEX`。
- 结果是图文样本的答案 label 被全部 mask，之前训练几乎没有真实文本答案监督。

修复：

```python
total_len = int(target.shape[0])
```

涉及文件：

```text
/home/ma-user/work/CoastGPT/Dataset/cap_dataset.py
```

验证：

```text
/home/ma-user/work/CoastGPT/tests/test_llama2_preprocess_labels.py
```

该测试验证当 `pad_token == eos_token` 时，`rural` 答案 label 不会被全部 mask。

## 5. 已运行的关键实验和结论

### 5.1 修复前 full LR 评测

Run：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lorafix_dsparams_full_from_output_stage2_8npu_bs1_20260621_1325
```

LR eval：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lorafix_dsparams_lr_eval_20260621_1518
```

结果：

```json
{
  "total": 572,
  "label_exact_rate_pct": 42.30769230769231,
  "label_recall_by_answer": {
    "urban": 1.0,
    "rural": 0.0
  },
  "label_confusion": {
    "urban": {"urban": 242},
    "rural": {"urban": 330}
  },
  "target_pass": false
}
```

结论：模型全预测 urban。

该失败 run 的大权重已删除。

### 5.2 只修 TextLoRA unwrap 后的 64 样本 overfit

Run：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_unwrapfix_8npu_20260621_162058
```

Eval：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_unwrapfix_eval_20260621_163228/LR/eval_summary.json
```

结果：

```json
{
  "total": 64,
  "label_exact_rate": 0.5,
  "label_balanced_accuracy": 0.5,
  "label_recall_by_answer": {
    "urban": 1.0,
    "rural": 0.0
  }
}
```

结论：仍然全预测 urban。该 run 大权重已删除。

### 5.3 修复 label mask 后的 64 样本 overfit

Run：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_labelmaskfix_8npu_20260621_1645
```

Eval：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_labelmaskfix_eval_20260621_165452/LR/eval_summary.json
```

结果：

```json
{
  "total": 64,
  "label_exact_rate": 0.5,
  "label_balanced_accuracy": 0.5,
  "label_recall_by_answer": {
    "urban": 1.0,
    "rural": 0.0
  }
}
```

结论：

- label 监督已经恢复，但 200 iter 仍全 urban。
- 说明 label mask 是真实根因之一，但不是全部问题。
- 该 run 大权重已删除。

### 5.4 clean-loss 64 样本 overfit

目的：

- 关闭 MoE 辅助损失、physics、随机增强。
- 排除“辅助损失压住文本学习”的因素。

配置：

```text
/home/ma-user/work/CoastGPT/Configs/step2_dual_bf16_clean_overfit64.yaml
```

Run：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_cleanloss_8npu_20260621_170337
```

训练证据：

- `mm_moe_aux_loss_weighted = 0`
- `non_text_loss = 0`
- 最后 `text_loss` 降到约 `5.796e-06`

Eval：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_cleanloss_eval_20260621_1726/lr_overfit64_summary_compact.json
```

结果：

```json
{
  "total": 64,
  "exact_rate_pct": 6.25,
  "contains_rate_pct": 51.5625,
  "label_exact_rate_pct": 50.0,
  "label_balanced_accuracy_pct": 50.0,
  "all_unk_rate_pct": 0,
  "label_recall_by_answer": {
    "urban": 0.0,
    "rural": 1.0
  },
  "label_confusion": {
    "urban": {"rural": 32},
    "rural": {"rural": 32}
  }
}
```

结论：

- 训练 teacher-forcing loss 能接近 0，说明训练路径能记住答案 token。
- 生成评测却全部输出 rural。
- 这说明当前核心问题不是“训练完全没有监督”，而是训练 forward/teacher-forcing 与 generate 推理路径不一致，或视觉/问题条件在生成阶段没有有效影响首 token。
- 该 run 大权重已删除。

### 5.5 logits probe

对 clean-loss checkpoint 做首 token logits probe。

结果摘要：

- 多个 urban/rural 图片上，生成首 token 都是 `rural`。
- `rural` token score 约 `30-32`。
- `urban` token score 约 `2.7-4.9`。
- 图片变化没有改变决策方向。

示例：

```json
{
  "sample_id": "364.tif#0:0",
  "answer": "urban",
  "generated": "rural",
  "scores": {
    "urban": 4.875,
    "rural": 30.375
  }
}
```

注意：带前导空格的 `" urban"` / `" rural"` 首 token 都是空格 token `29871`，比较无意义；应比较无前导空格的 `urban` / `rural` token。

结论：

- 生成首 token 几乎被固定偏向 rural。
- 视觉条件没有表现出足够影响。
- 当前最可疑点：训练中 `self.multimodal(data, image_embedding=image_seq)` 与生成中 `self.multimodal.encode_test(...)` 行为不一致，或者 `LanguageModel.generate -> prepare_inputs_for_multimodal` 在生成时的 embedding/attention/position 处理与训练不一致。

## 6. 当前根因判断

目前已经定位到多个真实问题：

1. TextLoRA 曾经被双重 PEFT 包装。
2. Stage2 LLaMA-2 label 曾经因为 `pad_token_id == eos_token_id` 被全部 mask。
3. 修复以上问题后，teacher-forcing overfit 能把 loss 降到接近 0，但生成仍塌缩到单一类别。

当前最重要的未解根因：

```text
训练 teacher-forcing 路径与 generate 推理路径不一致，导致训练能记住答案，但生成时视觉/问题条件没有有效控制首 token。
```

这已经具有架构/推理路径问题特征，不能继续简单加数据或全量训练，否则会浪费 8 卡时间。

## 7. 推荐下一步

优先级从高到低：

### 7.1 固化 logits probe 为脚本或测试

把临时 logits probe 固化到：

```text
Tools/probe_stage2_first_token_logits.py
```

用途：

- 输入 checkpoint、dataset json、image root。
- 输出每个样本首 token 中 `urban`/`rural` 分数。
- 比较同 prompt 不同图片时 logits 是否变化。
- 比较同图片不同 prompt/route text 时 logits 是否变化。

通过这个脚本快速验证后续修复是否真正让视觉条件影响生成。

### 7.2 对齐训练和生成的 multimodal 路径

重点检查：

```text
Models/coastgpt.py
```

训练路径：

```python
image_seq, fused_spatial, pyramid_raw = self.vision.encode_with_spatial(...)
multimodal_embedding = self.multimodal(data, image_embedding=image_seq)
output = self.language(data, multimodal_embedding=multimodal_embedding)
```

生成路径：

```python
image_embedding = self.vision.encode(image)
image_embedding = self.multimodal.encode_test(
    image_embedding,
    task_text_embs=task_text_embs,
    element_text_embs=element_text_embs,
)
return self.language.generate(...)
```

应验证：

- `self.multimodal(...)` 和 `encode_test(...)` 是否完全等价。
- task/element route 在训练和生成是否使用相同文本。
- MoE gate 在生成时是否坍塌到单专家。
- image embedding shape、dtype、数值范围是否一致。

### 7.3 对齐 prompt/token/attention

重点检查：

```text
Dataset/cap_dataset.py
Tools/run_geojson_batch_eval.py
Models/language_model.py
Dataset/conversation.py
```

要确认：

- 训练 prompt 与 eval prompt 完全一致。
- `<image>` token 在训练和生成中位置一致。
- `attention_mask` 插入 image token 后长度和值一致。
- `prepare_inputs_for_multimodal` 在 labels=None 的生成路径没有错误截断。
- prompt 没有因为 `model_max_length` 截断掉关键上下文。

### 7.4 只在 64 样本生成 overfit 通过后再进入全量训练

门槛建议：

- 64 样本 train-set generation `label_exact_rate_pct >= 95`
- urban/rural recall 都接近 1.0
- logits probe 中 urban 图片对 `urban` 分数高于 `rural`，rural 图片反之

如果这个门槛不过，不应启动全量 Stage2 repair。

### 7.5 全量训练策略

在 64 样本 generation overfit 通过后，再回到：

```text
/home/ma-user/work/Stage2TargetRepairData_firstonly_bal_20260620_0135
```

建议先跑小步数 sanity，再跑全量：

1. 20 iter：确认 TextLoRA 和 multimodal 权重更新。
2. 500-1000 iter：先评 LR-BEN。
3. LR-BEN 接近目标后，再评 HR-BEN、DIOR-RSVG、RSVG。
4. 不达标则删除失败轮次大权重。

## 8. 当前远端状态

当前没有活跃的 `train_stage_two.py` 或 `run_stage2_batch_eval.py` 进程。

已清理失败 run 的大权重：

- `stage2_lr_overfit64_dsparams_8npu_20260621_160456`
- `stage2_lr_overfit64_unwrapfix_8npu_20260621_162058`
- `stage2_lr_overfit64_labelmaskfix_8npu_20260621_1645`
- `stage2_lr_overfit64_cleanloss_8npu_20260621_170337`
- `stage2_lorafix_dsparams_full_from_output_stage2_8npu_bs1_20260621_1325`

保留内容：

- `train.log`
- `eval.log`
- `eval_summary.json`
- `predictions.jsonl`
- run 脚本
- 小型诊断 summary

本地临时目录：

```text
work\remote-edit
```

当前不存在。

## 9. 用户规则

必须继续遵守：

1. 本地临时文件用完删除。
2. 不删除 `work\modelarts-ssh`。
3. 每轮训练完成并评测后，如果没有达到目标，删除该轮大权重。
4. 保留日志、预测、summary、manifest、launch script。
5. 不删除源码、原始数据根目录、评测 summary、日志。
6. 不用 full training 替代小样本生成 overfit 的根因验证。

## 10. 常用路径速查

远端代码：

```text
/home/ma-user/work/CoastGPT
```

原始 Stage2 数据：

```text
/home/ma-user/work/Stage2Data
```

修复训练数据：

```text
/home/ma-user/work/Stage2TargetRepairData_firstonly_bal_20260620_0135
```

64 样本诊断数据：

```text
/home/ma-user/work/Stage2LR_Overfit64_20260621_160456
```

最新关键 eval summary：

```text
/home/ma-user/work/CoastGPT/runs/stage2_lr_overfit64_cleanloss_eval_20260621_1726/lr_overfit64_summary_compact.json
```

关键源码：

```text
/home/ma-user/work/CoastGPT/Models/coastgpt.py
/home/ma-user/work/CoastGPT/Models/language_model.py
/home/ma-user/work/CoastGPT/Models/embedding_model_r1.py
/home/ma-user/work/CoastGPT/Dataset/cap_dataset.py
/home/ma-user/work/CoastGPT/Tools/run_stage2_batch_eval.py
/home/ma-user/work/CoastGPT/Tools/run_geojson_batch_eval.py
```

关键测试：

```text
/home/ma-user/work/CoastGPT/tests/test_stage2_lora_trainable.py
/home/ma-user/work/CoastGPT/tests/test_llama2_preprocess_labels.py
/home/ma-user/work/CoastGPT/tests/test_deepspeed_trainable_parameters.py
```

## 11. 接手人应避免的错误

- 不要直接启动全量训练验证运气。
- 不要用约束解码强制 urban/rural 来掩盖根因。
- 不要只看 teacher-forcing loss，必须看 generation eval。
- 不要把 50% balanced accuracy 误认为模型部分可用；当前是单类塌缩。
- 不要保留失败轮次大权重。
- 不要用旧 heartbeat 指向的 `stage2_lr_overfit64_unwrapfix...` 作为当前状态；它已经过时。

## 12. 一句话结论

Stage2 目前已经修复了 TextLoRA 双包装和 label 全 mask 两个真实 bug，但仍存在更核心的训练/生成链路不一致问题：模型能在 teacher-forcing 下把 64 样本 loss 降到接近 0，却在 generation 下全部输出单一类别。因此下一步必须优先定位 `multimodal.encode_test` / `LanguageModel.generate` / prompt-attention 对齐问题，再恢复全量训练。

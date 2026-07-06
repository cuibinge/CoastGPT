# CoastGPT Stage1 修复交接文档 - 2026-07-06

## 1. 当前结论

旧服务器现在连不上。本地已经保留了本轮所有关键补丁文件，路径在：

```text
C:\Users\16606\Documents\Codex\2026-06-26\ssh-i-c-users-16606-ck\work\remote_patch
```

请注意状态分两类：

1. 已上传到旧服务器并验证过的改动：数据增强、评估脚本修复、Stage0 LayerNorm 迁移、semantic adapter 改残差、`pre_out_norm`。
2. 最后一步只保存在本地、还没来得及上传旧服务器的改动：`stage0_global_residual_scale` 全局 Stage0 语义锚点。

本地最终补丁文件已通过：

```powershell
python -m py_compile `
  work\remote_patch\Models__common_arch.py `
  work\remote_patch\Models__embedding_model_r1.py `
  work\remote_patch\Models__stage0_transfer.py `
  work\remote_patch\eval_stage1_m1m6.py `
  work\remote_patch\m1_probe.py
```

## 2. 必须迁移的本地文件

新服务器上把这些文件复制到 CoastGPT 仓库对应位置：

| 本地文件 | 目标路径 |
|---|---|
| `work\remote_patch\Dataset\task_augmentation.py` | `Dataset/task_augmentation.py` |
| `work\remote_patch\Dataset\cap_dataset.py` | `Dataset/cap_dataset.py` |
| `work\remote_patch\Dataset\build_loader.py` | `Dataset/build_loader.py` |
| `work\remote_patch\eval_stage1_m1m6.py` | `scripts/eval_stage1_m1m6.py` |
| `work\remote_patch\m1_probe.py` | `scripts/m1_probe.py` |
| `work\remote_patch\Models__common_arch.py` | `Models/common_arch.py` |
| `work\remote_patch\Models__embedding_model_r1.py` | `Models/embedding_model_r1.py` |
| `work\remote_patch\Models__stage0_transfer.py` | `Models/stage0_transfer.py` |
| `work\remote_patch\Configs__train_dual.yaml` | `Configs/train_dual.yaml` |

重要：`work\remote_patch\Configs__train_dual.yaml` 是最终本地版，包含 `stage0_global_residual_scale`。不要用 `work\remote_patch\Configs\train_dual.yaml` 覆盖它，那个是较早阶段的副本。

## 3. 已做过的空间清理

旧服务器上做过两轮清理：

1. 清理早期 Stage0/Stage1 中间 checkpoint：
   - 删除旧 run 中所有中间 `iter_*` shard 和非最终 consolidated 权重。
   - 保留 `FINAL.pt`、最终 consolidated、日志和评估 JSON。
   - 释放约 `653 GB`。
   - 清理后 `/home/ma-user/work` 约 `578G/2.2T`，占用 `27%`。

2. 停掉失败的新 Stage1 run 并清理：
   - run：`runs/stage1_v3_8npu_20260630_204414`
   - 原大小约 `73G`。
   - 删除 `iter_249`、`iter_499` shard 目录以及对应 consolidated 权重。
   - 保留 `eval_iter249_m1_only/results.json`、`eval_iter499_m1_only/results.json`、日志、配置。
   - 清理后该 run 约 `3.3M`，磁盘回到 `578G/2.2T`。

## 4. 训练和评估中发现的问题

### 4.1 Stage1 数据增强实际没有生效

配置里虽然有：

```yaml
task_augmentation:
  enabled: true
```

但 `Dataset/build_loader.py` 只把 task augmentation 传给 `stage >= 2` 的 `InstructDatasetWithTaskId`。Stage1 实际使用的是 `CaptionDatasetVQA`，所以 yes/no、多选、hard negative 并没有进入训练。

修复：

- `Dataset/build_loader.py` 增加 `_task_aug_dataset_kwargs(config)`。
- Stage1 `CaptionDatasetVQA` 和 Stage>=2 `InstructDatasetWithTaskId` 都接收 augmentation 参数。
- `Dataset/cap_dataset.py` 的 `CaptionDatasetVQA` 里真正调用 `TaskAugmenter`。
- `Dataset/task_augmentation.py` 增加 `caption_keep_ratio`，当前配置是 caption/yes-no/multi-choice/hard-neg = `30/30/30/10`。

已验证过的增强统计：

```text
total_after_aug 977381
counts {'分类': 293214, 'caption': 293215, '问答': 293214, '校验': 97738}
```

### 4.2 `prompt_template: plain` 会让分类/问答问题不可见

`preprocess_plain()` 会把问题替换成只有 `<image>`，导致 yes/no 和 multi-choice 的文本题目进不了模型。

修复：

```yaml
prompt_template: "llava_llama_2"
```

同时在 `CaptionDatasetVQA.__getitem__()` 中把 `task_text` / `element_text` 元数据从 conversation 中剥离，只作为 batch metadata 保留，否则 tokenizer 前处理会被 metadata 干扰。

验证过的 tokenizer/collator sanity：

```text
sample 0 task_text= 分类 element_text= 林地 ids=(109,) answer_tokens=3
sample 1 task_text= 描述 element_text= 无 ids=(107,) answer_tokens=36
sample 3 task_text= 问答 element_text= other ids=(75,) answer_tokens=3
sample 27 task_text= 校验 element_text= mixed ids=(189,) answer_tokens=68
batch_input_ids (4,189)
task_text_ids (4,16)
element_text_ids (4,16)
```

### 4.3 评估脚本一开始没有真正加载项目格式 checkpoint

`iter_*_consolidated.pt` 顶层是：

```text
vision_ckpt
other_ckpt
```

原评估脚本把这个 dict 直接和 `model.state_dict()` 对 key，实际几乎没有加载到训练权重，评估结论不可信。

修复：

- `scripts/eval_stage1_m1m6.py`
- `scripts/m1_probe.py`

遇到 `vision_ckpt` / `rgb_ckpt` / `module` 时，改用：

```python
model.custom_load_state_dict(checkpoint_path, strict=False)
```

并把 `cfg.stage` 设为 `1`，不是 Stage0。

评估日志里现在会显示：

```text
[Eval] Loading project-format checkpoint via CoastGPT.custom_load_state_dict
After loading vision: Missing: []. Unexpected: []
After loading multimodal: Missing: []. Unexpected: []
After loading embed_tokens: Missing: []. Unexpected: []
```

### 4.4 正确加载后，Stage1 仍然 M1 坍塌

失败 run：

```text
runs/stage1_v3_8npu_20260630_204414
```

可信 M1-only 评估结果：

```text
iter_249:
gate_feat_cos_mean = 0.999082
proj_emb_cos_mean  = 0.999764
m1_pass = false

iter_499:
gate_feat_cos_mean = 0.999490
proj_emb_cos_mean  = 0.999762
m1_pass = false
```

这违反 `docs/stage1通过标准.md` 的硬门槛：

```text
M1 proj_emb 跨图 cosine < 0.90
```

因此我停掉了这轮训练，没有让它继续空跑。

## 5. Stage1 坍塌的结构性根因

Stage0 学到的是：

```text
mean(image_seq) -> LayerNorm -> Linear(1024 -> 4096)
```

Stage1 原先只把 Stage0 的 `visual_proj.1.weight` 复制到每个 MoE expert 的 `out_proj.weight`，但没有复制 `visual_proj.0` 的 LayerNorm weight/bias。

同时，`ImageConditionedPooler.forward()` 里有一个更关键的问题：

```python
query_tokens = self.semantic_adapter(query_tokens) * self.adapter_scale
out = self.out_proj(query_tokens)
```

这不是 residual adapter，而是用随机 MLP 输出覆盖了 content query + residual 路径。这样 Stage0 复制来的 Linear 前面接的是随机分布，不是 Stage0 训练时的 `LN(mean(image_seq))` 分布，Stage0 语义初始化基本被洗掉。

## 6. 已做的结构修复

### 6.1 `Models/common_arch.py`

已改：

- `semantic_adapter` 改为小残差：

```python
adapter_delta = self.semantic_adapter(query_tokens) * self.adapter_scale
query_tokens = query_tokens + adapter_delta
```

- 新增 `pre_out_norm`，让 `out_proj` 前的分布兼容 Stage0：

```python
self.pre_out_norm = norm_cls(hidden_size)
out = self.out_proj(self.pre_out_norm(query_tokens))
```

- 新增 `semantic_adapter_scale_init`，默认 `0.05`，避免启动时 adapter 覆盖 Stage0 语义。

- 最后一步本地新增但未上传旧服务器：`stage0_global_residual_scale`：

```python
global_content = image_embs.float().mean(dim=1).to(dtype=image_embs.dtype)
global_out = self.out_proj(self.pre_out_norm(global_content)).unsqueeze(1)
out = out + global_out * self.stage0_global_residual_scale
```

这个全局锚点的目的：显式保留 Stage0 的 `mean(image_seq)->LN->Linear` 语义路径，避免 MoE query-token 路径把 Stage0 初始化稀释掉。

### 6.2 `Models/stage0_transfer.py`

已改：

- 继续复制 `visual_proj.1.weight` 到每个 expert 的 `out_proj.weight`。
- 新增复制：

```text
visual_proj.0.weight -> expert.pre_out_norm.weight
visual_proj.0.bias   -> expert.pre_out_norm.bias
```

这样 Stage1 expert 的 `pre_out_norm + out_proj` 与 Stage0 projector 保持一致。

### 6.3 `Models/embedding_model_r1.py`

已改：

- 把配置里的 `semantic_adapter_scale_init` 传给 `MoEProjection`。
- 本地最终版还把 `stage0_global_residual_scale` 传给 `MoEProjection`。

### 6.4 `Configs/train_dual.yaml`

最终本地版包含：

```yaml
prompt_template: "llava_llama_2"

task_augmentation:
  enabled: true
  caption_keep_ratio: 0.30
  yn_ratio: 0.30
  mc_ratio: 0.30
  hard_neg_ratio: 0.10
  seed: 42

moe_proj:
  semantic_adapter_scale_init: 0.05
  stage0_global_residual_scale: 1.0
```

## 7. 已完成的验证

### 7.1 本地最终补丁语法检查

通过：

```powershell
python -m py_compile `
  work\remote_patch\Models__common_arch.py `
  work\remote_patch\Models__embedding_model_r1.py `
  work\remote_patch\Models__stage0_transfer.py `
  work\remote_patch\eval_stage1_m1m6.py `
  work\remote_patch\m1_probe.py
```

### 7.2 旧服务器结构验证

在旧服务器上，`pre_out_norm` 和 Stage0 norm 迁移版本跑过结构验证：

```text
structure_transfer_ok
```

这个验证确认：

- `ImageConditionedPooler.forward()` 输出 shape 正常。
- `adapter_scale` 初始值为 `0.05`。
- `load_stage0_projection_into_moe()` 能复制 `out_proj.weight`。
- `load_stage0_projection_into_moe()` 能复制 `pre_out_norm.weight/bias`。

注意：这个验证发生在 `stage0_global_residual_scale` 最后本地补丁之前；最后的 global residual 只做过本地 `py_compile`。

### 7.3 MoE routing 回归测试

旧服务器通过：

```bash
python -m pytest -q \
  tests/test_moe_projection_routing.py::test_training_balanced_topk_keeps_each_branch_expert_active_after_warmup \
  tests/test_moe_projection_routing.py::test_hard_load_balance_loss_penalizes_topk_expert_starvation -q
```

结果：

```text
.. [100%]
```

### 7.4 初始化态 M1 探针

在旧服务器上，`pre_out_norm + residual adapter + Stage0 norm transfer` 版本的初始化态探针结果：

```text
copied_experts 4
image_seq cos_mean   = 0.956641
gate_feat cos_mean   = 0.924804
expert0 cos_mean     = 0.959290
moe_proj cos_mean    = 0.929191
```

对比旧训练 checkpoint 的 `moe_proj=0.999+`，已经明显改善，但仍未达到 M1 `<0.90`。所以最后我补了 `stage0_global_residual_scale=1.0`，但旧服务器断连前还没来得及上传/实测。

## 8. 新服务器建议执行顺序

### 8.1 复制补丁文件

把第 2 节表格里的本地文件复制到新服务器 CoastGPT 仓库。

### 8.2 基础检查

```bash
cd /home/ma-user/work/CoastGPT
python -m py_compile \
  Models/common_arch.py \
  Models/stage0_transfer.py \
  Models/embedding_model_r1.py \
  scripts/eval_stage1_m1m6.py \
  scripts/m1_probe.py
```

### 8.3 跑结构验证

建议先跑一个小脚本，确认：

```text
ImageConditionedPooler has pre_out_norm
adapter_scale == 0.05
load_stage0_projection_into_moe copies:
  out_proj.weight
  pre_out_norm.weight
  pre_out_norm.bias
```

旧服务器上这个验证的核心输出是：

```text
structure_transfer_ok
```

### 8.4 跑初始化态 M1 探针

目标：不训练，构建 Stage1、执行 `reset_gate_and_experts()`、加载 Stage0 FINAL、复制 Stage0 projector，然后计算：

```text
image_seq / stage0_direct / gate_feat / expert0 / moe_proj cosine
```

如果 `moe_proj cos_mean` 仍然 `> 0.90`，不要开全量训练。先调：

```yaml
moe_proj:
  stage0_global_residual_scale: 1.0
```

可以尝试 `1.5` 或 `2.0`，但要观察 token diversity，避免所有 token 变成同一个 global token。

### 8.5 2-step 8NPU quick sanity

```bash
cd /home/ma-user/work/CoastGPT
MAX_DEBUG_ITERS=2 NUM_GPUS=8 BATCH_SIZE=8 WORKERS=4 bash scripts/train_stage_one.sh
```

期望：

- 能构建 dataset。
- 能打印 `Copied Stage0 visual_proj.1.weight into 4 MoE experts.`
- 最好补一条日志，确认也复制了 `visual_proj.0` LayerNorm。
- 2 iter 正常退出，无 Traceback/OOM。

### 8.6 正式 Stage1 重训

```bash
cd /home/ma-user/work/CoastGPT
NUM_GPUS=8 BATCH_SIZE=8 WORKERS=4 bash scripts/train_stage_one.sh
```

第一个 checkpoint 通常是 `iter_249`。合并完成后立刻跑：

```bash
python scripts/eval_stage1_m1m6.py \
  --checkpoint runs/<new_stage1_run>/checkpoints/iter_249_consolidated.pt \
  --config runs/<new_stage1_run>/config.json \
  --device npu:0 \
  --max-images 4 \
  --m1-only \
  --output runs/<new_stage1_run>/checkpoints/eval_iter249_m1_only
```

判定：

- `proj_emb_cos_mean < 0.90`：继续训练到后续 checkpoint，并开始 M2-M6。
- `proj_emb_cos_mean >= 0.90`：立刻停训，不要空跑。优先调 `stage0_global_residual_scale` 或检查 global residual 是否真的在目标服务器版本中生效。

## 9. 旧服务器相关路径

SSH 旧连接命令原来是：

```text
ssh -i "C:\Users\16606\Documents\Codex\2026-06-26\ssh-i-c-users-16606-ck\work\ssh\CK.codex.1a9587270caf424286a82461d8648462.pem" -p 30629 -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com
```

旧服务器 repo：

```text
/home/ma-user/work/CoastGPT
```

关键 Stage0 权重：

```text
runs/stage0_mse_8npu_20260629_121909/checkpoints/FINAL.pt
```

旧服务器上已失败并停止的 Stage1 run：

```text
runs/stage1_v3_8npu_20260630_204414
```

## 10. 交接提醒

最重要的坑：

1. 不要再用 `prompt_template: plain` 训练 Stage1 yes/no、多选任务。
2. 不要相信未修复评估脚本得到的 M1-M6，因为它可能没有真正加载项目格式 checkpoint。
3. 不要只复制 Stage0 Linear，不复制 LayerNorm。
4. 不要让 `semantic_adapter` 覆盖视觉 content token；它必须是 residual。
5. 第一份 checkpoint 出来就跑 M1，`proj_emb` 还在 `0.999+` 就停。


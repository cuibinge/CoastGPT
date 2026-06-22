# CoastGPT MoE 升级路线：从全局语义路由到 Query-Level Local Routing

## 1. 当前设计定位

你现有的 MoE 路由不是简单的“图像 mean pooling 后直接交给专家”，而是一个更合理的结构：

```text
image_embs [B,196,768]
   └─ mean(dim=1) → img_gate [B,256]

task_text_embs [B,16,768]
   └─ attention pool + projection → t_embs [B,256]

element_text_embs [B,16,768]
   └─ attention pool + projection → e_embs [B,256]

physical_prompts [B,64,768]
   └─ projection → phy_gate [B,256]
```

当前两阶段路由大致为：

```text
img_gate + t_embs + phy_gate → task_gate    → expert[0], expert[1]
img_gate + e_embs + phy_gate → element_gate → expert[2], expert[3]
```

然后对 4 个专家做 `top-k(k=2)`，每个被选中的 expert 处理的是完整 token 序列：

```text
z_tilde = [image_embs | physical_prompts]
        = [B, 196 + Lp, 768]
```

每个 expert 内部通过 AttnPooler 输出：

```text
expert_i(z_tilde) → [B,144,4096]
```

最终输出：

```text
final_output = Σ dispatch_weights[i] * expert_i(z_tilde)
             → [B,144,4096]
```

因此，当前架构更准确地说是：

> **语义条件化的全局 MoE 路由。**

它的优点是稳定、可解释、实现成本低；主要短板是所有 144 个输出 query token 共享同一组 expert 权重，路由粒度仍然是样本级，而不是区域级或 query 级。

---

## 2. 升级目标

不建议直接推翻当前结构，改成复杂 token-dispatch MoE。更合理的目标是：

```text
保留当前全局语义路由的稳定性
+
增加 query-level local routing 的空间适应性
+
增加 shared expert residual 的兜底能力
```

目标结构：

```text
Global Semantic Gate: 负责给整张图提供专家先验
Local Query Gate:     负责给每个 spatial query 分配专家权重
Shared Expert:        负责在 gate 选错时保留基础视觉信息
```

最终从：

```text
dispatch_weights: [B,4]
```

升级为：

```text
dispatch_weights: [B,144,4]
```

也就是每个输出 query token 都可以拥有不同的专家混合权重。

---

## 3. 总体架构图

```text
image_embs [B,196,768]
physical_prompts [B,64,768]
task_text / element_text
        │
        ├───────────────────────────────────────────────┐
        │                                               │
        ▼                                               ▼
Global Semantic Gate                             z_tilde 构造
输入: img_gate + task + element + phy             [image_embs | physical_prompts]
输出: global_logits [B,4]                         [B,196+Lp,768]
        │                                               │
        │                                               ├───────────────┐
        │                                               │               │
        │                                               ▼               ▼
        │                                      Shared Expert       Expert[0..3]
        │                                      [B,144,4096]       each [B,144,4096]
        │                                               │               │
        │                                               │               ▼
        │                                               │        expert_outs
        │                                               │        [B,144,4,4096]
        │                                               │
        ▼                                               ▼
Local Query Gate  ◄──────────────────────────── query_context / shared_out
输出: local_logits [B,144,4]
        │
        ▼
routing_logits = local_logits + α * global_logits[:,None,:]
        │
        ▼
routing_weights = softmax(routing_logits, dim=-1)
        │
        ▼
moe_out = Σ routing_weights[:,:,e,None] * expert_outs[:,:,e,:]
        │
        ▼
final_output = shared_out + β * moe_out
        │
        ▼
[B,144,4096] → LLaMA / Cross-Attention / GeoJSON generation
```

---

## 4. 升级路线

### Step 1：增加 Shared Expert Residual

当前结构是：

```python
final_output = sum(dispatch_weights[:, i] * expert_outputs[i]
                   for i in selected_experts)
```

建议先改成：

```python
shared_out = self.shared_expert(z_tilde)  # [B,144,4096]
moe_out = sum(dispatch_weights[:, i, None, None] * expert_outputs[i]
              for i in selected_experts)

final_output = shared_out + beta * moe_out
```

作用：

```text
1. gate 选错时，shared expert 仍然提供基础视觉表达；
2. 降低 MoE 路由早期训练不稳定；
3. 对坐标、GeoJSON、图像描述任务都有兜底作用；
4. 改动小，适合作为第一步。 
```

建议参数：

```text
beta 初始值：0.1 ~ 0.3
训练稳定后可升到：0.5 ~ 1.0
```

---

### Step 2：训练期 Soft Routing，推理期 Top-K

当前如果训练期就直接 `topk(k=2)`，容易出现早期路由选错后专家无法学习的问题。

建议：

```text
训练前期：soft routing，4 个 expert 都参与
训练中期：soft top-k 或 temperature annealing
训练后期：top-k(k=2)
推理阶段：top-k(k=2)
```

伪代码：

```python
if self.training and global_step < warmup_steps:
    weights = torch.softmax(gate_logits / tau, dim=-1)  # [B,4]
    selected_experts = all_experts
else:
    top_weights, top_indices = torch.topk(gate_weights, k=2, dim=-1)
```

建议 temperature 调度：

```text
tau = 2.0 → 1.0 → 0.7
```

作用：

```text
1. 防止早期 hard top-k 把正确专家屏蔽掉；
2. 让所有专家都有训练机会；
3. 后期仍能保留 top-k 的计算效率。
```

---

### Step 3：把 144 个 Query 改成有空间含义的 Spatial Queries

当前 expert 输出为：

```text
[B,144,4096]
```

如果 144 个 query 是普通 learned queries，它们未必有明确空间含义。对于坐标和 GeoJSON 任务，更推荐：

```text
144 query = 12 × 12 spatial queries
```

每个 query 对应图像中的一个固定区域：

```text
query_0_0   → 左上区域
query_0_1   → 上方偏左区域
...
query_11_11 → 右下区域
```

建议 query 构造：

```python
# base_query: [12,12,D]
# pos_2d:    [12,12,D]
spatial_queries = base_query + pos_2d
spatial_queries = spatial_queries.reshape(144, D)
```

作用：

```text
1. 让 [B,144,4096] 具有空间可解释性；
2. 让 LLaMA 更容易把某个 summary token 和图像位置对应起来；
3. 为后续 query-level local gate 提供清晰路由单位。
```

---

### Step 4：增加 Query-Level Local Gate

当前路由权重是：

```text
dispatch_weights: [B,4]
```

所有 144 个 query token 共享同一组专家权重。

升级后改为：

```text
dispatch_weights: [B,144,4]
```

也就是每个 query token 都能选择不同专家。

伪代码：

```python
# expert_outputs: [B,144,4,4096]
# shared_out:     [B,144,4096]
# global_logits:  [B,4]

local_logits = self.local_gate(shared_out)  # [B,144,4]

routing_logits = local_logits + alpha * global_logits[:, None, :]
routing_weights = torch.softmax(routing_logits, dim=-1)  # [B,144,4]

moe_out = (routing_weights[..., None] * expert_outputs).sum(dim=2)
final_output = shared_out + beta * moe_out
```

作用：

```text
1. 不同空间 query 可以走不同 expert；
2. 混合场景中，养殖区、水体、岸线、建筑可以拥有不同专家偏好；
3. 仍然不需要做复杂 token dispatch，因为所有专家仍可并行输出 [B,144,D]；
4. 比完整 token-level MoE 更容易实现和训练。
```

---

### Step 5：Global Gate 作为 Soft Bias，而不是 Hard Mask

不建议：

```text
Global Gate Top-K → Local Gate 只能在 Top-K 中选择
```

因为一旦 global gate 漏掉正确专家，local gate 无法补救。

推荐：

```python
routing_logits = local_logits + alpha * global_logits[:, None, :]
```

含义：

```text
local_logits:  每个 query 自己判断专家

global_logits: 整张图提供场景级专家先验

alpha:         控制 global prior 的影响强度
```

建议 alpha 调度：

```text
训练早期：alpha = 0.0
训练中期：alpha = 0.2
训练后期：alpha = 0.5
```

不要一开始设置过大。早期 gate 没学好时，过强 global prior 会干扰 local routing。

---

### Step 6：可选：统一 Global Gate 输入，减少专家硬分组

当前结构中：

```text
task_gate    → expert[0], expert[1]
element_gate → expert[2], expert[3]
```

这个设计可解释性强，但灵活性稍弱。可以升级为：

```python
global_input = torch.cat([img_gate, t_embs, e_embs, phy_gate], dim=-1)
global_logits = self.global_gate(global_input)  # [B,4]
```

如果仍想保留 task/element 分工，可以采用 bias 形式：

```python
base_logits = self.global_gate(global_input)  # [B,4]
task_bias = self.task_gate(torch.cat([img_gate, t_embs, phy_gate], dim=-1))
elem_bias = self.element_gate(torch.cat([img_gate, e_embs, phy_gate], dim=-1))

global_logits = base_logits + task_bias + elem_bias
```

作用：

```text
1. 专家不再被强行固定为 task 组和 element 组；
2. 允许任务、要素、物理信息共同影响所有专家；
3. 保留原有 task/element 语义，但降低硬分组限制。
```

---

## 5. 推荐最终结构

最终建议版本：

```text
Visual Tokens X [B,196,768]
Physical Prompts P [B,Lp,768]
Task Text / Element Text
        │
        ├── Global Semantic Gate
        │       输入: image_global + task_emb + element_emb + physical_emb
        │       输出: global_logits [B,4]
        │
        ├── Shared Expert
        │       输入: z_tilde = [X | P]
        │       输出: shared_out [B,144,4096]
        │
        ├── Expert[0..3]
        │       输入: z_tilde = [X | P]
        │       输出: expert_outs [B,144,4,4096]
        │
        └── Query-Level Local Gate
                输入: shared_out 或 spatial query context
                输出: local_logits [B,144,4]

routing_logits = local_logits + α * global_logits[:,None,:]
routing_weights = softmax(routing_logits, dim=-1)

final_output = shared_out + β * Σ routing_weights[:,:,e,None] * expert_outs[:,:,e,:]
```

---

## 6. 损失函数建议

总损失可以设计为：

```text
L_total =
    L_lm
  + λ_coord   * L_coord
  + λ_detmask * L_det_or_mask
  + λ_balance * L_balance
  + λ_entropy * L_entropy
  + λ_zloss   * L_zloss
  + λ_align   * L_attention_align
```

各项作用：

| Loss | 作用 |
|---|---|
| `L_lm` | 语言生成 / GeoJSON 生成主损失 |
| `L_coord` | 坐标 token 或中心点监督 |
| `L_det_or_mask` | 检测、分割、边界监督，可选 |
| `L_balance` | 防止专家负载塌缩 |
| `L_entropy` | 防止 gate 过早变得过于自信 |
| `L_zloss` | 稳定 gate logits |
| `L_attention_align` | 可选，让坐标 token 生成时关注目标区域 |

建议初始权重：

```text
λ_balance = 0.01 ~ 0.05
λ_entropy = 0.001 ~ 0.01
λ_zloss   = 0.001
λ_coord   = 0.5 ~ 1.0
λ_align   = 0.1，可选
```

---

## 7. 训练策略

### 阶段 1：稳定当前 MoE

目标：验证 shared expert 和 soft routing 是否改善稳定性。

```text
1. 保留当前 global routing；
2. 加 shared expert residual；
3. 训练期 soft routing，推理期 top-k；
4. 不引入 local gate。
```

推荐实验：

```text
A0: 当前模型
A1: 当前模型 + shared expert
A2: 当前模型 + soft routing
A3: 当前模型 + shared expert + soft routing
```

---

### 阶段 2：Spatial Query 化

目标：让 144 个输出 token 具备空间含义。

```text
1. 将 144 learned queries 改为 12×12 spatial queries；
2. 给每个 query 加 2D position embedding；
3. 保持路由仍为 [B,4]，先不加 local gate。
```

推荐实验：

```text
B0: learned queries
B1: 12×12 spatial queries
B2: 12×12 spatial queries + 2D position embedding
```

主要指标：

```text
坐标误差
GeoJSON polygon IoU
边界 Hausdorff distance
region grounding accuracy
```

---

### 阶段 3：Query-Level Local Gate

目标：从样本级专家权重升级到 query 级专家权重。

```text
1. 增加 local_gate(shared_out) → [B,144,4]
2. global_logits 作为 soft bias；
3. 不使用 hard global mask；
4. 保留 shared expert residual。
```

推荐实验：

```text
C0: global routing [B,4]
C1: local routing [B,144,4]
C2: local routing + global soft bias
C3: local routing + global soft bias + shared expert
```

---

### 阶段 4：坐标与 GeoJSON 专项训练

目标：让模型真正学会“目标在哪里”。

```text
1. 使用归一化图像坐标，而不是绝对经纬度；
2. 坐标离散成 token，例如 <x_0123><y_0345>；
3. 使用 random crop / resize / flip / copy-paste，并同步更新坐标标签；
4. 如果有 mask 或 bbox，加入 detection / segmentation 辅助 loss；
5. 最终 GeoJSON 坐标由几何模块或后处理转换，不建议完全依赖 LLM 直接吐 polygon 顶点。
```

---

## 8. 推荐 Ablation 表

| 实验编号 | Shared Expert | Soft Routing | Spatial Query | Local Gate | Global Bias | 目标 |
|---|---:|---:|---:|---:|---:|---|
| A0 | 否 | 否 | 否 | 否 | 否 | 当前 baseline |
| A1 | 是 | 否 | 否 | 否 | 否 | 验证兜底路径 |
| A2 | 是 | 是 | 否 | 否 | 否 | 验证训练稳定性 |
| B1 | 是 | 是 | 是 | 否 | 否 | 验证空间 query |
| C1 | 是 | 是 | 是 | 是 | 否 | 验证 local routing |
| C2 | 是 | 是 | 是 | 是 | 是 | 最终推荐版本 |

---

## 9. 预期收益与风险

### 预期收益

```text
1. gate 选错时不再全局崩溃；
2. 不同空间 query 可以使用不同专家；
3. 对混合遥感场景更友好；
4. 对坐标预测和 GeoJSON 输出更友好；
5. 保留当前架构的语义路由可解释性。
```

### 主要风险

```text
1. local gate 可能导致专家负载不均；
2. 训练初期 routing 容易抖动；
3. 计算量增加，因为多个专家可能都要输出 [B,144,D]；
4. 如果 spatial query 没有设计好，local gate 的空间含义仍然不强。
```

### 对应缓解

```text
1. 使用 shared expert residual；
2. 训练前期 soft routing；
3. global gate 只做 soft bias，不做 hard mask；
4. 加 load balance / entropy / zloss；
5. alpha warmup；
6. query 使用 12×12 spatial queries + 2D position embedding。
```

---

## 10. 最小可行改动顺序

如果只做最小改动，建议按这个顺序：

```text
1. 加 shared expert residual
2. 训练期 soft routing，推理期 top-k
3. 144 query 改为 12×12 spatial queries
4. 加 query-level local gate [B,144,4]
5. global_logits 作为 local gate 的 soft bias
6. 再考虑统一 global gate，减少 task/element 硬分组
```

最推荐的中期目标：

```text
shared_out + query-level local routing + global soft bias
```

这不是推翻现有 MoE，而是在当前设计上增加局部适应能力。

---

## 11. 结论

当前架构是一个稳定、可解释的 **semantic-conditioned global MoE routing**。它的 expert 输入仍然是完整视觉 token，因此并没有把空间信息直接池化掉。

但它的 routing 权重是 `[B,4]`，所有输出 query token 共享同一组专家权重。对于遥感图像中的养殖区、水体、岸线、建筑等混合场景，这种全局路由会限制局部区域的专家分工。

推荐升级路线是：

```text
当前全局 MoE
→ shared expert residual
→ soft routing
→ 12×12 spatial queries
→ query-level local gate [B,144,4]
→ global gate as soft bias
```

最终目标不是复杂的 token-dispatch MoE，而是一个更稳的折中版本：

> **Global Semantic Prior + Query-Level Local Routing + Shared Expert Residual**

这版最适合在不大幅推翻 CoastGPT 现有结构的前提下，提高坐标定位、空间理解和 GeoJSON 生成能力。

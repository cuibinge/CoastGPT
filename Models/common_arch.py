from_pretrained
collections
import OrderedDict
from typing import Callable, List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch_npu


class TextProjHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if hidden_dim != in_dim:
            self.in_proj = nn.Linear(in_dim, hidden_dim)
            self.in_norm = nn.LayerNorm(hidden_dim)
        else:
            self.in_proj = self.in_norm = None
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        if self.in_proj is not None:
            x = self.in_norm(self.in_proj(x))
        return self.proj(x)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        in_dim = kwargs.pop("in_dim", 4096)
        hidden_dim = kwargs.pop("hidden_dim", 768)
        out_dim = kwargs.pop("out_dim", 768)
        model = cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        ckpt = torch.load(path)
        if "text_projection.weight" in ckpt.keys():
            new_ckpt = {"weight": ckpt["text_projection.weight"]}
            model.proj.load_state_dict(new_ckpt)
            del new_ckpt
        del ckpt
        return model


class RgbProjHead(TextProjHead):
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        in_dim = kwargs.pop("in_dim", 4096)
        hidden_dim = kwargs.pop("hidden_dim", 1024)
        out_dim = kwargs.pop("out_dim", 768)
        model = cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        ckpt = torch.load(path)
        if "visual_projection.weight" in ckpt.keys():
            new_ckpt = {"weight": ckpt["visual_projection.weight"]}
            model.proj.load_state_dict(new_ckpt)
            del new_ckpt
        del ckpt

        return model


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2):
        super().__init__()
        self.layers = [nn.Linear(in_channels, out_channels)]
        for _ in range(layers):
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(out_channels, out_channels))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        org_dtype = x.dtype
        org_module_dtype = self.layers[0].weight.dtype

        self.layers.to(torch.float32)
        x = x.to(torch.float32)

        x = self.layers(x)
        x.to(org_dtype)
        self.layers.to(org_module_dtype)
        return x


class AttnPooler(nn.Module):
    """
    Attention Pooler

    Args:
        hidden_size: hidden size of the model
        num_layers: number of layers
        num_attention_heads: number of attention heads
        encoder_hidden_size: hidden size of the encoder
        num_query: number of query vectors
        norm_layer: normalization layer
        output_size: output size of the model
    """

    def __init__(
            self,
            num_query: int,
            num_layers: int,
            num_attention_heads: int,
            encoder_hidden_size: int,
            hidden_size: int,
            output_size: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            checkpoint: bool = False,
            stage_num: Union[List, int] = [64, 48, 32],  # [64, 48, 32]
            split_part: List = [256, 256, 256],  # [256,256, 256]
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.stage_num = stage_num
        self.split_part = split_part

        self.query = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.01, mean=0.0)

        if encoder_hidden_size != hidden_size:
            self.in_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.in_proj = None

        self.layers = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    is_cross_attention=True,
                    ls_init_value=1e-1,  # 🌟 关键：极小的初始化值防止能量爆炸
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            image_embs: torch.Tensor,
    ) -> torch.Tensor:
        if self.in_proj is not None:
            image_embs = self.in_proj(image_embs)

        query_tokens = self.query.expand(image_embs.size(0), -1, -1)

        if isinstance(self.stage_num, int):
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.num_query // self.stage_num, dim=1
            )
            stage_query_sizes = [
                self.num_query // self.stage_num,
                self.num_query // self.stage_num,
                self.num_query // self.stage_num,
            ]
        else:
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.stage_num, dim=1
            )
            stage_query_sizes = list(self.stage_num)

        # Robust image split: if preset split_part doesn't match sequence length, derive sizes proportionally to query splits
        L = image_embs.size(1)
        preset_sum = sum(self.split_part) if isinstance(self.split_part, (list, tuple)) else None
        if isinstance(self.split_part, (list, tuple)) and preset_sum == L:
            split_sizes = list(self.split_part)
        else:
            base = sum(stage_query_sizes)
            # Avoid division by zero
            if base == 0:
                split_sizes = [L // 3, L // 3, L - 2 * (L // 3)]
            else:
                # Proportional allocation, ensure exact sum to L
                sizes = [int(round(L * s / base)) for s in stage_query_sizes]
                diff = L - sum(sizes)
                # Adjust last bucket to absorb rounding difference
                sizes[-1] += diff
                split_sizes = sizes

        stage1_image, stage2_image, stage3_image = torch.split(
            image_embs, split_sizes, dim=1
        )

        all_tokens = []
        spatial_attns = []
        for sub_token, sub_image in zip(
                [stage1_query, stage2_query, stage3_query],
                [stage1_image, stage2_image, stage3_image],
        ):
            q_len = sub_token.size(1)
            cat_embs = torch.cat([sub_token, sub_image], dim=1)
            # cat_embs = sub_image
            cat_embs = cat_embs.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
            sub_token = sub_token.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

            for layer in self.layers:
                sub_token = layer(sub_token, cat_embs, cat_embs)

            if not self.training and hasattr(self.layers[-1], "_last_attn_weights"):
                # attn 形状: [B, q_len, q_len + img_len]
                attn = self.layers[-1]._last_attn_weights
                # 只保留对图像部分 (sub_image) 的注意力，并对 query 维度求平均
                img_attn = attn[:, :, q_len:].mean(dim=1)  # -> [B, img_len]
                spatial_attns.append(img_attn)

            sub_token = sub_token.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)
            all_tokens.append(sub_token)

        if not self.training and len(spatial_attns) == 3:
            full_spatial_attn = torch.cat(spatial_attns, dim=1)  # -> [B, L]
            self._last_spatial_attn = full_spatial_attn[0].cpu()

        query_tokens = torch.cat(all_tokens, dim=1)
        out = self.out_proj(query_tokens)
        return out

        # cat_embs = torch.cat([query_tokens, image_embs], dim=1)

        # cat_embs = cat_embs.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
        # query_tokens = query_tokens.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

        # for layer in self.layers:
        #     if self.checkpoint and not torch.jit.is_scripting():
        #         query_tokens = checkpoint(layer, query_tokens, cat_embs, cat_embs)
        #     else:
        #         query_tokens = layer(query_tokens, cat_embs, cat_embs)

        # query_tokens = query_tokens.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)
        # out = self.out_proj(query_tokens)

        # return out


# 在 common_arch.py 中定义 RMSNorm (类似 Llama-2 实现)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 只缩放，不平移，保留特征显著性
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


class MoEProjection(nn.Module):
    """
    任务感知的稀疏混合专家投影层 (Sparse MoE)
    每个 token 根据任务和图像特征选择 top-k 专家，专家独立处理被分配到的 token。
    """

    def __init__(
            self,
            num_experts: int,
            num_query: int,
            num_layers: int,
            num_attention_heads: int,
            encoder_hidden_size: int,
            hidden_size: int,
            output_size: int,
            num_tasks: int = 3,
            task_dim: int = 256,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            checkpoint: bool = False,
            top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保不超过专家数
        self.num_tasks = num_tasks
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.output_size = output_size

        # 1. 专家列表
        self.experts = nn.ModuleList([
            AttnPooler(
                num_query=num_query,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                encoder_hidden_size=encoder_hidden_size,
                hidden_size=hidden_size,
                output_size=output_size,
                norm_layer=norm_layer,
                checkpoint=checkpoint,
            ) for _ in range(num_experts)
        ])

        # 2. 任务嵌入
        self.task_embedding = nn.Embedding(num_tasks, task_dim)

        # 3. 联合门控网络
        combined_dim = encoder_hidden_size + task_dim
        self.gate_norm = nn.LayerNorm(combined_dim)
        self.gate = nn.Linear(combined_dim, num_experts)

        # 4. 输出增益和归一化
        self.output_gain = nn.Parameter(torch.ones(output_size) * 0.3)
        self.final_norm = RMSNorm(output_size)

        # 5. 辅助损失系数（可调节）
        self.aux_balance_coef = 1.0
        self.aux_entropy_coef = 0.00001
        self.aux_zloss_coef = 0.00001

        # 用于缓存统计信息
        self._gate_stats = {}

    def forward(self, image_embs: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        image_embs: [B, L, C] 视觉特征序列
        task_ids: [B] 任务ID
        """
        B, L, C = image_embs.shape

        # --- 任务特征融合 ---
        t_embs = self.task_embedding(task_ids)  # [B, task_dim]
        t_embs_expanded = t_embs.unsqueeze(1).expand(-1, L, -1)  # [B, L, task_dim]
        combined = torch.cat([image_embs, t_embs_expanded], dim=-1)  # [B, L, C+task_dim]

        # --- 门控计算 ---
        # 使用 float32 计算门控以提高稳定性
        gate_input = combined.float()
        gate_logits = self.gate(self.gate_norm(gate_input))  # [B, L, num_experts]
        gate_logits = torch.clamp(gate_logits, min=-15.0, max=15.0)  # 防止溢出

        # 计算 softmax 权重（用于辅助损失和最终加权）
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, L, E]

        # --- 稀疏路由：选择 top-k 专家 ---
        top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # [B, L, k], [B, L, k]

        # --- 计算辅助损失统计（所有 token 参与，不梯度传递）---
        with torch.no_grad():
            # importance: 每个专家的平均权重（所有 token 上的 softmax 权重）
            importance = gate_weights.view(-1, self.num_experts).mean(dim=0)  # [E]

            # load: 每个专家被选为 top-1 的频率（用于负载均衡）
            top1_indices = top_indices[..., 0]  # [B, L]
            load = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
            for i in range(self.num_experts):
                load[i] = (top1_indices == i).float().mean()

            # 熵和 z-loss
            probs = gate_weights.float()
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-6), dim=-1))
            zloss = torch.mean(torch.logsumexp(gate_logits.float(), dim=-1) ** 2)

        self._gate_stats = {
            "importance": importance,
            "load": load,
            "entropy": entropy,
            "zloss": zloss,
        }

        # --- 初始化输出累加器 ---
        final_output = torch.zeros(B, self.num_query, self.output_size, device=image_embs.device,
                                   dtype=image_embs.dtype)

        # --- 对于每个专家，收集被分配到的 token 并处理 ---
        for expert_id in range(self.num_experts):
            # 找出哪些 token 选择了当前专家（在 top-k 中）
            mask = (top_indices == expert_id)  # [B, L, k] -> 但需要沿着 k 维度检查，所以用 any
            # 更高效：生成 token 级别的选择掩码
            # 我们遍历所有 token，但为了向量化，可以构建一个索引列表
            # 方法：将 top_indices 展开，构建 (batch, token) -> expert 的映射
            # 我们使用 torch.where 来收集所有选中当前专家的 token 索引

            # 生成所有 token 的坐标
            batch_idx = torch.arange(B, device=image_embs.device).view(-1, 1).expand(B, L).reshape(-1)
            token_idx = torch.arange(L, device=image_embs.device).view(1, -1).expand(B, L).reshape(-1)
            # 对应的 top-k 专家和权重
            expert_ids = top_indices.reshape(-1, self.top_k)  # [B*L, k]
            expert_weights = top_weights.reshape(-1, self.top_k)  # [B*L, k]

            # 找到当前专家被选中的位置
            selected = (expert_ids == expert_id)  # [B*L, k]
            if not selected.any():
                continue

            # 获取对应的 batch 和 token 索引，以及权重
            row_indices, col_indices = torch.where(selected)  # row 是展开后的 token 索引，col 是 k 维度
            # 对应的原始 batch 和 token
            b = batch_idx[row_indices]
            t = token_idx[row_indices]
            w = expert_weights[row_indices, col_indices]  # 对应权重

            # 收集被选中的 token 特征
            selected_features = image_embs[b, t]  # [num_selected, C]

            expert_input = torch.zeros(B, L, C, device=image_embs.device, dtype=image_embs.dtype)
            expert_input[b, t] = selected_features

            # 前向专家
            expert_out = self.experts[expert_id](expert_input)  # [B, num_query, output_size]
            if selected.any():
                avg_weight = w.mean()
            else:
                avg_weight = 0.0

            final_output += expert_out * avg_weight

        b, l, c = image_embs.shape
        # --- A. 获取并融合任务上下文 (Task Context Fusion) ---
        # 1. 获取当前 Batch 的任务特征 [B, task_dim]
        t_embs = self.task_embedding(task_ids)

        # 2. 在空间维度 L 上进行广播 (Broadcast)，让每个 Patch 都感知到任务
        # t_embs_expanded 形状: [B, L, task_dim]
        t_embs_expanded = t_embs.unsqueeze(1).expand(-1, l, -1)

        # 3. 拼接图像特征与任务特征 (Channel-wise Concatenation)
        # combined_features 形状: [B, L, c + task_dim]
        combined_features = torch.cat([image_embs, t_embs_expanded], dim=-1)

        # --- B. 计算任务与空间联合门控权重 ---
        x_fp32 = combined_features.float()
        self.gate_norm.to(torch.float32)
        self.gate.to(torch.float32)
        # token_logits 形状: [B, L, num_experts]
        token_logits = self.gate(self.gate_norm(x_fp32))
        token_logits = torch.clamp(token_logits, min=-15.0, max=15.0)
        # --- 最终归一化和增益 ---
        final_output = self.final_norm(final_output) * self.output_gain
        # 计算 Token 级别的权重分配
        token_weights = torch.softmax(token_logits, dim=-1)  # [B, L, E]
        if not self.training:
            self._cached_routing_weights = token_weights[0].detach().cpu()

        return final_output

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        return self._gate_stats

    def get_aux_loss(self) -> torch.Tensor:
        """计算辅助损失：负载均衡损失 + 可选熵损失 + z-loss"""
        imp = self._gate_stats.get("importance")
        load = self._gate_stats.get("load")
        if imp is None or load is None:
            return torch.tensor(0.0, device=self.gate.weight.device)

        balance_loss = self.num_experts * torch.sum(imp * load)
        entropy_loss = self._gate_stats.get("entropy", 0.0)
        zloss = self._gate_stats.get("zloss", 0.0)

        total = (self.aux_balance_coef * balance_loss +
                 self.aux_entropy_coef * entropy_loss +
                 self.aux_zloss_coef * zloss)
        return total.to(self.gate.weight.dtype)

    def set_aux_coefficients(self, balance=1.0, entropy=0.00001, zloss=0.00001):
        self.aux_balance_coef = balance
        self.aux_entropy_coef = entropy
        self.aux_zloss_coef = zloss


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        need_weights = not self.training
        attn_out, attn_weights = self.attn(
            q_x, k_x, v_x, need_weights=need_weights, attn_mask=attn_mask
        )

        if need_weights:
            # attn_weights 形状: [Batch, Target_Seq_Len, Source_Seq_Len]
            self._last_attn_weights = attn_weights.detach()
        return attn_out

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )

        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

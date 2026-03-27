from collections import OrderedDict
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
    Attention Pooler (Expert 序列压缩专家)
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
            stage_num: Union[List, int] = [64, 48, 32],
            split_part: List = [256, 256, 256],
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
                    ls_init_value=1e-1,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, image_embs: torch.Tensor) -> torch.Tensor:
        # 这里的 image_embs 在 T-MoE 阶段实际上是联合特征 Z_tilde (视觉特征 + 物理提示)
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

        # 动态切分，兼容拼接了物理特征后长度 L 变长的情况
        L = image_embs.size(1)
        preset_sum = sum(self.split_part) if isinstance(self.split_part, (list, tuple)) else None
        if isinstance(self.split_part, (list, tuple)) and preset_sum == L:
            split_sizes = list(self.split_part)
        else:
            base = sum(stage_query_sizes)
            if base == 0:
                split_sizes = [L // 3, L // 3, L - 2 * (L // 3)]
            else:
                sizes = [int(round(L * s / base)) for s in stage_query_sizes]
                diff = L - sum(sizes)
                sizes[-1] += diff
                split_sizes = sizes

        stage1_image, stage2_image, stage3_image = torch.split(image_embs, split_sizes, dim=1)

        all_tokens = []
        spatial_attns = []
        for sub_token, sub_image in zip(
                [stage1_query, stage2_query, stage3_query],
                [stage1_image, stage2_image, stage3_image],
        ):
            q_len = sub_token.size(1)
            cat_embs = torch.cat([sub_token, sub_image], dim=1)
            cat_embs = cat_embs.permute(1, 0, 2)
            sub_token = sub_token.permute(1, 0, 2)

            for layer in self.layers:
                sub_token = layer(sub_token, cat_embs, cat_embs)

            if not self.training and hasattr(self.layers[-1], "_last_attn_weights"):
                attn = self.layers[-1]._last_attn_weights
                img_attn = attn[:, :, q_len:].mean(dim=1)
                spatial_attns.append(img_attn)

            sub_token = sub_token.permute(1, 0, 2)
            all_tokens.append(sub_token)

        if not self.training and len(spatial_attns) == 3:
            full_spatial_attn = torch.cat(spatial_attns, dim=1)
            self._last_spatial_attn = full_spatial_attn[0].cpu()

        query_tokens = torch.cat(all_tokens, dim=1)
        out = self.out_proj(query_tokens)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


# =========================================================================
# 核心重构：任务与要素双驱动、物理先验复用的稀疏混合专家投影层 (T-MoE)
# =========================================================================
class MoEProjection(nn.Module):
    """
    对应论文 2.2 节：
    2.2.1 任务与要素双驱动的动态门控路由
    2.2.2 物理先验复用与联合特征投影
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
            num_tasks: int = 5,  # 任务种类 (如分类、问答等)
            num_elements: int = 10,  # 要素种类 (如船只、波浪、云等)
            task_dim: int = 256,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            checkpoint: bool = False,
            top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size

        # 1. 专家组：基于交叉注意力的序列压缩器 (AttnPooler)
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

        # 2. 任务与要素嵌入表 (对应 h_task, h_element)
        self.task_embedding = nn.Embedding(num_tasks, task_dim)
        self.element_embedding = nn.Embedding(num_elements, task_dim)

        # 3. 双驱动门控网络 (全局视觉特征 + 任务特征 + 要素特征)
        combined_dim = encoder_hidden_size + task_dim + task_dim
        self.gate_norm = nn.LayerNorm(combined_dim)
        self.gate = nn.Linear(combined_dim, num_experts)

        # 4. 稳定性输出增益和归一化
        self.output_gain = nn.Parameter(torch.ones(output_size) * 0.3)
        self.final_norm = RMSNorm(output_size)

        # 5. 辅助损失系数 (负载均衡)
        self.aux_balance_coef = 1.0
        self.aux_entropy_coef = 0.00001
        self.aux_zloss_coef = 0.00001
        self._gate_stats = {}

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_prompts: torch.Tensor,
            task_ids: torch.Tensor,
            element_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        image_embs: [B, L_v, C] 多尺度视觉特征序列 (Z_visual)
        physical_prompts: [B, L_p, C] 物理先验提示序列 (H_prior)
        task_ids: [B] 任务意图 ID
        element_ids: [B] 具体要素 ID
        """
        B, L_v, C = image_embs.shape

        # ==========================================
        # 1. 任务与要素双驱动的动态门控计算 (Sequence-level Routing)
        # ==========================================
        # 获取宏观全局图像上下文 z_bar_visual
        img_global = image_embs.mean(dim=1)  # [B, C]

        # 获取任务和要素上下文 h_task, h_element
        t_embs = self.task_embedding(task_ids)  # [B, task_dim]
        e_embs = self.element_embedding(element_ids)  # [B, task_dim]

        # 拼接用于联合门控评估 [z_bar_visual; h_task; h_element]
        gate_input = torch.cat([img_global, t_embs, e_embs], dim=-1).float()

        # 计算专家倾向性
        gate_logits = self.gate(self.gate_norm(gate_input))  # [B, num_experts]
        gate_logits = torch.clamp(gate_logits, min=-15.0, max=15.0)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, num_experts]

        # 选取 Top-K 专家
        top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # [B, k]

        # 权重重归一化
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 记录辅助损失状态
        if self.training:
            with torch.no_grad():
                importance = gate_weights.mean(dim=0)
                top1_indices = top_indices[:, 0]
                load = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
                for i in range(self.num_experts):
                    load[i] = (top1_indices == i).float().mean()

                entropy = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-6), dim=-1))
                zloss = torch.mean(torch.logsumexp(gate_logits.float(), dim=-1) ** 2)

            self._gate_stats = {
                "importance": importance,
                "load": load,
                "entropy": entropy,
                "zloss": zloss,
            }

        # ==========================================
        # 2. 物理先验复用与联合特征构建 (Physical Prior Reuse)
        # ==========================================
        # 沿着序列维度拼接视觉特征与物理提示: Z_tilde = Concat(Z_visual, H_prior)
        if physical_prompts is not None:
            # physical_prompts shape expected to be [B, L_p, C]
            z_tilde = torch.cat([image_embs, physical_prompts], dim=1)  # [B, L_v + L_p, C]
        else:
            z_tilde = image_embs

        # ==========================================
        # 3. 专家分发与投影重采样 (Dispatch & Combine)
        # ==========================================
        final_output = torch.zeros(B, self.num_query, self.output_size, device=image_embs.device,
                                   dtype=image_embs.dtype)

        for i in range(self.top_k):
            expert_idx = top_indices[:, i]  # [B]
            expert_weight = top_weights[:, i]  # [B]

            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)  # 布尔掩码 [B]
                if not mask.any():
                    continue

                # 提取指派给当前专家的 "物理联合序列 Z_tilde"
                selected_z_tilde = z_tilde[mask]  # [num_selected, L_v + L_p, C]

                # AttnPooler 处理包含物理属性的长序列
                expert_out = self.experts[e_id](selected_z_tilde)  # [num_selected, num_query, output_size]

                # 加权融合
                selected_weights = expert_weight[mask].unsqueeze(1).unsqueeze(2)  # [num_selected, 1, 1]
                weighted_out = expert_out * selected_weights

                final_output[mask] += weighted_out

        # ==========================================
        # 4. 特征对齐与约束输出
        # ==========================================
        final_output = self.final_norm(final_output) * self.output_gain

        return final_output

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        return self._gate_stats

    def get_aux_loss(self) -> torch.Tensor:
        """计算负载均衡损失，确保各个专家被均匀激活"""
        if not self.training or not self._gate_stats:
            return torch.tensor(0.0, device=self.gate.weight.device)

        imp = self._gate_stats.get("importance")
        load = self._gate_stats.get("load")

        balance_loss = self.num_experts * torch.sum(imp * load)
        entropy_loss = self._gate_stats.get("entropy", 0.0)
        zloss = self._gate_stats.get("zloss", 0.0)

        total = (self.aux_balance_coef * balance_loss +
                 self.aux_entropy_coef * entropy_loss +
                 self.aux_zloss_coef * zloss)

        return total.to(self.gate.weight.dtype)


# ---------------------------------------------------------
# 以下底层依赖保持你的原版不动，用于保障模型计算不出错
# ---------------------------------------------------------

class PatchDropout(nn.Module):
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token

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
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
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
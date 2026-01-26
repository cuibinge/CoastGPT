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
        for sub_token, sub_image in zip(
            [stage1_query, stage2_query, stage3_query],
            [stage1_image, stage2_image, stage3_image],
        ):
            cat_embs = torch.cat([sub_token, sub_image], dim=1)
            # cat_embs = sub_image
            cat_embs = cat_embs.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
            sub_token = sub_token.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

            for layer in self.layers:
                sub_token = layer(sub_token, cat_embs, cat_embs)

            sub_token = sub_token.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)
            all_tokens.append(sub_token)

        query_tokens = torch.cat(all_tokens, dim=1)
        out = self.out_proj(query_tokens)
        return out


# 在 common_arch.py 中定义 RMSNorm (类似 Llama-2 实现)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 只缩放，不平移，保留特征显著性
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class MoEProjection(nn.Module):
    """
    针对遥感多模态对齐优化的专家混合投影层 (MoE Projection)
    新增：final_norm 与能量缩放逻辑，解决 LLM 激活过载问题
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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        checkpoint: bool = False,
        top_k: int = 2, # 建议设为 2，以增强专家间的协同
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 初始化 4 个独立的 AttnPooler 专家
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

        # 2. 空间门控网络：作用于每个序列 Token (L)
        self.gate_norm = nn.LayerNorm(encoder_hidden_size)
        self.gate = nn.Linear(encoder_hidden_size, num_experts)
        self.output_gain = nn.Parameter(torch.ones(output_size) * 0.3)
        
        # 🌟 核心防线：解决 549.0 能量过载的归一化层
        self.final_norm = RMSNorm(output_size)
        # 3. 统计缓存与损失系数
        self._aux_loss = torch.tensor(0.0)
        self._gate_stats = {}
        self._aux_coeff = {"balance": 1.0, "entropy": 0.00001, "zloss":  0.00001}

    def set_aux_coefficients(self, balance: float = 1.0, entropy: float = 0.00001, zloss: float = 0.00001):
        self._aux_coeff = dict(balance=balance, entropy=entropy, zloss=zloss)

    def get_aux_loss(self) -> torch.Tensor:
        # 整合了重复定义的方法，确保系数被正确应用
        imp = self._gate_stats.get("importance")
        load = self._gate_stats.get("load")
        if imp is None or load is None: return torch.tensor(0.0, device=self.gate.weight.device)
        
        balance = self.num_experts * torch.sum(imp * load)
        entropy = self._gate_stats.get("entropy",0.00001)
        zloss = self._gate_stats.get("zloss", 0.00001)
        
        return (self._aux_coeff["balance"] * balance + 
                self._aux_coeff["entropy"] * entropy + 
                self._aux_coeff["zloss"] * zloss).to(self.gate.weight.dtype)

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        return self._gate_stats

    def forward(self, image_embs: torch.Tensor) -> torch.Tensor:
        b, l, c = image_embs.shape

        # --- A. 计算局部空间权重图 [B, L, E] ---
        # 不再对 dim=1 取 mean，而是对每个补丁进行点名
        x_fp32 = image_embs.float()
        self.gate_norm.to(torch.float32)
        self.gate.to(torch.float32)
        
        # token_logits 形状: [B, L, num_experts]
        token_logits = self.gate(self.gate_norm(x_fp32))
        token_logits = torch.clamp(token_logits, min=-15.0, max=15.0)
        
        # 计算 Token 级别的权重分配
        token_weights = torch.softmax(token_logits, dim=-1) # [B, L, E]

        # --- B. 计算路由统计 (用于负载均衡 aux_loss) ---
        # 计算所有补丁对专家的平均重要性
        importance = token_weights.view(-1, self.num_experts).mean(dim=0)
        
        with torch.no_grad():
            _, top1_idx = torch.max(token_weights, dim=-1) # 每个补丁的首选专家
            load = torch.zeros(self.num_experts, device=image_embs.device, dtype=torch.float32)
            for i in range(self.num_experts):
                load[i] = (top1_idx == i).float().mean()

        # 缓存统计信息
        self._gate_stats = {
            "importance": importance.detach(), 
            "load": load.detach(),
            "entropy": -torch.mean(torch.sum(token_weights * torch.log(token_weights + 1e-6), dim=-1)).detach(),
            "zloss": torch.mean(torch.logsumexp(token_logits, dim=-1) ** 2).detach()
        }

        # --- C. 局部加权专家并行处理 ---
        token_weights = token_weights.to(image_embs.dtype)
        mixed_output = 0

        # 遍历专家，每个专家只处理它被“点名”的那些空间补丁
        for i in range(self.num_experts):
            # 获取第 i 个专家在所有 L 个位置的权重 [B, L, 1]
            # 这里的 w_i 充当了“空间权重图”，标识了该专家负责哪些区域
            w_i = token_weights[:, :, i:i+1]
            
            # 空间加权：专家 i 重点从权重高的补丁中提取信息
            gated_input = image_embs * w_i
            
            # 专家执行 AttnPooler，将 L 个 Token 聚合成 num_query 个 Token
            expert_out = self.experts[i](gated_input)
            mixed_output += expert_out

        # 1. 精度对齐并执行归一化 (这一步会将每个 Token 的 Norm 锁死在 64)
        if self.final_norm.weight.dtype != mixed_output.dtype:
            self.final_norm.to(mixed_output.dtype)
        
        mixed_output = self.final_norm(mixed_output)

        # 2. 🌟 关键点：在归一化之后执行缩放 🌟
        # 目标：将总能量从 768 降到 200 左右
        # 计算系数：200 / 768 ≈ 0.26
        # 我们使用 0.3 作为缩放因子
        return mixed_output *self.output_gain
        

    def set_aux_coefficients(self, balance: float = 1.0, entropy: float = 0.0, zloss: float = 0.0):
        """Optionally set coefficients to scale different components when queried."""
        self._aux_coeff = dict(balance=balance, entropy=entropy, zloss=zloss)

    def get_aux_loss(self) -> torch.Tensor:
        """Return the auxiliary load-balancing loss computed in the last forward.

        If coefficients have been set via `set_aux_coefficients`, they are applied.
        Otherwise, returns the raw balance-only loss.
        """
        if self._aux_loss is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        coeff = getattr(self, "_aux_coeff", None)
        if coeff is None:
            return self._aux_loss
        # reconstruct components from stats and logits proxies
        imp = self._gate_stats.get("importance")
        load = self._gate_stats.get("load")
        entropy = self._gate_stats.get("entropy")
        zloss = self._gate_stats.get("zloss")
        balance = self.num_experts * torch.sum(imp * load)
        return coeff["balance"] * balance + coeff["entropy"] * entropy + coeff["zloss"] * zloss

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        """Return cached gate statistics (importance, load, entropy, zloss)."""
        return self._gate_stats


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
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

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

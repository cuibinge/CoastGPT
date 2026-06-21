from collections import OrderedDict
from typing import Callable, List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


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
    """Attention pooler used by projection experts."""

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

    def forward(
            self,
            image_embs: torch.Tensor,
            physical_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool image tokens with optional generic condition queries."""
        if self.in_proj is not None:
            image_embs = self.in_proj(image_embs)
            if physical_queries is not None:
                physical_queries = self.in_proj(physical_queries)

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
            stage_query_sizes = list(self.stage_num)
            if len(stage_query_sizes) != 3:
                raise ValueError(f"stage_num must have 3 parts, got: {stage_query_sizes}")
            if sum(stage_query_sizes) != self.num_query:
                # Keep training robust when num_query changes from default 144.
                even = self.num_query // 3
                stage_query_sizes = [even, even, self.num_query - 2 * even]
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, stage_query_sizes, dim=1
            )

        # Split dynamically so the token length may vary after condition fusion.
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
        if physical_queries is not None:
            Lp = physical_queries.size(1)
            # Split condition queries using the same relative segment sizes.
            if L > 0:
                phy_sizes = [max(1, int(round(Lp * s / L))) for s in split_sizes]
                diff_phy = Lp - sum(phy_sizes)
                phy_sizes[-1] += diff_phy
            else:
                phy_sizes = [Lp // 3, Lp // 3, Lp - 2 * (Lp // 3)]
            stage1_phy, stage2_phy, stage3_phy = torch.split(physical_queries, phy_sizes, dim=1)
        else:
            stage1_phy = stage2_phy = stage3_phy = None

        all_tokens = []
        spatial_attns = []
        for sub_token, sub_image, sub_phy in zip(
                [stage1_query, stage2_query, stage3_query],
                [stage1_image, stage2_image, stage3_image],
                [stage1_phy, stage2_phy, stage3_phy],
        ):
            if sub_phy is not None:
                sub_token = torch.cat([sub_token, sub_phy], dim=1)
            # Key/value tokens come from image features and condition tokens.
            kv_parts = [sub_image]
            if sub_phy is not None:
                kv_parts.append(sub_phy)
            cat_embs = torch.cat(kv_parts, dim=1)
            cat_embs = cat_embs.permute(1, 0, 2)
            sub_token = sub_token.permute(1, 0, 2)

            for layer in self.layers:
                sub_token = layer(sub_token, cat_embs, cat_embs)

            if not self.training and hasattr(self.layers[-1], "_last_attn_weights"):
                attn = self.layers[-1]._last_attn_weights
                img_len = sub_image.size(1)
                img_attn = attn[:, :, :img_len].mean(dim=1)
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


class PhysicalPromptEncoder(nn.Module):
    """Encode generic physical metadata maps into token sequences."""

    def __init__(
            self,
            in_channels: int = 1,
            embed_dim: int = 768,
            pool_scales: Union[List[int], tuple] = (1, 2, 4),
    ):
        super().__init__()
        self.pool_scales = list(pool_scales) if pool_scales is not None else [1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            tsm: Optional[torch.Tensor],
            mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if tsm is None:
            return None

        # squeeze potential extra dimension from dataloader collate
        if tsm.dim() == 5:
            tsm = tsm.squeeze(1)
        if tsm.dim() == 3:
            tsm = tsm.unsqueeze(1)

        if mask is not None:
            if mask.dim() == 5:
                mask = mask.squeeze(1)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            tsm = tsm * mask

        prompts = []
        for s in self.pool_scales:
            pooled = F.adaptive_avg_pool2d(tsm, output_size=(s, s))
            tokens = self.proj(pooled).flatten(2).transpose(1, 2)
            prompts.append(self.norm(tokens))

        if len(prompts) == 0:
            return None
        return torch.cat(prompts, dim=1)


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

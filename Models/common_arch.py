import os
from collections import OrderedDict
from typing import Callable, List, Optional, Union

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
        nn.init.trunc_normal_(self.query, std=0.02, mean=0.0)

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
        self.disable_npu_sdpa = (
            os.environ.get("COASTGPT_DISABLE_NPU_SDPA", "1").lower()
            not in ("0", "false", "no")
        )

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

    def _should_use_manual_attention(self, q_x: torch.Tensor) -> bool:
        return q_x.device.type == "npu" and self.disable_npu_sdpa

    def _expand_attn_mask(
        self,
        attn_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        tgt_len: int,
        src_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attn_mask.to(device=device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            if mask.shape[0] == batch_size * num_heads:
                mask = mask.view(batch_size, num_heads, tgt_len, src_len)
            elif mask.shape[0] == batch_size:
                mask = mask.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unsupported 3D attention mask shape: {tuple(mask.shape)}"
                )
        elif mask.dim() != 4:
            raise ValueError(
                f"Unsupported attention mask dimension: {mask.dim()}"
            )

        if mask.shape[-2:] != (tgt_len, src_len):
            raise ValueError(
                "Attention mask shape does not match attention scores: "
                f"expected (*, *, {tgt_len}, {src_len}), got {tuple(mask.shape)}"
            )

        if mask.dtype == torch.bool:
            return mask
        return mask.to(dtype=dtype)

    def _manual_attention(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embed_dim = self.attn.embed_dim
        num_heads = self.attn.num_heads
        head_dim = embed_dim // num_heads
        tgt_len, batch_size, _ = q_x.shape
        src_len = k_x.shape[0]

        w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3, dim=0)
        if self.attn.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3, dim=0)

        q = F.linear(q_x, w_q, b_q)
        k = F.linear(k_x, w_k, b_k)
        v = F.linear(v_x, w_v, b_v)

        q = q.contiguous().view(tgt_len, batch_size, num_heads, head_dim)
        k = k.contiguous().view(src_len, batch_size, num_heads, head_dim)
        v = v.contiguous().view(src_len, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Keep score computation on the eager path to avoid torch_npu flash-attention kernels.
        attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        attn_scores = attn_scores * (head_dim ** -0.5)

        if attn_mask is not None:
            expanded_mask = self._expand_attn_mask(
                attn_mask=attn_mask,
                batch_size=batch_size,
                num_heads=num_heads,
                tgt_len=tgt_len,
                src_len=src_len,
                dtype=attn_scores.dtype,
                device=attn_scores.device,
            )
            if expanded_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(
                    expanded_mask, torch.finfo(attn_scores.dtype).min
                )
            else:
                attn_scores = attn_scores + expanded_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        if self.attn.dropout > 0:
            attn_probs = F.dropout(
                attn_probs, p=self.attn.dropout, training=self.training
            )
        attn_probs = attn_probs.to(dtype=v.dtype)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(tgt_len, batch_size, embed_dim)
        return F.linear(
            attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias
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

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_x.dtype)
        if self._should_use_manual_attention(q_x):
            return self._manual_attention(q_x, k_x, v_x, attn_mask=attn_mask)
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

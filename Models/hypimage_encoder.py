import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1 + x3, inplace=True)
        return out

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class ViT(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skip_connection = torch.nn.ModuleList([])
        for _ in range(depth - 2):
            self.skip_connection.append(torch.nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        last_output = []
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skip_connection[nl - 2](
                    torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x, mask=mask)
            x = ff(x)
            nl += 1
        return x

class HypImageEncoder(torch.nn.Module):
    def __init__(self, patch_size, bands, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 embed_dim=512, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()

        self.bands = bands
        self.patch_size = patch_size

        self.conv3d = residual_block(1, 8)
        self.x1 = self._get_layer_size()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.x1.shape[1] * self.x1.shape[2], out_channels=bands, kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True))

        patch_dim = patch_size ** 2

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, bands + 1, dim))
        self.patch_to_embedding = torch.nn.Linear(patch_dim, dim)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = torch.nn.Dropout(emb_dropout)
        self.vision_transformer = ViT(dim, depth, heads, dim_head, mlp_dim, dropout, bands)

        self.pool = pool
        self.to_latent = torch.nn.Identity()

        self.layer_norm = torch.nn.LayerNorm(dim)
        self.classification = torch.nn.Linear(dim, num_classes)
        self.fc = torch.nn.Linear(dim, embed_dim)

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.bands,
                             self.patch_size, self.patch_size))
            s = self.conv3d(x)
        return s

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.conv2d(x)
        x = rearrange(x, 'b c h w -> b c (h w)')

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.vision_transformer(x, mask)

        x = self.to_latent(x[:, 0])

        x = self.layer_norm(x)
        return self.classification(x), self.fc(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)











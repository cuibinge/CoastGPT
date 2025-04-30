import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat

# 3D卷积块，包含一个3x3x3卷积和BatchNorm层
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

# 自定义残差块，实现三卷积层结构和残差连接
class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # 输入形状示例：(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # 第一次卷积+ReLU
        x2 = F.relu(self.conv2(x1), inplace=True)  # 第二次卷积+ReLU
        x3 = self.conv3(x2)  # 第三次卷积，无激活

        out = F.relu(x1 + x3, inplace=True)  # 残差连接：x1 + x3，然后ReLU
        return out

# 残差连接模块，将输入添加到函数输出
class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 前置归一化模块，先进行层归一化再应用函数
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 前馈神经网络模块，包含两个线性层和GELU激活
class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),  # 高斯误差线性单元激活函数
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 多头自注意力模块
class Attention(torch.nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads  # 注意力头的总维度
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，防止点积过大

        # 一次性生成查询(Q)、键(K)、值(V)的线性变换
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]，b为批次大小，n为序列长度，dim为特征维度
        b, n, _, h = *x.shape, self.heads

        # 获取qkv元组:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 将q,k,v从 [b,n,head_num*head_dim] 重塑为 [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # 计算注意力分数: transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # 应用掩码（如果提供）
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 应用softmax归一化得到注意力矩阵
        attn = dots.softmax(dim=-1)
        # 注意力矩阵乘以值矩阵得到输出
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # 拼接所有注意力头的输出 -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# Vision Transformer (ViT) 主体架构
class ViT(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        # 堆叠Transformer层
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        # 跳跃连接层，用于连接不同深度的特征
        self.skip_connection = torch.nn.ModuleList([])
        for _ in range(depth - 2):
            self.skip_connection.append(torch.nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        last_output = []  # 保存每一层的输出，用于跳跃连接
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            # 从第三层开始应用跳跃连接
            if nl > 1:
                x = self.skip_connection[nl - 2](
                    torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x, mask=mask)  # 注意力层
            x = ff(x)  # 前馈网络层
            nl += 1
        return x

# 图像编码器，结合CNN和ViT处理图像数据
class ImageEncoder(torch.nn.Module):
    def __init__(self, patch_size, bands, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 embed_dim=512, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()

        self.bands = bands  # 波段数
        self.patch_size = patch_size  # 图像块大小

        # 3D卷积残差块，处理多光谱图像
        self.conv3d = residual_block(1, 8)
        self.x1 = self._get_layer_size()  # 计算中间特征尺寸
        # 2D卷积层，整合3D卷积输出
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.x1.shape[1] * self.x1.shape[2], out_channels=bands, kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True))

        patch_dim = patch_size ** 2  # 图像块的维度

        # 位置编码和分类令牌
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, bands + 1, dim))
        self.patch_to_embedding = torch.nn.Linear(patch_dim, dim)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = torch.nn.Dropout(emb_dropout)
        self.vision_transformer = ViT(dim, depth, heads, dim_head, mlp_dim, dropout, bands)

        self.pool = pool  # 池化方式，默认为使用分类令牌
        self.to_latent = torch.nn.Identity()  # 恒等映射，可替换为其他变换

        # 分类和特征提取层
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.classification = torch.nn.Linear(dim, num_classes)
        self.fc = torch.nn.Linear(dim, embed_dim)

    # 计算中间特征尺寸的辅助函数
    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.bands,
                             self.patch_size, self.patch_size))
            s = self.conv3d(x)
        return s

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)  # 添加通道维度 [batch, bands, height, width] -> [batch, 1, bands, height, width]
        x = self.conv3d(x)  # 3D卷积处理 [batch, 8, bands, height, width]
        x = rearrange(x, 'b c h w y -> b (c h) w y')  # 重塑张量 [batch, 8*bands, height, width]

        x = self.conv2d(x)  # 2D卷积处理 [batch, bands, height, width]
        # 重塑为图像块 [batch, bands, height*width]
        x = rearrange(x, 'b c h w -> b c (h w)')

        # 将图像块嵌入到特征空间 [batch, bands, embedding_dim]
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # 添加分类令牌和位置编码
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 复制分类令牌到每个样本
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接分类令牌 [batch, bands+1, embedding_dim]
        x += self.pos_embedding[:, :(n + 1)]  # 添加位置编码
        x = self.dropout(x)

        # 通过ViT处理 [batch, bands+1, embedding_dim] -> [batch, bands+1, embedding_dim]
        x = self.vision_transformer(x, mask)

        # 使用分类令牌进行分类 [batch, embedding_dim]
        x = self.to_latent(x[:, 0])

        # 层归一化后进行分类和特征提取
        x = self.layer_norm(x)
        return self.classification(x), self.fc(x)  # 返回分类结果和特征向量

# 自定义层归一化，处理FP16数据类型
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)











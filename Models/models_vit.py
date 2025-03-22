from functools import partial  # 导入partial用于创建偏函数

import torch
import torch.nn as nn

import timm.models.vision_transformer  # 导入timm库中的Vision Transformer实现

# 自定义Vision Transformer类，继承自timm的实现
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        """
        初始化Vision Transformer模型

        参数:
            global_pool (bool): 是否使用全局池化，默认为False
            **kwargs: 其他参数，用于传递给父类
        """
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool  # 是否启用全局池化
        if self.global_pool:
            norm_layer = kwargs['norm_layer']  # 获取归一化层类型
            embed_dim = kwargs['embed_dim']    # 获取嵌入维度
            self.fc_norm = norm_layer(embed_dim)  # 为全局池化添加归一化层

            del self.norm  # 如果使用全局池化，删除原始的归一化层

    def forward_features(self, x):
        """
        前向传播提取特征

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)

        返回:
            outcome (torch.Tensor): 提取的特征
        """
        B = x.shape[0]  # 获取批量大小
        x = self.patch_embed(x)  # 将图像分成补丁并嵌入，输出形状 (B, num_patches, embed_dim)

        # 扩展CLS token到批量大小，形状从 (1, 1, embed_dim) 变为 (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 感谢Phil Wang的实现
        x = torch.cat((cls_tokens, x), dim=1)  # 将CLS token与补丁嵌入拼接，形状 (B, num_patches+1, embed_dim)
        x = x + self.pos_embed  # 添加位置嵌入
        x = self.pos_drop(x)    # 应用位置丢弃（Dropout）

        # 通过所有的Transformer块处理特征
        for blk in self.blocks:
            x = blk(x)  # 每个块包含多头自注意力和前馈网络

        if self.global_pool:
            # 如果启用全局池化，取补丁部分的平均值（排除CLS token）
            x = x[:, 1:, :].mean(dim=1)  # 形状从 (B, num_patches+1, embed_dim) 变为 (B, embed_dim)
            outcome = self.fc_norm(x)    # 应用归一化
        else:
            # 默认情况下，使用CLS token作为输出
            x = self.norm(x)  # 应用原始归一化层
            outcome = x[:, 0]  # 取出CLS token，形状为 (B, embed_dim)

        return outcome

# 创建基础版ViT模型（Base型号）
def vit_base_patch16(**kwargs):
    """
    创建ViT-Base模型（补丁大小16）

    参数:
        **kwargs: 其他参数，用于传递给VisionTransformer

    返回:
        model: 配置好的VisionTransformer模型实例
    """
    model = VisionTransformer(
        patch_size=16,          # 补丁大小为16x16
        embed_dim=768,          # 嵌入维度为768
        depth=12,              # Transformer层数为12
        num_heads=12,          # 多头注意力的头数为12
        mlp_ratio=4,           # MLP隐藏层与嵌入维度的比例为4
        qkv_bias=True,         # 在QKV投影中启用偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用LayerNorm，设置eps为1e-6
        **kwargs)
    return model

# 创建大模型ViT（Large型号）
def vit_large_patch16(**kwargs):
    """
    创建ViT-Large模型（补丁大小16）

    参数:
        **kwargs: 其他参数，用于传递给VisionTransformer

    返回:
        model: 配置好的VisionTransformer模型实例
    """
    model = VisionTransformer(
        patch_size=16,          # 补丁大小为16x16
        embed_dim=1024,         # 嵌入维度为1024
        depth=24,              # Transformer层数为24
        num_heads=16,          # 多头注意力的头数为16
        mlp_ratio=4,           # MLP隐藏层与嵌入维度的比例为4
        qkv_bias=True,         # 在QKV投影中启用偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用LayerNorm，设置eps为1e-6
        **kwargs)
    return model

# 创建超大模型ViT（Huge型号）
def vit_huge_patch14(**kwargs):
    """
    创建ViT-Huge模型（补丁大小14）

    参数:
        **kwargs: 其他参数，用于传递给VisionTransformer

    返回:
        model: 配置好的VisionTransformer模型实例
    """
    model = VisionTransformer(
        patch_size=14,          # 补丁大小为14x14
        embed_dim=1280,         # 嵌入维度为1280
        depth=32,              # Transformer层数为32
        num_heads=16,          # 多头注意力的头数为16
        mlp_ratio=4,           # MLP隐藏层与嵌入维度的比例为4
        qkv_bias=True,         # 在QKV投影中启用偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用LayerNorm，设置eps为1e-6
        **kwargs)
    return model
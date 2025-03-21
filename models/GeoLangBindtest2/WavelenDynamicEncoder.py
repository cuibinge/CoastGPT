import torch
import torch.nn as nn
from einops import rearrange


class WavelengthDynamicEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=256, latent_dim=64):
        super().__init__()
        # 波长编码器 (假设输入为通道中心波长的归一化值)
        self.wavelength_encoder = nn.Sequential(
            nn.Linear(1, latent_dim),  # 输入为每个通道的中心波长
            nn.GELU(),
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4),
            nn.Linear(latent_dim, in_channels * 3 * 3)  # 生成动态卷积核参数
        )

        # 共享的基础卷积
        self.base_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, wavelengths):
        """
        x: [B, C, H, W] 输入数据
        wavelengths: [C, 1] 每个通道的中心波长
        """
        # 生成动态卷积核
        kernel_params = self.wavelength_encoder(wavelengths)  # [C, C*3 * 3]
        kernel = rearrange(kernel_params, 'c (k ch) -> ch c 3 3', k=in_channels)

        # 动态卷积
        x = nn.functional.conv2d(
            x, kernel, bias=None, padding=1, groups=x.shape[1]
        )
        # 与基础卷积结果融合
        x_base = self.base_conv(x)
        return x + x_base


class ModalityAwareAggregation(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # 模态提示生成
        self.prompt_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # 自适应层归一化
        self.adap_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, wavelengths):
        """
        x: [B, D] 特征向量
        wavelengths: [C] 波长信息
        """
        # 生成模态提示
        prompt = self.prompt_generator(wavelengths.mean(dim=0))  # [D]

        # 自适应归一化参数
        gamma = self.gamma(prompt)
        beta = self.beta(prompt)

        # 应用调制
        x = self.adap_norm(x)
        return x * gamma + beta


def multi_teacher_loss(student_feat, teachers):
    """
    teachers: 包含SigLIP/DINOv2/ViT教师模型特征的字典
    """
    loss = 0
    for teacher_name, teacher_feat in teachers.items():
        # 特征对齐损失
        cos_loss = 1 - nn.functional.cosine_similarity(student_feat, teacher_feat)
        l1_loss = nn.functional.l1_loss(student_feat, teacher_feat)
        mse_loss = nn.functional.mse_loss(student_feat, teacher_feat)

        loss += (0.3 * cos_loss + 0.5 * l1_loss + 0.2 * mse_loss)

    return loss
class GeoChatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WavelengthDynamicEncoder()
        self.maaa = ModalityAwareAggregation()
        self.text_proj = nn.Linear(768, 512)  # 文本编码投影

    def forward(self, image, text, wavelengths):
        # 图像编码
        visual_feat = self.encoder(image, wavelengths)
        visual_feat = self.maaa(visual_feat.mean(dim=[2, 3]), wavelengths)

        # 文本编码
        text_feat = self.text_proj(text)

        # 对比学习
        logits = visual_feat @ text_feat.T
        return logits
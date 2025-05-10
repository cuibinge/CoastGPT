import ml_collections  # 导入ml_collections库，用于管理配置
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from transformers import CLIPVisionModel  # 从transformers库导入CLIP视觉模型
from Models.GeoLangBindtest2.WavelenDynamicEncoder import WavelengthDynamicEncoder, ModalityAwareAggregation  # 导入波长动态编码器

# 视觉模型类，继承自nn.Module
class VisionModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化视觉模型

        参数:
            config (ml_collections.ConfigDict): 配置对象，包含模型参数和设置
        """
        super(VisionModel, self).__init__()  # 调用父类nn.Module的初始化方法

        self.embedding_dim = config.vision.embedding_dim  # 设置嵌入维度，从配置中获取
        # 从预训练模型加载CLIP视觉编码器，模型名称由配置中的vit_name指定
        self.encoder = CLIPVisionModel.from_pretrained(config.vit_name)

        self.extract_stage = [
            self.encoder.config.num_hidden_layers // 3 - 1,
            self.encoder.config.num_hidden_layers // 3 * 2 - 1,
            self.encoder.config.num_hidden_layers - 2,
        ]
        
        # 添加波长动态编码器
        self.use_wavelength_encoder = config.vision.get('use_wavelength_encoder', True)
        if self.use_wavelength_encoder:
            # 获取CLIP输出的隐藏状态维度
            hidden_size = self.encoder.config.hidden_size
            self.wavelength_encoder = WavelengthDynamicEncoder(
                in_channels=hidden_size,
                out_channels=hidden_size,
                latent_dim=config.vision.get('wavelength_latent_dim', 64)
            )
            
            # 添加模态感知聚合
            self.modality_aggregation = ModalityAwareAggregation(embed_dim=hidden_size)
            
            # 添加多模态适配层
            self.modality_adapters = nn.ModuleDict({
                'rgb': nn.Identity(),  # RGB图像不需要特殊处理
                'infrared': nn.Conv2d(3, 3, kernel_size=1),  # 红外图像适配
                'radar': nn.Conv2d(3, 3, kernel_size=1),     # 雷达图像适配
                'multispectral': nn.Conv2d(10, 3, kernel_size=1)  # 多光谱图像适配(假设10个波段)
            })
            
            # 为不同模态定义默认波长信息
            self.default_wavelengths = {
                'rgb': torch.tensor([[0.45], [0.54], [0.65]]),  # RGB波长(微米)
                'infrared': torch.tensor([[1.4], [3.0], [5.0]]),  # 红外波长
                'radar': torch.tensor([[0.8], [3.0], [10.0]]),   # 雷达波长(厘米)
                'multispectral': None  # 多光谱需要在运行时提供
            }

    def encode(self, x: torch.Tensor, modality: str = 'rgb', wavelengths=None):
        """
        对输入图像进行编码

        参数:
            x (torch.Tensor): 输入图像张量，形状通常为 (B, C, H, W)
            modality (str): 图像模态类型，可选'rgb', 'infrared', 'radar', 'multispectral'
            wavelengths (torch.Tensor, optional): 波长信息，形状为 (C, 1)

        返回:
            image_embeds (torch.Tensor): 图像嵌入张量，形状为 (B, S, D)
        """
        # 如果启用波长编码器
        if self.use_wavelength_encoder:
            # 如果没有提供波长信息，使用默认波长
            if wavelengths is None and modality in self.default_wavelengths:
                wavelengths = self.default_wavelengths[modality]
                if wavelengths is not None:
                    wavelengths = wavelengths.to(x.device)
            
            # 应用模态特定的适配层
            if modality in self.modality_adapters:
                x = self.modality_adapters[modality](x)
        
        # 使用CLIP视觉编码器处理输入图像
        outputs = self.encoder(
            x,
            return_dict=True,         # 以字典形式返回输出
            output_hidden_states=True, # 返回所有隐藏状态
        )
        # 从输出中提取最后一层隐藏状态，去掉CLS token（第0个位置），仅保留补丁嵌入
        image_embeds = outputs.hidden_states[11][:, 1:, :]

        # 如果启用波长编码器且提供了波长信息，则应用波长动态编码
        if self.use_wavelength_encoder and wavelengths is not None:
            # 重塑图像嵌入以适应卷积操作 [B, S, D] -> [B, D, H, W]
            batch_size, seq_len, hidden_dim = image_embeds.shape
            h = w = int(seq_len ** 0.5)  # 假设序列长度是一个完全平方数
            
            # 重塑为 [B, D, H, W] 格式用于卷积操作
            reshaped_embeds = image_embeds.transpose(1, 2).reshape(batch_size, hidden_dim, h, w)
            
            # 应用波长动态编码
            encoded_embeds = self.wavelength_encoder(reshaped_embeds, wavelengths)
            
            # 重塑回原始形状 [B, D, H, W] -> [B, S, D]
            image_embeds = encoded_embeds.reshape(batch_size, hidden_dim, seq_len).transpose(1, 2)
            
            # 应用模态感知聚合
            pooled_embeds = image_embeds.mean(dim=1)  # [B, D]
            enhanced_embeds = self.modality_aggregation(pooled_embeds, wavelengths)
            
            # 将增强的特征广播回原始形状
            enhanced_scale = enhanced_embeds.unsqueeze(1) / pooled_embeds.unsqueeze(1)
            image_embeds = image_embeds * enhanced_scale.unsqueeze(1)

        return image_embeds

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (dict): 输入字典，包含图像数据和相关信息
                - 必须包含一个模态键(如"rgb", "infrared"等)对应的图像张量
                - 可选包含"modality"键指定模态类型
                - 可选包含"wavelengths"键提供波长信息

        返回:
            torch.Tensor: 编码后的图像嵌入
        """
        # 确定输入模态
        modality = x.get("modality", "rgb")
        
        # 获取对应模态的输入数据
        if modality in x:
            modal_input = x[modality]
        else:
            # 如果没有找到对应模态的数据，默认使用rgb
            modal_input = x.get("rgb", None)
            if modal_input is None:
                # 尝试找到任何可用的图像数据
                for key in ["infrared", "radar", "multispectral"]:
                    if key in x:
                        modal_input = x[key]
                        modality = key
                        break
                        
        wavelengths = x.get("wavelengths", None)  # 获取波长信息
        
        return self.encode(modal_input, modality, wavelengths)  # 调用encode方法进行编码
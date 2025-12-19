#加入地理信息注意力机制
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_npu
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import random
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
import torch.nn.functional as F

# 假设这些自定义模块已正确实现
from geodataset import GeoDataset
from Geo_features import MultiScaleGeoEncoder, GeoEncoderConfig

# 设置随机种子
def set_seed(seed=3667):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

set_seed(3667)

# 设置设备
device = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------------- 地理位置感知的视觉编码器架构 --------------------------

class MultiScaleFeaturePyramid(nn.Module):
    """多尺度特征金字塔网络 - 专门为地理定位设计"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels_list = in_channels_list
        
        # 横向连接：将不同尺度的特征映射到统一维度
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) 
            for in_channels in in_channels_list
        ])
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """
        参数:
            features: 按分辨率从高到低排序的特征列表 [feat1, feat2, feat3, feat4]
        """
        if not features:
            return None
        
        # 验证输入通道数是否匹配
        for i, feat in enumerate(features):
            if feat.shape[1] != self.in_channels_list[i]:
                raise RuntimeError(f"特征{i}的通道数{feat.shape[1]}与期望的{self.in_channels_list[i]}不匹配")
            
        # 目标分辨率：最高分辨率
        target_size = features[0].shape[-2:]
        
        # 处理每个尺度特征
        resized_features = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            # 通道数调整
            feature = lateral_conv(feature)
            
            # 分辨率对齐（上采样到最高分辨率）
            if i > 0:
                feature = F.interpolate(
                    feature, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            
            resized_features.append(feature)
        
        # 特征拼接和融合
        fused_features = torch.cat(resized_features, dim=1)
        output = self.fusion_conv(fused_features)
        
        return output

class GeographicAttention(nn.Module):
    """地理注意力机制 - 让模型关注与位置相关的区域"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # 自注意力机制
        self.to_query = nn.Conv2d(feature_dim, feature_dim, 1)
        self.to_key = nn.Conv2d(feature_dim, feature_dim, 1)
        self.to_value = nn.Conv2d(feature_dim, feature_dim, 1)
        
        # 输出投影
        self.output_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        # 生成查询、键、值
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)
        
        # 重塑为多头注意力格式
        query = self._reshape_to_multihead(query)
        key = self._reshape_to_multihead(key)
        value = self._reshape_to_multihead(value)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        attended_value = torch.matmul(attention_weights, value)
        attended_value = self._reshape_from_multihead(attended_value, H, W)
        
        # 输出投影和残差连接
        output = self.output_proj(attended_value)
        output = self.gamma * output + x
        
        return output
    
    def _reshape_to_multihead(self, x):
        """将特征重塑为多头注意力格式"""
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_dim, H, W)
        x = x.flatten(3).transpose(2, 3)  # [B, heads, H*W, head_dim]
        return x
    
    def _reshape_from_multihead(self, x, H, W):
        """从多头注意力格式恢复"""
        B, heads, HW, head_dim = x.shape
        x = x.transpose(2, 3).view(B, heads, head_dim, H, W)
        x = x.contiguous().view(B, self.feature_dim, H, W)
        return x

class GeoAwareCLIPVisionEncoder(nn.Module):
    """地理位置感知的CLIP视觉编码器"""
    def __init__(self, embed_dim=512, 
                 clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
        super().__init__()
        
        # 加载CLIP配置和模型
        self.clip_config = CLIPVisionConfig.from_pretrained(clip_model_name)
        self.clip_config.image_size = 256
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_model_name,
            config=self.clip_config,
            ignore_mismatched_sizes=True
        )
        
        # 启用隐藏状态输出以获取中间层特征
        self.clip_vision_model.config.output_hidden_states = True
        
        # 动态获取CLIP的隐藏层维度（关键修复）
        self.clip_hidden_dim = self.clip_config.hidden_size
        print(f"CLIP模型隐藏层维度: {self.clip_hidden_dim}")
        
        # 选择最后4层的特征（从深层到浅层）
        self.selected_layers = [-4, -3, -2, -1]
        
        # 多尺度特征金字塔（使用动态获取的通道数）
        self.fpn = MultiScaleFeaturePyramid(
            in_channels_list=[self.clip_hidden_dim] * len(self.selected_layers),
            out_channels=256
        )
        
        # 地理注意力机制
        self.geo_attention = GeographicAttention(256)
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
        
        self.normalize = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 获取CLIP的多层隐藏状态
        clip_output = self.clip_vision_model(pixel_values=x, output_hidden_states=True)
        
        # 获取隐藏状态并验证层数
        hidden_states = clip_output.hidden_states
        if len(hidden_states) < abs(min(self.selected_layers)):
            raise RuntimeError(f"CLIP模型仅返回{len(hidden_states)}层隐藏状态，不足以选择{len(self.selected_layers)}层")
        
        # 提取并处理选定的层
        spatial_features = []
        patch_size = int(self.clip_config.image_size / self.clip_config.patch_size)
        
        for layer_idx in self.selected_layers:
            # 获取隐藏状态 [batch, seq_len, dim]
            hidden_state = hidden_states[layer_idx]
            batch_size, seq_len, dim = hidden_state.shape
            
            # 验证通道数
            if dim != self.clip_hidden_dim:
                raise RuntimeError(f"层{layer_idx}的通道数{dim}与期望的{self.clip_hidden_dim}不匹配")
            
            # 第一个token是CLS token，其余是patch tokens
            patch_tokens = hidden_state[:, 1:, :]  # 移除CLS token
            
            # 重塑为空间特征图 [batch, dim, height, width]
            # 确保patch数量等于 patch_size * patch_size
            expected_patch_num = patch_size * patch_size
            if patch_tokens.shape[1] != expected_patch_num:
                raise RuntimeError(f"Patch数量{patch_tokens.shape[1]}与期望的{expected_patch_num}不匹配")
            
            spatial_feat = patch_tokens.transpose(1, 2).contiguous()
            spatial_feat = spatial_feat.view(batch_size, dim, patch_size, patch_size)
            
            spatial_features.append(spatial_feat)
        
        # 多尺度特征融合
        if spatial_features:
            fused_features = self.fpn(spatial_features)
            
            # 地理注意力增强
            attended_features = self.geo_attention(fused_features)
            
            # 投影到嵌入空间
            embeddings = self.feature_projection(attended_features)
            embeddings = self.normalize(embeddings)
            
            # 归一化
            embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        else:
            # 回退到原始CLIP方法
            cls_feat = clip_output.last_hidden_state[:, 0, :]
            embeddings = self.feature_projection(cls_feat.unsqueeze(-1).unsqueeze(-1))
            embeddings = self.normalize(embeddings)
            embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        
        return embeddings

# -------------------------- 损失函数（保持不变） --------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点之间的测地线距离（单位：公里），输入为弧度制，无学习参数"""
    R = 6371.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance

def geodesic_loss(pred_lonlat, true_lonlat):
    """测地线距离损失函数"""
    pred_lon_rad = torch.deg2rad(pred_lonlat[:, 0])
    pred_lat_rad = torch.deg2rad(pred_lonlat[:, 1])
    true_lon_rad = torch.deg2rad(true_lonlat[:, 0])
    true_lat_rad = torch.deg2rad(true_lonlat[:, 1])
    
    distances = haversine_distance(pred_lon_rad, pred_lat_rad, true_lon_rad, true_lat_rad)
    return torch.mean(distances)

def lonlat_to_unit_vector(lonlat):
    """将经纬度转换为单位球面上的三维向量"""
    lon_rad = torch.deg2rad(lonlat[:, 0])
    lat_rad = torch.deg2rad(lonlat[:, 1])
    
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    
    unit_vec = torch.stack([x, y, z], dim=1)
    unit_vec = unit_vec / torch.norm(unit_vec, dim=-1, keepdim=True)
    return unit_vec

class SphericalCosineLoss(nn.Module):
    """基于球面余弦相似度的损失函数"""
    def __init__(self, earth_radius=6371.0):
        super().__init__()
        self.earth_radius = earth_radius

    def forward(self, pred_lonlat, true_lonlat):
        v_pred = lonlat_to_unit_vector(pred_lonlat)
        v_true = lonlat_to_unit_vector(true_lonlat)
        
        cos_theta = torch.sum(v_pred * v_true, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        
        loss = 1 - cos_theta
        theta_rad = torch.arccos(cos_theta)
        geodesic_distance = self.earth_radius * theta_rad
        
        return torch.mean(loss), torch.mean(geodesic_distance)

# -------------------------- 数据加载和辅助函数（保持不变） --------------------------
def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    coords_squeezed = coords_batch.squeeze(1)
    batch_size, num_points, _ = coords_squeezed.shape
    
    coords_flat = coords_squeezed.reshape(-1, 2)
    lon_deg_list = coords_flat[:, 0].cpu().numpy()
    lat_deg_list = coords_flat[:, 1].cpu().numpy()
    
    point_feats = []
    for lon, lat in zip(lon_deg_list, lat_deg_list):
        feat_dict = geo_encoder.encode(lon_deg=lon, lat_deg=lat)
        feat_vec = geo_encoder.to_vector(feat_dict)
        point_feats.append(feat_vec)
    
    point_feats = np.stack(point_feats, axis=0)
    point_feats = torch.tensor(point_feats, dtype=torch.float32).to(device)
    point_feats = point_feats.reshape(batch_size, num_points, -1)
    
    if agg_method == "mean":
        encoded_tensor = point_feats.mean(dim=1)
    elif agg_method == "concat":
        encoded_tensor = point_feats.reshape(batch_size, -1)
    else:
        raise ValueError("agg_method must be 'mean' or 'concat'")
    
    return encoded_tensor

def get_dataloaders(train_root, val_root, batch_size=16, img_size=256):
    clip_processor = CLIPImageProcessor.from_pretrained(
        "/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    )
    clip_processor.size["shortest_edge"] = img_size
    clip_processor.crop_size["height"] = img_size
    clip_processor.crop_size["width"] = img_size

    train_dataset = GeoDataset(root_dir=train_root)
    val_dataset = GeoDataset(root_dir=val_root)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader, clip_processor

# -------------------------- 模型定义（使用新的视觉编码器） --------------------------
class CoordinateEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.normalize = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)

class LonLatDecoder(nn.Module):
    """经纬度Decoder"""
    def __init__(self, embed_dim=512, hidden_dim=512, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, image_embeds):
        raw_output = self.decoder(image_embeds)
        lon = raw_output[:, 0] * 180.0
        lat = raw_output[:, 1] * 90.0
        return torch.stack([lon, lat], dim=1)

class GeoAwareEncoderDecoder(nn.Module):
    """使用地理位置感知视觉编码器的完整模型"""
    def __init__(self, coord_input_dim, embed_dim=512, temperature=0.07):
        super().__init__()
        # 使用新的地理位置感知视觉编码器
        self.image_encoder = GeoAwareCLIPVisionEncoder(embed_dim=embed_dim)
        self.lonlat_decoder = LonLatDecoder(embed_dim=embed_dim)
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.temperature = temperature
        
    def forward(self, images, coordinates):
        image_embeds = self.image_encoder(images)
        pred_lonlat = self.lonlat_decoder(image_embeds)
        coord_embeds = self.coord_encoder(coordinates)
        return image_embeds, coord_embeds, pred_lonlat

# -------------------------- 损失函数和训练逻辑（保持不变） --------------------------
def contrastive_loss(image_embeds, coord_embeds, temperature=0.07):
    batch_size = image_embeds.shape[0]
    logits = torch.matmul(image_embeds, coord_embeds.t()) / temperature
    labels = torch.arange(batch_size, device=image_embeds.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

def train_epoch(model, dataloader, optimizer, spherical_loss_fn, device, geo_encoder, scheduler=None):
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_spherical_loss = 0.0
    total_geodesic_distance = 0.0
    total_batches = 0
    
    with tqdm(dataloader, unit="batch", desc="Training") as tepoch:
        for images, coordinates in tepoch:
            images = images.to(device)
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            true_coords = coordinates.squeeze(1).to(device)
            true_lonlat = true_coords.mean(dim=1)
            
            optimizer.zero_grad()
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            loss_contrastive = contrastive_loss(image_embeds, coord_embeds)
            loss_spherical, _ = spherical_loss_fn(pred_lonlat, true_lonlat)
            
            loss = 0.7 * loss_contrastive + 0.3 * loss_spherical
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive_loss += loss_contrastive.item()
            total_spherical_loss += loss_spherical.item()
            total_batches += 1
            
            tepoch.set_postfix(
                total_loss=loss.item(),
                contrastive_loss=loss_contrastive.item(),
                spherical_loss=loss_spherical.item()
            )
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / total_batches
    avg_contrastive_loss = total_contrastive_loss / total_batches
    avg_spherical_loss = total_spherical_loss / total_batches
    return avg_loss, avg_contrastive_loss, avg_spherical_loss

def validate_epoch(model, dataloader, spherical_loss_fn, device, geo_encoder):
    model.eval()
    total_val_loss = 0.0
    total_val_contrastive_loss = 0.0
    total_val_spherical_loss = 0.0
    total_val_geodesic_distance = 0.0
    total_samples = 0
    
    with torch.no_grad(), tqdm(dataloader, unit="batch", desc="Validation") as tepoch:
        for images, coordinates in tepoch:
            batch_size = images.shape[0]
            total_samples += batch_size
            
            images = images.to(device)
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            true_coords = coordinates.squeeze(1).to(device)
            true_lonlat = true_coords.mean(dim=1)
            
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            val_contrastive_loss = contrastive_loss(image_embeds, coord_embeds)
            val_spherical_loss, val_geodesic_distance = spherical_loss_fn(pred_lonlat, true_lonlat)
            
            val_total_loss = 0.7 * val_contrastive_loss.item() + 0.3 * val_spherical_loss.item()
            
            total_val_loss += val_total_loss
            total_val_contrastive_loss += val_contrastive_loss.item()
            total_val_spherical_loss += val_spherical_loss.item()
            total_val_geodesic_distance += val_geodesic_distance.item() * batch_size
            
            tepoch.set_postfix(
                val_total_loss=total_val_loss / (tepoch.n + 1),
                val_geodesic_dist=val_geodesic_distance.item()
            )
    
    avg_val_loss = total_val_loss / len(dataloader)
    avg_val_contrastive_loss = total_val_contrastive_loss / len(dataloader)
    avg_val_spherical_loss = total_val_spherical_loss / len(dataloader)
    avg_val_geodesic_distance = total_val_geodesic_distance / total_samples
    
    return avg_val_loss, avg_val_contrastive_loss, avg_val_spherical_loss, avg_val_geodesic_distance

def train(model, train_dataloader, val_dataloader, num_epochs, device, geo_encoder, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    spherical_loss_fn = SphericalCosineLoss()
    
    history = {
        'train_total_loss': [], 'train_contrastive_loss': [], 'train_spherical_loss': [],
        'val_total_loss': [], 'val_contrastive_loss': [], 'val_spherical_loss': [], 'val_geodesic_distance': []
    }
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    best_val_loss = float('inf')
    best_val_geodesic_distance = float('inf')
    
    print(f"开始训练（使用地理位置感知编码器），共 {num_epochs} 个epoch")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        train_metrics = train_epoch(model, train_dataloader, optimizer, spherical_loss_fn, device, geo_encoder, scheduler)
        val_metrics = validate_epoch(model, val_dataloader, spherical_loss_fn, device, geo_encoder)
        
        # 记录历史
        history['train_total_loss'].append(train_metrics[0])
        history['train_contrastive_loss'].append(train_metrics[1])
        history['train_spherical_loss'].append(train_metrics[2])
        history['val_total_loss'].append(val_metrics[0])
        history['val_contrastive_loss'].append(val_metrics[1])
        history['val_spherical_loss'].append(val_metrics[2])
        history['val_geodesic_distance'].append(val_metrics[3])

        # 打印指标
        print(
            f"【训练】总损失: {train_metrics[0]:.4f}, 对比损失: {train_metrics[1]:.4f}, 球面损失: {train_metrics[2]:.4f}\n"
            f"【验证】总损失: {val_metrics[0]:.4f}, 对比损失: {val_metrics[1]:.4f}, 球面损失: {val_metrics[2]:.4f}\n"
            f"【验证】平均测地线距离: {val_metrics[3]:.2f} 公里"
        )
        
        # 保存模型
        if save_path is not None:
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics, 'val_metrics': val_metrics
                }, os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
            
            if val_metrics[0] < best_val_loss:
                best_val_loss = val_metrics[0]
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_val_total_loss': best_val_loss,
                    'corresponding_geodesic_distance': val_metrics[3]
                }, os.path.join(save_path, "best_model_by_loss.pth"))
                print(f" 更新最佳损失模型！当前最佳验证总损失: {best_val_loss:.4f}")
            
            if val_metrics[3] < best_val_geodesic_distance:
                best_val_geodesic_distance = val_metrics[3]
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_val_geodesic_distance': best_val_geodesic_distance,
                    'corresponding_total_loss': val_metrics[0]
                }, os.path.join(save_path, "best_model_by_geodesic.pth"))
                print(f"✅ 更新最佳测地线距离模型！当前最佳测地线距离: {best_val_geodesic_distance:.2f} 公里")
    
    if save_path is not None:
        torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
        np.save(os.path.join(save_path, "training_history.npy"), history)
    
    return history

# -------------------------- 主函数 --------------------------
def main():
    # 配置参数
    train_root = "/home/ma-user/work/data/data_geoclip/train"
    val_root = "/home/ma-user/work/data/data_geoclip/val"
    batch_size = 32  # 减小batch_size以适应更大的模型
    embed_dim = 512
    num_epochs = 50
    img_size = 256
    save_path = f"geo_aware_clip_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("加载数据...")
    train_dataloader, val_dataloader, clip_processor = get_dataloaders(
        train_root, val_root, batch_size, img_size
    )
    
    # 查看样本形状
    sample_train_imgs, sample_train_coords = next(iter(train_dataloader))
    print(f"训练集样本图像形状: {sample_train_imgs.shape}")
    print(f"训练集样本坐标形状: {sample_train_coords.shape}")

    print("初始化地理编码器...")
    geo_encoder_cfg = GeoEncoderConfig(
        lon_harmonics=4,
        lat_harmonics=3,
        add_unitvec=True
    )
    geo_encoder = MultiScaleGeoEncoder(cfg=geo_encoder_cfg)

    print("计算坐标编码维度...")
    sample_encoded = batch_encode_coords(sample_train_coords[:1], geo_encoder)
    coord_input_dim = sample_encoded.shape[1]
    print(f"编码后的坐标维度: {coord_input_dim}")

    print("初始化地理位置感知的CLIP模型...")
    model = GeoAwareEncoderDecoder(coord_input_dim, embed_dim=embed_dim).to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n启动训练+验证...")
    history = train(
        model, 
        train_dataloader, 
        val_dataloader, 
        num_epochs, 
        device, 
        geo_encoder, 
        save_path
    )
    
    print("\n训练+验证完成!")
    print(f"训练历史已保存至: {os.path.join(save_path, 'training_history.npy')}")

if __name__ == "__main__":
    main()

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


from CorDataset_1 import GeoDataset
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
device = torch.device("npu:1" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")
def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点之间的测地线距离（单位：公里），输入为弧度制，无学习参数"""
    R = 6371.0  # 固定地球平均半径（公里），无学习参数
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance

def geodesic_loss(pred_lonlat, true_lonlat):
    """
    测地线距离损失函数（纯函数，无学习参数）
    输入：pred_lonlat/true_lonlat - (batch_size, 2)，格式[经度, 纬度]（角度制）
    输出：批次平均测地线距离（公里）
    """
    # 角度制转弧度制（固定数学转换，无学习参数）
    pred_lon_rad = torch.deg2rad(pred_lonlat[:, 0])
    pred_lat_rad = torch.deg2rad(pred_lonlat[:, 1])
    true_lon_rad = torch.deg2rad(true_lonlat[:, 0])
    true_lat_rad = torch.deg2rad(true_lonlat[:, 1])
    
    # 计算每个样本的测地线距离（固定哈弗辛公式，无学习参数）
    distances = haversine_distance(pred_lon_rad, pred_lat_rad, true_lon_rad, true_lat_rad)
    return torch.mean(distances)
# -------------------------- 核心优化：基于SLERP的损失函数 --------------------------
def lonlat_to_unit_vector(lonlat):
    """
    将经纬度转换为单位球面上的三维向量。
    输入: lonlat - (batch_size, 2)，格式[经度, 纬度]（角度制）
    输出: unit_vec - (batch_size, 3)，单位球面向量
    """
    lon_rad = torch.deg2rad(lonlat[:, 0])
    lat_rad = torch.deg2rad(lonlat[:, 1])
    
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    
    # 确保向量是单位长度（处理数值误差）
    unit_vec = torch.stack([x, y, z], dim=1)
    unit_vec = unit_vec / torch.norm(unit_vec, dim=-1, keepdim=True)
    return unit_vec

class SphericalCosineLoss(nn.Module):
    """
    基于球面余弦相似度的损失函数。
    它优化的是预测点和真实点在单位球面上的夹角，间接等价于优化测地线距离。
    """
    def __init__(self, earth_radius=6371.0):
        super().__init__()
        self.earth_radius = earth_radius

    def forward(self, pred_lonlat, true_lonlat):
        """
        输入: pred_lonlat, true_lonlat - (batch_size, 2)，角度制经纬度
        输出: loss (优化用), avg_geodesic_distance (评估用, 单位：公里)
        """
        v_pred = lonlat_to_unit_vector(pred_lonlat)
        v_true = lonlat_to_unit_vector(true_lonlat)
        
        # 计算余弦相似度，clip避免数值问题导致的超出范围
        cos_theta = torch.sum(v_pred * v_true, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        
        # 损失函数：1 - cos(theta)，当theta=0时损失为0，theta增大损失增大
        loss = 1 - cos_theta
        
        # 计算实际的测地线距离，用于评估
        theta_rad = torch.arccos(cos_theta)
        geodesic_distance = self.earth_radius * theta_rad
        
        return torch.mean(loss), torch.mean(geodesic_distance)

# -------------------------- 原有辅助函数（保持不变） --------------------------
def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    #coords_squeezed = coords_batch.squeeze(1)  # (batch_size, num_points, 2)
    coords_squeezed = coords_batch  
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
        encoded_tensor = point_feats.mean(dim=1)  # (batch_size, D)
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

# -------------------------- 模型定义（优化Decoder输出） --------------------------
class CLIPViTImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
        super().__init__()
        self.clip_config = CLIPVisionConfig.from_pretrained(clip_model_name)
        self.clip_config.image_size = 256
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_model_name,
            config=self.clip_config,
            ignore_mismatched_sizes=True
        )
        self.clip_hidden_dim = self.clip_config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(self.clip_hidden_dim, self.clip_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.clip_hidden_dim // 2, embed_dim)
        )
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, x):
        clip_output = self.clip_vision_model(pixel_values=x)
        cls_feat = clip_output.last_hidden_state[:, 0, :]
        x = self.projection(cls_feat)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)

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
    """经纬度Decoder：增加输出约束，防止数值异常"""
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
        
        # 输出层：使用tanh将输出限制在[-1, 1]，再缩放至经纬度范围
        layers.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, image_embeds):
        raw_output = self.decoder(image_embeds)
        # 经度范围: [-180, 180], 纬度范围: [-90, 90]
        lon = raw_output[:, 0] * 180.0  # tanh输出[-1,1] → 经度[-180,180]
        lat = raw_output[:, 1] * 90.0   # tanh输出[-1,1] → 纬度[-90,90]
        return torch.stack([lon, lat], dim=1)

class EncoderDecoderCLIP(nn.Module):
    def __init__(self, coord_input_dim, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = CLIPViTImageEncoder(embed_dim=embed_dim)
        self.lonlat_decoder = LonLatDecoder(embed_dim=embed_dim)
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.temperature = temperature
        
    def forward(self, images, coordinates):
        image_embeds = self.image_encoder(images)
        pred_lonlat = self.lonlat_decoder(image_embeds)
        coord_embeds = self.coord_encoder(coordinates)
        return image_embeds, coord_embeds, pred_lonlat

# -------------------------- 损失函数和训练逻辑 --------------------------
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
            #true_lonlat = true_coords.mean(dim=1)
            true_lonlat=true_coords
            optimizer.zero_grad()
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 计算损失
            loss_contrastive = contrastive_loss(image_embeds, coord_embeds)
            loss_spherical, _ = spherical_loss_fn(pred_lonlat, true_lonlat)
            
            # 组合损失
            loss = 0.7 * loss_contrastive + 0.3 * loss_spherical
            
            loss.backward()
            optimizer.step()
            
            # 累计指标
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
            #true_lonlat = true_coords.mean(dim=1)
            true_lonlat =true_coords
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            val_contrastive_loss = contrastive_loss(image_embeds, coord_embeds)
            val_spherical_loss, val_geodesic_distance = spherical_loss_fn(pred_lonlat, true_lonlat)
            # val_spherical_loss= geodesic_loss(pred_lonlat, true_lonlat)
            # val_geodesic_distance=val_spherical_loss
            val_total_loss = 0.7 * val_contrastive_loss.item() + 0.3 * val_spherical_loss.item()
            
            # 累计指标
            total_val_loss += val_total_loss
            total_val_contrastive_loss += val_contrastive_loss.item()
            total_val_spherical_loss += val_spherical_loss.item()
            total_val_geodesic_distance += val_geodesic_distance.item() * batch_size # 累计总距离
            
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
    
    print(f"开始训练（使用 SphericalCosineLoss），共 {num_epochs} 个epoch")
    print("指标说明：val_geodesic_distance为平均测地线距离（单位：公里），越小表示预测越准")
    
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
            # 每5个epoch保存一次
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics, 'val_metrics': val_metrics
                }, os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
            
            # 保存基于总损失的最佳模型
            if val_metrics[0] < best_val_loss:
                best_val_loss = val_metrics[0]
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_val_total_loss': best_val_loss,
                    'corresponding_geodesic_distance': val_metrics[3]
                }, os.path.join(save_path, "best_model_by_loss.pth"))
                print(f" 更新最佳损失模型！当前最佳验证总损失: {best_val_loss:.4f}")
            
            # 保存基于测地线距离的最佳模型
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
    train_root = "/home/ma-user/work/data/data_geoclip/train_1"
    val_root = "/home/ma-user/work/data/data_geoclip/val_1"
    batch_size = 64
    embed_dim = 512
    num_epochs = 50
    img_size = 256
    save_path = f"encoder_decoder_clip_models_slerp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    print("初始化Encoder-Decoder CLIP模型...")
    model = EncoderDecoderCLIP(coord_input_dim, embed_dim=embed_dim).to(device)
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
    print(f"最佳损失模型: {os.path.join(save_path, 'best_model_by_loss.pth')}")
    print(f"最佳测地线距离模型: {os.path.join(save_path, 'best_model_by_geodesic.pth')}")

if __name__ == "__main__":
    main()
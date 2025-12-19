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
device = torch.device("npu:1" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------------- 测地线距离计算及统计（训练用归一化） --------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点之间的测地线距离（单位：公里），输入为弧度制"""
    R = 6371.0  # 地球平均半径（公里）
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_geodesic_stats(train_dataloader, device):
    """预计算训练集真实坐标的测地线距离分布（用于训练损失归一化）"""
    all_distances = []
    with torch.no_grad():
        for _, coordinates in tqdm(train_dataloader, desc="计算测地线距离分布"):
            true_coords = coordinates.squeeze(1).to(device)  # (batch_size, num_points, 2)
            true_lonlat = true_coords.mean(dim=1)  # 同训练逻辑的真实坐标 (batch_size, 2)
            
            # 计算同批次内样本间的距离，模拟预测误差分布
            batch_size = true_lonlat.shape[0]
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    lon1, lat1 = true_lonlat[i]
                    lon2, lat2 = true_lonlat[j]
                    dist = haversine_distance(
                        torch.deg2rad(lon1), torch.deg2rad(lat1),
                        torch.deg2rad(lon2), torch.deg2rad(lat2)
                    )
                    all_distances.append(dist.item())
    
    # 处理空列表情况（极端小数据集）
    if not all_distances:
        return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 10000.0}
    
    all_distances = np.array(all_distances)
    return {
        "mean": np.mean(all_distances),
        "std": np.std(all_distances),
        "min": np.min(all_distances),
        "max": np.max(all_distances)
    }

class NormalizedGeodesicLoss(nn.Module):
    """训练用：支持归一化的测地线损失模块（返回归一化损失和原始距离）"""
    def __init__(self, stats_dict, norm_type="standardize"):
        super().__init__()
        self.stats = stats_dict
        self.norm_type = norm_type  # 标准化(standardize)或归一化(minmax)
        self.eps = 1e-6  # 避免除以零

    def forward(self, pred_lonlat, true_lonlat):
        # 计算原始测地线距离（公里）
        pred_lon_rad = torch.deg2rad(pred_lonlat[:, 0])
        pred_lat_rad = torch.deg2rad(pred_lonlat[:, 1])
        true_lon_rad = torch.deg2rad(true_lonlat[:, 0])
        true_lat_rad = torch.deg2rad(true_lonlat[:, 1])
        
        distances = haversine_distance(pred_lon_rad, pred_lat_rad, true_lon_rad, true_lat_rad)
        original_mean = torch.mean(distances)  # 原始距离均值（用于监控）
        
        # 归一化处理（仅用于损失计算）
        if self.norm_type == "standardize":
            normed_distances = (distances - self.stats["mean"]) / (self.stats["std"] + self.eps)
        elif self.norm_type == "minmax":
            normed_distances = (distances - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + self.eps)
        else:
            raise ValueError("norm_type must be 'standardize' or 'minmax'")
        
        # 返回归一化损失（用于反向传播）和原始距离均值（用于日志）
        return torch.mean(torch.abs(normed_distances)), original_mean

class GeodesicLoss(nn.Module):
    """验证用：仅计算原始测地线距离（无归一化）"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_lonlat, true_lonlat):
        pred_lon_rad = torch.deg2rad(pred_lonlat[:, 0])
        pred_lat_rad = torch.deg2rad(pred_lonlat[:, 1])
        true_lon_rad = torch.deg2rad(true_lonlat[:, 0])
        true_lat_rad = torch.deg2rad(true_lonlat[:, 1])
        
        distances = haversine_distance(pred_lon_rad, pred_lat_rad, true_lon_rad, true_lat_rad)
        return torch.mean(distances)

# -------------------------- 坐标编码函数（保持不变） --------------------------
def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    coords_squeezed = coords_batch.squeeze(1)  # (batch_size, num_points, 2)
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

# -------------------------- 数据集加载（训练验证共用处理器，单视图） --------------------------
def get_dataloaders(train_root, val_root, batch_size=16, img_size=256):
    # 统一使用相同的处理器（单视图处理，无Ten Crop预处理）
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

# -------------------------- 模型定义（保持不变） --------------------------
class CLIPViTImageEncoder(nn.Module):
    """图像Encoder：负责提取图像特征"""
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
    """坐标Encoder：用于对比学习的坐标嵌入生成"""
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
    """经纬度Decoder：接收图像Encoder输出，预测经纬度"""
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
        
        layers.append(nn.Linear(hidden_dim, 2))  # 输出：经度、纬度
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, image_embeds):
        return self.decoder(image_embeds)

class EncoderDecoderCLIP(nn.Module):
    """Encoder-Decoder核心模型"""
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

# -------------------------- 损失函数（训练/验证分离） --------------------------
def contrastive_loss(image_embeds, coord_embeds, temperature=0.07):
    batch_size = image_embeds.shape[0]
    logits = torch.matmul(image_embeds, coord_embeds.t()) / temperature
    labels = torch.arange(batch_size, device=image_embeds.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

# 训练用：归一化测地线损失
train_lonlat_criterion = None  # 动态初始化（需要训练集统计信息）
# 验证用：原始测地线损失
val_lonlat_criterion = GeodesicLoss()

# -------------------------- 训练循环（保持归一化损失逻辑） --------------------------
def train_epoch(model, dataloader, optimizer, device, geo_encoder, scheduler=None):
    global train_lonlat_criterion
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_normed_geodesic_loss = 0.0  # 训练用：归一化损失
    total_raw_geodesic_loss = 0.0     # 训练用：原始距离（监控用）
    total_batches = 0
    
    with tqdm(dataloader, unit="batch", desc="Training") as tepoch:
        for images, coordinates in tepoch:
            batch_size = images.shape[0]
            images = images.to(device)  
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            # 真实经纬度（平均坐标）
            true_coords = coordinates.squeeze(1).to(device)
            true_lonlat = true_coords.mean(dim=1)  # (batch_size, 2)
            
            optimizer.zero_grad()
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 损失计算（训练用归一化损失）
            loss_contrastive = contrastive_loss(image_embeds, coord_embeds)
            normed_geo_loss, raw_geo_loss = train_lonlat_criterion(pred_lonlat, true_lonlat)
            loss = 0.3 * loss_contrastive + 0.7 * normed_geo_loss  # 量级对齐加权
            
            loss.backward()
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_contrastive_loss += loss_contrastive.item()
            total_normed_geodesic_loss += normed_geo_loss.item()
            total_raw_geodesic_loss += raw_geo_loss.item()
            total_batches += 1
            
            tepoch.set_postfix(
                total_loss=loss.item(),
                contrastive_loss=loss_contrastive.item(),
                normed_geo_loss=normed_geo_loss.item(),
                raw_geo_loss=f"{raw_geo_loss.item():.2f}km"
            )
    
    if scheduler is not None:
        scheduler.step()
    
    # 计算平均指标
    avg_loss = total_loss / total_batches
    avg_contrastive_loss = total_contrastive_loss / total_batches
    avg_normed_geo_loss = total_normed_geodesic_loss / total_batches
    avg_raw_geo_loss = total_raw_geodesic_loss / total_batches
    return avg_loss, avg_contrastive_loss, avg_normed_geo_loss, avg_raw_geo_loss

# -------------------------- 验证循环（替换为单视图直接预测，用原始损失） --------------------------
def validate_epoch(model, dataloader, device, geo_encoder):
    model.eval()
    total_val_loss = 0.0
    total_val_contrastive_loss = 0.0
    total_val_geodesic_loss = 0.0  # 验证用：原始测地线距离
    total_samples = 0
    total_val_batches = 0
    
    with torch.no_grad(), tqdm(dataloader, unit="batch", desc="Validation") as tepoch:
        for images, coordinates in tepoch:
            batch_size = images.shape[0]
            total_samples += batch_size
            
            images = images.to(device)
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            # 真实经纬度
            true_coords = coordinates.squeeze(1).to(device)
            true_lonlat = true_coords.mean(dim=1)  # (batch_size, 2)
            
            # 单视图直接预测（无Ten Crop）
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 损失计算（验证用原始损失）
            val_contrastive_loss = contrastive_loss(image_embeds, coord_embeds)
            val_geo_loss = val_lonlat_criterion(pred_lonlat, true_lonlat).item()
            val_total_loss = 0.3 * val_contrastive_loss.item() + 0.7 * val_geo_loss  # 保持权重一致
            
            # 累计指标
            total_val_loss += val_total_loss
            total_val_contrastive_loss += val_contrastive_loss.item()
            total_val_geodesic_loss += val_geo_loss * batch_size  # 按样本数累计
            total_val_batches += 1
            
            tepoch.set_postfix(
                val_total_loss=total_val_loss / total_val_batches,
                val_contrastive_loss=val_contrastive_loss.item(),
                val_geo_loss=f"{val_geo_loss:.2f}km"
            )
    
    # 计算平均指标
    avg_val_loss = total_val_loss / total_val_batches
    avg_val_contrastive_loss = total_val_contrastive_loss / total_val_batches
    avg_val_geo_loss = total_val_geodesic_loss / total_samples  # 按样本数平均
    return avg_val_loss, avg_val_contrastive_loss, avg_val_geo_loss

# -------------------------- 训练主逻辑 --------------------------
def train(model, train_dataloader, val_dataloader, num_epochs, device, geo_encoder, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {
        'train_total_loss': [],
        'train_contrastive_loss': [],
        'train_normed_geodesic_loss': [],  # 训练：归一化损失
        'train_raw_geodesic_loss': [],     # 训练：原始距离
        'val_total_loss': [],
        'val_contrastive_loss': [],
        'val_geodesic_loss': []            # 验证：原始距离
    }
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    best_val_loss = float('inf')
    best_val_geo_loss = float('inf')
    
    print(f"开始训练（训练用归一化测地线损失，验证用单视图原始损失），共 {num_epochs} 个epoch")
    print("指标说明：")
    print(" - 训练：normed_geo_loss（归一化损失，用于反向传播），raw_geo_loss（原始公里数，监控用）")
    print(" - 验证：val_geodesic_loss（平均原始测地线距离，公里）")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # 训练（保持归一化损失逻辑）
        train_total_loss, train_contrastive_loss, train_normed_geo, train_raw_geo = train_epoch(
            model, train_dataloader, optimizer, device, geo_encoder, scheduler
        )
        
        # 验证（单视图原始损失）
        val_total_loss, val_contrastive_loss, val_geo_loss = validate_epoch(
            model, val_dataloader, device, geo_encoder
        )
        
        # 记录历史
        history['train_total_loss'].append(train_total_loss)
        history['train_contrastive_loss'].append(train_contrastive_loss)
        history['train_normed_geodesic_loss'].append(train_normed_geo)
        history['train_raw_geodesic_loss'].append(train_raw_geo)
        history['val_total_loss'].append(val_total_loss)
        history['val_contrastive_loss'].append(val_contrastive_loss)
        history['val_geodesic_loss'].append(val_geo_loss)
        
        # 打印指标
        print(
            f"【训练】总损失: {train_total_loss:.4f}, 对比损失: {train_contrastive_loss:.4f}, "
            f"归一化测地线损失: {train_normed_geo:.4f}, 原始测地线距离: {train_raw_geo:.2f} 公里"
        )
        print(
            f"【验证】总损失: {val_total_loss:.4f}, 对比损失: {val_contrastive_loss:.4f}, "
            f"平均测地线距离: {val_geo_loss:.2f} 公里"
        )
        
        # 模型保存
        if save_path is not None:
            # 每5个epoch保存一次
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': {
                        'total_loss': train_total_loss,
                        'contrastive_loss': train_contrastive_loss,
                        'normed_geo_loss': train_normed_geo,
                        'raw_geo_loss': train_raw_geo
                    },
                    'val_metrics': {
                        'total_loss': val_total_loss,
                        'contrastive_loss': val_contrastive_loss,
                        'geo_loss': val_geo_loss
                    }
                }, os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
            
            # 保存最佳总损失模型
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_total_loss': best_val_loss,
                    'corresponding_geo_loss': val_geo_loss
                }, os.path.join(save_path, "best_model_by_loss.pth"))
                print(f"✅ 更新最佳总损失模型！当前最佳: {best_val_loss:.4f}")
            
            # 保存最佳测地线距离模型
            if val_geo_loss < best_val_geo_loss:
                best_val_geo_loss = val_geo_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_geo_loss': best_val_geo_loss,
                    'corresponding_total_loss': val_total_loss
                }, os.path.join(save_path, "best_model_by_geo.pth"))
                print(f"✅ 更新最佳测地线距离模型！当前最佳: {best_val_geo_loss:.2f} 公里")
    
    if save_path is not None:
        torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
        np.save(os.path.join(save_path, "training_history.npy"), history)
    
    return history

# -------------------------- main函数 --------------------------
def main():
    global train_lonlat_criterion
    train_root = "/home/ma-user/work/data/data_geoclip/train_1"
    val_root = "/home/ma-user/work/data/data_geoclip/val_1"
    batch_size = 64
    embed_dim = 512
    num_epochs = 50
    img_size = 256
    save_path = f"encoder_decoder_clip_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("加载数据...")
    train_dataloader, val_dataloader, clip_processor = get_dataloaders(
        train_root, val_root, batch_size, img_size
    )
    
    sample_train_imgs, sample_train_coords = next(iter(train_dataloader))
    print(f"训练集样本图像形状: {sample_train_imgs.shape}")
    print(f"训练集样本坐标形状: {sample_train_coords.shape}")

    # 计算训练集测地线距离统计（用于训练损失归一化）
    print("计算训练集测地线距离统计信息...")
    geodesic_stats = calculate_geodesic_stats(train_dataloader, device)
    print(f"测地线距离统计：")
    print(f"  均值: {geodesic_stats['mean']:.2f}km, 标准差: {geodesic_stats['std']:.2f}km")
    print(f"  最小值: {geodesic_stats['min']:.2f}km, 最大值: {geodesic_stats['max']:.2f}km")

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
    
    # 初始化训练用归一化损失
    train_lonlat_criterion = NormalizedGeodesicLoss(geodesic_stats, norm_type="standardize")
    
    print("\n启动训练+验证...")
    history = train(
        model, 
        train_dataloader, 
        val_dataloader, 
        num_epochs, 
        device, 
        geo_encoder,
        save_path=save_path
    )
    
    print("\n训练+验证完成!")
    print(f"训练历史已保存至: {os.path.join(save_path, 'training_history.npy')}")
    print(f"最佳模型保存至: {save_path}")

if __name__ == "__main__":
    main()
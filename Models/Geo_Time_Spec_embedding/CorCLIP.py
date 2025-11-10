import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")

# 坐标批量编码函数
def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    coords_squeezed = coords_batch.squeeze(1)  # 形状：(batch_size, num_points, 2)
    batch_size, num_points, _ = coords_squeezed.shape
    encoded_list = []
    
    for i in range(batch_size):
        sample_points = coords_squeezed[i].cpu().numpy()
        point_feats = []
        for point in sample_points:
            lon_deg, lat_deg = point[0], point[1]
            feat_dict = geo_encoder.encode(lon_deg=lon_deg, lat_deg=lat_deg)
            feat_vec = geo_encoder.to_vector(feat_dict)
            point_feats.append(feat_vec)
        
        if agg_method == "mean":
            aggregated = np.mean(point_feats, axis=0)
        elif agg_method == "concat":
            aggregated = np.concatenate(point_feats, axis=0)
        else:
            raise ValueError("agg_method must be 'mean' or 'concat'")
        
        encoded_list.append(aggregated)
    
    encoded_tensor = torch.tensor(encoded_list, dtype=torch.float32).to(device)
    return encoded_tensor

# -------------------------- 数据集加载（按指定路径加载训练/验证集） --------------------------
def get_dataloaders(train_root, val_root, batch_size=16, img_size=256):
    # 1. 初始化CLIP图像处理器，强制resize到256×256
    clip_processor = CLIPImageProcessor.from_pretrained(
        "/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    )
    clip_processor.size["shortest_edge"] = img_size  # 最短边=256
    clip_processor.crop_size["height"] = img_size    # 裁剪高度=256
    clip_processor.crop_size["width"] = img_size     # 裁剪宽度=256

    # 2. 加载训练集和验证集（传入CLIP预处理）
    train_dataset = GeoDataset(root_dir=train_root)
    val_dataset = GeoDataset(root_dir=val_root)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 3. 创建DataLoader
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

# -------------------------- 基于transformers-CLIP实现图像编码器 --------------------------
class CLIPViTImageEncoder(nn.Module):
    """使用transformers库的CLIP ViT作为图像编码器，输入尺寸256×256"""
    def __init__(self, embed_dim=512, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
        super().__init__()
        self.clip_config = CLIPVisionConfig.from_pretrained(clip_model_name)
        self.clip_config.image_size = 256  # 强制模型接受256×256输入
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
        cls_feat = clip_output.last_hidden_state[:, 0, :]  # 取cls-token特征
        x = self.projection(cls_feat)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)  # L2归一化

# 坐标编码器
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

# -------------------------- 新增坐标解码器（用于反推经纬度） --------------------------
class CoordinateDecoder(nn.Module):
    """从坐标嵌入反推真实经纬度（lon, lat），输出单位为度"""
    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # 输出：[lon, lat]（2维）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 防止过拟合

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        lonlat = self.fc3(x)  # 直接输出经纬度（度）
        return lonlat

# -------------------------- CLIP模型整合解码器 --------------------------
class CLIPModel(nn.Module):
    def __init__(self, coord_input_dim, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = CLIPViTImageEncoder(embed_dim=embed_dim)
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.coord_decoder = CoordinateDecoder(embed_dim)  # 新增解码器
        self.temperature = temperature
        
    def forward(self, images, coordinates):
        image_embeds = self.image_encoder(images)
        coord_embeds = self.coord_encoder(coordinates)
        pred_lonlat = self.coord_decoder(coord_embeds)  # 反推预测经纬度
        return image_embeds, coord_embeds, pred_lonlat

# 对比损失函数
def contrastive_loss(image_embeds, coord_embeds, temperature=0.07):
    batch_size = image_embeds.shape[0]
    logits = torch.matmul(image_embeds, coord_embeds.t()) / temperature
    labels = torch.arange(batch_size, device=image_embeds.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2

# -------------------------- 训练循环适配解码器（新增经纬度预测损失） --------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, geo_encoder, scheduler=None):
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_lonlat_mae_loss = 0.0
    total_batches = 0
    
    # 经纬度MAE损失（用于优化解码器）
    lonlat_criterion = nn.L1Loss()
    
    with tqdm(dataloader, unit="batch", desc="Training") as tepoch:
        for images, coordinates in tepoch:
            batch_size = images.shape[0]
            images = images.to(device)  
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            # 计算真实经纬度（与编码时的mean聚合一致：取num_points个点的均值）
            true_coords_np = coordinates.squeeze(1).cpu().numpy()  # (batch_size, num_points, 2)
            true_lonlat = np.mean(true_coords_np, axis=1)  # (batch_size, 2)：[lon, lat]
            true_lonlat = torch.tensor(true_lonlat, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            # 前向传播（新增pred_lonlat输出）
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 混合损失：对比损失 + 经纬度MAE损失（权重可调整）
            loss_contrastive = criterion(image_embeds, coord_embeds)
            loss_lonlat_mae = lonlat_criterion(pred_lonlat, true_lonlat)
            loss = loss_contrastive + 0.1 * loss_lonlat_mae  # 0.1为MAE损失权重
            
            loss.backward()
            optimizer.step()
            
            # 累计各类损失
            total_loss += loss.item()
            total_contrastive_loss += loss_contrastive.item()
            total_lonlat_mae_loss += loss_lonlat_mae.item()
            total_batches += 1
            
            # 进度条显示混合损失和MAE损失
            tepoch.set_postfix(
                total_loss=loss.item(),
                contrastive_loss=loss_contrastive.item(),
                lonlat_mae=loss_lonlat_mae.item()
            )
    
    if scheduler is not None:
        scheduler.step()
    
    # 返回平均损失
    avg_loss = total_loss / total_batches
    avg_contrastive_loss = total_contrastive_loss / total_batches
    avg_lonlat_mae = total_lonlat_mae_loss / total_batches
    return avg_loss, avg_contrastive_loss, avg_lonlat_mae

# -------------------------- 验证循环删除top-1准确率计算 --------------------------
def validate_epoch(model, dataloader, criterion, device, geo_encoder, temperature=0.07):
    model.eval()
    total_val_loss = 0.0
    total_val_contrastive_loss = 0.0
    total_val_lonlat_mae = 0.0  # 经纬度平均绝对误差
    total_mean_error = 0.0
    total_samples = 0
    total_val_batches = 0
    
    lonlat_criterion = nn.L1Loss()  # 计算经纬度MAE
    
    with torch.no_grad(), tqdm(dataloader, unit="batch", desc="Validation") as tepoch:
        for images, coordinates in tepoch:
            batch_size = images.shape[0]
            total_samples += batch_size
            
            images = images.to(device)
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            # 计算真实经纬度（与训练时一致：取num_points个点的均值）
            true_coords_np = coordinates.squeeze(1).cpu().numpy()
            true_lonlat = np.mean(true_coords_np, axis=1)
            true_lonlat = torch.tensor(true_lonlat, dtype=torch.float32).to(device)
            
            # 前向传播（含经纬度预测）
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 1. 计算各类损失
            val_contrastive_loss = criterion(image_embeds, coord_embeds)
            val_lonlat_mae = lonlat_criterion(pred_lonlat, true_lonlat).item()
            val_total_loss = val_contrastive_loss.item() + 0.1 * val_lonlat_mae
            
            # 2. 计算正样本对平均误差（保留，仅删除top-1相关）
            img_to_coord_logits = torch.matmul(image_embeds, coord_embeds.t()) / temperature
            pos_similarities = torch.diagonal(img_to_coord_logits) * temperature
            pos_errors = 1 - pos_similarities
            batch_mean_error = torch.mean(pos_errors).item()
            total_mean_error += batch_mean_error * batch_size
            
            # 累计指标
            total_val_loss += val_total_loss
            total_val_contrastive_loss += val_contrastive_loss.item()
            total_val_lonlat_mae += val_lonlat_mae * batch_size  # 经纬度MAE加权累计
            total_val_batches += 1
            
            # 进度条显示验证指标（删除top1_acc）
            tepoch.set_postfix(
                val_total_loss=val_total_loss,
                val_contrastive_loss=val_contrastive_loss.item(),
                val_lonlat_mae=val_lonlat_mae,
                mean_error=batch_mean_error
            )
    
    # 计算全局平均指标
    avg_val_loss = total_val_loss / total_val_batches
    avg_val_contrastive_loss = total_val_contrastive_loss / total_val_batches
    avg_val_lonlat_mae = total_val_lonlat_mae / total_samples  # 经纬度平均绝对误差
    avg_mean_error = total_mean_error / total_samples
    
    # 返回值删除top1_acc
    return avg_val_loss, avg_val_contrastive_loss, avg_mean_error, avg_val_lonlat_mae

# -------------------------- 训练主逻辑删除top-1相关所有内容 --------------------------
def train(model, train_dataloader, val_dataloader, num_epochs, device, geo_encoder, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # 历史记录删除top-1相关字段
    history = {
        'train_total_loss': [],
        'train_contrastive_loss': [],
        'train_lonlat_mae': [],
        'val_total_loss': [],
        'val_contrastive_loss': [],
        'val_mean_error': [],
        'val_lonlat_mae': []  # 经纬度平均绝对误差
    }
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # 删除最佳top-1准确率相关记录
    best_val_loss = float('inf')
    best_val_mean_error = float('inf')
    best_val_lonlat_mae = float('inf')  # 经纬度MAE越小越好
    
    print(f"开始训练，共 {num_epochs} 个epoch")
    print("指标说明：val_lonlat_mae为经纬度平均绝对误差（单位：度），越小表示预测越准")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # 训练阶段
        train_total_loss, train_contrastive_loss, train_lonlat_mae = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            contrastive_loss, 
            device, 
            geo_encoder, 
            scheduler
        )
        
        # 验证阶段
        val_total_loss, val_contrastive_loss, val_mean_error, val_lonlat_mae = validate_epoch(
            model, 
            val_dataloader, 
            contrastive_loss, 
            device, 
            geo_encoder,
            temperature=model.temperature
        )
        
        # 记录历史指标
        history['train_total_loss'].append(train_total_loss)
        history['train_contrastive_loss'].append(train_contrastive_loss)
        history['train_lonlat_mae'].append(train_lonlat_mae)
        history['val_total_loss'].append(val_total_loss)
        history['val_contrastive_loss'].append(val_contrastive_loss)
        history['val_mean_error'].append(val_mean_error)
        history['val_lonlat_mae'].append(val_lonlat_mae)
        
        # 打印当前epoch所有指标
        print(
            f"【训练】总损失: {train_total_loss:.4f}, 对比损失: {train_contrastive_loss:.4f}, 经纬度MAE: {train_lonlat_mae:.4f}\n"
            f"【验证】总损失: {val_total_loss:.4f}, 对比损失: {val_contrastive_loss:.4f}\n"
            f"【验证】正样本平均误差: {val_mean_error:.4f}\n"
            f"【验证】经纬度平均绝对误差: {val_lonlat_mae:.4f} 度"
        )
        
        # 保存每5个epoch的完整模型
        if save_path is not None and (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': {
                    'total_loss': train_total_loss,
                    'contrastive_loss': train_contrastive_loss,
                    'lonlat_mae': train_lonlat_mae
                },
                'val_metrics': {
                    'total_loss': val_total_loss,
                    'contrastive_loss': val_contrastive_loss,
                    'mean_error': val_mean_error,
                    'lonlat_mae': val_lonlat_mae
                }
            }, os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
        
        # 保存基于最佳验证总损失的模型
        if save_path is not None and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_total_loss': best_val_loss,
                'corresponding_metrics': {
                    'mean_error': val_mean_error,
                    'lonlat_mae': val_lonlat_mae
                }
            }, os.path.join(save_path, "best_model_by_loss.pth"))
            print(f" 更新最佳损失模型！当前最佳验证总损失: {best_val_loss:.4f}")
        
        # 保存基于最佳经纬度MAE的模型
        if save_path is not None and val_lonlat_mae < best_val_lonlat_mae:
            best_val_lonlat_mae = val_lonlat_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_lonlat_mae': best_val_lonlat_mae,
                'corresponding_metrics': {
                    'total_loss': val_total_loss,
                    'mean_error': val_mean_error
                }
            }, os.path.join(save_path, "best_model_by_lonlat_mae.pth"))
            print(f"✅ 更新最佳经纬度MAE模型！当前最佳经纬度MAE: {best_val_lonlat_mae:.4f} 度")
    
    # 保存最终模型和完整训练历史
    if save_path is not None:
        torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
        np.save(os.path.join(save_path, "training_history.npy"), history)
    
    return history


def main():
    # 配置参数
    train_root = "/home/ma-user/work/data/data_geoclip/train"
    val_root = "/home/ma-user/work/data/data_geoclip/val"
    batch_size = 64
    embed_dim = 512
    num_epochs = 50
    img_size = 256
    save_path = f"clip_geo_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1. 加载数据
    print("加载数据...")
    train_dataloader, val_dataloader, clip_processor = get_dataloaders(
        train_root, val_root, batch_size, img_size
    )
    
    # 验证数据形状
    sample_train_imgs, sample_train_coords = next(iter(train_dataloader))
    print(f"训练集样本图像形状: {sample_train_imgs.shape}")
    print(f"训练集样本坐标形状: {sample_train_coords.shape}")  # (batch_size, 1, num_points, 2)

    # 2. 初始化地理编码器
    print("初始化地理编码器...")
    geo_encoder_cfg = GeoEncoderConfig(
        lon_harmonics=4,
        lat_harmonics=3,
        add_unitvec=True
    )
    geo_encoder = MultiScaleGeoEncoder(cfg=geo_encoder_cfg)

    # 3. 计算坐标编码维度
    print("计算坐标编码维度...")
    sample_encoded = batch_encode_coords(sample_train_coords[:1], geo_encoder)
    coord_input_dim = sample_encoded.shape[1]
    print(f"编码后的坐标维度: {coord_input_dim}")

    # 4. 初始化CLIP模型（含解码器）
    print("初始化CLIP模型（含经纬度解码器）...")
    model = CLIPModel(coord_input_dim, embed_dim=embed_dim).to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 启动训练+全维度验证（删除top-1准确率）
    print("\n启动训练+验证（含损失、正样本误差、经纬度MAE）...")
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
    print(f"训练历史（含所有指标）已保存至: {os.path.join(save_path, 'training_history.npy')}")
    print(f"最佳损失模型: {os.path.join(save_path, 'best_model_by_loss.pth')}")
    print(f"最佳经纬度MAE模型: {os.path.join(save_path, 'best_model_by_lonlat_mae.pth')}")

if __name__ == "__main__":
    main()
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
device = torch.device("npu:3" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")


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

            
            true_coords = coordinates.squeeze(1).to(device)
            true_lonlat = true_coords.mean(dim=1)
            
            optimizer.zero_grad()
            image_embeds, coord_embeds, pred_lonlat = model(images, encoded_coords)
            
            # 计算损失

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


def train(model, train_dataloader, val_dataloader, num_epochs, device, geo_encoder, save_path=None):
    
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
    train_root = "/home/ma-user/work/data/data_geoclip/train"
    val_root = "/home/ma-user/work/data/data_geoclip/val"
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
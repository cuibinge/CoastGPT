import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch_npu
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

# -------------------------- 关键修改1：引入transformers的CLIP相关类 --------------------------
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor

# 从自定义脚本导入（确保路径正确）
from geodataset import GeoDataset  
from Geo_features import MultiScaleGeoEncoder, GeoEncoderConfig  


# 设置设备（无修改）
device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
print(f"使用设备: {device}")




# -------------------------- 主函数（适配数据集划分和测试逻辑） --------------------------
def main():
    # 配置参数
    dataset_root = "/home/ma-user/work/data/caption/images_rgb"
    batch_size = 64
    embed_dim = 512
    num_epochs = 50
    img_size = 256  # 固定输入图像尺寸为256×256
    save_path = f"clip_geo_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1. 加载数据（修改为获取训练/测试两个dataloader）
    print("加载数据并划分训练集/测试集（80%/20%）...")
    train_dataloader, test_dataloader, clip_processor = get_dataloaders(
        dataset_root, 
        batch_size, 
        img_size, 
        train_ratio=0.8, 
        seed=42
    )
    
    # 验证数据集划分结果
    print(f"训练集样本数: {len(train_dataloader.dataset)}, 测试集样本数: {len(test_dataloader.dataset)}")
    sample_train_batch = next(iter(train_dataloader))
    print(f"训练集样本图像形状: {sample_train_batch[0].shape}")

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
    sample_encoded = batch_encode_coords(sample_train_batch[1][:1], geo_encoder)
    coord_input_dim = sample_encoded.shape[1]
    print(f"编码后的坐标维度: {coord_input_dim}")

    # 4. 初始化CLIP模型（使用transformers-CLIP ViT编码器）
    print("初始化CLIP模型（基于transformers-CLIP ViT）...")
    model = CLIPModel(coord_input_dim, embed_dim=embed_dim).to(device)
    
    # 5. 启动训练（传入训练和测试dataloader）
    print("启动训练+测试...")
    history = train(
        model, 
        train_dataloader, 
        test_dataloader, 
        num_epochs, 
        device, 
        geo_encoder, 
        save_path
    )
    
    print("训练完成!")

if __name__ == "__main__":
    # 注意：需先安装transformers库：pip install transformers
    main()
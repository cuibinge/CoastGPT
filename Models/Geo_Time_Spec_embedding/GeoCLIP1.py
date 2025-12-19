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


# 坐标批量编码函数（无修改）
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


# -------------------------- 关键修改2：适配CLIP预处理+数据集划分（80%训练/20%测试） --------------------------
def get_dataloaders(dataset_root, batch_size=16, img_size=256, train_ratio=0.8, seed=42):
    # 1. 初始化CLIP图像处理器，强制resize到256×256
    clip_processor = CLIPImageProcessor.from_pretrained("/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/")
    clip_processor.size["shortest_edge"] = img_size  # 最短边=256（确保输出256×256）
    clip_processor.crop_size["height"] = img_size    # 裁剪高度=256
    clip_processor.crop_size["width"] = img_size     # 裁剪宽度=256

    # 2. 加载完整数据集（需确保GeoDataset的__init__支持`transform`参数）
    full_dataset = GeoDataset(
        root_dir=dataset_root
    )

    # 3. 划分训练集和测试集（80%/20%）
    torch.manual_seed(seed)  # 固定种子保证划分可复现
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 4. 创建训练和测试DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, test_dataloader, clip_processor  # 返回训练/测试加载器+处理器


# -------------------------- 关键修改3：基于transformers-CLIP实现图像编码器 --------------------------
class CLIPViTImageEncoder(nn.Module):
    """使用transformers库的CLIP ViT作为图像编码器，输入尺寸256×256"""
    def __init__(self, embed_dim=512, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
        super().__init__()
        # 1. 加载CLIP视觉模型配置，指定输入图像尺寸为256×256
        self.clip_config = CLIPVisionConfig.from_pretrained(clip_model_name)
        self.clip_config.image_size = 256  # 强制模型接受256×256输入
        # 2. 加载预训练CLIP视觉模型（仅保留视觉部分，不含文本部分）
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_model_name,
            config=self.clip_config,
            ignore_mismatched_sizes=True
        )
        # 3. CLIP视觉模型的输出维度（如base模型为768，large模型为1024）
        self.clip_hidden_dim = self.clip_config.hidden_size
        # 4. 投影层：将CLIP特征映射到目标嵌入维度（如512）
        self.projection = nn.Sequential(
            nn.Linear(self.clip_hidden_dim, self.clip_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.clip_hidden_dim // 2, embed_dim)
        )
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # CLIP模型输入：(batch_size, 3, 256, 256)
        # CLIP模型输出：last_hidden_state shape为(batch_size, num_tokens, hidden_size)
        # 取cls-token（第一个token）的特征：(batch_size, hidden_size)
        clip_output = self.clip_vision_model(pixel_values=x)
        cls_feat = clip_output.last_hidden_state[:, 0, :]  # 核心特征：cls-token
        # 投影到目标维度并归一化
        x = self.projection(cls_feat)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)  # L2归一化


# 坐标编码器（无修改）
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


# CLIP模型组合（修改图像编码器为新实现）
class CLIPModel(nn.Module):
    def __init__(self, coord_input_dim, embed_dim=512, temperature=0.07):
        super().__init__()
        # 关键：使用transformers-CLIP ViT编码器
        self.image_encoder = CLIPViTImageEncoder(embed_dim=embed_dim)
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.temperature = temperature
        
    def forward(self, images, coordinates):
        image_embeds = self.image_encoder(images)
        coord_embeds = self.coord_encoder(coordinates)
        return image_embeds, coord_embeds


# 对比损失函数（无修改）
def contrastive_loss(image_embeds, coord_embeds, temperature=0.07):
    batch_size = image_embeds.shape[0]
    logits = torch.matmul(image_embeds, coord_embeds.t()) / temperature
    labels = torch.arange(batch_size, device=image_embeds.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# 训练epoch函数（无修改）
def train_epoch(model, dataloader, optimizer, criterion, device, geo_encoder, scheduler=None):
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for images, coordinates in tepoch:
            images = images.to(device)  
            encoded_coords = batch_encode_coords(coordinates, geo_encoder)
            
            optimizer.zero_grad()
            image_embeds, coord_embeds = model(images, encoded_coords)
            loss = criterion(image_embeds, coord_embeds)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            tepoch.set_postfix(loss=loss.item())
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total_batches


# -------------------------- 关键修改4：新增测试epoch函数 --------------------------
def test_epoch(model, dataloader, criterion, device, geo_encoder):
    model.eval()  # 模型设为评估模式
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省资源
        with tqdm(dataloader, unit="batch", desc="Testing") as tepoch:
            for images, coordinates in tepoch:
                images = images.to(device)
                encoded_coords = batch_encode_coords(coordinates, geo_encoder)
                
                # 仅前向传播计算损失
                image_embeds, coord_embeds = model(images, encoded_coords)
                loss = criterion(image_embeds, coord_embeds)
                
                total_loss += loss.item()
                total_batches += 1
                tepoch.set_postfix(test_loss=loss.item())
    
    return total_loss / total_batches


# -------------------------- 关键修改5：修改训练主逻辑，支持测试+保存最佳模型 --------------------------
def train(model, train_dataloader, test_dataloader, num_epochs, device, geo_encoder, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = {'train_loss': [], 'test_loss': []}
    
    # 初始化最佳测试损失（设为极大值）
    best_test_loss = float('inf')
    best_epoch = 0
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    print(f"开始训练，共 {num_epochs} 个epoch")
    print(f"训练集批次数量: {len(train_dataloader)}, 测试集批次数量: {len(test_dataloader)}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # 训练步骤
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            contrastive_loss, 
            device, 
            geo_encoder, 
            scheduler
        )
        history['train_loss'].append(train_loss)
        print(f"训练损失: {train_loss:.4f}")
        
        # 测试步骤
        test_loss = test_epoch(
            model, 
            test_dataloader, 
            contrastive_loss, 
            device, 
            geo_encoder
        )
        history['test_loss'].append(test_loss)
        print(f"测试损失: {test_loss:.4f}")
        
        # 保存测试损失最佳的模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            if save_path is not None:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_loss': best_test_loss,
                    'train_loss': train_loss,
                }, os.path.join(save_path, "best_model.pth"))
                print(f"✅ 更新最佳模型（epoch {best_epoch}），最佳测试损失: {best_test_loss:.4f}")
        
        # 每5个epoch保存一次当前模型（可选保留）
        if save_path is not None and (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
    
    # 训练结束后保存最终模型和训练历史
    if save_path is not None:
        torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
        np.save(os.path.join(save_path, "training_history.npy"), history)
    
    print(f"\n训练完成!")
    print(f"最佳模型来自epoch {best_epoch}，最佳测试损失: {best_test_loss:.4f}")
    print(f"最佳模型已保存至: {os.path.join(save_path, 'best_model.pth')}")
    
    return history


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
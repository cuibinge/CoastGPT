import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor  # 匹配训练的CLIP组件
from torchvision import transforms
from geodataset import GeoDataset  # 复用训练时的数据集类（仅参考预处理逻辑）
from Geo_features import MultiScaleGeoEncoder, GeoEncoderConfig  # 复用地理编码器
import numpy as np
from PIL import Image
import os
import torch_npu

class CLIPViTImageEncoder(nn.Module):
    """完全复用训练时的CLIP ViT图像编码器，无任何修改"""
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
        cls_feat = clip_output.last_hidden_state[:, 0, :]  # 取CLIP的CLS-token特征
        x = self.projection(cls_feat)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)  # L2归一化


class CoordinateEncoder(nn.Module):
    """完全复用训练时的坐标编码器，无任何修改"""
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


class CLIPModel(nn.Module):
    """复用训练时的CLIP组合模型，新增推理专用接口"""
    def __init__(self, coord_input_dim, embed_dim=512, temperature=0.07, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
        super().__init__()
        self.image_encoder = CLIPViTImageEncoder(embed_dim=embed_dim, clip_model_name=clip_model_name)  # 训练用的CLIP ViT编码器
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.temperature = temperature  # 训练时的温度参数，推理时暂不使用
        
    def forward_image(self, images):
        """新增：仅生成图片embedding的接口（输入为CLIPProcessor处理后的pixel_values）"""
        return self.image_encoder(images)
    
    def forward_coord(self, coordinates):
        """新增：仅生成坐标embedding的接口"""
        return self.coord_encoder(coordinates)


def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    """完全复用训练时的坐标批量编码函数，无任何修改"""
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
    
    encoded_tensor = torch.tensor(encoded_list, dtype=torch.float32).to(geo_encoder.device)
    return encoded_tensor


def init_device():
    """初始化推理设备（与训练一致，优先NPU）"""
    device = torch.device("npu:1" if torch_npu.npu.is_available() else "cpu")
    print(f"推理使用设备: {device}")
    return device


def load_geo_encoder(geo_cfg, device):
    """加载地理编码器（配置必须与训练时完全一致）"""
    geo_encoder = MultiScaleGeoEncoder(cfg=geo_cfg)
    geo_encoder.device = device  # 绑定设备
    return geo_encoder


def load_clip_model(model_path, geo_encoder, device, embed_dim=512, agg_method="mean", clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
    """加载训练好的CLIP模型（适配CLIP ViT编码器）"""
    # 1. 计算坐标编码维度（与训练时逻辑完全一致）
    sample_coords = torch.tensor([[[[116.39, 39.90]]]], dtype=torch.float32)  # 测试坐标格式：(1,1,1,2)
    sample_encoded = batch_encode_coords(sample_coords, geo_encoder, agg_method)
    coord_input_dim = sample_encoded.shape[1]
    
    # 2. 初始化CLIP模型（使用训练时的CLIP ViT路径）
    model = CLIPModel(
        coord_input_dim=coord_input_dim,
        embed_dim=embed_dim,
        clip_model_name=clip_model_name
    ).to(device)
    
    # 3. 加载训练权重（兼容训练时的checkpoint格式）
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # 切换推理模式（冻结BatchNorm/禁用Dropout）
    print(f"模型加载完成：坐标编码维度={coord_input_dim}，CLIP ViT路径={clip_model_name}")
    return model


def preprocess_image(image_path, device, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/", img_size=256):
    """适配训练时的CLIP预处理（使用CLIPImageProcessor，确保与训练一致）"""
    # 1. 初始化CLIP图像处理器（与训练时get_dataloader的配置完全匹配）
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_processor.size["shortest_edge"] = img_size  # 最短边=256
    clip_processor.crop_size["height"] = img_size    # 裁剪高度=256
    clip_processor.crop_size["width"] = img_size     # 裁剪宽度=256
    
    # 2. 读取并预处理图片
    image = Image.open(image_path).convert("RGB")  # 强制3通道（避免灰度图问题）
    processed = clip_processor(images=image, return_tensors="pt")  # 返回PyTorch张量
    pixel_values = processed["pixel_values"].to(device)  # CLIP模型输入为pixel_values
    
    return pixel_values  # 形状：(1, 3, 256, 256)


def generate_candidate_coords(coord_range, num_candidates=1000):
    """生成候选坐标库（与之前逻辑一致，适配batch_encode_coords输入格式）"""
    #(min_lon, max_lon), (min_lat, max_lat) = coord_range
    # 生成均匀分布的候选坐标
    min_lon, min_lat = coord_range[0]
    max_lon, max_lat = coord_range[1]
    lons = np.random.uniform(min_lon, max_lon, num_candidates)
    lats = np.random.uniform(min_lat, max_lat, num_candidates)
    print("lons: ",lons)
    print("lats: ",lats)
    # 整理为 (num_candidates, 1, 1, 2) 格式（匹配batch_encode_coords的输入要求）
    candidate_coords = torch.tensor(
        np.stack([lons, lats], axis=-1).reshape(-1, 1, 1, 2),
        dtype=torch.float32
    )
    return candidate_coords


def match_best_coord(image_path, model, geo_encoder, candidate_coords, device, clip_model_name, img_size=256, agg_method="mean"):
    """核心匹配逻辑（适配CLIP ViT的embedding生成）"""
    # 1. 预处理图片并生成图片embedding
    pixel_values = preprocess_image(image_path, device, clip_model_name, img_size)
    with torch.no_grad():  # 推理禁用梯度，节省内存
        image_embed = model.forward_image(pixel_values)  # (1, 512)
    
    # 2. 编码候选坐标并生成坐标embedding
    candidate_encoded = batch_encode_coords(candidate_coords, geo_encoder, agg_method)  # (num_candidates, coord_input_dim)
    with torch.no_grad():
        coord_embeds = model.forward_coord(candidate_encoded)  # (num_candidates, 512)
    
    # 3. 计算余弦相似度（越高匹配度越好）
    similarities = torch.matmul(image_embed, coord_embeds.T).squeeze(0)  # (num_candidates,)
    max_sim_idx = torch.argmax(similarities).item()  # 取相似度最高的索引
    
    # 4. 提取最佳坐标与相似度
    best_coord = candidate_coords[max_sim_idx].squeeze().cpu().numpy()  # (lon, lat)
    max_similarity = similarities[max_sim_idx].item()
    # if(max_similarity<0.7):
    #     match_best_coord(image_path, model, geo_encoder, candidate_coords, device, clip_model_name, img_size=256, agg_method="mean")
    # else:
    #     return best_coord, max_similarity
    return best_coord,max_similarity

        


# -------------------------- 3. 端到端推理主函数（可直接调用） --------------------------
def infer_image_to_coord(
    image_path,
    model_path,
    geo_cfg,
    coord_range,
    clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/",
    device=None,
    img_size=256,
    agg_method="mean",
    num_candidates=1000
):
    """
    端到端推理：输入图片→输出最佳匹配坐标（完全适配训练逻辑）
    Args:
        clip_model_name: CLIP ViT模型路径（必须与训练一致）
        其他参数含义与之前推理函数一致
    Returns:
        best_coord: 最佳匹配坐标 (lon, lat)
        max_similarity: 最高匹配相似度 (0~1)
    """
    # 1. 初始化依赖组件
    if device is None:
        device = init_device()
    geo_encoder = load_geo_encoder(geo_cfg, device)
    model = load_clip_model(
        model_path=model_path,
        geo_encoder=geo_encoder,
        device=device,
        clip_model_name=clip_model_name,
        agg_method=agg_method
    )
    
    # 2. 生成候选坐标库
    print(f"生成 {num_candidates} 个候选坐标（范围：{coord_range}）...")
    candidate_coords = generate_candidate_coords(coord_range, num_candidates)
    
    # 3. 匹配最佳坐标
    print(f"正在匹配图片：{os.path.basename(image_path)}...")
    best_coord, max_similarity = match_best_coord(
        image_path=image_path,
        model=model,
        geo_encoder=geo_encoder,
        candidate_coords=candidate_coords,
        device=device,
        clip_model_name=clip_model_name,
        img_size=img_size,
        agg_method=agg_method
    )
    
    # 4. 输出结果
    print(f"\n=== 推理结果 ===")
    print(f"最佳匹配坐标（经度, 纬度）: {np.round(best_coord, 4)}")
    print(f"最高匹配相似度: {np.round(max_similarity, 4)}")
    print(f"候选坐标数量: {num_candidates}")
    print(f"CLIP ViT模型路径: {os.path.basename(clip_model_name)}")
    return best_coord, max_similarity


if __name__ == "__main__":
    # 关键配置：必须与训练时的main函数参数完全一致！
    GEO_CONFIG = GeoEncoderConfig(
        lon_harmonics=4,    # 训练时的经度谐波数
        lat_harmonics=3,    # 训练时的纬度谐波数
        add_unitvec=True    # 训练时是否保留球面单位向量
    )
    CLIP_MODEL_NAME = "/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"  # 训练用CLIP路径
    MODEL_PATH = "./clip_geo_models_20251017_220246/final_model.pth"  # 训练保存的权重路径
    COORD_RANGE = [(0.0, 0.0), (180.0, 180.0)]  # 候选坐标范围（与训练数据区域匹配）
    IMAGE_PATH = "/home/ma-user/work/CoastGPT/Images/5.jpg"  # 测试图片路径
    NUM_CANDIDATES = 10000 # 候选坐标数量（越多越准，但耗时更长）
    
    # 启动端到端推理
    best_coord, max_similarity = infer_image_to_coord(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        geo_cfg=GEO_CONFIG,
        coord_range=COORD_RANGE,
        clip_model_name=CLIP_MODEL_NAME,
        num_candidates=NUM_CANDIDATES
    )


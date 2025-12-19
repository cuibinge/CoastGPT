import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
from torchvision import transforms
from geodataset import GeoDataset
from Geo_features import MultiScaleGeoEncoder, GeoEncoderConfig
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
        cls_feat = clip_output.last_hidden_state[:, 0, :]
        x = self.projection(cls_feat)
        x = self.normalize(x)
        return x / torch.norm(x, dim=-1, keepdim=True)


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
        self.image_encoder = CLIPViTImageEncoder(embed_dim=embed_dim, clip_model_name=clip_model_name)
        self.coord_encoder = CoordinateEncoder(coord_input_dim, embed_dim)
        self.temperature = temperature
        
    def forward_image(self, images):
        return self.image_encoder(images)
    
    def forward_coord(self, coordinates):
        return self.coord_encoder(coordinates)


def batch_encode_coords(coords_batch, geo_encoder, agg_method="mean"):
    """完全复用训练时的坐标批量编码函数，无任何修改"""
    coords_squeezed = coords_batch.squeeze(1)
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
    geo_encoder.device = device
    return geo_encoder


def load_clip_model(model_path, geo_encoder, device, embed_dim=512, agg_method="mean", clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"):
    """加载训练好的CLIP模型（适配CLIP ViT编码器）"""
    sample_coords = torch.tensor([[[[116.39, 39.90]]]], dtype=torch.float32)
    sample_encoded = batch_encode_coords(sample_coords, geo_encoder, agg_method)
    coord_input_dim = sample_encoded.shape[1]
    
    model = CLIPModel(
        coord_input_dim=coord_input_dim,
        embed_dim=embed_dim,
        clip_model_name=clip_model_name
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"模型加载完成：坐标编码维度={coord_input_dim}，CLIP ViT路径={clip_model_name}")
    return model


def preprocess_image(image_path, device, clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/", img_size=256):
    """适配训练时的CLIP预处理（使用CLIPImageProcessor，确保与训练一致）"""
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_processor.size["shortest_edge"] = img_size
    clip_processor.crop_size["height"] = img_size
    clip_processor.crop_size["width"] = img_size
    
    image = Image.open(image_path).convert("RGB")
    processed = clip_processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)
    
    return pixel_values


def generate_candidate_coords(coord_range, num_candidates=1000):
    """生成候选坐标库（与之前逻辑一致，适配batch_encode_coords输入格式）"""
    min_lon, min_lat = coord_range[0]
    max_lon, max_lat = coord_range[1]
    # 修正经纬度范围逻辑（避免min > max）
    min_lon, max_lon = (min_lon, max_lon) if min_lon < max_lon else (max_lon, min_lon)
    min_lat, max_lat = (min_lat, max_lat) if min_lat < max_lat else (max_lat, min_lat)
    
    lons = np.random.uniform(min_lon, max_lon, num_candidates)
    lats = np.random.uniform(min_lat, max_lat, num_candidates)
    
    candidate_coords = torch.tensor(
        np.stack([lons, lats], axis=-1).reshape(-1, 1, 1, 2),
        dtype=torch.float32
    )
    return candidate_coords


def match_best_coord(image_path, model, geo_encoder, candidate_coords, device, clip_model_name, img_size=256, agg_method="mean"):
    """核心匹配逻辑（适配CLIP ViT的embedding生成）"""
    # 1. 生成图片embedding
    pixel_values = preprocess_image(image_path, device, clip_model_name, img_size)
    with torch.no_grad():
        image_embed = model.forward_image(pixel_values)
    
    # 2. 生成坐标embedding
    candidate_encoded = batch_encode_coords(candidate_coords, geo_encoder, agg_method)
    with torch.no_grad():
        coord_embeds = model.forward_coord(candidate_encoded)
    
    # 3. 计算相似度并取最佳
    similarities = torch.matmul(image_embed, coord_embeds.T).squeeze(0)
    max_sim_idx = torch.argmax(similarities).item()
    
    best_coord = candidate_coords[max_sim_idx].squeeze().cpu().numpy()
    max_similarity = similarities[max_sim_idx].item()
    
    return best_coord, max_similarity


# -------------------------- 核心修改：端到端推理主函数（新增阈值与范围缩小逻辑） --------------------------
def infer_image_to_coord(
    image_path,
    model_path,
    geo_cfg,
    coord_range,
    clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/",
    device=None,
    img_size=256,
    agg_method="mean",
    num_candidates=1000,
    similarity_threshold=0.7,  # 新增：相似度阈值（低于此值则缩小范围）
    max_retries=3,             # 新增：最大重试缩小次数（避免无限循环）
    shrink_ratio=0.5           # 新增：范围缩小比例（每次缩小为原范围的50%）
):
    """
    端到端推理：输入图片→输出最佳匹配坐标（新增阈值判断与范围动态缩小）
    Args:
        similarity_threshold: 相似度阈值，低于此值触发范围缩小
        max_retries: 最大缩小重试次数，防止无限循环
        shrink_ratio: 范围缩小比例，每次以最佳坐标为中心缩小
        其他参数含义与之前一致
    Returns:
        best_coord: 最佳匹配坐标 (lon, lat)
        max_similarity: 最高匹配相似度 (0~1)
        final_coord_range: 最终使用的候选坐标范围
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
    
    # 2. 初始化循环参数
    current_coord_range = coord_range  # 当前候选坐标范围
    current_retry = 0                  # 当前重试次数
    best_coord = None
    max_similarity = 0.0
    
    # 3. 循环匹配（直到满足阈值或达到最大重试次数）
    while current_retry <= max_retries:
        # 生成当前范围的候选坐标
        print(f"\n=== 第 {current_retry + 1} 次匹配 ===")
        print(f"当前候选范围：{np.round(current_coord_range, 4)}，候选数量：{num_candidates}")
        candidate_coords = generate_candidate_coords(current_coord_range, num_candidates)
        
        # 执行匹配
        current_best_coord, current_max_sim = match_best_coord(
            image_path=image_path,
            model=model,
            geo_encoder=geo_encoder,
            candidate_coords=candidate_coords,
            device=device,
            clip_model_name=clip_model_name,
            img_size=img_size,
            agg_method=agg_method
        )
        
        # 更新全局最佳结果
        if current_max_sim > max_similarity:
            max_similarity = current_max_sim
            best_coord = current_best_coord
        
        # 检查是否满足阈值
        if max_similarity >= similarity_threshold:
            print(f"相似度 {np.round(max_similarity, 4)} ≥ 阈值 {similarity_threshold}，匹配完成！")
            break
        else:
            print(f"相似度 {np.round(max_similarity, 4)} < 阈值 {similarity_threshold}，准备缩小范围...")
            current_retry += 1
            
            # 计算缩小后的新范围（以当前最佳坐标为中心）
            if current_retry <= max_retries:
                lon, lat = best_coord
                # 原范围宽度
                lon_range = current_coord_range[1][0] - current_coord_range[0][0]
                lat_range = current_coord_range[1][1] - current_coord_range[0][1]
                # 新范围半宽（缩小后的一半）
                new_lon_half = (lon_range * shrink_ratio) / 2
                new_lat_half = (lat_range * shrink_ratio) / 2
                # 更新范围（确保不超出地理逻辑）
                new_min_lon = max(-180.0, lon - new_lon_half)
                new_max_lon = min(180.0, lon + new_lon_half)
                new_min_lat = max(-90.0, lat - new_lat_half)
                new_max_lat = min(90.0, lat + new_lat_half)
                current_coord_range = [(new_min_lon, new_min_lat), (new_max_lon, new_max_lat)]
            else:
                print(f"已达到最大重试次数 {max_retries}，停止缩小范围")
    
    # 4. 输出最终结果
    print(f"\n=== 最终推理结果 ===")
    print(f"最佳匹配坐标（经度, 纬度）: {np.round(best_coord, 4)}")
    print(f"最高匹配相似度: {np.round(max_similarity, 4)}")
    print(f"候选坐标数量: {num_candidates}")
    print(f"最终候选范围: {np.round(current_coord_range, 4)}")
    print(f"CLIP ViT模型路径: {os.path.basename(clip_model_name)}")
    return best_coord, max_similarity, current_coord_range


if __name__ == "__main__":
    # 关键配置：必须与训练时的main函数参数完全一致！
    GEO_CONFIG = GeoEncoderConfig(
        lon_harmonics=4,
        lat_harmonics=3,
        add_unitvec=True
    )
    CLIP_MODEL_NAME = "/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    MODEL_PATH = "./clip_geo_models_20251017_220246/final_model.pth"
    # 修正初始坐标范围（确保min < max，示例：青岛附近范围）
    COORD_RANGE = [(0.0, 0.0), (180.0, 180.0)]  # (min_lon, min_lat), (max_lon, max_lat)
    #COORD_RANGE = [(118.0, 37.0), (119.0, 38.0)]  # (min_lon, min_lat), (max_lon, max_lat)

    IMAGE_PATH = "/home/ma-user/work/CoastGPT/Images/5.jpg"
    NUM_CANDIDATES = 10000  # 建议增加候选数量以提升初始匹配精度
    
    # 启动端到端推理（新增阈值相关参数）
    best_coord, max_similarity, final_range = infer_image_to_coord(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        geo_cfg=GEO_CONFIG,
        coord_range=COORD_RANGE,
        clip_model_name=CLIP_MODEL_NAME,
        num_candidates=NUM_CANDIDATES,
        similarity_threshold=0.9,  # 可根据模型性能调整
        max_retries=20,             # 最多缩小n次范围
        shrink_ratio=0.8         # 每次缩小为原范围的50%
    )
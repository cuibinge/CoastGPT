import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
from torchvision import transforms
from geodataset import GeoDataset
from Geo_features import MultiScaleGeoEncoder, GeoEncoderConfig
import numpy as np
from PIL import Image
import time
import os
import torch_npu

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


def generate_candidate_coords(coord_ranges, num_candidates):
    """
    生成候选坐标库（支持多区域采样，适配两步策略）
    Args:
        coord_ranges: 坐标范围列表，格式为 [(min_lon1, min_lat1, max_lon1, max_lat1), ...]
        num_candidates: 总候选坐标数量（均匀分配到各区域）
    Returns:
        candidate_coords: 候选坐标张量，形状为 [num_candidates, 1, 1, 2]
    """
    num_ranges = len(coord_ranges)
    candidates_per_range = num_candidates // num_ranges
    remaining = num_candidates % num_ranges  # 处理余数，前remaining个区域多1个候选
    
    all_candidates = []
    for i, (min_lon, min_lat, max_lon, max_lat) in enumerate(coord_ranges):
        # 修正范围逻辑（确保min < max）
        min_lon, max_lon = (min_lon, max_lon) if min_lon < max_lon else (max_lon, min_lon)
        min_lat, max_lat = (min_lat, max_lat) if min_lat < max_lat else (max_lat, min_lat)
        
        # 确保不超出地理坐标范围
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)
        
        # 分配候选数量
        current_num = candidates_per_range + 1 if i < remaining else candidates_per_range
        if current_num <= 0:
            continue
        
        # 生成该区域的候选坐标
        lons = np.random.uniform(min_lon, max_lon, current_num)
        lats = np.random.uniform(min_lat, max_lat, current_num)
        range_candidates = np.stack([lons, lats], axis=-1)
        all_candidates.append(range_candidates)
    
    # 合并所有区域的候选并调整形状
    all_candidates = np.concatenate(all_candidates, axis=0)
    candidate_coords = torch.tensor(
        all_candidates.reshape(-1, 1, 1, 2),
        dtype=torch.float32
    )
    return candidate_coords


def match_best_coord(image_embed, candidate_coords, model, geo_encoder, device, clip_model_name, img_size=256, agg_method="mean"):
    """核心匹配逻辑（适配CLIP ViT的embedding生成，支持传入预计算的image_embed）"""
    # 生成坐标embedding
    candidate_encoded = batch_encode_coords(candidate_coords, geo_encoder, agg_method)
    with torch.no_grad():
        coord_embeds = model.forward_coord(candidate_encoded)
    
    # 计算相似度并取最佳
    similarities = torch.matmul(image_embed, coord_embeds.T).squeeze(0)
    max_sim_idx = torch.argmax(similarities).item()
    
    best_coord = candidate_coords[max_sim_idx].squeeze().cpu().numpy()
    max_similarity = similarities[max_sim_idx].item()
    # 返回所有相似度和坐标（用于筛选高潜力区域）
    all_similarities = similarities.cpu().numpy()
    all_coords = candidate_coords.squeeze().cpu().numpy()
    
    return best_coord, max_similarity, all_similarities, all_coords


def get_high_potential_ranges(all_coords, all_similarities, top_k_ratio=0.1, local_range=0.5):
    """
    从全局匹配结果中筛选高潜力区域
    Args:
        all_coords: 全局所有候选坐标，形状为 [num_candidates, 2]
        all_similarities: 全局所有候选的相似度，形状为 [num_candidates]
        top_k_ratio: 取前k%的高相似度坐标作为高潜力中心（默认10%）
        local_range: 每个高潜力中心的局部范围（默认±0.5°，即1°×1°区域）
    Returns:
        merged_ranges: 合并后的高潜力区域列表，格式为 [(min_lon, min_lat, max_lon, max_lat), ...]
    """
    # 1. 筛选前k%的高相似度坐标
    num_top = max(1, int(len(all_similarities) * top_k_ratio))  # 至少保留1个
    top_indices = np.argsort(all_similarities)[-num_top:]  # 相似度从低到高排序，取后num_top个
    top_coords = all_coords[top_indices]
    
    # 2. 为每个高潜力坐标生成局部范围
    potential_ranges = []
    for (lon, lat) in top_coords:
        min_lon = lon - local_range
        max_lon = lon + local_range
        min_lat = lat - local_range
        max_lat = lat + local_range
        potential_ranges.append((min_lon, min_lat, max_lon, max_lat))
    
    # 3. 合并重叠或相邻的区域（简化处理，取所有区域的整体包围盒）
    # 注：若需更精细的合并，可实现区间合并算法，此处为兼顾效率采用包围盒策略
    if not potential_ranges:
        return []
    all_min_lon = min([r[0] for r in potential_ranges])
    all_min_lat = min([r[1] for r in potential_ranges])
    all_max_lon = max([r[2] for r in potential_ranges])
    all_max_lat = max([r[3] for r in potential_ranges])
    merged_ranges = [(all_min_lon, all_min_lat, all_max_lon, all_max_lat)]
    
    print(f"筛选出 {num_top} 个高潜力坐标，合并为 {len(merged_ranges)} 个局部区域")
    for i, (min_lon, min_lat, max_lon, max_lat) in enumerate(merged_ranges):
        print(f"  高潜力区域{i+1}: 经度[{np.round(min_lon,4)}, {np.round(max_lon,4)}], 纬度[{np.round(min_lat,4)}, {np.round(max_lat,4)}]")
    
    return merged_ranges


# -------------------------- 两步采样端到端推理主函数 --------------------------
def infer_image_to_coord(
    image_path,
    model_path,
    geo_cfg,
    coord_range,
    clip_model_name="/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/",
    device=None,
    img_size=256,
    agg_method="mean",
    # 全局粗采样参数
    global_num_candidates=500,    # 全局候选数量（少量，确保覆盖范围）
    top_k_ratio=0.1,              # 高潜力区域筛选比例（前10%）
    # 局部精采样参数
    local_num_candidates=1000,    # 局部候选数量（大量，提升密度）
    local_range=0.5,              # 高潜力中心的局部范围（±0.5°）
    similarity_threshold=0.9      # 相似度阈值（用于结果评估）
):
    """
    端到端推理：输入图片→输出最佳匹配坐标（采用“全局粗采样+局部精采样”两步策略）
    Args:
        global_num_candidates: 全局粗采样的候选数量（建议500-1000）
        top_k_ratio: 从全局结果中筛选高潜力区域的比例（建议0.05-0.1）
        local_num_candidates: 局部精采样的候选数量（建议1000-5000，需大于全局数量）
        local_range: 每个高潜力坐标的局部范围（单位：度，建议0.3-1.0）
        similarity_threshold: 相似度阈值（用于评估结果可靠性）
        其他参数含义与之前一致
    Returns:
        best_coord: 最佳匹配坐标 (lon, lat)
        max_similarity: 最高匹配相似度 (0~1)
        final_range: 最终局部精采样的坐标范围
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
    
    # 2. 预计算图片Embedding（仅计算一次，避免重复处理）
    pixel_values = preprocess_image(image_path, device, clip_model_name, img_size)
    with torch.no_grad():
        image_embed = model.forward_image(pixel_values)
    print(f"\n图片预处理完成：尺寸={img_size}×{img_size}，Embedding维度={image_embed.shape[1]}")
    
    # -------------------------- 第一步：全局粗采样 --------------------------
    print(f"\n=== 第一步：全局粗采样匹配 ===")
    # 转换初始范围格式（适配generate_candidate_coords的多区域输入）
    init_min_lon, init_min_lat = coord_range[0]
    init_max_lon, init_max_lat = coord_range[1]
    global_ranges = [(init_min_lon, init_min_lat, init_max_lon, init_max_lat)]
    
    # 生成全局候选坐标并匹配
    global_candidates = generate_candidate_coords(global_ranges, global_num_candidates)
    global_best_coord, global_max_sim, all_sims, all_coords = match_best_coord(
        image_embed=image_embed,
        candidate_coords=global_candidates,
        model=model,
        geo_encoder=geo_encoder,
        device=device,
        clip_model_name=clip_model_name,
        img_size=img_size,
        agg_method=agg_method
    )
    
    print(f"全局粗匹配结果：")
    print(f"  最佳坐标（经度, 纬度）: {np.round(global_best_coord, 4)}")
    print(f"  最高相似度: {np.round(global_max_sim, 4)}")
    print(f"  全局候选数量: {len(global_candidates)}")
    print(f"  初始范围: {np.round(coord_range, 4)}")
    
    # 筛选高潜力区域（用于第二步精采样）
    high_potential_ranges = get_high_potential_ranges(
        all_coords=all_coords,
        all_similarities=all_sims,
        top_k_ratio=top_k_ratio,
        local_range=local_range
    )
    if not high_potential_ranges:
        print("未筛选出高潜力区域，直接返回全局匹配结果")
        return global_best_coord, global_max_sim, coord_range
    
    # -------------------------- 第二步：局部精采样 --------------------------
    print(f"\n=== 第二步：局部精采样匹配 ===")
    # 生成局部候选坐标并匹配
    local_candidates = generate_candidate_coords(high_potential_ranges, local_num_candidates)
    local_best_coord, local_max_sim, _, _ = match_best_coord(
        image_embed=image_embed,
        candidate_coords=local_candidates,
        model=model,
        geo_encoder=geo_encoder,
        device=device,
        clip_model_name=clip_model_name,
        img_size=img_size,
        agg_method=agg_method
    )
    
    print(f"局部精匹配结果：")
    print(f"  最佳坐标（经度, 纬度）: {np.round(local_best_coord, 4)}")
    print(f"  最高相似度: {np.round(local_max_sim, 4)}")
    print(f"  局部候选数量: {len(local_candidates)}")
    print(f"  精采样范围: {np.round(high_potential_ranges[0], 4)}")
    
    # -------------------------- 结果整合与输出 --------------------------
    # 选择全局和局部中的最优结果
    final_best_coord = local_best_coord if local_max_sim > global_max_sim else global_best_coord
    final_max_sim = max(local_max_sim, global_max_sim)
    final_range = high_potential_ranges[0]  # 最终范围为局部精采样范围
    
    # 结果评估提示
    sim_feedback = "（高可靠性）" if final_max_sim >= similarity_threshold else "（建议调整参数或扩大范围）"
    
    print(f"\n=== 最终推理结果 ===")
    print(f"最佳匹配坐标（经度, 纬度）: {np.round(final_best_coord, 4)}")
    print(f"最高匹配相似度: {np.round(final_max_sim, 4)} {sim_feedback}")
    print(f"全局候选数量: {global_num_candidates}, 局部候选数量: {local_num_candidates}")
    print(f"最终精采样范围: 经度[{np.round(final_range[0],4)}, {np.round(final_range[2],4)}], 纬度[{np.round(final_range[1],4)}, {np.round(final_range[3],4)}]")
    print(f"CLIP ViT模型路径: {os.path.basename(clip_model_name)}")
    return final_best_coord, final_max_sim, final_range


if __name__ == "__main__":
    # 关键配置：必须与训练时的main函数参数完全一致！
    GEO_CONFIG = GeoEncoderConfig(
        lon_harmonics=4,
        lat_harmonics=3,
        add_unitvec=True
    )
    CLIP_MODEL_NAME = "/home/ma-user/work/CoastGPT/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    MODEL_PATH = "./clip_geo_models_20251017_220246/final_model.pth"
    # 初始全局范围（示例：中国东部沿海）
    COORD_RANGE = [(0.0, 0.0), (180.0, 180.0)]  # (min_lon, min_lat), (max_lon, max_lat)

    IMAGE_PATH = "/home/ma-user/work/CoastGPT/Images/5.jpg"
    # 启动两步采样端到端推理
    best_coord, max_similarity, final_range = infer_image_to_coord(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        geo_cfg=GEO_CONFIG,
        coord_range=COORD_RANGE,
        clip_model_name=CLIP_MODEL_NAME,
        # 全局粗采样参数
        global_num_candidates=10000,    # 全局候选：800个（覆盖大范围）
        top_k_ratio=0.08,             # 取前8%作为高潜力区域
        # 局部精采样参数
        local_num_candidates=10000,    # 局部候选：3000个（高密度采样）
        local_range=0.4,              # 每个高潜力中心的范围：±0.4°
        similarity_threshold=0.85     # 相似度阈值：根据模型性能调整
    )
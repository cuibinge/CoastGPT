import math
import numpy as np
from datetime import datetime
from typing import Tuple, Union


class GeoTimeEmbedding:
    """地理时间信息嵌入层"""

    def __init__(self, grid_size: float = 1.0, embed_dim: int = 16):
        """
        初始化嵌入层
        参数:
            grid_size: 网格大小(度)
            embed_dim: 嵌入向量维度
        """
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # 计算网格总数
        self.num_lon_grids = int(360 / grid_size)  # 经度网格数 [-180, 180)
        self.num_lat_grids = int(180 / grid_size)  # 纬度网格数 [-90, 90)
        self.total_grids = self.num_lon_grids * self.num_lat_grids

        # 初始化网格嵌入矩阵
        self.grid_embedding = np.random.randn(self.total_grids, embed_dim) * 0.01

        # 月份嵌入矩阵 (12个月)
        self.month_embedding = np.random.randn(12, embed_dim // 2) * 0.01

        # 季节嵌入矩阵 (4个季节)
        self.season_embedding = np.random.randn(4, embed_dim // 2) * 0.01

    def grid_to_index(self, lon: float, lat: float) -> int:
        """将经纬度转换为网格索引"""
        lon_grid = int((lon + 180) // self.grid_size)
        lat_grid = int((lat + 90) // self.grid_size)

        # 确保索引在有效范围内
        lon_grid = np.clip(lon_grid, 0, self.num_lon_grids - 1)
        lat_grid = np.clip(lat_grid, 0, self.num_lat_grids - 1)

        return lon_grid * self.num_lat_grids + lat_grid

    def get_grid_embedding(self, lon: float, lat: float) -> np.ndarray:
        """获取经纬度对应的网格嵌入向量"""
        grid_idx = self.grid_to_index(lon, lat)
        return self.grid_embedding[grid_idx]

    def get_month_embedding(self, month: int) -> np.ndarray:
        """获取月份嵌入向量 (1-12)"""
        return self.month_embedding[month - 1]

    def get_season_embedding(self, season: int) -> np.ndarray:
        """获取季节嵌入向量 (1-4)"""
        return self.season_embedding[season - 1]


def geo_time_vectorization(
        lon: float,
        lat: float,
        date_str: str,
        method: str = 'cartesian',
        embedder: Union[GeoTimeEmbedding, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    地理时间信息向量化函数，支持嵌入处理
    参数:
        lon: 经度(度), [-180, 180]
        lat: 纬度(度), [-90, 90]
        date_str: 日期字符串, 格式'YYYY-MM-DD'
        method: 空间编码方法, ['periodic', 'cartesian', 'grid']
        embedder: 嵌入层实例, 当method='grid'时需要

    返回:
        geo_vector: 地理向量
        time_vector: 时间向量
        combined_vector: 地理时间联合向量
    """
    # 确保使用grid方法时有嵌入层
    if method == 'grid' and embedder is None:
        raise ValueError("使用grid方法时必须提供embedder参数")

    # ==================== 1. 经纬度向量化 ====================
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)

    if method == 'periodic':  # 周期性编码
        # 经度编码
        lon_sin = math.sin(math.pi * lon / 180)
        lon_cos = math.cos(math.pi * lon / 180)

        # 纬度编码
        lat_sin = math.sin(math.pi * lat / 180)
        lat_cos = math.cos(math.pi * lat / 180)

        geo_vector = np.array([lon_sin, lon_cos, lat_sin, lat_cos])

    elif method == 'cartesian':  # 球面笛卡尔坐标
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        geo_vector = np.array([x, y, z])

    elif method == 'grid':  # 网格分桶+嵌入
        # 使用嵌入层获取网格向量
        geo_vector = embedder.get_grid_embedding(lon, lat)

    # ==================== 2. 时间向量化 ====================
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # 计算年内天数(DOY)
    date_start = datetime(year, 1, 1)
    doy = (date_obj - date_start).days + 1

    # 时间戳(1970-01-01基准)
    timestamp = (date_obj - datetime(1970, 1, 1)).total_seconds()

    # 周期性编码
    day_sin = math.sin(2 * math.pi * doy / 365.25)
    day_cos = math.cos(2 * math.pi * doy / 365.25)

    # 多特征分解
    # 季节 (1:春, 2:夏, 3:秋, 4:冬)
    season = (month % 12 + 3) // 3

    # 闰年标记
    is_leap = 1 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 0

    # 基础时间特征
    base_time_features = np.array([
        timestamp,
        year,
        doy,
        day_sin,
        day_cos,
        is_leap
    ])

    # 如果提供了嵌入层，使用嵌入向量
    if embedder is not None:
        month_embed = embedder.get_month_embedding(month)
        season_embed = embedder.get_season_embedding(season)
        time_vector = np.concatenate([base_time_features, month_embed, season_embed])
    else:
        # 否则使用独热编码
        month_onehot = np.zeros(12)
        month_onehot[month - 1] = 1

        season_onehot = np.zeros(4)
        season_onehot[season - 1] = 1

        time_vector = np.concatenate([base_time_features, month_onehot, season_onehot])

    # ==================== 3. 融合与标准化 ====================
    # 地理和时间向量拼接
    combined_vector = np.concatenate([geo_vector, time_vector])

    # 标准化 - 在实际训练中应使用训练集的均值和标准差
    # 这里提供标准化代码示例，但默认不执行
    # mean = np.mean(combined_vector)
    # std = np.std(combined_vector)
    # combined_vector = (combined_vector - mean) / (std + 1e-8)

    return geo_vector, time_vector, combined_vector

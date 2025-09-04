# -*- coding: utf-8 -*-
"""
经纬度信息编码（分带/分区 + 可选海拔），可直接对接深度学习模型。
- 基础连续编码：经纬度周期性、安全处理（避免经度180°不连续）
- 分带（纬度带）离散编码：热带/副热带/温带/亚寒带/极地
- 可选：DEM 采样获得海拔、局部地形粗糙度（标准差）
- 可选：矢量分区（如气候带/生态区）点选查询

依赖：
- 基础功能：仅标准库
- DEM/矢量查询（可选）：rasterio、shapely（若未安装会自动跳过）
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List


# ============ 基础数学编码 ============
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def safe_sin_cos(angle_rad: float) -> Tuple[float, float]:
    return math.sin(angle_rad), math.cos(angle_rad)


def lonlat_to_unitvec(lon_deg: float, lat_deg: float) -> Tuple[float, float, float]:
    """ 把经纬度映射到地球单位球面坐标 (x,y,z)，避免经度不连续问题 """
    lon, lat = deg2rad(lon_deg), deg2rad(lat_deg)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return x, y, z


def fourier_cyclic(value: float, period: float, harmonics: int = 3) -> List[float]:
    """
    连续周期变量的多谐波编码（类似 Time2Vec/positional encoding）
    value: 同一单位的标量（如经度度数、或归一化后的纬度）
    period: 周期长度（经度=360）
    return: [sin(1x), cos(1x), sin(2x), cos(2x), ...]
    """
    theta = 2.0 * math.pi * (value / period)
    feats = []
    for k in range(1, harmonics + 1):
        feats.append(math.sin(k * theta))
        feats.append(math.cos(k * theta))
    return feats


# ============ 分带（纬度带） ============
@dataclass
class LatBeltConfig:
    # 经典阈值（单位度）：回归线 23.44、35、55、极圈 66.56
    belts: Tuple[float, float, float, float] = (23.44, 35.0, 55.0, 66.56)
    # 名称仅注释用：Tropical, Subtropical, Temperate, Subpolar, Polar
    names: Tuple[str, str, str, str, str] = ("tropical", "subtropical", "temperate", "subpolar", "polar")


def lat_belt_index(lat_deg: float, cfg: LatBeltConfig = LatBeltConfig()) -> int:
    a, b, c, d = cfg.belts
    alat = abs(lat_deg)
    if alat < a:
        return 0
    if alat < b:
        return 1
    if alat < c:
        return 2
    if alat < d:
        return 3
    return 4


def one_hot(idx: int, num_classes: int) -> List[int]:
    v = [0] * num_classes
    v[idx] = 1
    return v


# ============ 可选：DEM 采样（rasterio） ============
def sample_dem_elevation(dem_path: str, lon_deg: float, lat_deg: float) -> Optional[float]:
    """
    从 DEM GeoTIFF 读取点高程（米）。需要 rasterio。
    若不可用或采样失败，返回 None。
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
        with rasterio.open(dem_path) as ds:
            # 经度/纬度在 WGS84 下的像元行列
            row, col = rowcol(ds.transform, lon_deg, lat_deg)
            val = ds.read(1, window=((row, row + 1), (col, col + 1)))
            if val is None or val.size == 0:
                return None
            x = float(val[0, 0])
            if ds.nodata is not None and x == ds.nodata:
                return None
            # 某些 DEM 用 -32768 等 nodata，也可能存在异常小值
            return None if (x < -10000) else x
    except Exception:
        return None


def sample_dem_stats(dem_path: str, lon_deg: float, lat_deg: float,
                     half_window_px: int = 3) -> Optional[Dict[str, float]]:
    """
    以目标点为中心，在 (2*half_window_px+1)^2 窗口内计算均值/标准差/极差。
    用于粗糙度/起伏度特征（米）。需要 rasterio。
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
        import numpy as np
        with rasterio.open(dem_path) as ds:
            r, c = rowcol(ds.transform, lon_deg, lat_deg)
            r0, r1 = max(0, r - half_window_px), min(ds.height, r + half_window_px + 1)
            c0, c1 = max(0, c - half_window_px), min(ds.width, c + half_window_px + 1)
            arr = ds.read(1, window=((r0, r1), (c0, c1))).astype("float32")
            if ds.nodata is not None:
                arr = np.where(arr == ds.nodata, np.nan, arr)
            if not np.isfinite(arr).any():
                return None
            m = float(np.nanmean(arr))
            s = float(np.nanstd(arr))
            rng = float(np.nanmax(arr) - np.nanmin(arr))
            return {"elev_mean": m, "elev_std": s, "elev_range": rng}
    except Exception:
        return None


# ============ 可选：矢量分区查询（shapely） ============
def point_in_zones(lon_deg: float, lat_deg: float, shp_path: str,
                   attr_name: str = None) -> Optional[Any]:
    """
    在 Shapefile/GeoPackage 等矢量中查找包含该点的分区。
    优先返回属性 attr_name；若为 None 则返回 (layer_name, feature_id)。
    需要 shapely + fiona 或 geopandas。
    """
    try:
        from shapely.geometry import Point, shape
        import fiona
        pt = Point(lon_deg, lat_deg)
        with fiona.open(shp_path, "r") as src:
            for feat in src:
                geom = shape(feat["geometry"])
                if geom is not None and geom.contains(pt):
                    if attr_name and attr_name in feat["properties"]:
                        return feat["properties"][attr_name]
                    return (src.name, feat["id"])
        return None
    except Exception:
        return None


# ============ 主编码器 ============
@dataclass
class GeoEncoderConfig:
    lon_harmonics: int = 4     # 经度周期谐波数（360°）
    lat_harmonics: int = 3     # 绝对纬度作为 [0, 180] 周期的谐波数（或对 [-90,90] 归一后 2π 周期）
    add_unitvec: bool = True   # 是否添加球面单位向量 (x,y,z)
    add_abs_lat: bool = True   # |lat|（度）
    add_equator_dist: bool = True  # 到赤道的角距（=|lat|，度）
    add_tropics_dist: bool = True  # 到回归线的角距（min(|lat|-23.44) 的绝对值）
    add_polarcircle_dist: bool = True  # 到极圈的角距
    belt_config: LatBeltConfig = LatBeltConfig()


class MultiScaleGeoEncoder:
    """
    把 (lon, lat) 编码成一个向量：
      - 连续周期编码：经度（360°周期）、纬度（对称处理）
      - 球面单位向量：避免经度不连续
      - 分带 one-hot：热带/副热带/温带/亚寒带/极地
      - 可选：DEM 海拔/粗糙度、矢量分区索引（外部传入）
    """
    def __init__(self, cfg: GeoEncoderConfig = GeoEncoderConfig()):
        self.cfg = cfg

    def encode(self, lon_deg: float, lat_deg: float,
               dem_path: Optional[str] = None,
               zone_vector_path: Optional[str] = None,
               zone_attr: Optional[str] = None) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}

        # -------- 连续周期编码 --------
        # 经度 360° 周期
        feats["lon_fourier"] = fourier_cyclic(lon_deg % 360.0, period=360.0, harmonics=self.cfg.lon_harmonics)

        # 纬度的处理：使用对称性（sin/cos φ），再补充绝对纬度的周期编码
        lat_rad = deg2rad(lat_deg)
        sphi, cphi = safe_sin_cos(lat_rad)
        feats["lat_sin_cos"] = [sphi, cphi]

        # 绝对纬度作 360° 周期的一半（0..180），避免南北符号对称性问题
        feats["abs_lat_fourier"] = fourier_cyclic(abs(lat_deg), period=180.0, harmonics=self.cfg.lat_harmonics)

        # 球面单位向量（避免经度跳变）
        if self.cfg.add_unitvec:
            feats["unitvec_xyz"] = list(lonlat_to_unitvec(lon_deg, lat_deg))

        # 衍生标量
        if self.cfg.add_abs_lat:
            feats["abs_lat_deg"] = abs(lat_deg)
        if self.cfg.add_equator_dist:
            feats["dist_equator_deg"] = abs(lat_deg)  # 到赤道角距
        if self.cfg.add_tropics_dist:
            feats["dist_tropics_deg"] = abs(abs(lat_deg) - 23.44)
        if self.cfg.add_polarcircle_dist:
            feats["dist_polarcircle_deg"] = abs(abs(lat_deg) - 66.56)

        # -------- 分带 one-hot --------
        belt_idx = lat_belt_index(lat_deg, self.cfg.belt_config)
        feats["lat_belt_index"] = belt_idx
        feats["lat_belt_onehot"] = one_hot(belt_idx, 5)

        # -------- 可选：DEM 海拔/粗糙度 --------
        if dem_path:
            elev = sample_dem_elevation(dem_path, lon_deg, lat_deg)
            feats["elevation_m"] = elev
            stats = sample_dem_stats(dem_path, lon_deg, lat_deg, half_window_px=3)
            if stats:
                feats.update(stats)  # elev_mean/std/range

        # -------- 可选：矢量分区（如 Köppen、生态区、海区等） --------
        if zone_vector_path:
            z = point_in_zones(lon_deg, lat_deg, zone_vector_path, attr_name=zone_attr)
            feats["zone_attr"] = z  # 可直接存字符串标签或类别编码（自行映射）

        return feats

    def to_vector(self, feat_dict: Dict[str, Any]) -> List[float]:
        """
        把字典中的数值字段拼成一个扁平向量（便于喂给网络）。
        非数值/None 会被跳过。离散 zone_attr 可在外部映射到 id/one-hot 再拼。
        """
        vec: List[float] = []

        def add(v):
            if v is None:
                return
            if isinstance(v, (int, float)):
                vec.append(float(v))
                return
            if isinstance(v, (list, tuple)):
                for x in v:
                    if x is None:
                        continue
                    if isinstance(x, (int, float)):
                        vec.append(float(x))

        add(feat_dict.get("lon_fourier"))
        add(feat_dict.get("lat_sin_cos"))
        add(feat_dict.get("abs_lat_fourier"))
        add(feat_dict.get("unitvec_xyz"))
        add(feat_dict.get("abs_lat_deg"))
        add(feat_dict.get("dist_equator_deg"))
        add(feat_dict.get("dist_tropics_deg"))
        add(feat_dict.get("dist_polarcircle_deg"))
        add(feat_dict.get("lat_belt_onehot"))
        add(feat_dict.get("elevation_m"))
        add(feat_dict.get("elev_mean"))
        add(feat_dict.get("elev_std"))
        add(feat_dict.get("elev_range"))
        return vec


# ============ 小示例 ============
if __name__ == "__main__":
    enc = MultiScaleGeoEncoder()
    # 例：舟山群岛附近
    lon, lat = 122.20, 29.95
    feats = enc.encode(lon, lat, dem_path=None, zone_vector_path=None)
    vec = enc.to_vector(feats)
    print("Feature keys:", list(feats.keys()))
    print("Vector length:", len(vec))
    print("Vector: ",vec)
    print("lat_belt:", enc.cfg.belt_config.names[feats['lat_belt_index']])
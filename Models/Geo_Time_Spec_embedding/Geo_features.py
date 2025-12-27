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


#print("lat_belt:", enc.cfg.belt_config.names[feats['lat_belt_index']])
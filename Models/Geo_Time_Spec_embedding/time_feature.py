# -*- coding: utf-8 -*-
"""
多尺度时间特征预处理（无第三方依赖）
输入：UTC 时间 + 经纬度（可选时区），输出各类周期性时间元数据
"""

from dataclasses import dataclass
from typing import Optional, Dict
import math
from datetime import datetime, timezone, timedelta



# -------------------------
# 示例：执行测试
# -------------------------
if __name__ == "__main__":
    # 测试输入：2023年夏至（UTC时间）+ 北京坐标（116.38°E，39.90°N）
    test_dt_utc = datetime(2023, 6, 21, 3, 30, tzinfo=timezone.utc)
    test_lon, test_lat = 116.38, 39.90
    # 生成特征
    time_feats = build_time_features(test_dt_utc, test_lon, test_lat, tz_str=None)
    print("time_feats: ",time_feats)
    # 打印结果
    print("时间特征计算结果：")
    for key, value in time_feats.items():
        # 索引类特征保留整数，其他保留2位小数
        if "idx" in key:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.2f}")
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
# 工具：基本历法/天文近似
# -------------------------

def julian_day(dt_utc: datetime) -> float:
    """计算UTC时间对应的儒略日（JD）"""
    assert dt_utc.tzinfo is not None, "需要 timezone-aware 的 UTC datetime"
    dt_utc = dt_utc.astimezone(timezone.utc)
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day + (dt_utc.hour + (dt_utc.minute + dt_utc.second/60.0)/60.0)/24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 5
    jd = int(365.25*(y + 4716)) + int(30.6001*(m + 1)) + d + B - 1524.5
    return jd


def solar_longitude_deg(dt_utc: datetime) -> float:
    """近似计算太阳黄经（度），精度满足节气/季节划分"""
    jd = julian_day(dt_utc)
    T = (jd - 2451545.0) / 36525.0  # 世纪数
    # 平黄经 L0、平近点角 M（单位度）
    L0 = (280.46646 + 36000.76983*T + 0.0003032*T*T) % 360.0
    M = (357.52911 + 35999.05029*T - 0.0001537*T*T) % 360.0
    Mr = math.radians(M)
    # 地心黄经（忽略黄道章动与小项）
    lam = L0 + (1.914602 - 0.004817*T - 0.000014*T*T)*math.sin(Mr) \
          + (0.019993 - 0.000101*T)*math.sin(2*Mr) \
          + 0.000289*math.sin(3*Mr)
    return lam % 360.0


# 朔望月参数：平均朔望月长度 + 参考新月儒略日
SYNODIC_MONTH = 29.530588853  # 天
REF_NEW_MOON_JD = 2451550.1   # 2000-01-06 18:14 UTC 附近新月

def lunar_age_days(dt_utc: datetime) -> float:
    """计算朔望月月龄（天，0~29.53）"""
    jd = julian_day(dt_utc)
    age = (jd - REF_NEW_MOON_JD) % SYNODIC_MONTH
    if age < 0:
        age += SYNODIC_MONTH
    return age


# -------------------------
# 配置与类别名称
# -------------------------

SOLAR_TERMS = [
    "立春", "雨水", "惊蛰", "春分", "清明", "谷雨",
    "立夏", "小满", "芒种", "夏至", "小暑", "大暑",
    "立秋", "处暑", "白露", "秋分", "寒露", "霜降",
    "立冬", "小雪", "大雪", "冬至", "小寒", "大寒"
]  # 对应黄经：315°,330°,345°,0°,15°,...,300°

ZODIAC = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
SEASONS_4 = ["春", "夏", "秋", "冬"]


# -------------------------
# 配置类
# -------------------------

@dataclass
class TimeFeatureConfig:
    """时间特征计算配置"""
    # 是否用经度近似时区（round(lon/15)）
    tz_offset_hours_from_lon: bool = True
    # 生肖年切换规则：'lìchūn'（立春）或 'chūnjié'（春节，简化用立春）
    zodiac_year_rule: str = "lìchūn"
    # 早中晚分桶边界（本地时，小时）：[夜,5)→夜，[5,10)→早，[10,14)→中，[14,18)→晚，[18,24)+[0,5)→夜
    tod_buckets: tuple = (5, 10, 14, 18)


# -------------------------
# 核心计算函数
# -------------------------

def local_datetime(dt_utc: datetime, lon_deg: float, tz_str: Optional[str], cfg: TimeFeatureConfig) -> datetime:
    """将UTC时间转换为本地时间（优先时区字符串，其次经度近似）"""
    if tz_str:
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            return dt_utc.astimezone(ZoneInfo(tz_str))
        except Exception:
            pass
    # 经度近似时区：时区 = round(经度/15)（每15°对应1小时时差）
    if cfg.tz_offset_hours_from_lon:
        offset = round(lon_deg / 15.0)
        return dt_utc.astimezone(timezone(timedelta(hours=offset)))
    # 若均不满足，返回UTC时间（不推荐）
    return dt_utc


def season_index_from_month(month: int) -> int:
    """按月份计算四季索引：0=春(3-5)，1=夏(6-8)，2=秋(9-11)，3=冬(12-2)"""
    if month in (3, 4, 5):
        return 0
    elif month in (6, 7, 8):
        return 1
    elif month in (9, 10, 11):
        return 2
    else:
        return 3


def solar_term_index(dt_utc: datetime) -> int:
    """按太阳黄经计算24节气索引（0~23，立春为0）"""
    lam = solar_longitude_deg(dt_utc)  # 太阳黄经（0°=春分）
    # 旋转黄经：将立春（315°）作为起点
    lam_rot = (lam - 315.0) % 360.0
    idx = int(math.floor(lam_rot / 15.0)) % 24  # 每15°一个节气
    return idx


def zodiac_index_from_year(dt_local: datetime, rule: str = "lìchūn") -> int:
    """计算生肖索引（0~11，0=鼠），默认以立春为年界"""
    year = dt_local.year
    # 若在当年立春前，归属上一年（简化：2月5日前视为立春前）
    if rule == "lìchūn" and (dt_local.month, dt_local.day) < (2, 5):
        year -= 1
    # 1984年为甲子鼠年，作为基准
    return (year - 1984) % 12


def time_of_day_bucket(hour_local: float, buckets=(5, 10, 14, 18)) -> int:
    """将本地时（0~24）分桶：0=早，1=中，2=晚，3=夜"""
    h = hour_local % 24.0
    b0, b1, b2, b3 = buckets
    if h < b0 or h >= b3:
        return 3  # 夜
    elif h < b1:
        return 0  # 早
    elif h < b2:
        return 1  # 中
    else:
        return 2  # 晚


def day_of_year(dt_local: datetime) -> float:
    """计算积日（当年第几天，1~365/366，含小数小时）"""
    start = datetime(dt_local.year, 1, 1, tzinfo=dt_local.tzinfo)
    delta = dt_local - start
    return delta.days + delta.seconds / 86400.0 + 1.0  # 1为1月1日


def fortnight_phase_days(dt_utc: datetime) -> float:
    """计算半月潮相位（0~14.77天，简化为朔望月的一半）"""
    return lunar_age_days(dt_utc) % (SYNODIC_MONTH / 2.0)


# -------------------------
# 对外主接口
# -------------------------

def build_time_features(
    dt_utc: datetime,
    lon_deg: float,
    lat_deg: float,
    tz_str: Optional[str] = None,
    cfg: TimeFeatureConfig = TimeFeatureConfig()
) -> Dict[str, float]:
    """
    生成多尺度时间特征字典
    返回键：
        - hour_local: 本地时（浮点小时，0~24）
        - day_of_year: 积日（1~365/366）
        - lunar_age_days: 朔望月月龄（天）
        - fortnight_phase_days: 半月潮相位（天，0~14.77）
        - season_idx: 四季索引（0=春，1=夏，2=秋，3=冬）
        - solar_term_idx: 24节气索引（0~23）
        - zodiac_idx: 生肖索引（0~11）
        - tod_bucket_idx: 早中晚夜索引（0~3）
        - solar_longitude_deg: 太阳黄经（度）
    """
    assert dt_utc.tzinfo is not None, "dt_utc必须带UTC时区，如datetime(..., tzinfo=timezone.utc)"
    dt_local = local_datetime(dt_utc, lon_deg, tz_str, cfg)

    # 计算各特征
    hour_loc = dt_local.hour + (dt_local.minute + dt_local.second / 60.0) / 60.0
    doy = day_of_year(dt_local)
    lam = solar_longitude_deg(dt_utc)
    st_idx = solar_term_index(dt_utc)
    sea_idx = season_index_from_month(dt_local.month)
    zod_idx = zodiac_index_from_year(dt_local, cfg.zodiac_year_rule)
    tod_idx = time_of_day_bucket(hour_loc, cfg.tod_buckets)
    la = lunar_age_days(dt_utc)
    fn = fortnight_phase_days(dt_utc)

    return {
        "hour_local": hour_loc,
        "day_of_year": doy,
        "lunar_age_days": la,
        "fortnight_phase_days": fn,
        "season_idx": sea_idx,
        "solar_term_idx": st_idx,
        "zodiac_idx": zod_idx,
        "tod_bucket_idx": tod_idx,
        "solar_longitude_deg": lam
    }



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
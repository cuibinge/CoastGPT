import os
import json
import re
import math
from datetime import datetime
import pytz
from dateutil import tz
from geopy.geocoders import Nominatim

sig_dir = r"C:\Users\zhhq9\Desktop\suaeda_ecotypes\red"
output_geojson = os.path.join(r"C:\Users\zhhq9\Desktop\suaeda_ecotypes", "red_suaeda.geojson")
geolocator = Nominatim(user_agent="geojson_generator")

# 节气计算
def get_solar_term(date: datetime) -> str:
    solar_terms = [
        ("小寒", 1, 5), ("大寒", 1, 20), ("立春", 2, 4), ("雨水", 2, 19),
        ("惊蛰", 3, 6), ("春分", 3, 21), ("清明", 4, 5), ("谷雨", 4, 20),
        ("立夏", 5, 6), ("小满", 5, 21), ("芒种", 6, 6), ("夏至", 6, 21),
        ("小暑", 7, 7), ("大暑", 7, 22), ("立秋", 8, 7), ("处暑", 8, 23),
        ("白露", 9, 7), ("秋分", 9, 23), ("寒露", 10, 8), ("霜降", 10, 23),
        ("立冬", 11, 7), ("小雪", 11, 22), ("大雪", 12, 7), ("冬至", 12, 22)
    ]
    for term, month, day in reversed(solar_terms):
        if (date.month > month) or (date.month == month and date.day >= day):
            return term
    return "小寒"

# 基于视角+拍摄距离估算实际图像覆盖面积
def calculate_area_and_scale():
    distance_m = 1.0  # 假设拍摄距离 1 米
    fov_deg = 30      # 假设水平视角 30°
    width_m = 2 * distance_m * math.tan(math.radians(fov_deg / 2))  # ≈ 0.536 m
    height_m = width_m * (240 / 320)  # 保持图像纵横比
    area_km2 = (width_m * height_m) / 1e6  # 转为 km²
    area_m2 = width_m * height_m
    return area_m2, area_km2, "Patch Level"

# 解析 .sig 文件
def extract_metadata(file_path):
    metadata = {"bands": [], "coordinates": None, "timestamp": None}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 提取经纬度和时间（在文件头部）
    for line in lines:
        if line.lower().startswith("longitude"):
            match = re.search(r"(\d+\.\d+)[Ee]", line)
            if match:
                dmm = float(match.group(1))
                lon = round(int(dmm / 100) + (dmm % 100) / 60, 6)
        elif line.lower().startswith("latitude"):
            match = re.search(r"(\d+\.\d+)[Nn]", line)
            if match:
                dmm = float(match.group(1))
                lat = round(int(dmm / 100) + (dmm % 100) / 60, 6)
        elif line.lower().startswith("time="):
            match = re.search(r"(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}:\d{2})", line)
            if match:
                dt = datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S")
                metadata["timestamp"] = dt.astimezone(pytz.utc).isoformat()

    for i in range(27, min(1020, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                wavelength = float(parts[0])
                reflectance = float(parts[1])
                metadata["bands"].append((wavelength, reflectance))
            except ValueError:
                # 非数值行跳过
                continue

    if 'lon' in locals() and 'lat' in locals():
        metadata["coordinates"] = [lon, lat]
    return metadata

# 生成 GeoJSON
features = []
for file in os.listdir(sig_dir):
    if not file.endswith(".sig"): continue
    metadata = extract_metadata(os.path.join(sig_dir, file))
    if not metadata["coordinates"] or not metadata["timestamp"]:
        print(f"跳过: {file}")
        continue

    dt = datetime.fromisoformat(metadata["timestamp"])
    local_dt = dt.astimezone(tz.gettz("Asia/Shanghai"))
    season = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
              "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter"][local_dt.month - 1]
    part_of_day = ("Night" if local_dt.hour < 6 else "Morning" if local_dt.hour < 10
                   else "Noon" if local_dt.hour < 14 else "Afternoon" if local_dt.hour < 18 else "Evening")
    solar_term = get_solar_term(local_dt)
    area_m2, area_km2, scale = calculate_area_and_scale()
    wavelengths = [w[0] for w in metadata["bands"]]
    reflectances = [w[1] for w in metadata["bands"]]

    try:
        loc = geolocator.reverse((metadata["coordinates"][1], metadata["coordinates"][0]), language="zh-CN", timeout=10)
        addr = loc.raw.get("address", {})
    except:
        addr = {}

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": metadata["coordinates"]
        },
        "properties": {
            "instrument": "HI: 9162087 (HR-1024i)",
            "label": "盐地碱蓬",
            "color": "浅红、赤红或紫红",
            "height": "矮小",
            "leaf_shape": "肉质化",
            "salinity_range": "1%~1.6%",
            "ecotype": "红色矮小",
            "country_code": "CHN",
            "area(km²)": round(area_km2, 8),
            "area(m²)": round(area_m2, 2),
            "spatial_scale": scale,
            "location_info": {
                "province": addr.get("state", ""),
                "city": addr.get("city", addr.get("town", ""))
            },
            "temporal_info": {
                "timestamp": metadata["timestamp"],
                "datetime_local": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "season": season,
                "part_of_day": part_of_day,
                "weekday": local_dt.strftime("%A"),
                "timezone": "Asia/Shanghai",
                "solar_term": solar_term
            },
            "bands": {
                "range_nm": f"{min(wavelengths):.1f} ~ {max(wavelengths):.1f}",
                "center_nm": round(sum(wavelengths)/len(wavelengths), 1),
                "reflectance_range": f"{min(reflectances):.2f} ~ {max(reflectances):.2f}",
                "total_bands": len(metadata["bands"])
            },
            "caption": "盐地碱蓬呈现出红色，植株矮小，叶片粗短肉质化，生于高盐泥滩区。",
        }
    }
    features.append(feature)

# 保存 GeoJSON
geojson = {
    "type": "FeatureCollection",
    "name": "Suaeda_Spectral_Samples",
    "features": features
}

with open(output_geojson, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=4)

print(f"GeoJSON 文件已生成: {output_geojson}")

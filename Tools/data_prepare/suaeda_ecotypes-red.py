import os
import json
import re
import math
from datetime import datetime
from dateutil import tz
import pytz
from geopy.geocoders import Nominatim
import time

# 输入/输出目录
sig_dir = r"C:\Users\zhhq9\Desktop\suaeda_ecotypes\红碱蓬"
geojson_output_dir = r"C:\Users\zhhq9\Desktop\suaeda_ecotypes\红碱蓬\json"
os.makedirs(geojson_output_dir, exist_ok=True)

geolocator = Nominatim(user_agent="geojson_generator")

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

def calculate_area_and_scale():
    distance_m = 1.0
    fov_deg = 30
    width_m = 2 * distance_m * math.tan(math.radians(fov_deg / 2))
    height_m = width_m * (240 / 320)
    return width_m * height_m, (width_m * height_m) / 1e6, "Patch Level"

def extract_metadata(file_path):
    metadata = {"coordinates": None, "timestamp": None, "data": []}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        if "longitude" in line.lower():
            match = re.search(r"(\d+\.\d+)[Ee]", line)
            if match:
                dmm = float(match.group(1))
                lon = round(int(dmm / 100) + (dmm % 100) / 60, 6)
        elif "latitude" in line.lower():
            match = re.search(r"(\d+\.\d+)[Nn]", line)
            if match:
                dmm = float(match.group(1))
                lat = round(int(dmm / 100) + (dmm % 100) / 60, 6)
        elif line.lower().startswith("time="):
            match = re.search(r"(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}:\d{2})", line)
            if match:
                dt = datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S")
                metadata["timestamp"] = dt.astimezone(pytz.utc).isoformat()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                metadata["data"].append([float(p) for p in parts])
            except:
                continue

    if 'lat' in locals() and 'lon' in locals():
        metadata["coordinates"] = [lon, lat]

    return metadata

# 主循环：每个.sig生成一个GeoJSON
for file in os.listdir(sig_dir):
    if not file.endswith(".sig"):
        continue

    sig_path = os.path.join(sig_dir, file)
    metadata = extract_metadata(sig_path)

    if not metadata["coordinates"] or not metadata["timestamp"]:
        print(f"跳过无效文件: {file}")
        continue

    dt = datetime.fromisoformat(metadata["timestamp"])
    local_dt = dt.astimezone(tz.gettz("Asia/Shanghai"))
    season = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
              "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter"][local_dt.month - 1]
    part_of_day = ("Night" if local_dt.hour < 6 else "Morning" if local_dt.hour < 10
                   else "Noon" if local_dt.hour < 14 else "Afternoon" if local_dt.hour < 18 else "Evening")
    solar_term = get_solar_term(local_dt)
    area_m2, area_km2, scale = calculate_area_and_scale()

    wavelengths = [d[0] for d in metadata["data"]]
    reference = [d[1] for d in metadata["data"]]
    target = [d[2] for d in metadata["data"]]
    reflectance = [d[3] for d in metadata["data"]]

    try:
        time.sleep(1)  # 防止请求过快
        loc = geolocator.reverse((metadata["coordinates"][1], metadata["coordinates"][0]), language="zh-CN", timeout=10)
        addr = loc.raw.get("address", {})
    except Exception as e:
        print(f"地理信息失败：{file} - {e}")
        addr = {}

    province = addr.get("state") or addr.get("state_district") or addr.get("region", "")
    city = addr.get("city") or addr.get("town") or addr.get("county", "")

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
                "province": province,
                "city": city
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
                "wavelengths_nm_range": f"{min(wavelengths):.1f} ~ {max(wavelengths):.1f}",
                "reference_values_range": f"{min(reference):.2f} ~ {max(reference):.2f}",
                "target_values_range": f"{min(target):.2f} ~ {max(target):.2f}",
                "reflectance_range": f"{min(reflectance):.2f} ~ {max(reflectance):.2f}",
                "total_bands": len(metadata["data"]),
                "data": metadata["data"]
            },
            "caption": "盐地碱蓬呈现出红色，植株矮小，叶片粗短肉质化，生于高盐泥滩区。"
        }
    }

    geojson_data = {
        "type": "FeatureCollection",
        "name": file.replace(".sig", ".sig.jpg"),
        "features": [feature]
    }

    output_path = os.path.join(geojson_output_dir, file.replace(".sig", ".geojson"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"已生成: {output_path}")

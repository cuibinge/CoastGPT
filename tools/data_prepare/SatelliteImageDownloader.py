import json
import requests
import math
import os

# Google Maps API Key
API_KEY = "AIzaSyAXVvMxySW3a-NE_baAYhCUcOUIzhJf5U4"

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# 解析 JSON 数据
def parse_json_data(json_data):
    images_info = []

    for item in json_data["data"]:
        name_parts = item["name"].split("_")
        if len(name_parts) < 5:
            continue

        country = item["info"]["location"][0]
        city = item["info"]["location"][1]
        image_id = name_parts[0]

        ul_lon = float(name_parts[1])  # 左上角经度
        ul_lat = float(name_parts[2])  # 左上角纬度
        lr_lon = float(name_parts[3])  # 右下角经度
        lr_lat = float(name_parts[4])  # 右下角纬度

        # 生成符合要求的文件名
        filename = f"{country}_{image_id}_{city}_{ul_lon}_{ul_lat}_{lr_lon}_{lr_lat}.jpg"

        images_info.append({
            "filename": filename,
            "ul_lon": ul_lon,
            "ul_lat": ul_lat,
            "lr_lon": lr_lon,
            "lr_lat": lr_lat
        })

    return images_info


# 计算中心点
def get_center(ul_lat, ul_lon, lr_lat, lr_lon):
    return (ul_lat + lr_lat) / 2, (ul_lon + lr_lon) / 2

# 经纬度 -> 瓦片坐标
def lat_lon_to_tile_coords(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

# 创建 `session_token`
def create_session():
    url = f"https://tile.googleapis.com/v1/createSession?key={API_KEY}"
    session_data = {
        "mapType": "satellite",
        "language": "en-US",
        "region": "US"
    }
    response = requests.post(url, json=session_data)
    if response.status_code == 200:
        return response.json().get("session", "")
    return None

# 下载卫星瓦片
def download_tile(session_token, zoom, x, y, filename):
    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}?session={session_token}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs("downloaded_images", exist_ok=True)
        with open(f"downloaded_images/{filename}", "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {filename}")

# 处理 JSON 批量下载
def process_json(json_file):
    data = load_json(json_file)
    images_info = parse_json_data(data)
    session_token = create_session()  # 获取 session_token
    if not session_token:
        print("Failed to create session token.")
        return

    for image in images_info:
        lat_center, lon_center = get_center(image["ul_lat"], image["ul_lon"], image["lr_lat"], image["lr_lon"])
        zoom = 18  # 设置缩放级别
        x_tile, y_tile = lat_lon_to_tile_coords(lat_center, lon_center, zoom)

        download_tile(session_token, zoom, x_tile, y_tile, image["filename"])

# 运行批量处理
json_file_path = r"D:\LHRS\OSCapAnn(Stage1)\captions_01.json"  # 你的 JSON 文件
process_json(json_file_path)
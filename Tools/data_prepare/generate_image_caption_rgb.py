import os
import re
import json
import base64
import numpy as np
from PIL import Image
from datetime import datetime
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import rasterio
from groq import Groq
import requests
# 加载环境变量
load_dotenv()

# 配置路径
image_dir = r"C:\Users\zhhq9\Desktop\gf1"  # 遥感图像目录
output_json = "RS_Caption_Dataset.json"

# 初始化地理编码器
geolocator = Nominatim(user_agent="rs_vision", timeout=10)

def parse_filename_metadata(filename):
    # 支持所有观测到的格式：
    # GF1B_PMS_E120.9_N28.0_20230130_L1A1228249417-MUX
    # GF2_PM51_E119.4_N34.7_20231021_L1A13413411001-MSS1
    # GF1_WFV4_E123.1_N46.7_20231005_L1A13115564001
    pattern = r"""
        ^GF(?P<satellite>\d+[A-Z]?)               # 卫星编号（支持GF1B这种格式）
        _(?P<sensor>PMS|PM51|WFV\d)               # 传感器类型
        (?:_E(?P<lon>\d+\.\d+)                    # 可选经度部分
           _N(?P<lat>\d+\.\d+)                    # 可选纬度部分
           _(?P<date>\d{8})?                      # 可选日期
        )?
        _(?P<product_id>L1A[\d_]+)                # 产品编号（支持带下划线的版本）
        (?:-(?P<resolution_type>MSS|MUX)          # 可选分辨率类型
           (?P<resolution>\d*))?                  # 可选分辨率数值
        \..*$                                     # 文件扩展名
    """

    match = re.search(pattern, filename, re.IGNORECASE | re.VERBOSE)

    metadata = {
        "satellite": "Unknown",
        "sensor": "Unknown",
        "lon": None,
        "lat": None,
        "date": "Unknown",
        "resolution": "Unknown",
        "product_id": "Unknown"
    }

    if not match:
        print(f"正则无法匹配的文件名: {filename}")
        return metadata

    groups = match.groupdict()

    try:
        # 1. 处理卫星和传感器
        metadata["satellite"] = f"GF{groups['satellite']}" if groups.get('satellite') else "Unknown"
        metadata["sensor"] = groups.get('sensor', "Unknown")
        metadata["product_id"] = groups.get('product_id', "Unknown")

        # 2. 处理坐标
        if groups.get("lon") and groups.get("lat"):
            try:
                metadata["lon"] = float(groups["lon"])
                metadata["lat"] = float(groups["lat"])
            except (ValueError, TypeError):
                pass

        # 3. 处理日期
        if groups.get("date"):
            try:
                metadata["date"] = datetime.strptime(groups["date"], "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pass

        # 4. 处理分辨率（完全重写的逻辑）
        resolution_rules = {
            "GF1": {"PMS": "2m", "WFV1": "16m", "WFV2": "16m", "WFV3": "16m", "WFV4": "16m"},
            "GF1B": {"PMS": "2m"},
            "GF1C": {"PMS": "2m"},
            "GF1D": {"PMS": "2m"},
            "GF2": {"PM51": {"MSS1": "0.8m", "MSS2": "3.2m", "MUX": "2m"}, "PMS": "0.8m"},
            "GF6": {"PMS": "2m", "WFV": "16m"},
            "GF7": {"PMS": "0.5m"}
        }

        sat = metadata["satellite"]
        sen = metadata["sensor"]
        res_type = groups.get("resolution_type", "")
        res_value = groups.get("resolution", "")

        # 查找最匹配的规则
        if sat in resolution_rules:
            # 尝试完全匹配传感器
            if sen in resolution_rules[sat]:
                rule = resolution_rules[sat][sen]
                if isinstance(rule, dict):
                    key = f"{res_type}{res_value}" if res_type else ""
                    metadata["resolution"] = rule.get(key, rule.get("", "Unknown"))
                else:
                    metadata["resolution"] = rule
            else:
                # 尝试部分匹配传感器
                for sensor_pattern, rule in resolution_rules[sat].items():
                    if sensor_pattern in sen or sen in sensor_pattern:
                        if isinstance(rule, dict):
                            key = f"{res_type}{res_value}" if res_type else ""
                            metadata["resolution"] = rule.get(key, rule.get("", "Unknown"))
                        else:
                            metadata["resolution"] = rule
                        break

    except Exception as e:
        print(f"处理 {filename} 时出现意外错误: {str(e)}")
        import traceback
        traceback.print_exc()

    return metadata


def process_3band_image(filepath):
    """通用三波段图像处理器，支持TIF/PNG"""
    try:
        if filepath.lower().endswith('.tif'):
            with rasterio.open(filepath) as src:
                # 使用下采样读取（降低分辨率）
                scale_factor = 4  # 根据实际需求调整
                width = src.width // scale_factor
                height = src.height // scale_factor

                bands = [
                    src.read(1, out_shape=(1, height, width)),
                    src.read(2, out_shape=(1, height, width)),
                    src.read(3, out_shape=(1, height, width))
                ]
                rgb = np.dstack(bands)

                # 自动对比度拉伸
                percentiles = np.percentile(rgb, [2, 98])
                rgb = np.clip(rgb, percentiles[0], percentiles[1])
                rgb = ((rgb - percentiles[0]) /
                       (percentiles[1] - percentiles[0]) * 255).astype(np.uint8)

        # 处理PNG文件
        elif filepath.lower().endswith('.png'):
            with Image.open(filepath) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    raise ValueError("Unsupported image mode")
                # 移除alpha通道
                rgb = np.array(img.convert('RGB'))

        # 统一后处理
        img = Image.fromarray(rgb)
        img = img.resize((1024, 1024))

        # 保存为临时文件
        img.save("temp.jpg", "JPEG", quality=95)
        with open("temp.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        print(f"图像处理失败: {str(e)}")
        return ""


def generate_rs_description(image_base64, metadata):
    # 构建坐标描述
    coord_info = "Coordinates: Unknown"
    if metadata["lat"] and metadata["lon"]:
        coord_info = f"Coordinates: {metadata['lat']:.4f}°N, {metadata['lon']:.4f}°E"

    prompt = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        As a professional remote sensing analyst, generate JSON containing:
        1. cap: 1-sentence summary
        2. caption: Within 200 words
        3. attrs: Array of land attributes using these keys:
           - landuse (e.g. farmland, residential)
           - landcover (e.g. vegetation, water)
           - natural (e.g. wood, ground)
           Format example:
           {{
             "attrs": [
                 {{"landuse": "farmland"}},
                 {{"landcover": "bare_ground", "natural": "ground"}},
                 {{"landuse": "industrial"}}
             ]
           }}
        Metadata:
        - Satellite: {metadata['satellite']}
        - Resolution: {metadata['resolution']}
        - Coordinates:  {coord_info}
        - Date: {metadata['date']}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Analyze this satellite image.<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {{
            "cap": """

    try:
        from groq import Groq
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response = json.loads(completion.choices[0].message.content)

        # 属性后处理验证
        if "attrs" in response:
            for attr in response["attrs"]:
                if not isinstance(attr, dict):
                    print("属性格式错误，正在修正...")
                    response["attrs"] = [{"landuse": str(x)} for x in response["attrs"] if isinstance(x, str)]
                    break
        else:
            response["attrs"] = []

        return response

    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {
            "cap": "Automated description",
            "caption": "Analysis unavailable",
            "attrs": []
        }


def main():
    dataset = {"data": []}

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.tif', '.png')):
            continue

        filepath = os.path.join(image_dir, filename)
        try:
            # 元数据提取
            metadata = parse_filename_metadata(filename)
            # 图像处理
            image_base64 = process_3band_image(filepath)
            if not image_base64:
                continue

            # 地理编码
            # 在main函数的地理编码部分添加：
            # 地理编码
            if metadata["lat"] and metadata["lon"]:
                try:
                    location = geolocator.reverse(
                        (metadata["lat"], metadata["lon"]),
                        exactly_one=True,
                        language="en",
                        timeout=15
                    )
                    if location:
                        addr = location.raw.get("address", {})
                        metadata["location"] = {
                            "country": addr.get("country", "Unknown"),
                            "province": addr.get("state", "Unknown"),
                            "city": addr.get("city", addr.get("county", "Unknown"))
                        }
                    else:
                        metadata["location"] = {"error": "No location found"}
                except Exception as e:
                    print(f"地理编码失败 ({filename}): {str(e)}")
                    metadata["location"] = {"error": str(e)}
            else:
                metadata["location"] = {"error": "Invalid coordinates"}

            # 生成描述
            ai_response = generate_rs_description(image_base64, metadata)

            # 构建数据项
            dataset["data"].append({
                "name": filename.rsplit(".", 1)[0],
                "info": {
                    "satellite": metadata["satellite"],
                    "sensor": metadata["sensor"],
                    "acquisition_date": metadata["date"],
                    "resolution": metadata["resolution"],
                    "location": metadata["location"],
                    "coordinates": {
                        "lat": metadata["lat"],
                        "lon": metadata["lon"]
                    }
                },
                    "attrs": ai_response.get("attrs", []),
                    "cap": ai_response.get("cap", ""),
                    "caption": ai_response.get("caption", "")
            })

        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")
            continue

    # 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"成功处理 {len(dataset['data'])} 个遥感影像")


if __name__ == "__main__":
    main()
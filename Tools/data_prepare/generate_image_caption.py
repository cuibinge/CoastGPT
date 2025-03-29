import os
import re
import json
import base64
import requests
import numpy as np
from PIL import Image
from datetime import datetime
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import rasterio
from groq import Groq

# 加载环境变量
load_dotenv()

# 配置路径
image_dir = r"C:\Users\zhhq9\Desktop\Download\tif"
output_json = "CaptionGeneration_dataset.json"

# 初始化地理编码器
geolocator = Nominatim(user_agent="rs_vision", timeout=10)


def parse_filename_metadata(filename):
    pattern = r"""
        ^.*?
        (?:(?P<satellite>GF\d+)[_\-]?)?
        (?:(?P<sensor>PMS\d+)[_\-]?)?
        (?:E(?P<lon>\d+\.\d+)[_\-]?)?
        (?:N(?P<lat>\d+\.\d+)[_\-]?)?
        (?:(?P<date>\d{8})[_\-]?)?
        .*?
        (?:-MSS(?P<resolution>\d+\.\d+))
        .*?
        (?:\.tif$)
    """
    match = re.search(pattern, filename, re.IGNORECASE | re.VERBOSE)
    metadata = {}
    if match:
        groups = match.groupdict()
        # 卫星和传感器
        metadata["satellite"] = groups.get("satellite", "Unknown")
        metadata["sensor"] = groups.get("sensor", "Unknown")

        # 分辨率处理（优先使用文件名解析结果）
        if groups.get("resolution"):
            metadata["resolution"] = f"{groups['resolution']}m"
        elif metadata["satellite"] == "GF2" and metadata["sensor"] == "PMS1":
            metadata["resolution"] = "1m"  # 卫星传感器后备方案
        else:
            metadata["resolution"] = "Unknown"

        # 坐标处理
        try:
            metadata["lon"] = float(groups.get("lon", 0))
            metadata["lat"] = float(groups.get("lat", 0))
        except:
            metadata["lon"] = 0.0
            metadata["lat"] = 0.0

        # 日期处理
        if groups.get("date"):
            try:
                metadata["date"] = datetime.strptime(groups["date"], "%Y%m%d").strftime("%Y-%m-%d")
            except:
                metadata["date"] = "Unknown"
        else:
            metadata["date"] = "Unknown"

    return metadata

def process_tif_image(filepath):
    with rasterio.open(filepath) as src:
        # 提取RGB波段（根据实际数据调整波段索引）
        rgb = np.stack([src.read(3), src.read(2), src.read(1)], axis=0)
        rgb = np.moveaxis(rgb, 0, -1)
        rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        # 转换为JPEG并编码为Base64
        img = Image.fromarray(rgb)
        img = img.resize((1024, 1024))  # 调整尺寸以适应模型
        img.save("temp.jpg", "JPEG")
        with open("temp.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

def generate_vision_description(image_base64, metadata):
    """调用多模态模型生成包含attrs属性的描述"""
    prompt = f"""<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    As a professional remote sensing analyst, generate JSON containing:
    1. cap: 1-sentence summary
    2. caption: 3-5 sentence description
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
    - Coordinates: {metadata['lat']:.4f}°N, {metadata['lon']:.4f}°E
    - Date: {metadata['date']}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Analyze this satellite image.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {{
        "cap": """

    try:
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
        if not filename.lower().endswith(".tif"):
            continue
        filepath = os.path.join(image_dir, filename)
        try:
            metadata = parse_filename_metadata(filename)
            if not metadata:
                continue

            image_base64 = process_tif_image(filepath)

            # # 地理编码获取三级地址
            # location = geolocator.reverse(
            #     (metadata["lat"], metadata["lon"]),
            #     language="en"
            # )
            # if location:
            #     address = location.raw.get("address", {})
            #     metadata["location"] = [
            #         address.get("state", "Unknown"),
            #         address.get("city", address.get("county", "Unknown")),
            #         address.get("county", address.get("town", "Unknown"))
            #     ]
            # else:
            #     metadata["location"] = ["Unknown", "Unknown", "Unknown"]
            # 地理编码
            location = geolocator.reverse(
                (metadata["lat"], metadata["lon"]),
                language="en"
            )
            metadata["location"] = [
                location.raw["address"].get("country", "Unknown"),
                location.raw["address"].get("state", "Unknown")
            ] if location else ["Unknown", "Unknown"]
            # 调用多模态模型
            ai_response = generate_vision_description(image_base64, metadata)

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
                    },
                    "attrs": ai_response.get("attrs", [])
                },
                "analysis": {
                    "cap": ai_response.get("cap", ""),
                    "caption": ai_response.get("caption", "")
                }
            })

        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")
            continue

    # 保存结果（按处理顺序）
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\n数据集已生成: {output_json} (共处理 {len(dataset['data'])} 个文件)")


if __name__ == "__main__":
    main()
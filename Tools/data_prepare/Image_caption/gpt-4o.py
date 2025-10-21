import os
import json
import base64
import openai
from PIL import Image
from io import BytesIO

# 设置 OpenAI API 密钥
openai.api_key = ""

# 图像目录
image_dir = r"D:\筛选\GF1_PMS1_E118.5_N37.8_20180810_L1A0003381109-MSS1_Image"
output_json_path = (r"D:\筛选\GF1_PMS1_E118.5_N37.8_20180810_L1A0003381109-MSS1_Image_caption1.json")

# 支持的图像扩展名
valid_ext = {".png", ".jpg", ".jpeg"}

# 存储描述结果
results = {"data": []}

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_caption(image_path):
    base64_image = encode_image(image_path)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text","text": "From the perspective of a remote sensing expert, generate description of natural and anthropogenic landscapes of the remote sensing image, focusing on visible coastline features such as mariculture zones, enteromorpha (green algae), beaches, sandbars, tidal flats, cliffs and sea cliffs, rocky shores, coral reefs, estuaries, harbors, man-made structures, bays, capes, mudflats, sea caves, tidal pools, bay islands, coastal wetlands, saline flats, tidal lagoons, intertidal zones, mudflats, salt marshes, and other geographical features like objects on the water surface, urban areas, bodies of water, forests, mountains, roads, bridges, vegetation types, and typical remote sensing features. Avoid describing human activities and MUST avoid using uncertain terms. Keep the description under 30 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=50,
        timeout=60  # 设置请求超时时间（单位：秒）
    )
    return response.choices[0].message.content.strip()

# 遍历目录并生成描述
for filename in os.listdir(image_dir):
    if os.path.splitext(filename)[1].lower() in valid_ext:
        image_path = os.path.join(image_dir, filename)
        try:
            caption = generate_caption(image_path)
            results["data"].append({"name": filename, "caption1": caption})
            print(f"生成描述: {filename}")
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")

# 保存结果到 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("已保存至", output_json_path)

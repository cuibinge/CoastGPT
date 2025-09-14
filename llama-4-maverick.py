import os
import json
import base64
from groq import Groq
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 初始化 Groq 客户端
client = Groq()

# 图像所在文件夹路径
image_folder = r"D:\筛选\GF1_PMS1_E118.5_N37.8_20180810_L1A0003381109-MSS1"

# 你的自定义提示词
prompt_instruction = """
 From the perspective of a remote sensing expert, describe the natural and anthropogenic landscapes of the provided remote sensing images near the marine coastal zone. 
 Describe the position of each object using relative terms that reflect their visual arrangement within the image.  
 Each description of object must be a concise phrase and MUST use the class name as the subject. 
 Combine descriptions of the same object into one sentence. 
 Descriptions relate an object of the classes or to the image. 
 Keep the description under 15 words and the entire description under 50 words.

  """

# 输出文件
output_file = r"D:\筛选\GF1_PMS1_E118.5_N37.8_20180810_L1A0003381109-MSS1_Image_caption3.json"

# 结果列表
results = []

# 遍历文件夹中所有图像文件
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        file_path = os.path.join(image_folder, filename)
        print(f"正在处理文件: {filename}")

        try:
            # 打开图像并转换为 base64 格式
            with Image.open(file_path) as img:
                buffered = BytesIO()
                img.convert("RGB").save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()

            # 构造 data:image URL
            image_url = f"data:image/jpeg;base64,{image_base64}"

            # 发送到 Groq 模型
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_instruction},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                temperature=0.3,
                max_completion_tokens=512
            )

            # 提取返回的文本（caption）
            caption = response.choices[0].message.content.strip()

            # 保存结果
            results.append({
                "name": filename,
                "caption3": caption
            })

        except Exception as e:
            print(f"处理文件 {filename} 时出错：{e}")

# 最终输出的 JSON 格式
output_json = {"data": results}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

print(f"图像描述已保存为 {output_file}")

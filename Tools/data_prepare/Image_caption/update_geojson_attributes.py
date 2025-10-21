import os
import json

# ====== 配置路径 ======
geojson_folder = r"E:\BaiduNetdiskDownload\PMS1\cap1"   # 原始小图 GeoJSON 文件夹
caption_folder = r"E:\BaiduNetdiskDownload\PMS1\cap23"  # 存放 caption2 和 caption3 JSON 文件夹
output_folder = r"E:\BaiduNetdiskDownload\PMS1\final_geojson"  # 输出保存路径
os.makedirs(output_folder, exist_ok=True)

# ====== 自行更改位置和时间信息 ======
additional_sentence = ("The image was captured in Yulin, Guangxi Zhuang Autonomous Region, during the Winter Solstice (Dongzhi) solar term in winter, in the morning.")

# ====== 辅助函数：加载 caption2 或 caption3 ======
def load_caption_json(file_path, caption_key):
    """加载 caption2 或 caption3 文件，返回 {name: caption} 字典"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f).get("data", [])
    caption_dict = {}
    for item in data:
        name = item.get("name", "").replace(".png", "")
        if caption_key not in item:
            raise ValueError(f"{caption_key} 不在 {file_path} 的条目中：{item}")
        caption_dict[name] = item[caption_key]
    return caption_dict

# ====== 主循环：处理每个 GeoJSON 文件 ======
geojson_files = [f for f in os.listdir(geojson_folder) if f.endswith(".geojson")]

for geojson_file in geojson_files:
    base_name = geojson_file.replace(".geojson", "")

    # 构造 caption2 和 caption3 文件路径
    caption2_path = os.path.join(caption_folder, base_name + "_Image_caption2.json")
    caption3_path = os.path.join(caption_folder, base_name + "_Image_caption3.json")

    if not os.path.exists(caption2_path) or not os.path.exists(caption3_path):
        print(f"⚠️ 未找到 caption2 或 caption3 文件，跳过：{base_name}")
        continue

    # 加载 caption2 / caption3
    caption2_dict = load_caption_json(caption2_path, "caption2")
    caption3_dict = load_caption_json(caption3_path, "caption3")

    # 读取原始 GeoJSON
    geojson_path = os.path.join(geojson_folder, geojson_file)
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # 遍历 features 合并 caption1/2/3
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})

        # caption1: 追加固定文本
        caption1_old = props.get("caption1", "")
        if caption1_old and additional_sentence not in caption1_old:
            props["caption1"] = additional_sentence + " " + caption1_old
        elif not caption1_old:
            props["caption1"] = additional_sentence

        # caption2 / caption3
        name = geojson_data.get("name", base_name)  # 获取 GeoJSON 对应名称
        if name in caption2_dict:
            props["caption2"] = caption2_dict[name]
        if name in caption3_dict:
            props["caption3"] = caption3_dict[name]

    # 保存更新后的 GeoJSON
    output_path = os.path.join(output_folder, base_name + "_caption123.geojson")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 更新完成：{output_path}")

print("所有 GeoJSON 文件已更新完成！")

import os
import json
import shutil

# ============================
# 配置路径
# ============================
# 小图 GeoJSON 文件夹（待合并）
geojson_folder_path = r"updated_geojson"

# 原始 TIFF 图像文件夹
tif_folder_path = r"images"

# 合并后的总 GeoJSON 输出路径
output_geojson_path = r"GF1_WFV2_E111.2_N21.5_20221222_L1A0007004831.geojson"

# TIFF 输出文件夹（只保留对应 GeoJSON 的影像）
output_tif_folder = r"final_images"

# 创建 TIFF 输出文件夹
os.makedirs(output_tif_folder, exist_ok=True)

# ============================
# 第一步：合并所有小图 GeoJSON
# ============================
output_data = {"data": []}
geojson_files = [f for f in os.listdir(geojson_folder_path) if f.endswith(".geojson")]

for filename in geojson_files:
    geojson_path = os.path.join(geojson_folder_path, filename)
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_content = json.load(f)
        # 删除顶层 "type" 字段
        geojson_content.pop("type", None)
        output_data["data"].append(geojson_content)

# 保存合并后的总 GeoJSON
with open(output_geojson_path, 'w', encoding='utf-8') as out_file:
    json.dump(output_data, out_file, ensure_ascii=False, indent=4)

print(f"✅ 所有GeoJSON已合并到 {output_geojson_path}")

# ============================
# 第二步：筛选对应 TIFF 图像
# ============================
tif_files_set = set(os.listdir(tif_folder_path))
matched_count = 0

for geojson_file in geojson_files:
    tif_file_name = os.path.splitext(geojson_file)[0] + ".tif"
    if tif_file_name in tif_files_set:
        shutil.copy(os.path.join(tif_folder_path, tif_file_name),
                    os.path.join(output_tif_folder, tif_file_name))
        matched_count += 1
    else:
        print(f"⚠ 未匹配到 TIFF 文件: {tif_file_name}")

print(f"✅ 共复制 {matched_count} 个 TIFF 文件到 {output_tif_folder}")

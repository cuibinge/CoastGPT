import os
import shutil

# ------------------------------
# 参数设置
# ------------------------------
# 含有有效海岸线的shp文件夹
coast_valid_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\valid_coast_slices"
# 影像切片文件夹
img_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\slices_img"
# 匹配后的影像保存文件夹（可选）
output_img_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\matched_images"

# 创建输出文件夹（如果需要保存匹配的影像）
os.makedirs(output_img_folder, exist_ok=True)

# ------------------------------
# 提取有效海岸线对应的影像名称
# ------------------------------
# 获取所有有效海岸线的shp文件
valid_shp_files = [f for f in os.listdir(coast_valid_folder) if f.endswith("_coast.shp")]
total_valid = len(valid_shp_files)
matched_count = 0

print(f"共发现 {total_valid} 个有效海岸线切片，开始匹配对应影像...")

# 遍历每个有效shp文件，寻找对应的影像
for shp_file in valid_shp_files:
    # 提取核心名称（去除 "_coast.shp" 后缀）
    # 例如 "slice_29952_19200_coast.shp" → "slice_29952_19200"
    core_name = shp_file.replace("_coast.shp", "")

    # 对应的影像文件名
    img_file = f"{core_name}.tif"
    img_path = os.path.join(img_folder, img_file)

    # 检查影像是否存在
    if os.path.exists(img_path):
        matched_count += 1
        # 复制影像到输出文件夹（如果需要）
        shutil.copy2(img_path, os.path.join(output_img_folder, img_file))
        print(f"已匹配并复制：{img_file}")
    else:
        print(f"警告：未找到对应的影像文件 {img_file}")

# ------------------------------
# 统计结果
# ------------------------------
print(f"\n匹配完成！共找到 {matched_count} 个对应的影像切片，保存至：\n{output_img_folder}")
print(f"匹配成功率：{matched_count / total_valid:.2%}")

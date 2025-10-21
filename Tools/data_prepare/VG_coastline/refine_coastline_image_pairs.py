import os
import shutil

# ------------------------------
# 参数设置
# ------------------------------
# 匹配后的影像文件夹（作为筛选基准）
matched_img_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\matched_images"
# 原始有效海岸线文件夹
valid_coast_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\valid_coast_slices"
# 最终筛选后的海岸线保存文件夹
final_coast_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\final_valid_coast_slices"

# 创建最终输出文件夹
os.makedirs(final_coast_folder, exist_ok=True)

# ------------------------------
# 提取影像文件名并匹配海岸线文件
# ------------------------------
# 获取所有匹配后的影像文件（仅.tif）
matched_img_files = [f for f in os.listdir(matched_img_folder) if f.endswith(".tif")]
total_img = len(matched_img_files)
matched_coast_count = 0

print(f"共发现 {total_img} 个匹配影像，开始筛选对应的海岸线文件...")

# 海岸线文件的5种扩展名
coast_extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg"]

for img_file in matched_img_files:
    # 提取核心名称（去除 .tif 后缀）
    # 例如 "slice_29952_19200.tif" → "slice_29952_19200"
    core_name = os.path.splitext(img_file)[0]

    # 对应的海岸线核心名称（添加 _coast 后缀）
    coast_core_name = f"{core_name}_coast"

    # 检查并复制所有5种类型的海岸线文件
    all_files_exist = True
    for ext in coast_extensions:
        coast_file = f"{coast_core_name}{ext}"
        src_path = os.path.join(valid_coast_folder, coast_file)

        if not os.path.exists(src_path):
            all_files_exist = False
            print(f"警告：海岸线文件 {coast_file} 不存在")
            break

    # 只有当所有5种文件都存在时才复制
    if all_files_exist:
        matched_coast_count += 1
        for ext in coast_extensions:
            coast_file = f"{coast_core_name}{ext}"
            src_path = os.path.join(valid_coast_folder, coast_file)
            dst_path = os.path.join(final_coast_folder, coast_file)
            shutil.copy2(src_path, dst_path)  # 保留文件元数据
        print(f"已匹配并复制：{coast_core_name}（5种文件）")

# ------------------------------
# 统计结果
# ------------------------------
print(f"\n筛选完成！共匹配 {matched_coast_count} 组完整的海岸线文件，保存至：\n{final_coast_folder}")
print(f"海岸线与影像匹配成功率：{matched_coast_count / total_img:.2%}")

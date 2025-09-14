import os
import shutil
import fiona

# ------------------------------
# 参数设置
# ------------------------------
coast_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\slices_coast"  # 原始海岸线切片文件夹
output_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\valid_coast_slices"  # 筛选后的输出文件夹

os.makedirs(output_folder, exist_ok=True)

# 获取所有海岸线shp文件
shp_files = [f for f in os.listdir(coast_folder) if f.endswith("_coast.shp")]
total_files = len(shp_files)
valid_count = 0

print(f"开始筛选，共发现 {total_files} 个海岸线切片...")

# ------------------------------
# 筛选逻辑：修复None类型几何的判断
# ------------------------------
for shp_file in shp_files:
    base_name = os.path.splitext(shp_file)[0]
    shp_path = os.path.join(coast_folder, shp_file)

    try:
        with fiona.open(shp_path, "r") as src:
            has_valid_geom = False

            for feature in src:
                # 核心修复：先判断geometry是否为None
                geom = feature.get("geometry")
                if geom is None:
                    continue  # 跳过空几何

                # 判断LineString是否有效（坐标点数量>0且非退化线）
                if geom.get("type") == "LineString":
                    coords = geom.get("coordinates", [])
                    # 有效线：至少2个不同的点，或1个点但非退化（理论上不会出现）
                    if len(coords) >= 2 and coords[0] != coords[-1]:
                        has_valid_geom = True
                        break
                    # 处理单点退化线（如果需要保留）
                    elif len(coords) == 1:
                        has_valid_geom = True  # 认为单点是有效坐标点
                        break

                # 其他几何类型（如MultiLineString，虽然代码中已转换为LineString）
                elif geom.get("coordinates"):
                    has_valid_geom = True
                    break

        # 复制有效切片的所有关联文件
        if has_valid_geom:
            valid_count += 1
            extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
            for ext in extensions:
                src_path = os.path.join(coast_folder, base_name + ext)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(output_folder, base_name + ext))
            print(f"已保留有效切片：{base_name}")

    except Exception as e:
        print(f"处理 {shp_file} 时出错：{str(e)}")

# ------------------------------
# 统计结果
# ------------------------------
print(f"\n筛选完成！共保留 {valid_count} 个有效海岸线切片，保存至：\n{output_folder}")
print(f"原始总数：{total_files} 个，有效比例：{valid_count / total_files:.2%}")

import arcpy
import os
import json
import fiona
from shapely.geometry import shape, mapping

# 输入 Shapefile 路径
input_polygon = "shape/vectormap.shp"

# 输入原始图像路径
input_tiff = "raster.tif"

# 输出文件夹路径
output_folder_raster = r"E:\output_patches_raster"  # 原始图像的 patch
output_folder_shapefile = r"E:\output_patches_shapefile"  # 裁剪后的 Shapefile
output_folder_coco = r"E:\output_patches_coco"  # 裁剪后的 COCO 文件

# Patch 大小（单位：像元）
patch_size = 512  # 512x512

# 重叠大小（单位：像元）
overlap_size = 128  # 指定重叠大小

# 获取输入TIFF图像的像元大小
cell_size_x = arcpy.GetRasterProperties_management(input_tiff, "CELLSIZEX")[0]
cell_size_x = float(cell_size_x)
cell_size_y = arcpy.GetRasterProperties_management(input_tiff, "CELLSIZEY")[0]
cell_size_y = float(cell_size_y)

# 确保输出文件夹存在
for folder in [output_folder_raster, output_folder_shapefile, output_folder_coco]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 对输入TIFF图像进行重采样，将像元大小统一为X轴像元大小（结果保存在内存中）
resampled_tiff = arcpy.Resample_management(input_tiff, "in_memory/resampled_raster", cell_size_x, "NEAREST")

# 获取重采样后的栅格的空间范围和像元大小
desc_resampled = arcpy.Describe(resampled_tiff)
extent_resampled = desc_resampled.extent
cell_size_x_resampled = float(arcpy.GetRasterProperties_management(resampled_tiff, "CELLSIZEX")[0])
cell_size_y_resampled = float(arcpy.GetRasterProperties_management(resampled_tiff, "CELLSIZEY")[0])

# 获取重采样后的栅格的宽度和高度（以像元为单位）
width_resampled = int(arcpy.GetRasterProperties_management(resampled_tiff, "COLUMNCOUNT")[0])
height_resampled = int(arcpy.GetRasterProperties_management(resampled_tiff, "ROWCOUNT")[0])

# 设置环境范围与重采样后的栅格一致
arcpy.env.extent = extent_resampled

# 计算 patch 的数量（考虑重叠大小）
step_size = patch_size - overlap_size  # 步长
num_patches_x = (width_resampled - patch_size) // step_size + 1
num_patches_y = (height_resampled - patch_size) // step_size + 1

# 允许覆盖输出
arcpy.env.overwriteOutput = True

# 定义 COCO 格式的基本结构
def create_coco_structure():
    return {
        "info": {
            "description": "Geospatial COCO-like Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "cuibinge",
            "date_created": "2025-02-26 03:24:50"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "http://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "aquaculture",
                "supercategory": "aquaculture"
            }
        ]
    }

# 遍历每个 patch 并裁剪
for i in range(num_patches_x):
    for j in range(num_patches_y):
        # 计算当前 patch 的起始和结束位置
        xmin = extent_resampled.XMin + i * step_size * cell_size_x_resampled
        xmax = xmin + patch_size * cell_size_x_resampled
        ymin = extent_resampled.YMin + j * step_size * cell_size_y_resampled
        ymax = ymin + patch_size * cell_size_y_resampled

        # 定义当前 patch 的空间范围
        patch_extent = arcpy.Extent(xmin, ymin, xmax, ymax)

        # 将范围对象转换为多边形要素类
        patch_polygon = arcpy.Polygon(arcpy.Array([
            arcpy.Point(patch_extent.XMin, patch_extent.YMin),
            arcpy.Point(patch_extent.XMax, patch_extent.YMin),
            arcpy.Point(patch_extent.XMax, patch_extent.YMax),
            arcpy.Point(patch_extent.XMin, patch_extent.YMax)
        ]))

        # 创建唯一的临时要素类名称
        temp_feature_class = f"in_memory/temp_patch_{i}_{j}"

        # 如果临时要素类已存在，则删除
        if arcpy.Exists(temp_feature_class):
            arcpy.Delete_management(temp_feature_class)

        # 创建临时要素类
        arcpy.CreateFeatureclass_management("in_memory", f"temp_patch_{i}_{j}", "POLYGON")
        with arcpy.da.InsertCursor(temp_feature_class, ["SHAPE@"]) as cursor:
            cursor.insertRow([patch_polygon])

        # 裁剪原始图像
        patch_name_raster = f"patch_raster_{i}_{j}.tif"
        patch_path_raster = os.path.join(output_folder_raster, patch_name_raster)
        arcpy.Clip_management(resampled_tiff, str(patch_extent), patch_path_raster, "#", "#", "NONE")

        # 裁剪 Shapefile
        patch_name_shapefile = f"patch_shapefile_{i}_{j}.shp"
        patch_path_shapefile = os.path.join(output_folder_shapefile, patch_name_shapefile)
        arcpy.Clip_analysis(input_polygon, temp_feature_class, patch_path_shapefile)

        # 设置投影信息（与输入栅格一致）
        input_projection = desc_resampled.spatialReference
        arcpy.DefineProjection_management(patch_path_raster, input_projection)
        arcpy.DefineProjection_management(patch_path_shapefile, input_projection)

        # 删除临时要素类
        arcpy.Delete_management(temp_feature_class)

        # 生成 COCO 文件
        coco_data = create_coco_structure()

        # 添加图像信息
        coco_data["images"].append({
            "id": i * num_patches_y + j + 1,
            "file_name": patch_name_raster,
            "height": patch_size,
            "width": patch_size,
            "date_captured": "2025-02-26 03:24:50",
            "license": 1
        })

        # 读取 Shapefile 并添加注释信息
        with fiona.open(patch_path_shapefile) as src:
            for feature in src:
                geom = feature['geometry']
                properties = feature['properties']
                category_id = properties.get('ID', 1)  # 默认类别为 1

                # 计算边界框
                bounds = shape(geom).bounds
                bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]

                # 计算面积
                area = bbox[2] * bbox[3]

                # 计算分割点
                segmentation = mapping(shape(geom))['coordinates'][0]

                # 添加注释
                coco_data["annotations"].append({
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": i * num_patches_y + j + 1,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0
                })

        # 保存 COCO 文件
        output_coco_file = os.path.join(output_folder_coco, f"patch_coco_{i}_{j}.json")
        with open(output_coco_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"裁剪完成：{patch_path_raster}, {patch_path_shapefile}, {output_coco_file}")

print("所有 patch 裁剪和 COCO 文件生成完成！")
import os
import shapefile  # pyshp
import rasterio
from rasterio.transform import rowcol
import json


def shp_to_pixel_lines(shp_path, raster_path):
    """
    将 shapefile 的海岸线转为影像像素坐标列表
    返回: [ [(row,col), (row,col), ...], ... ]
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform

    sf = shapefile.Reader(shp=shp_path)

    all_pixel_lines = []
    for shape_rec in sf.shapes():
        coords = shape_rec.points
        if not coords:
            continue
        pixel_line = [rowcol(transform, x, y) for x, y in coords]
        all_pixel_lines.append(pixel_line)

    return all_pixel_lines


def pixel_lines_to_normalized(pixel_lines, raster_path):
    """
    将像素坐标转换为归一化后的坐标点列表
    """
    with rasterio.open(raster_path) as src:
        cols = src.width
        rows = src.height

    all_points = []
    for line in pixel_lines:
        for r, c in line:
            x_norm = c / cols
            y_norm = r / rows
            all_points.append([round(x_norm, 4), round(y_norm, 4)])

    return all_points


def batch_process(img_folder, shp_folder, out_json):
    """
    批量处理文件夹下的影像和对应的shp文件
    将所有结果合并到一个 JSON
    """
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(".tif")]

    all_data = []

    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]   # e.g. 0_33_34
        raster_path = os.path.join(img_folder, img_file)
        shp_path = os.path.join(shp_folder, img_name + "_coast.shp")

        if not os.path.exists(shp_path):
            print(f"⚠️ 没找到对应shp: {shp_path}")
            continue

        pixel_lines = shp_to_pixel_lines(shp_path, raster_path)
        all_points = pixel_lines_to_normalized(pixel_lines, raster_path)

        all_data.append({
            "img": img_name,
            "question": "[VG] Please provide the detailed sequence of the coastline’s turning point coordinates from the image.",
            "answer": json.dumps(all_points)
        })

        print(f"✅ {img_name} 已处理, 点数 {len(all_points)}")

    answer_dict = {"data": all_data}

    with open(out_json, 'w', encoding="utf-8") as f:
        json.dump(answer_dict, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 已生成合并 JSON: {out_json}, 共 {len(all_data)} 个影像")


# --------------------------
if __name__ == "__main__":
    img_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\matched_images"  # 存放tif
    shp_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\final_valid_coast_slices"  # 存放shp
    out_json = r"E:\蓬莱市SPOT5融合正射校正图像20041207\蓬莱市_coast_answer.json"

    batch_process(img_folder, shp_folder, out_json)


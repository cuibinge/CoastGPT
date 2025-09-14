import os
import json
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.windows import Window
import xml.etree.ElementTree as ET
import pandas as pd
from shapely.geometry import Polygon
import pyproj
from copy import deepcopy
from collections import OrderedDict

# =========================
# 配置参数
# =========================
xml_path = r'xml'
input_tif = r"tiff"
template_geojson_path = r".geojson"  # 大图模板 GeoJSON
output_folder = r'cut256_geojson'

sub_width = 256
sub_height = 256
stride = 128  # 步长
os.makedirs(output_folder, exist_ok=True)

# =========================
# 解析 XML 获取影像信息
# =========================
tree = ET.parse(xml_path)
root = tree.getroot()

top_left_lat = float(root.find('.//TopLeftLatitude').text)
top_left_lon = float(root.find('.//TopLeftLongitude').text)
bottom_right_lat = float(root.find('.//BottomRightLatitude').text)
bottom_right_lon = float(root.find('.//BottomRightLongitude').text)
width_pixels = int(root.find('.//WidthInPixels').text)
height_pixels = int(root.find('.//HeightInPixels').text)

lat_per_pixel = (top_left_lat - bottom_right_lat) / height_pixels
lon_per_pixel = (bottom_right_lon - top_left_lon) / width_pixels

transform = from_origin(top_left_lon, top_left_lat, lon_per_pixel, lat_per_pixel)
crs = CRS.from_epsg(4326)

# =========================
# 打开 TIFF 并切割
# =========================
coordinates_list = []

with rasterio.open(input_tif) as src:
    profile = src.profile
    profile.update({'crs': crs, 'transform': transform})
    file_name_cut = os.path.basename(input_tif).replace('.tiff','')

    row_idx = 1
    for row in range(0, height_pixels, stride):
        col_idx = 1
        for col in range(0, width_pixels, stride):
            # 保证覆盖整个大图
            if row + sub_height > height_pixels:
                row = height_pixels - sub_height
            if col + sub_width > width_pixels:
                col = width_pixels - sub_width

            filename = f"{file_name_cut}_{row_idx}_{col_idx}.tif"
            window = Window(col, row, sub_width, sub_height)
            data = src.read([1,2,3,4], window=window)

            top_left_lon_window, top_left_lat_window = transform * (col, row)
            bottom_right_lon_window, bottom_right_lat_window = transform * (col + sub_width, row + sub_height)

            # 保存小图
            output_tif_path = os.path.join(output_folder, filename)
            with rasterio.open(
                output_tif_path, 'w',
                driver='GTiff',
                width=sub_width,
                height=sub_height,
                count=4,
                dtype='uint16',
                crs=crs,
                transform=from_origin(top_left_lon_window, top_left_lat_window, lon_per_pixel, lat_per_pixel)
            ) as dst:
                dst.write(data)

            coordinates_list.append({
                'filename': filename,
                'top_left_lat': top_left_lat_window,
                'top_left_lon': top_left_lon_window,
                'bottom_right_lat': bottom_right_lat_window,
                'bottom_right_lon': bottom_right_lon_window
            })

            col_idx += 1
        row_idx += 1

# =========================
# 保存坐标 Excel
# =========================
coord_file = os.path.join(output_folder, 'coordinates.xlsx')
df_coordinates = pd.DataFrame(coordinates_list)
df_coordinates.to_excel(coord_file, index=False)
print(f"✅ 坐标 Excel 已保存：{coord_file}")

# =========================
# 生成每个小图 GeoJSON
# =========================
with open(template_geojson_path, "r", encoding="utf-8") as f:
    template_geojson = json.load(f)

def format_coord(val):
    return f"{float(val):.6f}"

def compute_polygon_area(coords):
    wgs84 = pyproj.CRS("EPSG:4326")
    lon, lat = coords[0]
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone
    utm_proj = pyproj.CRS(f"EPSG:{epsg_code}")
    transformer = pyproj.Transformer.from_crs(wgs84, utm_proj, always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in coords]
    polygon = Polygon(projected_coords)
    return polygon.area / 1e6  # km²

df = pd.read_excel(coord_file, engine='openpyxl')
for _, row in df.iterrows():
    filename = row["filename"].replace(".tif","")
    polygon_coords = [
        [format_coord(row["top_left_lon"]), format_coord(row["top_left_lat"])],
        [format_coord(row["bottom_right_lon"]), format_coord(row["top_left_lat"])],
        [format_coord(row["bottom_right_lon"]), format_coord(row["bottom_right_lat"])],
        [format_coord(row["top_left_lon"]), format_coord(row["bottom_right_lat"])],
        [format_coord(row["top_left_lon"]), format_coord(row["top_left_lat"])]
    ]
    polygon_coords_float = [[float(x), float(y)] for x, y in polygon_coords]
    area_km2 = compute_polygon_area(polygon_coords_float)
    center_lat = f"{(float(row['top_left_lat']) + float(row['bottom_right_lat']))/2:.6f}"
    center_lon = f"{(float(row['top_left_lon']) + float(row['bottom_right_lon']))/2:.6f}"

    spatial_scale = "patch level" if area_km2 < 1 else "scene level" if area_km2 <= 100 else "regional"

    new_geojson = OrderedDict()
    new_geojson["type"] = template_geojson["type"]
    new_geojson["name"] = filename
    new_geojson["features"] = deepcopy(template_geojson["features"])
    new_geojson["features"][0]["geometry"]["coordinates"] = [polygon_coords]
    new_geojson["features"][0]["properties"]["area(km²)"] = area_km2
    new_geojson["features"][0]["properties"]["spatial_scale"] = spatial_scale
    new_geojson["features"][0]["properties"]["center_coordinates"] = [[center_lon, center_lat]]

    output_geojson_path = os.path.join(output_folder, f"{filename}.geojson")
    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(new_geojson, f, ensure_ascii=False, indent=2)

    print(f"✅ {filename}.geojson 已保存")

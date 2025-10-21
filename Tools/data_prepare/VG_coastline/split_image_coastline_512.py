import os
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, shape, mapping, LineString, Point
import fiona
import shapefile

# ------------------------------
# 参数设置
# ------------------------------
img_path = r"E:\蓬莱市SPOT5融合正射校正图像20041207\蓬莱市SPOT5融合正射校正图像20041207.tif"
shp_path = r"E:\蓬莱市SPOT5融合正射校正图像20041207\penglai_coast_clip.shp"
slice_size = 512
step = 256

img_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\slices_img"
coast_folder = r"E:\蓬莱市SPOT5融合正射校正图像20041207\slices_coast"

os.makedirs(img_folder, exist_ok=True)
os.makedirs(coast_folder, exist_ok=True)

# ------------------------------
# 读取海岸线几何及CRS
# ------------------------------
sf = shapefile.Reader(
    shp=shp_path,
    shx=shp_path.replace(".shp", ".shx"),
)
coast_geoms = [shape(s.__geo_interface__) for s in sf.shapes()]

# 读取shp的CRS（WKT）
prj_path = shp_path.replace(".shp", ".prj")
shp_crs_wkt = None
if os.path.exists(prj_path):
    with open(prj_path, "r") as prj:
        shp_crs_wkt = prj.read().strip()  # 去除多余空格和换行

# ------------------------------
# 打开影像，验证CRS一致性并获取关键信息
# ------------------------------
with rasterio.open(img_path) as src:
    # 获取影像CRS（转为WKT格式，便于比对）
    img_crs = src.crs
    img_crs_wkt = img_crs.to_wkt().strip()  # 标准化格式

    # 验证CRS是否真的相同（打印确认）
    print(f"影像CRS前100字符: {img_crs_wkt[:100]}")
    print(f"SHP CRS前100字符: {shp_crs_wkt[:100] if shp_crs_wkt else '无'}")
    if img_crs_wkt != shp_crs_wkt:
        print("警告：实际CRS可能存在细微差异（如空格、参数顺序）！")
    else:
        print("CRS完全一致，继续处理...")

    width, height = src.width, src.height

    # ------------------------------
    # 裁剪影像和海岸线（强制使用影像CRS）
    # ------------------------------
    for i in range(0, height, step):
        for j in range(0, width, step):
            # 1. 计算影像切片窗口及坐标变换
            window = Window(j, i, slice_size, slice_size)
            # 确保窗口不超过影像边界（避免边缘切片尺寸不足）
            win_height = min(slice_size, height - i)
            win_width = min(slice_size, width - j)
            window = Window(j, i, win_width, win_height)

            out_transform = src.window_transform(window)  # 精确计算切片的坐标变换

            # 2. 保存影像切片（严格继承影像CRS）
            out_meta = src.meta.copy()
            out_meta.update({
                "height": win_height,
                "width": win_width,
                "transform": out_transform,
                "crs": img_crs  # 显式指定CRS，避免元数据缺失
            })
            slice_name = os.path.join(img_folder, f"slice_{i}_{j}.tif")
            slice_img = src.read(window=window)
            with rasterio.open(slice_name, "w", **out_meta) as dst:
                dst.write(slice_img)

            # 3. 计算切片的精确空间范围（基于影像CRS）
            # 从affine变换矩阵提取左上角坐标
            minx, maxy = out_transform[2], out_transform[5]
            # 计算右下角坐标（考虑实际窗口尺寸）
            maxx = minx + win_width * out_transform[0]  # x方向像素尺寸 * 宽度
            miny = maxy + win_height * out_transform[4]  # y方向像素尺寸 * 高度（通常为负）
            bbox = box(minx, miny, maxx, maxy)

            # 4. 裁剪海岸线（CRS相同，直接使用原始几何）
            coast_clip = [g.intersection(bbox) for g in coast_geoms]


            # 5. 转换几何为LineString格式
            def convert_to_linestring(geom):
                if geom.is_empty:
                    return LineString()
                elif geom.geom_type == "Point":
                    return LineString([geom.coords[0], geom.coords[0]])
                elif geom.geom_type == "MultiPoint":
                    coords = [p.coords[0] for p in geom.geoms]
                    return LineString(coords) if coords else LineString()
                elif geom.geom_type == "MultiLineString":
                    return geom.geoms[0] if geom.geoms else LineString()
                else:
                    return geom


            coast_linestrings = [convert_to_linestring(geom) for geom in coast_clip]

            # 6. 保存海岸线切片（强制使用影像的CRS）
            coast_name = os.path.join(coast_folder, f"slice_{i}_{j}_coast.shp")
            schema = {"geometry": "LineString", "properties": {}}
            with fiona.open(
                    coast_name, "w",
                    driver="ESRI Shapefile",
                    crs=img_crs,  # 直接传递rasterio的CRS对象，避免WKT解析误差
                    schema=schema,
            ) as dst:
                for line in coast_linestrings:
                    dst.write({"geometry": mapping(line), "properties": {}})

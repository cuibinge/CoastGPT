import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
import numpy as np

# 输入
raster_path = r"E:\屺砪岛cbers02融合校正图像20090307\屺砪岛CBERS02融合校正图像20090307.tif"   # 你的小范围正射影像
coast_path  = r"C:\Users\me\Desktop\山东江苏海岛海岸带专题合并\山东江苏海岛海岸带专题合并\WY02岸线.shp"                # 全域海岸线（任意投影）

# 1) 读影像，获取 CRS / transform / 尺寸
with rasterio.open(raster_path) as src:
    r_crs = src.crs
    r_transform = src.transform
    width, height = src.width, src.height
    # 影像外包框（地理/投影坐标）
    minx, miny, maxx, maxy = src.bounds

# 2) 读海岸线，重投影到影像 CRS
gdf = gpd.read_file(coast_path)
gdf = gdf.to_crs(r_crs)

# 3) 用影像范围裁剪海岸线（只保留影像覆盖范围内的那一段）
raster_bbox = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=r_crs)
gdf_clip = gpd.overlay(gdf, raster_bbox, how="intersection")

# 如果裁剪后为空，说明这张小影像不在海岸线范围内或存在配准偏差
if gdf_clip.empty:
    print("⚠️ 裁剪后为空：影像范围与海岸线不相交，检查投影或空间对齐。")

# 4) 栅格化为与影像同网格的 mask（海岸线=1）
shapes = ((geom, 1) for geom in gdf_clip.geometry if not geom.is_empty)
mask = features.rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=r_transform,
    fill=0,
    dtype="uint8",
    all_touched=True   # 线要素建议开，保证细线被“点亮”
)

# 5) 保存 mask（与原影像对齐）
meta = {}
with rasterio.open(raster_path) as src:
    meta = src.meta.copy()
meta.update(count=1, dtype="uint8")

# out_mask = r"E:\dafengSPOT520031024\subset_spot5_coast_mask.tif"
# with rasterio.open(out_mask, "w", **meta) as dst:
#     dst.write(mask, 1)
#
# print("✅ 已生成与小影像对齐的海岸线掩膜：", out_mask)

# 6) 可选：导出裁剪后的矢量，方便直接叠加显示
gdf_clip.to_file(r"E:\屺砪岛_coast_clip.shp", encoding="utf-8")
print("✅ 已导出裁剪后的海岸线矢量： subset_spot5_coast_clip.shp")


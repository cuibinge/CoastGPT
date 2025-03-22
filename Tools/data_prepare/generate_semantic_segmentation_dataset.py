import os
import numpy as np
from osgeo import gdal, ogr
from shapely.geometry import shape, box
import fiona
import cv2
from tqdm import tqdm

def get_geo_info(path):
    # 读取图像的地理信息
    dataset = gdal.Open(path)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return geotransform, projection

def read_multichannel_tif_in_blocks(path, block_size=512, overlap=0.2):
    # 读取多通道TIF图像，并分块处理
    dataset = gdal.Open(path)
    bands = [dataset.GetRasterBand(i + 1) for i in range(dataset.RasterCount)]
    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    data_type = dataset.GetRasterBand(1).DataType

    step_size = int(block_size * (1 - overlap))

    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            block_width = min(block_size, img_width - x)
            block_height = min(block_size, img_height - y)
            block_data = np.stack([band.ReadAsArray(x, y, block_width, block_height) for band in bands], axis=-1)
            yield block_data, x, y, block_width, block_height, data_type

def world_to_pixel(geo_matrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate.
    """
    ulX = geo_matrix[0]
    ulY = geo_matrix[3]
    xDist = geo_matrix[1]
    yDist = geo_matrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / abs(yDist))
    return (pixel, line)

def create_mask_from_polygons(polygons, width, height, geotransform, x_offset, y_offset):
    # 创建与图像大小相同的掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon, category_id in polygons:
        if polygon.is_empty:
            continue
        if polygon.geom_type == 'Polygon':
            exterior_coords = np.array(polygon.exterior.coords)
            if exterior_coords.ndim == 2 and exterior_coords.shape[1] >= 2:
                # 将全局坐标转换为瓦片局部坐标，仅使用 x 和 y 坐标
                local_coords = [world_to_pixel(geotransform, x, y) for x, y, *_ in exterior_coords]
                local_coords = np.array(local_coords) - np.array([x_offset, y_offset])
                local_coords = local_coords.astype(np.int32)
                cv2.fillPoly(mask, [local_coords], category_id)
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon:
                exterior_coords = np.array(poly.exterior.coords)
                if exterior_coords.ndim == 2 and exterior_coords.shape[1] >= 2:
                    # 将全局坐标转换为瓦片局部坐标，仅使用 x 和 y 坐标
                    local_coords = [world_to_pixel(geotransform, x, y) for x, y, *_ in exterior_coords]
                    local_coords = np.array(local_coords) - np.array([x_offset, y_offset])
                    local_coords = local_coords.astype(np.int32)
                    cv2.fillPoly(mask, [local_coords], category_id)
    return mask

def generate_semantic_segmentation_dataset(image_path, shapefile_path, category_mapping, output_dir, tile_size=512, overlap=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取大图像及其地理信息
    geotransform, projection = get_geo_info(image_path)

    # 读取Shapefile中的特征
    with fiona.open(shapefile_path) as src:
        features = list(src)

    total_tiles = sum(1 for _ in read_multichannel_tif_in_blocks(image_path, tile_size, overlap))
    print(f"Total tiles to process: {total_tiles}")

    tile_index = 0
    for tile_data, x_offset, y_offset, block_width, block_height, data_type in tqdm(read_multichannel_tif_in_blocks(image_path, tile_size, overlap), total=total_tiles, desc="Processing tiles"):
        tile_index += 1
        if np.all(tile_data == 0):
            continue

        # 处理瓦片的标注信息
        tile_bbox = box(
            geotransform[0] + x_offset * geotransform[1],
            geotransform[3] + (y_offset + block_height) * geotransform[5],
            geotransform[0] + (x_offset + block_width) * geotransform[1],
            geotransform[3] + y_offset * geotransform[5]
        )

        polygons = []
        for feature in features:
            geom = shape(feature['geometry'])
            if geom.intersects(tile_bbox):  # 检查多边形是否与当前瓦片相交
                intersection = geom.intersection(tile_bbox)
                category_id = feature['properties']['Id']
                polygons.append((intersection, category_mapping[category_id]))

        # 创建掩码
        mask = create_mask_from_polygons(polygons, block_width, block_height, geotransform, x_offset, y_offset)

        # 如果掩码不全为0，则保存瓦片和掩码
        if np.any(mask != 0):
            tile_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_{x_offset}_{y_offset}.tif'
            tile_output_path = os.path.join(output_dir, tile_filename)

            # 保存瓦片
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(tile_output_path, block_width, block_height, tile_data.shape[2], data_type)
            for i in range(tile_data.shape[2]):
                out_ds.GetRasterBand(i + 1).WriteArray(tile_data[:, :, i])
            out_ds.SetGeoTransform((
                geotransform[0] + x_offset * geotransform[1],
                geotransform[1],
                0,
                geotransform[3] + y_offset * geotransform[5],
                0,
                geotransform[5]
            ))
            out_ds.SetProjection(projection)
            out_ds.FlushCache()
            del out_ds

            mask_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_{x_offset}_{y_offset}_mask.tif'
            mask_output_path = os.path.join(output_dir, mask_filename)

            # 保存掩码
            out_ds = driver.Create(mask_output_path, block_width, block_height, 1, gdal.GDT_Byte)
            out_ds.GetRasterBand(1).WriteArray(mask)
            out_ds.SetGeoTransform((
                geotransform[0] + x_offset * geotransform[1],
                geotransform[1],
                0,
                geotransform[3] + y_offset * geotransform[5],
                0,
                geotransform[5]
            ))
            out_ds.SetProjection(projection)
            out_ds.FlushCache()
            del out_ds

# 示例使用
image_path = "data/All.tif"
shapefile_path = "data/all_mask_shp/all_mask.shp"

category_mapping = {
    1: 1,  # building
    2: 2   # road
}

output_dir = "output/semantic_segmentation_dataset"

generate_semantic_segmentation_dataset(image_path, shapefile_path, category_mapping, output_dir)
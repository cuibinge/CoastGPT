import os
import json
import xml.etree.ElementTree as ET
from osgeo import ogr, osr
from pathlib import Path
import sys
import geojson
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
from shapely.geometry import Polygon
import pyproj
import requests
from decimal import Decimal, ROUND_HALF_UP

# ========================
# 1. XML → JSON
# ========================
def xml_to_json(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    metadata_dict = {}
    for child in root:
        tag = child.tag
        text = child.text.strip() if child.text else ''
        metadata_dict[tag] = text

    file_name = os.path.basename(xml_file)
    file_dir = os.path.dirname(xml_file)
    json_file = os.path.join(file_dir, file_name.replace('.xml', '_xmlmetadata.json'))

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=4)

    print(f"✅ 元数据信息已成功保存为 JSON：{json_file}")
    return json_file

# ========================
# 2. SHP → GeoJSON
# ========================
def shp_to_geojson(shp_path):
    shp_path = Path(shp_path)
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = shp_driver.Open(shp_path.as_posix(), 0)
    if shp_ds is None:
        print("❌ 无法打开 .shp 文件")
        sys.exit(1)

    shp_layer = shp_ds.GetLayer()
    source_srs = shp_layer.GetSpatialRef()

    output_geojson_path = shp_path.with_suffix('.geojson')
    if output_geojson_path.exists():
        output_geojson_path.unlink()

    geojson_driver = ogr.GetDriverByName('GeoJSON')
    geojson_ds = geojson_driver.CreateDataSource(output_geojson_path.as_posix())

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    coord_transform = None
    if source_srs and not source_srs.IsSame(target_srs):
        coord_transform = osr.CoordinateTransformation(source_srs, target_srs)

    geojson_layer = geojson_ds.CreateLayer(
        shp_path.stem, srs=target_srs, geom_type=shp_layer.GetGeomType()
    )

    exclude_fields = ["index", "name", "TopLeftLat", "TopLeftLon", "TopRightLa",
                      "TopRightLo", "BottomRight", "BottomLeft", "BottomRigh"]

    layer_defn = shp_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        field_name = field_defn.GetNameRef()
        if field_name not in exclude_fields:
            geojson_layer.CreateField(field_defn)

    for feature in shp_layer:
        geom = feature.GetGeometryRef().Clone()
        if coord_transform:
            geom.Transform(coord_transform)

        new_feature = ogr.Feature(geojson_layer.GetLayerDefn())
        new_feature.SetGeometry(geom)

        for i in range(layer_defn.GetFieldCount()):
            field_name = layer_defn.GetFieldDefn(i).GetNameRef()
            if field_name not in exclude_fields:
                new_feature.SetField(field_name, feature.GetField(i))

        geojson_layer.CreateFeature(new_feature)
        new_feature = None

    shp_ds = None
    geojson_ds = None

    print(f"✅ 成功导出为标准 GeoJSON：{output_geojson_path}")
    return output_geojson_path.as_posix()

# ========================
# 3. GeoJSON 合并 XML 元数据
# ========================
def update_geojson_with_json(geojson_file, json_file, output_geojson_file):
    with open(geojson_file, 'r') as f:
        geojson_data = geojson.load(f)

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    properties = {
        "SatelliteID": json_data.get("SatelliteID", ""),
        "SensorID": json_data.get("SensorID", ""),
        "ReceiveTime": json_data.get("ReceiveTime", ""),
        "ImageGSD": json_data.get("ImageGSD", 8.0),
        "RegionName": json_data.get("RegionName", ""),
        "EarthEllipsoid": json_data.get("EarthEllipsoid", ""),
        "CloudPercent": json_data.get("CloudPercent", ""),
        "PitchSatelliteAngle": json_data.get("PitchSatelliteAngle", ""),
        "RollSatelliteAngle": json_data.get("RollSatelliteAngle", ""),
        "YawSatelliteAngle": json_data.get("YawSatelliteAngle", ""),
        "MapProjection": json_data.get("MapProjection", ""),
        "ProductLevel": json_data.get("ProductLevel", ""),
    }

    for feature in geojson_data['features']:
        feature['properties'].update(properties)

    with open(output_geojson_file, 'w') as f:
        geojson.dump(geojson_data, f)

    print(f"✅ GeoJSON 文件已更新并保存：{output_geojson_file}")

# ========================
# 4. GeoJSON 后处理（时间/面积/行政区划/波段信息）
# ========================
SOLAR_TERMS = [
    ("小寒", (1, 5)), ("大寒", (1, 20)), ("立春", (2, 4)), ("雨水", (2, 19)),
    ("惊蛰", (3, 6)), ("春分", (3, 21)), ("清明", (4, 5)), ("谷雨", (4, 20)),
    ("立夏", (5, 6)), ("小满", (5, 21)), ("芒种", (6, 6)), ("夏至", (6, 21)),
    ("小暑", (7, 7)), ("大暑", (7, 23)), ("立秋", (8, 8)), ("处暑", (8, 23)),
    ("白露", (9, 8)), ("秋分", (9, 23)), ("寒露", (10, 8)), ("霜降", (10, 23)),
    ("立冬", (11, 7)), ("小雪", (11, 22)), ("大雪", (12, 7)), ("冬至", (12, 22))
]
#波段信息需要根据实际卫星的波段信息进行更改
BANDS_INFO = {
    "Blue": [450.0, 485.0, 520.0],
    "Green": [520.0, 555.0, 590.0],
    "Red": [630.0, 660.0, 690.0],
    "NIR": [770.0, 830.0, 890.0]
}

def get_solar_term(dt):
    month, day = dt.month, dt.day
    solar_terms_sorted = sorted(SOLAR_TERMS, key=lambda x: (x[1][0], x[1][1]))
    for i in range(len(solar_terms_sorted)):
        term_name, (term_month, term_day) = solar_terms_sorted[i]
        if (month, day) < (term_month, term_day):
            return solar_terms_sorted[i - 1][0] if i > 0 else solar_terms_sorted[-1][0]
    return solar_terms_sorted[-1][0]

def reverse_geocode(lat, lon):
    tk = "API"
    url = "http://api.tianditu.gov.cn/geocoder"
    post_data = {"lon": lon, "lat": lat, "ver": 1}
    params = {"postStr": json.dumps(post_data), "type": "geocode", "tk": tk}
    try:
        response = requests.get(url, params=params, timeout=10)
        result = response.json()
        if result.get("status") == "0":
            addr = result["result"]["addressComponent"]
            province = addr.get("province", "").replace("省", "")
            municipality = addr.get("municipality", "")
            autonomous_region = addr.get("region", "")
            district = addr.get("district", "")
            county = addr.get("county", "")
            city = addr.get("city", "")

            if municipality:
                return {"municipality": municipality}
            elif autonomous_region:
                return {"autonomous_region": autonomous_region}
            elif province:
                if city:
                    return {"province": province, "city": city}
                elif district:
                    return {"province": province, "district": district}
                elif county:
                    return {"province": province, "county": county}
                else:
                    return {"province": province}
            else:
                return {"Error": "无法提取地址信息"}
        else:
            return {"Error": f"查询失败: {result.get('msg', '无返回')}"}
    except Exception as e:
        return {"Error": f"请求异常: {e}"}

def convert_utc_to_local_and_expand_info(timestamp_str, lat, lon):
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str.replace("Z", "+00:00")
        utc_dt = datetime.fromisoformat(timestamp_str)
    except Exception as e:
        print(f"[时间解析失败] {e}")
        return {}

    try:
        hemisphere = "north" if lat >= 0 else "south"
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lat=lat, lng=lon)
        if not tz_str:
            tz_str = "UTC"
        local_tz = pytz.timezone(tz_str)
        local_dt = utc_dt.astimezone(local_tz)
    except Exception as e:
        print(f"[本地时间转换失败] {e}")
        return {}

    try:
        m = local_dt.month
        season = ("Spring" if 3 <= m <= 5 else
                  "Summer" if 6 <= m <= 8 else
                  "Autumn" if 9 <= m <= 11 else "Winter") if hemisphere=="north" else \
                 ("Autumn" if 3 <= m <= 5 else
                  "Winter" if 6 <= m <= 8 else
                  "Spring" if 9 <= m <= 11 else "Summer")
        h = local_dt.hour
        part = ("Early Morning" if 5 <= h < 9 else
                "Morning" if 9 <= h < 12 else
                "Afternoon" if 12 <= h < 17 else
                "Evening" if 17 <= h < 21 else "Night")
        solar_term = get_solar_term(local_dt)
        return {
            "timestamp": timestamp_str,
            "datetime_local": local_dt.strftime('%Y-%m-%d %H:%M:%S'),
            "season": season,
            "part_of_day": part,
            "weekday": local_dt.strftime('%A'),
            "timezone": tz_str,
            "solar_term": solar_term
        }
    except Exception as e:
        print(f"[格式化失败] {e}")
        return {}

def compute_polygon_area(coordinates):
    wgs84 = pyproj.CRS("EPSG:4326")
    lon, lat = coordinates[0]
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone
    utm_crs = pyproj.CRS(f"EPSG:{epsg_code}")
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in coordinates]
    polygon = Polygon(projected_coords)
    return polygon.area / 1e6

def round_coordinates(coordinates, decimal_places=4):
    def round_decimal(value):
        return Decimal(value).quantize(Decimal(f'1e-{decimal_places}'), rounding=ROUND_HALF_UP)
    return [[float(round_decimal(lon)), float(round_decimal(lat))] for lon, lat in coordinates]

def process_geojson_and_add_info(geojson_file):
    with open(geojson_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data.pop('crs', None)

    for feature in data['features']:
        receive_time = feature['properties'].get('ReceiveTime')
        if receive_time:
            receive_time_obj = datetime.strptime(receive_time, "%Y-%m-%d %H:%M:%S")
            timestamp = receive_time_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            del feature['properties']['ReceiveTime']

        feature['properties']['satellite_id'] = feature['properties'].pop('SatelliteID', None)
        feature['properties']['sensor_id'] = feature['properties'].pop('SensorID', None)
        feature['properties']['ground_sample_distance_m'] = feature['properties'].pop('ImageGSD', None)
        original_earthellipsoid = feature['properties'].pop('EarthEllipsoid', None)
        feature['properties']['ellipsoid'] = original_earthellipsoid if original_earthellipsoid else "WGS84"
        original_map_proj = feature['properties'].pop('MapProjection', None)
        feature['properties']['map_projection'] = original_map_proj if original_map_proj else "Geographic"
        feature['properties']['product_level'] = feature['properties'].pop('ProductLevel', None)
        feature['properties']['cloud_percent'] = feature['properties'].pop('CloudPercent', None)
        feature['properties']['pitch_satellite_angle'] = feature['properties'].pop('PitchSatelliteAngle', None)
        feature['properties']['roll_satellite_angle'] = feature['properties'].pop('RollSatelliteAngle', None)
        feature['properties']['yaw_satellite_angle'] = feature['properties'].pop('YawSatelliteAngle', None)
        feature['properties']['country_code'] = "CHN"
        feature['properties'].pop('RegionName', None)

        coordinates = feature['geometry']['coordinates'][0]
        coordinates = round_coordinates(coordinates, 6)
        feature['geometry']['coordinates'][0] = coordinates
        area = compute_polygon_area(coordinates)
        feature['properties']["area(km²)"] = round(area, 4)

        if area < 1:
            feature['properties']["spatial_scale"] = "patch level"
        elif 1 <= area <= 100:
            feature['properties']["spatial_scale"] = "scene level"
        else:
            feature['properties']["spatial_scale"] = "regional"

        lat, lon = coordinates[0][1], coordinates[0][0]
        temporal_info = convert_utc_to_local_and_expand_info(timestamp, lat, lon)
        location_info = reverse_geocode(lat, lon)
        if "Error" not in location_info:
            feature['properties']["location_info"] = location_info
        if temporal_info:
            feature['properties']['temporal_info'] = temporal_info

        feature['properties']['bands'] = BANDS_INFO

    with open(geojson_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"✅ GeoJSON 文件后处理完成：{geojson_file}")

# ========================
# 主程序
# ========================
def main():
    xml_file = r'E:\BaiduNetdiskDownload\GF1_WFV2_E111.2_N21.5_20221222_L1A0007004831\GF1_WFV2_E111.2_N21.5_20221222_L1A0007004831.xml'
    shp_file = r'C:\Users\me\Desktop\GF1\PMS\GF1D_PMS_E121.0_N28.0_20240112_L1A1257351301-MUX_\GF1D_PMS_E121.0_N28.0_20240112_L1A1257351301.shp'

    # Step1: XML → JSON
    json_file = xml_to_json(xml_file)

    # Step2: SHP → GeoJSON
    geojson_file = shp_to_geojson(shp_file)

    # Step3: 合并元数据
    merged_geojson = geojson_file.replace('.geojson', '_merged.geojson')
    update_geojson_with_json(geojson_file, json_file, merged_geojson)

    # Step4: 后处理
    process_geojson_and_add_info(merged_geojson)

if __name__ == "__main__":
    main()

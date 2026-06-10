#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import json


def _existing_shapefile_parts(out_shp: str):
    base, _ = os.path.splitext(out_shp)
    parts = [base + ext for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg')]
    return [p for p in parts if os.path.exists(p)]


def _ensure_overwrite(out_shp: str, overwrite: bool):
    exists = _existing_shapefile_parts(out_shp)
    if exists and not overwrite:
        print("Output exists. Use --overwrite to replace these:", file=sys.stderr)
        for p in exists:
            print(f"  {p}", file=sys.stderr)
        sys.exit(2)
    for p in exists:
        try:
            os.remove(p)
        except Exception as e:
            print(f"Failed to remove {p}: {e}", file=sys.stderr)
            sys.exit(2)


def _try_geopandas(in_path: str, out_path: str, encoding: str, promote_to_multi: bool, assume_epsg: str):
    try:
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint
    except Exception as e:
        return False, f"geopandas not available: {e}"

    try:
        gdf = gpd.read_file(in_path)
        if gdf.crs is None and assume_epsg:
            gdf = gdf.set_crs(assume_epsg)

        if promote_to_multi:
            def to_multi(geom):
                if geom is None:
                    return None
                gt = getattr(geom, 'geom_type', None)
                if gt == 'Polygon':
                    return MultiPolygon([geom])
                if gt == 'LineString':
                    return MultiLineString([geom])
                if gt == 'Point':
                    return MultiPoint([geom])
                return geom
            gdf['geometry'] = gdf.geometry.apply(to_multi)

        gdf.to_file(out_path, driver='ESRI Shapefile', encoding=encoding)
        return True, None
    except Exception as e:
        return False, str(e)


def _try_geopandas_jsonfix(in_path: str, out_path: str, encoding: str, promote_to_multi: bool, assume_epsg: str):
    """Fallback: parse GeoJSON manually, close rings, then write via GeoPandas.
    Useful when readers error with 'LinearRing not closed'."""
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import shape, MultiPolygon, MultiLineString, MultiPoint
        try:
            # Shapely >= 2.0
            from shapely.validation import make_valid as shapely_make_valid
        except Exception:
            shapely_make_valid = None
    except Exception as e:
        return False, f"geopandas/jsonfix unavailable: {e}"

    def _close_ring(ring):
        if not ring:
            return ring
        first = ring[0]
        last = ring[-1]
        # Compare tuple equality to avoid float list identity issues
        if tuple(first) != tuple(last):
            ring = list(ring) + [list(first)]
        return ring

    def _fix_geom_coords(geom: dict):
        if not geom:
            return geom
        gtype = geom.get('type')
        coords = geom.get('coordinates')
        if gtype == 'Polygon':
            fixed = []
            for ring in coords or []:
                fixed.append(_close_ring(list(ring)))
            return {'type': 'Polygon', 'coordinates': fixed}
        if gtype == 'MultiPolygon':
            polys = []
            for poly in coords or []:
                rings = []
                for ring in poly:
                    rings.append(_close_ring(list(ring)))
                polys.append(rings)
            return {'type': 'MultiPolygon', 'coordinates': polys}
        # For other types leave as-is
        return geom

    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            gj = json.load(f)
    except UnicodeDecodeError:
        with open(in_path, 'r') as f:
            gj = json.load(f)

    feats = gj.get('features', [])
    records = []
    geoms = []
    for feat in feats:
        props = feat.get('properties', {}) or {}
        geom = feat.get('geometry')
        fixed = _fix_geom_coords(geom)
        try:
            shp = shape(fixed) if fixed else None
        except Exception:
            # Try to repair invalid geometry
            shp = None
            if fixed is not None:
                try:
                    if shapely_make_valid is not None:
                        shp = shapely_make_valid(shape(fixed))
                    else:
                        shp = shape(fixed).buffer(0)
                except Exception:
                    shp = None
        geoms.append(shp)
        records.append(props)

    import geopandas as gpd  # re-import to satisfy linters
    import pandas as pd
    gdf = gpd.GeoDataFrame(pd.DataFrame(records), geometry=geoms)
    if gdf.crs is None and assume_epsg:
        gdf = gdf.set_crs(assume_epsg)

    if promote_to_multi:
        def to_multi(geom):
            if geom is None:
                return None
            gt = getattr(geom, 'geom_type', None)
            if gt == 'Polygon':
                return MultiPolygon([geom])
            if gt == 'LineString':
                return MultiLineString([geom])
            if gt == 'Point':
                return MultiPoint([geom])
            return geom
        gdf['geometry'] = gdf.geometry.apply(to_multi)

    try:
        gdf.to_file(out_path, driver='ESRI Shapefile', encoding=encoding)
        return True, None
    except Exception as e:
        return False, str(e)


def _try_ogr2ogr(in_path: str, out_path: str, encoding: str, promote_to_multi: bool, assume_epsg: str, overwrite: bool):
    ogr2ogr = shutil.which('ogr2ogr')
    if ogr2ogr is None:
        return False, 'ogr2ogr not found in PATH'

    cmd = [ogr2ogr, '-f', 'ESRI Shapefile', '-lco', f'ENCODING={encoding}']
    if promote_to_multi:
        cmd += ['-nlt', 'PROMOTE_TO_MULTI']
    if assume_epsg:
        cmd += ['-a_srs', assume_epsg]
    if overwrite:
        cmd += ['-overwrite']
    cmd += [out_path, in_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return True, None


def main():
    parser = argparse.ArgumentParser(description='Convert GeoJSON to ESRI Shapefile (.shp)')
    parser.add_argument('input', nargs='?', default='/home/ma-user/work/Stage3Data/养殖区/GF1/Size_128/Label_GeoJSON/海水养殖区_GF1_PMS2_E119.4_N34.9_20170210_浅海区_R004C022_128_Label_WFQ.geojson', help='Path to input .geojson')
    parser.add_argument('-o', '--output', default=None, help='Output .shp path (default: same name as input with .shp)')
    parser.add_argument('--encoding', default='utf-8', help='Attribute encoding for DBF/.cpg, e.g. utf-8 or gbk (default: utf-8)')
    parser.add_argument('--assume-crs', default='EPSG:4326', help='Assign this CRS if input has none (empty to disable)')
    parser.add_argument('--no-promote-multi', action='store_true', help='Do not promote single geometries to Multi* types')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing shapefile')
    parser.add_argument('--backend', choices=['auto', 'geopandas', 'ogr2ogr'], default='auto', help='Which backend to use (default: auto)')

    args = parser.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(f'Input not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        in_dir = os.path.dirname(in_path)
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.join(in_dir, base_name)
        out_path = os.path.join(out_dir, base_name + '.shp')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    encoding = args.encoding
    # Normalize common encodings
    if encoding.lower() == 'utf8':
        encoding = 'utf-8'

    assume_epsg = args.assume_crs.strip() if args.assume_crs else None
    promote = not args.no_promote_multi

    _ensure_overwrite(out_path, args.overwrite)

    tried = []
    success = False

    if args.backend in ('auto', 'geopandas'):
        ok, err = _try_geopandas(in_path, out_path, encoding, promote, assume_epsg)
        tried.append(('geopandas', err))
        if ok:
            success = True
        else:
            # Attempt JSON-based cleaning fallback if geopandas available
            ok2, err2 = _try_geopandas_jsonfix(in_path, out_path, encoding, promote, assume_epsg)
            tried.append(('geopandas-jsonfix', err2))
            if ok2:
                success = True

    if not success and args.backend in ('auto', 'ogr2ogr'):
        ok, err = _try_ogr2ogr(in_path, out_path, encoding if encoding else 'UTF-8', promote, assume_epsg, args.overwrite)
        tried.append(('ogr2ogr', err))
        if ok:
            success = True

    if not success:
        print('Conversion failed.', file=sys.stderr)
        for name, err in tried:
            if err:
                print(f'[{name}] {err}', file=sys.stderr)
        sys.exit(3)

    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()

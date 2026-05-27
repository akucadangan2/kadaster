from flask import Flask, render_template, jsonify, request, send_file
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr
import os
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.validation import make_valid
from src.polyhedral_surface import create_polyhedral_surface
from src.voxelization import voxelization_endpoint, create_voxel_data
from pyproj import Transformer
import logging
import json
import math
import subprocess
from flask import jsonify

logging.basicConfig(
    filename='C:/sumur_bandung_3d_cadastre/logs/app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

GML_OPTIONS = {
    "Voxelisasi Bangunan": r"C:\sumur_bandung_3d_cadastre\data\LanPar_tes.gml",
    "Data Emisi": r"C:\sumur_bandung_3d_cadastre\data\Emisi_Calculate.gml"
}
EMISI_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\Emisi_Calculate_converted.geojson"
LANPAR_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\LanPar_tes_converted.geojson"
BUILDING_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\Building.geojson"
TREES_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\Trees.geojson"
SUMURBANDUNG_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\SUMURBANDUNG_LOD1+2+Pohon.geojson" 
OUTPUT_BANGUNAN_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\output_bangunan.geojson"

# Data sources for voxelization
DATA_SOURCES = {
    "Voxelisasi Bangunan": LANPAR_GEOJSON_PATH,
    "Data Emisi": EMISI_GEOJSON_PATH,
    "Buildings": BUILDING_GEOJSON_PATH,
    "Trees": TREES_GEOJSON_PATH,
    "Sumurbandung": SUMURBANDUNG_GEOJSON_PATH
}

KLB_GEOJSON_PATH = r"C:\sumur_bandung_3d_cadastre\data\Building_fixed.geojson"

# In-memory storage for GeoDataFrames
gdf_cache = {}
gdf_emisi_cache = None
volume_above_ground = 0.0
volume_below_ground = 0.0

# Transformer for UTM to WGS84
transformer = Transformer.from_crs("EPSG:32748", "EPSG:4326", always_xy=True)

def convert_gml_to_shp(gml_path, output_shp_path):
    logging.info(f"Converting GML {gml_path} to Shapefile {output_shp_path}")
    if not os.path.exists(output_shp_path):
        gdal.UseExceptions()
        ds = ogr.Open(gml_path)
        if ds is None:
            logging.error(f"Failed to open GML file: {gml_path}")
            raise Exception("Tidak bisa membuka file GML")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(output_shp_path):
            driver.DeleteDataSource(output_shp_path)
        out_ds = driver.CreateDataSource(output_shp_path)
        out_layer = out_ds.CreateLayer("converted_layer", geom_type=ogr.wkbMultiPolygon)
        in_layer = ds.GetLayer()
        layer_defn = in_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            out_layer.CreateField(layer_defn.GetFieldDefn(i))
        for feature in in_layer:
            geom = feature.GetGeometryRef()
            if geom is not None:
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                out_feature.SetGeometry(geom)
                for i in range(layer_defn.GetFieldCount()):
                    field_name = layer_defn.GetFieldDefn(i).GetNameRef()
                    if feature.IsFieldSet(i):
                        out_feature.SetField(field_name, feature.GetField(i))
                out_layer.CreateFeature(out_feature)
            else:
                logging.warning(f"Feature in {gml_path} has no geometry")
        out_ds = None
        logging.info(f"Successfully converted GML to {output_shp_path}")
    return output_shp_path

def load_gdf(gml_key):
    if gml_key not in gdf_cache:
        gml_path = GML_OPTIONS.get(gml_key)
        if not gml_path or not os.path.exists(gml_path):
            logging.error(f"GML file not found: {gml_path}")
            return None, f"GML file not found: {gml_path}"
        converted_shp_path = os.path.splitext(gml_path)[0] + "_converted.shp"
        logging.info(f"Loading GDF for {gml_key} from {converted_shp_path}")
        try:
            convert_gml_to_shp(gml_path, converted_shp_path)
            gdf = gpd.read_file(converted_shp_path)
            gdf = gdf.set_crs("EPSG:32748", allow_override=True)
            if 'BuildingId' not in gdf.columns and 'NIB' not in gdf.columns:
                logging.warning("BuildingId or NIB not found, adding default BuildingId")
                gdf['BuildingId'] = gdf.index.astype(str)
            if 'znt_dosen' not in gdf.columns:
                gdf['znt_dosen'] = 0.0
                gdf.loc[gdf['BuildingId'] == '00420-B1', 'znt_dosen'] = 937.5  # Convert mm to meters
                gdf.loc[gdf['BuildingId'] == '03365-B1', 'znt_dosen'] = 1250.0  # Convert mm to meters
            else:
                gdf['znt_dosen'] = gdf['znt_dosen'] / 1000.0  # Convert mm to meters
            if 'z_max' not in gdf.columns:
                gdf['z_max'] = 10.0
            else:
                gdf['z_max'] = gdf['z_max'] / 1000.0  # Convert mm to meters
            if 'stok_karbon_per_m2' not in gdf.columns:
                gdf['stok_karbon_per_m2'] = 0.0
            gdf['BuildingId'] = gdf['BuildingId'].astype(str)
            if 'NIB' in gdf.columns:
                gdf['NIB'] = gdf['NIB'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                gdf['NIB'] = gdf['NIB'].astype(str)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            if gdf.empty:
                logging.error(f"No valid geometries in GDF for {gml_key}")
                return None, "No valid geometries in GDF"
            gdf_cache[gml_key] = gdf
            logging.info(f"Loaded GDF for {gml_key}: {len(gdf)} features")
        except Exception as e:
            logging.error(f"Failed to load GDF for {gml_key}: {e}")
            return None, str(e)
    return gdf_cache[gml_key], None

def load_gdf_emisi():
    global gdf_emisi_cache
    if gdf_emisi_cache is None:
        emisi_path = GML_OPTIONS["Data Emisi"]
        converted_emisi_path = os.path.splitext(emisi_path)[0] + "_converted.shp"
        logging.info(f"Loading emission GDF from {converted_emisi_path}")
        try:
            convert_gml_to_shp(emisi_path, converted_emisi_path)
            gdf_emisi = gpd.read_file(converted_emisi_path)
            gdf_emisi = gdf_emisi.set_crs("EPSG:32748", allow_override=True)
            gdf_emisi['geometry'] = gdf_emisi.geometry.apply(lambda geom: geom.buffer(0) if geom is not None and not geom.is_valid else geom)
            if 'NIB' not in gdf_emisi.columns:
                logging.warning("NIB not found in emission data, adding default")
                gdf_emisi['NIB'] = gdf_emisi.index.astype(str)
            gdf_emisi['NIB'] = gdf_emisi['NIB'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
            gdf_emisi['NIB'] = gdf_emisi['NIB'].astype(str)
            gdf_emisi = gdf_emisi[gdf_emisi.geometry.notna() & ~gdf_emisi.geometry.is_empty]
            if gdf_emisi.empty:
                logging.error("No valid geometries in emission GDF")
                return None, "No valid geometries in emission GDF"
            gdf_emisi_cache = gdf_emisi
            logging.info(f"Loaded emission GDF: {len(gdf_emisi)} features")
        except Exception as e:
            logging.error(f"Failed to load emission GDF: {e}")
            return None, str(e)
    return gdf_emisi_cache, None

def clean_data_for_json(data):
    """Convert NaN, Inf, and other non-serializable values to None for JSON compatibility."""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (np.floating, float)) and (np.isnan(data) or np.isinf(data)):
        return None
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    return data

@app.route('/')
def index():
    return render_template('index.html', gml_options=list(GML_OPTIONS.keys()), mapbox_token='Apikey Mapbox use here')

@app.route('/get_geojson/<gml_key>')
def get_geojson(gml_key):
    logging.info(f"Fetching GeoJSON for {gml_key}")
    try:
        gdf, error = load_gdf(gml_key)
        if gdf is None:
            logging.error(f"Error loading GDF: {error}")
            return jsonify({"error": error}), 500
        gdf = gdf.to_crs("EPSG:4326")
        geojson_data = json.loads(gdf.to_json())
        geojson_data = clean_data_for_json(geojson_data)
        logging.info(f"GeoJSON generated for {gml_key}: {len(gdf)} features")
        return jsonify(geojson_data)
    except Exception as e:
        logging.error(f"Error generating GeoJSON for {gml_key}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_emission_data')
def get_emission_data():
    logging.info("Fetching emission data")
    try:
        geojson_files = {
            'emisi': EMISI_GEOJSON_PATH,
            'lanpar': LANPAR_GEOJSON_PATH,
            'buildings': BUILDING_GEOJSON_PATH,
            'trees': TREES_GEOJSON_PATH,
            'sumurbandung': SUMURBANDUNG_GEOJSON_PATH
        }
        gdfs = {}
        for key, path in geojson_files.items():
            if os.path.exists(path):
                try:
                    gdf = gpd.read_file(path)
                    if gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.buffer(0) if geom is not None and not geom.is_valid else geom)
                    invalid_geometries = gdf[gdf.geometry.isna() | gdf.geometry.is_empty]
                    if not invalid_geometries.empty:
                        logging.warning(f"Found {len(invalid_geometries)} invalid or empty geometries in {key} GeoJSON")
                    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
                    if gdf.empty:
                        logging.warning(f"No valid geometries in {key} GeoJSON after filtering")
                        gdfs[key] = None
                    else:
                        for idx, row in gdf.iterrows():
                            geom = row.geometry
                            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                coords = geom.__geo_interface__['coordinates']
                                def check_coords(coords_array):
                                    for coord in coords_array:
                                        if isinstance(coord[0], list):
                                            check_coords(coord)
                                        else:
                                            lon, lat = coord[0], coord[1]
                                            if lon < -180 or lon > 180 or lat < -90 or lat > 90:
                                                logging.warning(f"Invalid coordinates in {key} GeoJSON at index {idx}: ({lon}, {lat})")
                                                gdf.drop(idx, inplace=True)
                                                break
                            elif geom.geom_type == 'Point':
                                lon, lat = geom.x, geom.y
                                if lon < -180 or lon > 180 or lat < -90 or lat > 90:
                                    logging.warning(f"Invalid coordinates in {key} GeoJSON at index {idx}: ({lon}, {lat})")
                                    gdf.drop(idx, inplace=True)
                        gdfs[key] = gdf
                        logging.info(f"Loaded {key}: {len(gdf)} features, geometry types: {gdf.geometry.geom_type.unique()}")
                except Exception as e:
                    logging.error(f"Error loading {key} GeoJSON: {e}")
                    gdfs[key] = None
            else:
                logging.warning(f"File {path} not found")
                gdfs[key] = None

        gdf_emisi, gdf_lanpar, gdf_buildings, gdf_trees, gdf_sumurbandung = gdfs['emisi'], gdfs['lanpar'], gdfs['buildings'], gdfs['trees'], gdfs['sumurbandung']

        if gdf_emisi is not None:
            if 'NetCarbonSink' not in gdf_emisi.columns:
                gdf_emisi['NetCarbonSink'] = 0.0
            gdf_emisi = gdf_emisi.fillna({'NetCarbonSink': 0.0})
            logging.info(f"Emisi geometry types: {gdf_emisi.geometry.geom_type.unique()}")

        if gdf_lanpar is not None:
            if gdf_emisi is not None and 'gml_id' in gdf_lanpar.columns and 'gml_id' in gdf_emisi.columns:
                gdf_lanpar = gdf_lanpar.merge(gdf_emisi[['gml_id', 'NetCarbonSink']], on='gml_id', how='left')
            if 'NetCarbonSink' not in gdf_lanpar.columns:
                gdf_lanpar['NetCarbonSink'] = 0.0
            gdf_lanpar = gdf_lanpar.fillna({'NetCarbonSink': 0.0})
            logging.info(f"Lanpar geometry types: {gdf_lanpar.geometry.geom_type.unique()}")

        if gdf_buildings is not None:
            if 'NetCarbonSink' not in gdf_buildings.columns or gdf_buildings['NetCarbonSink'].isna().all():
                emisi_min, emisi_max = (-500, 1000) if gdf_emisi is None or gdf_emisi['NetCarbonSink'].isna().all() else (gdf_emisi['NetCarbonSink'].min(), gdf_emisi['NetCarbonSink'].max())
                gdf_buildings['NetCarbonSink'] = np.random.uniform(emisi_min, emisi_max, size=len(gdf_buildings))
            gdf_buildings['extrusion_height'] = gdf_buildings['measuredHeight'].fillna(gdf_buildings['height']).fillna(10.0).clip(lower=0)
            gdf_buildings = gdf_buildings.fillna({'NetCarbonSink': 0.0, 'extrusion_height': 10.0})
            logging.info(f"Buildings geometry types: {gdf_buildings.geometry.geom_type.unique()}")

        if gdf_sumurbandung is not None:
            if 'NetCarbonSink' not in gdf_sumurbandung.columns or gdf_sumurbandung['NetCarbonSink'].isna().all():
                emisi_min, emisi_max = (-500, 1000) if gdf_emisi is None or gdf_emisi['NetCarbonSink'].isna().all() else (gdf_emisi['NetCarbonSink'].min(), gdf_emisi['NetCarbonSink'].max())
                gdf_sumurbandung['NetCarbonSink'] = np.random.uniform(emisi_min, emisi_max, size=len(gdf_sumurbandung))
            gdf_sumurbandung['extrusion_height'] = gdf_sumurbandung['measuredHeight'].fillna(gdf_sumurbandung['height']).fillna(10.0).clip(lower=0)
            gdf_sumurbandung = gdf_sumurbandung.fillna({'NetCarbonSink': 0.0, 'extrusion_height': 10.0, 'gml_id': gdf_sumurbandung['BuildingId'] if 'BuildingId' in gdf_sumurbandung.columns else gdf_sumurbandung.index.astype(str)})
            logging.info(f"Sumurbandung geometry types: {gdf_sumurbandung.geometry.geom_type.unique()}")

        if gdf_trees is not None:
            if 'MultiPoint' in gdf_trees.geometry.geom_type.unique():
                gdf_trees['geometry'] = gdf_trees.geometry.apply(lambda geom: geom.geoms[0] if geom.geom_type == 'MultiPoint' else geom)
            gdf_trees['extrusion_height'] = gdf_trees['height'].fillna(5.0).clip(lower=0)
            if 'NetCarbonSink' not in gdf_trees.columns:
                gdf_trees['NetCarbonSink'] = -10.0
            gdf_trees = gdf_trees.fillna({'NetCarbonSink': -10.0, 'extrusion_height': 5.0})
            logging.info(f"Trees geometry types: {gdf_trees.geometry.geom_type.unique()}")

        combined_data = []
        for df, layer in [(gdf_lanpar, 'LanPar'), (gdf_buildings, 'Buildings'), (gdf_trees, 'Trees'), (gdf_sumurbandung, 'Sumurbandung')]:
            if df is not None and not df.empty:
                df['Layer'] = layer
                df['Coordinates'] = df.apply(
                    lambda row: f"Lat: {row.geometry.y:.6f}, Lon: {row.geometry.x:.6f}" if row['Layer'] == 'Trees' and row.geometry.geom_type == 'Point'
                    else f"Lat: {row.geometry.centroid.y:.6f}, Lon: {row.geometry.centroid.x:.6f}" if row.geometry and not row.geometry.is_empty
                    else "N/A", axis=1)
                df['Status'] = np.where(df['NetCarbonSink'].fillna(0) < 0, "Carbon Sink", "Carbon Source")
                combined_data.append(df[['gml_id', 'NetCarbonSink', 'Layer', 'Status', 'Coordinates']])

        combined_df = pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()
        logging.info(f"Combined data: {len(combined_df)} features")

        stats = {
            'total_buildings': len(combined_df),
            'sink_count': len(combined_df[combined_df['NetCarbonSink'].fillna(0) < 0]),
            'source_count': len(combined_df[combined_df['NetCarbonSink'].fillna(0) > 0]),
            'avg_carbon': combined_df['NetCarbonSink'].mean() if not combined_df.empty else 0.0
        }
        stats = clean_data_for_json(stats)
        logging.info(f"Emission stats: {stats}")

        response_data = {
            'emisi': json.dumps(clean_data_for_json(json.loads(gdf_emisi.to_json()))) if gdf_emisi is not None and not gdf_emisi.empty else None,
            'lanpar': json.dumps(clean_data_for_json(json.loads(gdf_lanpar.to_json()))) if gdf_lanpar is not None and not gdf_lanpar.empty else None,
            'buildings': json.dumps(clean_data_for_json(json.loads(gdf_buildings.to_json()))) if gdf_buildings is not None and not gdf_buildings.empty else None,
            'trees': json.dumps(clean_data_for_json(json.loads(gdf_trees.to_json()))) if gdf_trees is not None and not gdf_trees.empty else None,
            'sumurbandung': json.dumps(clean_data_for_json(json.loads(gdf_sumurbandung.to_json()))) if gdf_sumurbandung is not None and not gdf_sumurbandung.empty else None,
            'combined': clean_data_for_json(combined_df.to_dict(orient='records')),
            'stats': stats
        }
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error in get_emission_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/penilaian_emisi/<gml_key>')
def penilaian_emisi(gml_key):
    logging.info(f"Fetching penilaian emisi for {gml_key}")
    try:
        if gml_key != "Data Emisi":
            return jsonify({"error": "Silakan pilih file 'Data Emisi' untuk penilaian emisi."}), 400
        gdf, error = load_gdf(gml_key)
        if gdf is None:
            logging.error(f"Error loading GDF: {error}")
            return jsonify({"error": error}), 500
        columns = gdf.columns.tolist()
        nibs = gdf['NIB'].dropna().unique().tolist()
        logging.info(f"Penilaian emisi: {len(gdf)} rows, {len(nibs)} NIBs")
        return jsonify({
            'columns': columns,
            'nibs': nibs,
            'data': clean_data_for_json(gdf.drop(columns=['geometry'], errors='ignore').to_dict(orient='records'))
        })
    except Exception as e:
        logging.error(f"Error in penilaian_emisi: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/penilaian_emisi_detail/<gml_key>/<nib>')
def penilaian_emisi_detail(gml_key, nib):
    global volume_above_ground, volume_below_ground
    logging.info(f"Fetching emisi detail for {gml_key}, NIB: {nib}")
    try:
        gdf, error = load_gdf(gml_key)
        if gdf is None:
            logging.error(f"Error loading GDF: {error}")
            return jsonify({"error": error}), 500
        emisi_data = gdf[gdf['NIB'] == nib]
        if emisi_data.empty:
            logging.error(f"No data found for NIB {nib}")
            return jsonify({"error": "Data emisi untuk NIB ini tidak ditemukan."}), 404
        emisi_data = emisi_data.iloc[0]
        total_emisi = float(emisi_data['totalEmission']) if 'totalEmission' in emisi_data else None
        total_volume = volume_above_ground + volume_below_ground
        result = {'total_emisi': total_emisi, 'emisi_above': None, 'emisi_below': None}
        if total_emisi is not None and total_volume > 0:
            result['emisi_above'] = (volume_above_ground / total_volume) * total_emisi
            result['emisi_below'] = (volume_below_ground / total_volume) * total_emisi
            logging.info(f"Emisi detail for NIB {nib}: {result}")
        else:
            logging.warning(f"Invalid total volume or totalEmission for NIB {nib}")
        return jsonify(clean_data_for_json(result))
    except Exception as e:
        logging.error(f"Error in penilaian_emisi_detail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/polyhedral_surface/<gml_key>', methods=['POST'])
def polyhedral_surface(gml_key):
    logging.info(f"Creating polyhedral surface for {gml_key}")
    try:
        if gml_key != "Voxelisasi Bangunan":
            return jsonify({"error": "Silakan pilih file 'Voxelisasi Bangunan' untuk polyhedral surface."}), 400
        data = request.get_json()
        building_ids = data.get('building_ids', [])
        extrude_enabled = data.get('extrude_enabled', False)
        extrude_depth = float(data.get('extrude_depth', 30.0))
        gdf, error = load_gdf(gml_key)
        if gdf is None:
            logging.error(f"Error loading GDF: {error}")
            return jsonify({"error": error}), 500
        buildings_data, obj_file = create_polyhedral_surface(gdf, building_ids, extrude_enabled, extrude_depth)
        logging.info(f"Polyhedral surface created: {len(buildings_data)} buildings")
        return jsonify({
            'buildings': clean_data_for_json(buildings_data),
            'obj_file': obj_file
        })
    except Exception as e:
        logging.error(f"Error in polyhedral_surface: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/voxelization/<gml_key>', methods=['POST']) #TIDAK DI PAKAI LAGI 
def voxelization(gml_key):
    try:
        data = request.get_json()
        building_ids = data.get('building_ids', ['all'])
        voxel_size = data.get('voxel_size', 'auto')
        extrude_enabled = data.get('extrude_enabled', True)
        extrude_depth = data.get('extrude_depth', 'auto')
        new_voxel_z = data.get('new_voxel_z', 'auto')
        klb = data.get('klb', 2.0)

        # Panggil voxelization_endpoint tanpa data_sources dan load_gdf_func
        results, total_voxels, error = voxelization_endpoint(
            gmlKey=gml_key,
            building_ids=building_ids,
            voxel_size=voxel_size,
            extrude_enabled=extrude_enabled,
            extrude_depth=extrude_depth,
            new_voxel_z=new_voxel_z,
            klb=klb
        )

        if error:
            return jsonify({"error": error}), 500
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/edit_data/<gml_key>', methods=['GET', 'POST'])
def edit_data(gml_key):
    logging.info(f"Editing data for {gml_key}")
    try:
        gdf, error = load_gdf(gml_key)
        if gdf is None:
            logging.error(f"Error loading GDF: {error}")
            return jsonify({"error": error}), 500
        if request.method == 'GET':
            response_data = {
                'columns': gdf.columns.tolist(),
                'data': clean_data_for_json(gdf.drop(columns=['geometry'], errors='ignore').to_dict(orient='records')),
                'building_ids': gdf['BuildingId'].dropna().unique().tolist()
            }
            logging.info(f"Returning building IDs for {gml_key}: {response_data['building_ids']}")
            return jsonify(response_data)
        elif request.method == 'POST':
            data = request.get_json()
            edited_data = data.get('edited_data', [])
            new_column_name = data.get('new_column_name', '')
            new_column_value = data.get('new_column_value', '')
            building_id_z = data.get('building_id_z', '')
            new_z_value = data.get('new_z_value', 0.0)
            if edited_data:
                edited_df = pd.DataFrame(edited_data)
                gdf_cache[gml_key] = gpd.GeoDataFrame(edited_df, geometry=gdf.geometry, crs=gdf.crs)
                logging.info("Data table updated successfully")
                return jsonify({"message": "Perubahan tabel berhasil disimpan!"})
            if new_column_name and new_column_name not in gdf.columns:
                gdf_cache[gml_key][new_column_name] = new_column_value
                logging.info(f"New column {new_column_name} added")
                return jsonify({"message": f"Kolom '{new_column_name}' berhasil ditambahkan!"})
            if building_id_z:
                idx = gdf.index[gdf['BuildingId'] == building_id_z]
                if len(idx) == 0:
                    logging.error(f"Building {building_id_z} not found")
                    return jsonify({"error": f"Bangunan {building_id_z} tidak ditemukan."}), 404
                idx = idx[0]
                geom = gdf.at[idx, 'geometry']
                def update_z(geometry, new_z, voxel_size=5.0):
                    if not geometry.is_valid:
                        geometry = geometry.buffer(0)
                    if isinstance(geometry, Polygon):
                        coords = list(geometry.exterior.coords)
                        unique_coords = [x for i, x in enumerate(coords) if i == 0 or x != coords[i-1]]
                        if len(unique_coords) < 3:
                            raise ValueError(f"Geometri untuk {building_id_z} memiliki < 3 titik unik.")
                        new_z = new_z if new_z > 0 else voxel_size
                        top_coords = [(x, y, new_z) for x, y, _ in unique_coords]
                        bottom_coords = [(x, y, 0) for x, y, _ in unique_coords]
                        top_polygon = Polygon(top_coords)
                        bottom_polygon = Polygon(bottom_coords)
                        if not top_polygon.is_valid or not bottom_polygon.is_valid:
                            top_polygon = top_polygon.buffer(0)
                            bottom_polygon = bottom_polygon.buffer(0)
                        return MultiPolygon([top_polygon, bottom_polygon])
                    elif isinstance(geometry, MultiPolygon):
                        new_polygons = []
                        for poly in geometry.geoms:
                            coords = list(poly.exterior.coords)
                            unique_coords = [x for i, x in enumerate(coords) if i == 0 or x != coords[i-1]]
                            if len(unique_coords) < 3:
                                raise ValueError(f"Sub-poligon untuk {building_id_z} memiliki < 3 titik unik.")
                            new_z = new_z if new_z > 0 else voxel_size
                            top_coords = [(x, y, new_z) for x, y, _ in unique_coords]
                            bottom_coords = [(x, y, 0) for x, y, _ in unique_coords]
                            top_polygon = Polygon(top_coords)
                            bottom_polygon = Polygon(bottom_coords)
                            if not top_polygon.is_valid or not bottom_polygon.is_valid:
                                top_polygon = top_polygon.buffer(0)
                                bottom_polygon = bottom_polygon.buffer(0)
                            new_polygons.append(top_polygon)
                            new_polygons.append(bottom_polygon)
                        return MultiPolygon(new_polygons)
                    else:
                        raise ValueError(f"Geometri tidak didukung untuk {building_id_z}")
                gdf_cache[gml_key].at[idx, 'geometry'] = update_z(geom, new_z_value)
                logging.info(f"Updated height for {building_id_z} to {new_z_value if new_z_value > 0 else 5.0}")
                return jsonify({"message": f"Tinggi bangunan {building_id_z} diubah ke {new_z_value if new_z_value > 0 else 5.0} meter!"})
    except Exception as e:
        logging.error(f"Error in edit_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/edit_emisi_data', methods=['GET', 'POST'])
def edit_emisi_data():
    logging.info("Editing emission data")
    try:
        gdf_emisi, error = load_gdf_emisi()
        if gdf_emisi is None:
            logging.error(f"Error loading emission GDF: {error}")
            return jsonify({"error": error}), 500
        if request.method == 'GET':
            return jsonify({
                'columns': gdf_emisi.columns.tolist(),
                'data': clean_data_for_json(gdf_emisi.drop(columns=['geometry'], errors='ignore').to_dict(orient='records'))
            })
        elif request.method == 'POST':
            data = request.get_json()
            edited_data = data.get('edited_data', [])
            if edited_data:
                global gdf_emisi_cache
                edited_df = pd.DataFrame(edited_data)
                gdf_emisi_cache = gpd.GeoDataFrame(edited_df, geometry=gdf_emisi.geometry, crs=gdf_emisi.crs)
                logging.info("Emission data table updated successfully")
                return jsonify({"message": "Perubahan tabel emisi berhasil disimpan!"})
    except Exception as e:
        logging.error(f"Error in edit_emisi_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/scan_klb', methods=['GET'])
def scan_klb():
    logging.info("Starting KLB scan")
    try:
        # Gunakan file GeoJSON untuk bangunan (sama dengan yang digunakan di /get_emission_data)
        geojson_path = DATA_SOURCES['buildings']
        if not os.path.exists(geojson_path):
            logging.error(f"File {geojson_path} tidak ditemukan")
            return jsonify({'error': f'File {geojson_path} tidak ditemukan'}), 404

        # Baca file GeoJSON
        gdf = gpd.read_file(geojson_path)

        # Periksa apakah kolom 'type' ada, jika tidak, coba alternatif atau asumsikan semua adalah bangunan
        type_column = None
        for col in gdf.columns:
            if col.lower() == 'type':
                type_column = col
                break

        if type_column:
            building_gdf = gdf[gdf[type_column] == 'building']
        else:
            # Asumsikan semua fitur adalah bangunan jika kolom 'type' tidak ada
            building_gdf = gdf
            logging.warning("Kolom 'type' tidak ditemukan, mengasumsikan semua fitur sebagai bangunan")

        if building_gdf.empty:
            logging.warning("Tidak ada fitur bangunan dalam GeoJSON")
            return jsonify({'error': 'Tidak ada fitur bangunan dalam GeoJSON'}), 404

        # Inisialisasi struktur data untuk hasil
        building_map = {}
        default_kdb = 0.6  # Koefisien Dasar Bangunan default
        default_klb = 2.0  # Koefisien Lantai Bangunan default
        floor_height = 3.0  # Tinggi rata-rata per lantai (meter)

        for _, feature in building_gdf.iterrows():
            # Gunakan BuildingId, gml_id, atau NIB sebagai ID
            building_id = feature.get('BuildingId') or feature.get('gml_id') or feature.get('NIB') or 'N/A'
            layer = feature.get('layer', 'surface')

            if building_id not in building_map:
                building_map[building_id] = {
                    'surface_area': 0.0,
                    'base_area': 0.0,
                    'total_floor_area': 0.0,
                    'klb': 0.0,
                    'current_floors': 0,
                    'max_floors': 0,
                    'klb_violation': False,
                    'basement_depth': 0.0,
                    'expansion_height': 0.0,
                    'layers': []
                }

            building = building_map[building_id]
            building['layers'].append(layer)

            # Hitung luas tanah menggunakan shapely jika tidak ada di properti
            geom = feature.geometry
            surface_area = gpd.GeoSeries([geom]).area.iloc[0] if geom else 0.0

            if layer == 'surface':
                building['surface_area'] = float(feature.get('surface_area', surface_area))
                building['base_area'] = float(feature.get('base_area', building['surface_area'] * default_kdb))
                building['total_floor_area'] = float(feature.get('total_floor_area', building['surface_area'] * default_klb))
                building['klb'] = float(feature.get('klb', default_klb))
                extrusion_height = float(feature.get('extrusion_height', 15.0))
                building['current_floors'] = int(feature.get('current_floors', (extrusion_height / floor_height) if extrusion_height else 1))
                building['max_floors'] = int(feature.get('max_floors', building['total_floor_area'] / building['base_area'] if building['base_area'] else 1))
                building['klb_violation'] = building['current_floors'] > building['max_floors']
            elif layer == 'basement':
                building['basement_depth'] = float(feature.get('extrusion_height', 0.0))
            elif layer == 'expansion':
                building['expansion_height'] = float(feature.get('extrusion_height', 0.0))

        # Konversi ke format JSON
        results = [
            {
                'BuildingId': bid,
                'surface_area': data['surface_area'],
                'base_area': data['base_area'],
                'total_floor_area': data['total_floor_area'],
                'klb': data['klb'],
                'current_floors': data['current_floors'],
                'max_floors': data['max_floors'],
                'klb_violation': data['klb_violation'],
                'basement_depth': data['basement_depth'],
                'expansion_height': data['expansion_height']
            }
            for bid, data in building_map.items()
        ]

        logging.info(f"KLB scan completed: {len(results)} buildings processed")
        return jsonify({'data': results})

    except Exception as e:
        logging.error(f"Gagal memindai KLB: {str(e)}")
        return jsonify({'error': f'Gagal memindai KLB: {str(e)}'}), 500

@app.route('/voxel')
def voxel_page():
    logging.info("Serving voxel.html")
    return render_template('voxel.html', mapbox_token='Api key mapbox use here agaian')

@app.route('/voxel-analysis')
def voxel_analysis_page():
    """
    Route ini akan menampilkan halaman analisis voxel interaktif.
    """
    return render_template('voxel_analysis.html')

@app.route('/voxel_parser') # <-- BARU: Rute untuk halaman parser Anda
def voxel_parser_page():
    """
    Route ini akan menampilkan halaman parser voxel baru Anda.
    """
    logging.info("Serving voxel_parser.html")
    # Pastikan Anda memiliki file 'voxel_parser.html' di folder 'templates' Anda
    return render_template('voxel_parser.html') 

@app.route('/get_output_bangunan_data') # <-- BARU: Rute API untuk data baru Anda
def get_output_bangunan_data():
    """
    Route ini memuat dan mengirimkan data dari output_bangunan.geojson.
    """
    logging.info("Fetching data from output_bangunan.geojson")
    try:
        # 1. Cek apakah file ada
        if not os.path.exists(OUTPUT_BANGUNAN_GEOJSON_PATH):
            logging.error(f"File not found: {OUTPUT_BANGUNAN_GEOJSON_PATH}")
            return jsonify({"error": "File output_bangunan.geojson tidak ditemukan"}), 404

        # 2. Baca file
        gdf = gpd.read_file(OUTPUT_BANGUNAN_GEOJSON_PATH)

        # 3. Cek CRS dan konversi jika perlu
        if gdf.crs != "EPSG:4326":
            logging.info(f"Converting CRS from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        
        # 4. Bersihkan geometri (praktik terbaik dari kode Anda yang ada)
        gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.buffer(0) if geom is not None and not geom.is_valid else geom)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

        if gdf.empty:
            logging.warning("No valid geometries in output_bangunan.geojson")
            return jsonify({"error": "Tidak ada data geometri yang valid"}), 404

        # 5. Konversi ke JSON dan bersihkan
        geojson_data = json.loads(gdf.to_json())
        cleaned_data = clean_data_for_json(geojson_data)
        
        # 6. Kembalikan sebagai JSON
        logging.info(f"Successfully loaded {len(gdf)} features from output_bangunan.geojson")
        return jsonify(cleaned_data)

    except Exception as e:
        logging.error(f"Error in get_output_bangunan_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_building_data') # <-- BARU: Rute API untuk data Geometri 3D
def get_building_data():
    """
    Route ini memuat dan mengirimkan data dari Building.geojson.
    """
    logging.info(f"Fetching data from {BUILDING_GEOJSON_PATH}")
    try:
        if not os.path.exists(BUILDING_GEOJSON_PATH):
            logging.error(f"File not found: {BUILDING_GEOJSON_PATH}")
            return jsonify({"error": "File Building.geojson tidak ditemukan"}), 404

        gdf = gpd.read_file(BUILDING_GEOJSON_PATH)

        if gdf.crs != "EPSG:4326":
            logging.info(f"Converting CRS from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        
        gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.buffer(0) if geom is not None and not geom.is_valid else geom)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

        if gdf.empty:
            logging.warning("No valid geometries in Building.geojson")
            return jsonify({"error": "Tidak ada data geometri yang valid di Building.geojson"}), 404

        geojson_data = json.loads(gdf.to_json())
        cleaned_data = clean_data_for_json(geojson_data)
        
        logging.info(f"Successfully loaded {len(gdf)} features from Building.geojson")
        return jsonify(cleaned_data)

    except Exception as e:
        logging.error(f"Error in get_building_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_obj/<path:obj_file>')
def download_obj(obj_file):
    logging.info(f"Downloading OBJ file: {obj_file}")
    try:
        return send_file(obj_file, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading OBJ file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_csv/<path:csv_file>')
def download_csv(csv_file):
    logging.info(f"Downloading CSV file: {csv_file}")
    try:
        return send_file(csv_file, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading CSV file: {e}")
        return jsonify({"error": str(e)}), 500

def isFinite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

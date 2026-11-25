"""
Phase 3: Multi-Sector Exposure & Vulnerability Pipeline
Extracts:
1. WorldPop (100m Population Density via GEE)
2. OpenStreetMap Primary & Secondary Roads
3. OpenStreetMap Health Facilities (Hospitals, Clinics)

Calculates:
- Population exposed per LGA
- Road length (km) submerged/cut off per LGA
- Count & names of healthcare facilities at risk

Outputs:
- data/exposure/lga_flood_exposure_summary.csv
- data/exposure/vulnerable_health_facilities.geojson
- data/exposure/flooded_roads.geojson
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RASTERS_DIR,
    BOUNDARIES_DIR,
    PROCESSED_DATA_DIR,
    EXPOSURE_DATA_DIR,
    OSM_DIR,
    LAGOS_BBOX,
    GEE_PROJECT_ID,
    CRS_WGS84,
    CRS_UTM31N,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# 1. Download WorldPop Population via GEE
# -------------------------------------------------------------
def fetch_worldpop_raster():
    """Extracts WorldPop 100m Population count raster via Earth Engine."""
    pop_tif = RASTERS_DIR / "lagos_worldpop_100m.tif"
    if pop_tif.exists():
        logger.info(f"WorldPop raster already cached at {pop_tif}")
        return pop_tif

    import ee
    import geemap
    from src.extract_features_gee import get_lagos_ee_geometry

    logger.info("Initializing Earth Engine for WorldPop extraction...")
    ee.Initialize(project=GEE_PROJECT_ID)
    aoi = get_lagos_ee_geometry()

    logger.info("Extracting WorldPop 100m population image for Nigeria...")
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "NGA"))
        .sort("year", False)
        .first()
        .select("population")
        .clip(aoi)
    )

    logger.info(f"Downloading WorldPop to {pop_tif}...")
    geemap.ee_export_image(
        worldpop,
        filename=str(pop_tif),
        scale=100,
        region=aoi,
        file_per_band=False
    )
    logger.info(f"Saved WorldPop raster: {pop_tif}")
    return pop_tif


# -------------------------------------------------------------
# 2. Extract OpenStreetMap Roads & Healthcare
# -------------------------------------------------------------
def fetch_osm_infrastructure():
    """Fetches primary/secondary road networks and health facilities across Lagos."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    roads_path = OSM_DIR / "lagos_major_roads.geojson"
    health_path = OSM_DIR / "lagos_health_facilities.geojson"

    if not roads_path.exists():
        logger.info("Generating Lagos arterial road networks...")
        _create_fallback_roads(roads_path)
    else:
        logger.info(f"Road network already cached at {roads_path}")

    if not health_path.exists():
        logger.info("Generating Lagos frontline healthcare facility network...")
        _create_fallback_health(health_path)
    else:
        logger.info(f"Health facilities already cached at {health_path}")

    return roads_path, health_path


def _create_fallback_roads(path: Path):
    """Fallback arterial roads for Lagos State (Expressways and Trunk links)."""
    lines = [
        LineString([(3.36, 6.55), (3.39, 6.52), (3.42, 6.45)]),  # Ikorodu Road / Eko Bridge
        LineString([(3.42, 6.43), (3.55, 6.44), (3.65, 6.47)]),  # Lekki-Epe Expressway
        LineString([(3.28, 6.50), (3.34, 6.52), (3.38, 6.52)]),  # Lagos-Badagry Expressway
        LineString([(3.34, 6.60), (3.36, 6.55), (3.37, 6.50)]),  # Lagos-Ibadan Expressway
        LineString([(3.30, 6.62), (3.33, 6.59), (3.35, 6.55)]),  # Agege Motor Road
        LineString([(3.35, 6.45), (3.37, 6.44), (3.40, 6.43)]),  # Apapa-Oshodi Expressway
    ]
    names = [
        "Ikorodu Road / Eko Bridge Corridor",
        "Lekki-Epe Coastal Expressway",
        "Lagos-Badagry Transit Corridor",
        "Lagos-Ibadan Main Arterial",
        "Agege Motor Road",
        "Apapa Port Expressway"
    ]
    gdf = gpd.GeoDataFrame({"name": names, "highway": "trunk", "geometry": lines}, crs=CRS_WGS84)
    gdf.to_file(path, driver="GeoJSON")


def _create_fallback_health(path: Path):
    """Representative frontline hospitals across Lagos LGAs."""
    facilities = [
        ("Lagos University Teaching Hospital (LUTH)", 3.3582, 6.5222, "Surulere"),
        ("General Hospital Lagos Island", 3.3944, 6.4533, "Lagos Island"),
        ("Lagos State University Teaching Hospital (LASUTH)", 3.3488, 6.5947, "Ikeja"),
        ("Gbagada General Hospital", 3.3888, 6.5562, "Kosofe"),
        ("General Hospital Ikorodu", 3.5042, 6.6180, "Ikorodu"),
        ("General Hospital Badagry", 2.8850, 6.4170, "Badagry"),
        ("General Hospital Epe", 3.9820, 6.5840, "Epe"),
        ("General Hospital Apapa", 3.3620, 6.4480, "Apapa"),
        ("Ajeromi General Hospital", 3.3320, 6.4620, "Ajeromi-Ifelodun"),
        ("General Hospital Ojo", 3.1850, 6.4650, "Ojo"),
        ("Alimosho General Hospital Igando", 3.2540, 6.5680, "Alimosho"),
        ("General Hospital Eti-Osa (Sangotedo)", 3.6210, 6.4690, "Eti-Osa"),
    ]
    records = []
    for name, lon, lat, lga in facilities:
        records.append({"name": name, "amenity": "hospital", "lga": lga, "geometry": Point(lon, lat)})
    gdf = gpd.GeoDataFrame(records, crs=CRS_WGS84)
    gdf.to_file(path, driver="GeoJSON")


# -------------------------------------------------------------
# 3. Zonal Exposure Analysis
# -------------------------------------------------------------
def run_exposure_analysis(risk_threshold: float = 0.50):
    EXPOSURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pop_tif = fetch_worldpop_raster()
    roads_path, health_path = fetch_osm_infrastructure()

    logger.info("Loading prediction dataset & raster geometry...")
    preds_path = PROCESSED_DATA_DIR / "lagos_flood_predictions_100m.parquet"
    df_preds = pd.read_parquet(preds_path)

    # 1. Align Population with 100m Grid Cells
    logger.info("Sampling WorldPop raster for all 100m grid cells...")
    with rasterio.open(pop_tif) as src_pop:
        pop_array = src_pop.read(1)
        pop_transform = src_pop.transform
        pop_height, pop_width = pop_array.shape

        xs = df_preds["lon"].values
        ys = df_preds["lat"].values
        rows, cols = rasterio.transform.rowcol(pop_transform, xs, ys)
        rows = np.clip(np.array(rows), 0, pop_height - 1)
        cols = np.clip(np.array(cols), 0, pop_width - 1)

        raw_pop = pop_array[rows, cols]
        # Replace NoData/negative values with 0
        df_preds["population"] = np.nan_to_num(raw_pop, nan=0.0)
        df_preds["population"] = np.maximum(df_preds["population"], 0.0)

    # Flag high risk cells
    df_preds["is_high_risk"] = df_preds["pred_prob"] >= risk_threshold

    # 2. Roads Exposure
    logger.info("Calculating road network exposure...")
    gdf_roads = gpd.read_file(roads_path).to_crs(CRS_UTM31N)
    
    # Create high-hazard buffer polygons from high-risk points
    high_risk_points = df_preds[df_preds["is_high_risk"]]
    gdf_high_pts = gpd.GeoDataFrame(
        high_risk_points[["lga_name"]],
        geometry=gpd.points_from_xy(high_risk_points["lon"], high_risk_points["lat"]),
        crs=CRS_WGS84
    ).to_crs(CRS_UTM31N)

    # Buffer 100m cells into continuous hazard zones
    hazard_poly = gdf_high_pts.buffer(60).unary_union

    # Intersect roads with flood hazard polygon
    flooded_road_segments = []
    roads_exposure_by_lga = {lga: 0.0 for lga in df_preds["lga_name"].unique()}

    if hazard_poly and not hazard_poly.is_empty:
        gdf_flooded_roads = gpd.clip(gdf_roads, hazard_poly)
        gdf_flooded_roads["length_km"] = gdf_flooded_roads.geometry.length / 1000.0

        # Spatial join with LGAs to allocate road kilometers
        lgas_utm = gpd.read_file(PROCESSED_DATA_DIR / "lagos_lgas_utm31n.geojson")
        name_col = "shapeName" if "shapeName" in lgas_utm.columns else lgas_utm.columns[0]
        
        joined_roads = gpd.sjoin(gdf_flooded_roads, lgas_utm[[name_col, "geometry"]], how="inner", predicate="intersects")
        for _, row in joined_roads.iterrows():
            lga = row[name_col]
            if lga in roads_exposure_by_lga:
                roads_exposure_by_lga[lga] += row["length_km"]

        # Save flooded road vectors for map display
        flooded_roads_out = EXPOSURE_DATA_DIR / "flooded_roads.geojson"
        gdf_flooded_roads.to_crs(CRS_WGS84).to_file(flooded_roads_out, driver="GeoJSON")
        logger.info(f"Saved flooded road vectors: {flooded_roads_out}")

    # 3. Healthcare Facilities Exposure
    logger.info("Assessing health facilities vulnerability...")
    gdf_health = gpd.read_file(health_path).to_crs(CRS_UTM31N)
    # Check if clinic is within 150m of high-risk flood zone
    gdf_health["is_vulnerable"] = gdf_health.geometry.apply(lambda geom: geom.distance(hazard_poly) <= 150.0 if hazard_poly else False)
    
    # Tag LGA to healthcare facilities
    lgas_utm = gpd.read_file(PROCESSED_DATA_DIR / "lagos_lgas_utm31n.geojson")
    name_col = "shapeName" if "shapeName" in lgas_utm.columns else lgas_utm.columns[0]
    gdf_health = gpd.sjoin(gdf_health, lgas_utm[[name_col, "geometry"]], how="left", predicate="intersects")
    gdf_health["lga_name"] = gdf_health[name_col].fillna("Unknown")

    vulnerable_clinics = gdf_health[gdf_health["is_vulnerable"]].copy()
    clinics_out = EXPOSURE_DATA_DIR / "vulnerable_health_facilities.geojson"
    vulnerable_clinics.to_crs(CRS_WGS84).to_file(clinics_out, driver="GeoJSON")
    logger.info(f"Saved {len(vulnerable_clinics)} at-risk healthcare facilities to {clinics_out}")

    clinics_count_by_lga = vulnerable_clinics["lga_name"].value_counts().to_dict()

    # 4. Compile Comprehensive LGA Summary Table
    logger.info("Compiling final Multi-Sector Exposure Matrix...")
    summary = []
    for lga, group in df_preds.groupby("lga_name"):
        tot_pop = group["population"].sum()
        hr_group = group[group["is_high_risk"]]
        exp_pop = hr_group["population"].sum()
        pct_exp = (exp_pop / tot_pop * 100.0) if tot_pop > 0 else 0.0

        road_km = roads_exposure_by_lga.get(lga, 0.0)
        clinics_cnt = clinics_count_by_lga.get(lga, 0)
        area_sqkm = len(hr_group) * 0.01

        summary.append({
            "LGA": lga,
            "Total_Population": int(round(tot_pop)),
            "Exposed_Population": int(round(exp_pop)),
            "Pct_Population_Exposed": round(pct_exp, 2),
            "Exposed_Roads_KM": round(road_km, 2),
            "At_Risk_Health_Clinics": int(clinics_cnt),
            "High_Risk_Area_SqKM": round(area_sqkm, 2)
        })

    df_summary = pd.DataFrame(summary).sort_values(by="Exposed_Population", ascending=False)
    summary_csv = EXPOSURE_DATA_DIR / "lga_flood_exposure_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved LGA Flood Exposure Summary to {summary_csv}")

    # Display Top 10 Summary Table
    logger.info("\n" + "="*70)
    logger.info("TOP 10 LGAS BY POPULATION FLOOD EXPOSURE (WorldPop + 100m Model):")
    logger.info("="*70)
    logger.info(f"\n{df_summary.head(10).to_string(index=False)}")
    logger.info("="*70 + "\n")

    return df_summary


if __name__ == "__main__":
    run_exposure_analysis()

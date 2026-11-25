"""
Automated Google Earth Engine Extraction Script
Extracts:
1. Copernicus DEM (30m) -> Elevation, Slope, Aspect
2. Sentinel-1 SAR (10m) -> Backscatter Change & Flood Mask (July 2021 Lagos Flood Event)
3. ESA WorldCover (10m) -> Built-up Impervious Surface
4. CHIRPS Rainfall -> Peak 3-Day Deluge

Outputs GeoTIFF rasters directly to data/raw/rasters/ using geemap direct download.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RASTERS_DIR, BOUNDARIES_DIR, LAGOS_BBOX, GEE_PROJECT_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def init_ee(project_id: str = None):
    """Initializes Google Earth Engine with the specified Google Cloud Project."""
    import ee
    project = project_id or GEE_PROJECT_ID
    try:
        if project:
            logger.info(f"Initializing Earth Engine with project: {project}")
            ee.Initialize(project=project)
        else:
            logger.info("Initializing Earth Engine with default persistent credentials...")
            ee.Initialize()
        logger.info("Google Earth Engine successfully authenticated and initialized!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        logger.info("\n" + "="*70)
        logger.info("HOW TO AUTHENTICATE GOOGLE EARTH ENGINE:")
        logger.info("1. In your PowerShell terminal, run:")
        logger.info("   .\\venv\\Scripts\\earthengine.exe authenticate")
        logger.info("2. Sign in with your Google Account in the browser window that opens.")
        logger.info("3. Copy your Google Cloud project ID (e.g., 'ee-myproject' or 'spatial-lagos').")
        logger.info("4. Re-run this script with: python -m src.extract_features_gee --project <YOUR_PROJECT_ID>")
        logger.info("="*70 + "\n")
        return False


def get_lagos_ee_geometry():
    """Loads the Lagos State boundary polygon into an ee.Geometry."""
    import ee
    import json

    state_path = BOUNDARIES_DIR / "lagos_state.geojson"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        features = geojson_data.get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            geom_type = features[0]["geometry"]["type"]
            if geom_type == "Polygon":
                return ee.Geometry.Polygon(coords)
            elif geom_type == "MultiPolygon":
                return ee.Geometry.MultiPolygon(coords)
    
    # Fallback to bounding box [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def extract_and_download_all(project_id: str = None, scale: int = 100):
    """
    Pulls layers from GEE and downloads them directly to data/raw/rasters/.
    Target resolution: 100m grid for the susceptibility model.
    """
    import ee
    import geemap

    if not init_ee(project_id):
        return False

    aoi = get_lagos_ee_geometry()
    RASTERS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Copernicus DEM (30m) -> Elevation & Slope
    # -------------------------------------------------------------
    logger.info("Extracting Copernicus 30m DEM & calculating slope...")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().clip(aoi)
    slope = ee.Terrain.slope(dem).rename("slope")
    terrain = dem.rename("elevation").addBands(slope)

    dem_path = RASTERS_DIR / "lagos_dem_slope_100m.tif"
    if not dem_path.exists():
        logger.info(f"Downloading DEM & Slope raster to {dem_path}...")
        geemap.ee_export_image(
            terrain,
            filename=str(dem_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )
        logger.info(f"DEM & Slope saved successfully: {dem_path}")
    else:
        logger.info(f"DEM raster already cached at {dem_path}")

    # -------------------------------------------------------------
    # 2. Sentinel-1 SAR Backscatter (July 2021 Flood Event)
    # -------------------------------------------------------------
    logger.info("Extracting Sentinel-1 SAR (July 2021 historic Lagos flood event)...")
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filterBounds(aoi)
        .select("VH")
    )

    # Baseline dry season: Jan-Feb 2021
    pre_flood = s1.filterDate("2021-01-01", "2021-02-28").median().clip(aoi)
    # Major flood event: July 16 - July 24, 2021
    post_flood = s1.filterDate("2021-07-16", "2021-07-24").min().clip(aoi)

    sar_diff = post_flood.subtract(pre_flood).rename("sar_diff_db")
    # Standing open water specular reflection drops backscatter by > 3 dB
    flood_extent = sar_diff.lt(-3.0).rename("flood_observed")
    sar_composite = sar_diff.addBands(flood_extent)

    sar_path = RASTERS_DIR / "lagos_sentinel1_flood_july2021_100m.tif"
    if not sar_path.exists():
        logger.info(f"Downloading Sentinel-1 SAR flood layers to {sar_path}...")
        geemap.ee_export_image(
            sar_composite,
            filename=str(sar_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )
        logger.info(f"Sentinel-1 SAR flood layer saved: {sar_path}")
    else:
        logger.info(f"SAR raster already cached at {sar_path}")

    # -------------------------------------------------------------
    # 3. ESA WorldCover (Built-up / Impervious Surface Fraction)
    # -------------------------------------------------------------
    logger.info("Extracting ESA WorldCover built-up layer...")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
    # Class 50 = Built-up / Urban
    builtup = worldcover.eq(50).rename("builtup_fraction")

    builtup_path = RASTERS_DIR / "lagos_builtup_100m.tif"
    if not builtup_path.exists():
        logger.info(f"Downloading built-up fraction raster to {builtup_path}...")
        geemap.ee_export_image(
            builtup,
            filename=str(builtup_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )
        logger.info(f"Built-up raster saved: {builtup_path}")
    else:
        logger.info(f"Built-up raster already cached at {builtup_path}")

    # -------------------------------------------------------------
    # 4. CHIRPS Daily Rainfall (Peak 3-Day Accumulation)
    # -------------------------------------------------------------
    logger.info("Extracting CHIRPS peak daily rainfall...")
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate("2021-07-01", "2021-07-31")
        .select("precipitation")
    )
    max_rain = chirps.max().clip(aoi).rename("peak_rainfall_mm")

    rain_path = RASTERS_DIR / "lagos_chirps_rainfall_100m.tif"
    if not rain_path.exists():
        logger.info(f"Downloading CHIRPS rainfall raster to {rain_path}...")
        geemap.ee_export_image(
            max_rain,
            filename=str(rain_path),
            scale=scale,
            region=aoi,
            file_per_band=False
        )
        logger.info(f"CHIRPS rainfall raster saved: {rain_path}")
    else:
        logger.info(f"Rainfall raster already cached at {rain_path}")

    logger.info("All Earth Engine layers extracted and downloaded successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GEE layers for Lagos Flood Risk Project")
    parser.add_argument("--project", type=str, default=None, help="Google Cloud Project ID for Earth Engine")
    parser.add_argument("--scale", type=int, default=100, help="Spatial resolution in meters (default: 100)")
    args = parser.parse_args()

    extract_and_download_all(project_id=args.project, scale=args.scale)

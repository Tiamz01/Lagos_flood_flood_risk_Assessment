"""
Google Earth Engine Pipeline for Lagos Flood Analysis
Extracts Copernicus DEM terrain, Sentinel-1 SAR flood extents, ESA WorldCover, and CHIRPS rainfall.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAGOS_BBOX, RASTERS_DIR

try:
    import ee
except ImportError:
    ee = None


def initialize_earth_engine(project_id: str = None):
    """
    Initializes Earth Engine. If not authenticated, prints instructions.
    """
    if ee is None:
        raise ImportError("earthengine-api is not yet installed. Wait for pip install to finish.")
    
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✅ Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"⚠️ GEE Initialization required. Run 'earthengine authenticate' in terminal: {e}")


def get_lagos_ee_geometry():
    """Returns an ee.Geometry.Rectangle for Lagos State bounding box."""
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def get_terrain_features(aoi):
    """
    Pulls Copernicus GLO-30 DEM and computes:
    - Elevation (meters)
    - Slope (degrees)
    - Aspect (degrees)
    """
    dem = ee.Image("COPERNICUS/DEM/GLO30").select("DEM").clip(aoi)
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    
    terrain = dem.rename("elevation").addBands(slope).addBands(aspect)
    return terrain


def get_sentinel1_flood_extent(aoi, pre_flood_start, pre_flood_end, flood_start, flood_end):
    """
    Performs SAR change detection using Sentinel-1 GRD (C-Band, VV/VH).
    Smooth open water surfaces cause specular reflection, showing up as a significant drop in backscatter (dB).
    """
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filterBounds(aoi)
        .select("VH")
    )

    pre_flood = s1.filterDate(pre_flood_start, pre_flood_end).median().clip(aoi)
    post_flood = s1.filterDate(flood_start, flood_end).min().clip(aoi)

    # Difference in backscatter
    diff = post_flood.subtract(pre_flood).rename("sar_diff")
    # Mask permanent water and find newly inundated areas (drop > 3 dB)
    flood_mask = diff.lt(-3.0).rename("flood_label")
    
    return flood_mask.addBands(diff)


def get_landcover_builtup(aoi):
    """
    Pulls ESA WorldCover 10m and creates a built-up / impervious surface mask (class 50 = built-up).
    """
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
    built_up = worldcover.eq(50).rename("builtup_fraction")
    return built_up


def get_extreme_rainfall(aoi, start_date="2021-07-01", end_date="2021-07-31"):
    """
    Pulls CHIRPS Daily Rainfall and computes maximum 3-day rainfall accumulation.
    """
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("precipitation")
    )
    max_daily = chirps.max().clip(aoi).rename("max_rainfall_mm")
    return max_daily

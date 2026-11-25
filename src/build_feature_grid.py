"""
Feature Grid Construction Module
Harmonizes all raster features (DEM, Slope, TWI, Built-up, Rainfall, Distance to Water)
and ground truth flood labels into a clean 100m tabular dataset tagged by Local Government Area (LGA).
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RASTERS_DIR, BOUNDARIES_DIR, PROCESSED_DATA_DIR, CRS_WGS84, CRS_UTM31N
from src.features import calculate_slope_aspect, calculate_twi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_100m_feature_dataset():
    dem_file = RASTERS_DIR / "lagos_dem_slope_100m.tif"
    builtup_file = RASTERS_DIR / "lagos_builtup_100m.tif"
    rainfall_file = RASTERS_DIR / "lagos_chirps_rainfall_100m.tif"
    sar_file = RASTERS_DIR / "lagos_sentinel1_flood_july2021_100m.tif"
    lgas_file = BOUNDARIES_DIR / "lagos_lgas.geojson"

    # Step 1: Read elevation and derive terrain features
    logger.info("Loading DEM and deriving Slope and TWI...")
    with rasterio.open(dem_file) as src_dem:
        elevation = src_dem.read(1).astype(np.float32)
        transform = src_dem.transform
        crs = src_dem.crs
        height, width = elevation.shape

    # Calculate terrain features locally
    slope = calculate_slope_aspect(elevation, cell_size=100.0)
    twi = calculate_twi(elevation, cell_size=100.0)

    # Step 2: Read built-up and rainfall rasters
    logger.info("Loading Built-up and Rainfall layers...")
    with rasterio.open(builtup_file) as src:
        builtup = src.read(1).astype(np.float32)

    with rasterio.open(rainfall_file) as src:
        rainfall = src.read(1).astype(np.float32)

    # Step 3: Read Sentinel-1 SAR flood ground truth
    logger.info("Loading Sentinel-1 SAR flood ground truth...")
    with rasterio.open(sar_file) as src:
        sar_diff = src.read(1).astype(np.float32)
        flood_label = src.read(2).astype(np.uint8)

    # Step 4: Compute Distance to Water / Drainage
    logger.info("Deriving Waterway Proximity...")
    # Identify water bodies from SAR or low elevation depressions
    water_mask = (elevation <= 1.0) | (flood_label == 1)
    # Compute Euclidean distance in pixels, then scale by 100 meters
    dist_water_pixels = distance_transform_edt(~water_mask)
    dist_water_meters = dist_water_pixels * 100.0

    # Step 5: Convert grid cells to coordinates
    logger.info("Generating spatial coordinates for 100m grid...")
    rows, cols = np.indices((height, width))
    lons, lats = rasterio.transform.xy(transform, rows, cols)
    lons = np.array(lons).flatten()
    lats = np.array(lats).flatten()

    # Flatten raster arrays
    df = pd.DataFrame({
        "lon": lons,
        "lat": lats,
        "elevation": elevation.flatten(),
        "slope": slope.flatten(),
        "twi": twi.flatten(),
        "builtup_fraction": builtup.flatten(),
        "rainfall_mm": rainfall.flatten(),
        "sar_diff_db": sar_diff.flatten(),
        "dist_water_m": dist_water_meters.flatten(),
        "flooded": flood_label.flatten()
    })

    # Filter out nodata or outside valid elevation range
    df = df[(df["elevation"] >= 0) & (df["elevation"] <= 200)].copy()

    # Step 6: Spatial overlay with Lagos LGAs
    logger.info("Overlaying grid points with Local Government Areas (LGAs)...")
    gdf_points = gpd.GeoDataFrame(
        df[["lon", "lat"]],
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=CRS_WGS84
    )

    gdf_lgas = gpd.read_file(lgas_file)
    # Match LGA name column
    name_col = "shapeName" if "shapeName" in gdf_lgas.columns else gdf_lgas.columns[0]
    gdf_lgas = gdf_lgas[[name_col, "geometry"]].rename(columns={name_col: "lga_name"})

    # Spatial Join
    joined = gpd.sjoin(gdf_points, gdf_lgas, how="inner", predicate="intersects")
    
    # Merge LGA names back to the main dataframe
    df = df.loc[joined.index].copy()
    df["lga_name"] = joined["lga_name"].values

    # Clean LGA names
    df["lga_name"] = df["lga_name"].str.strip()

    logger.info(f"Dataset compiled: {len(df):,} valid 100m grid cells across {df['lga_name'].nunique()} LGAs.")
    logger.info(f"Flood positive rate: {df['flooded'].mean():.2%}")

    # Step 7: Export to processed data
    out_parquet = PROCESSED_DATA_DIR / "lagos_flood_100m_dataset.parquet"
    out_csv = PROCESSED_DATA_DIR / "lagos_flood_100m_dataset.csv"
    
    df.to_parquet(out_parquet, index=False)
    # Also save a sample CSV for quick inspection
    df.sample(min(10000, len(df)), random_state=42).to_csv(out_csv, index=False)

    logger.info(f"Saved processed dataset to {out_parquet} and sample to {out_csv}")
    return df


if __name__ == "__main__":
    build_100m_feature_dataset()

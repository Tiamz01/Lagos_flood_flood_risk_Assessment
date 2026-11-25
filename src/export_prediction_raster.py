"""
Export Model Predictions to a GeoTIFF Raster
Creates lagos_flood_probability_100m.tif for direct visualization in QGIS and GIS software.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RASTERS_DIR, PROCESSED_DATA_DIR


def export_prediction_geotiff():
    dem_path = RASTERS_DIR / "lagos_dem_slope_100m.tif"
    preds_path = PROCESSED_DATA_DIR / "lagos_flood_predictions_100m.parquet"
    out_tif = PROCESSED_DATA_DIR / "lagos_flood_probability_100m.tif"

    print("Reading reference DEM raster profile...")
    with rasterio.open(dem_path) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        height, width = ref.shape

    print("Loading model predictions from parquet...")
    df = pd.read_parquet(preds_path)

    # Initialize raster array with NoData (-1.0)
    prob_grid = np.full((height, width), -1.0, dtype=np.float32)

    # Convert geographic lon, lat to pixel rows and cols
    xs = df["lon"].values
    ys = df["lat"].values
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.array(rows)
    cols = np.array(cols)

    # Clip to valid bounds
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    prob_grid[rows[valid], cols[valid]] = df["pred_prob"].values[valid].astype(np.float32)

    profile.update(
        count=1,
        dtype=rasterio.float32,
        nodata=-1.0,
        compress="lzw"
    )

    print(f"Writing prediction GeoTIFF to {out_tif}...")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(prob_grid, 1)

    print(f"Successfully exported {out_tif} (Shape: {prob_grid.shape})")
    return out_tif


if __name__ == "__main__":
    export_prediction_geotiff()

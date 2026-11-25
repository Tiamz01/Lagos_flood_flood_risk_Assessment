"""
Data Ingestion Module for Lagos Flood Risk & Exposure Project
Handles downloading and processing administrative boundaries, OSM data, and base layers.
"""

import os
import sys
import json
import logging
from pathlib import Path
import urllib.request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BOUNDARIES_DIR,
    PROCESSED_DATA_DIR,
    CRS_WGS84,
    CRS_UTM31N,
    LAGOS_LGAS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# GeoBoundaries API Endpoints for Nigeria
GEOBOUNDARIES_API_ADM1 = "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/"
GEOBOUNDARIES_API_ADM2 = "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM2/"


def get_geoboundaries_download_url(api_url: str) -> str:
    """Fetch the resolved GeoJSON download URL from geoBoundaries API."""
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
        return meta.get("gjDownloadURL")


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with progress logging if not already cached."""
    if dest_path.exists():
        logger.info(f"File already exists at {dest_path}, skipping download.")
        return dest_path

    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    logger.info(f"Saved to {dest_path}")
    return dest_path


def get_lagos_boundaries():
    """
    Downloads Nigeria ADM1 and ADM2, filters for Lagos State & its LGAs,
    and saves clean GeoJSON files in WGS84 and UTM31N.
    """
    raw_adm1 = BOUNDARIES_DIR / "nigeria_adm1.geojson"
    raw_adm2 = BOUNDARIES_DIR / "nigeria_adm2.geojson"

    # Step 1: Resolve download URLs and fetch raw boundaries
    adm1_download_url = get_geoboundaries_download_url(GEOBOUNDARIES_API_ADM1)
    adm2_download_url = get_geoboundaries_download_url(GEOBOUNDARIES_API_ADM2)

    download_file(adm1_download_url, raw_adm1)
    download_file(adm2_download_url, raw_adm2)

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("GeoPandas not yet installed; parsing raw JSON directly.")
        _filter_geojson_fallback(raw_adm1, raw_adm2)
        return

    # Step 2: Filter ADM1 (Lagos State boundary)
    gdf_adm1 = gpd.read_file(raw_adm1)
    lagos_state = gdf_adm1[gdf_adm1["shapeName"].str.contains("Lagos", case=False, na=False)]
    if lagos_state.empty:
        raise ValueError("Could not find 'Lagos' in ADM1 dataset.")

    state_wgs84_path = BOUNDARIES_DIR / "lagos_state.geojson"
    lagos_state.to_file(state_wgs84_path, driver="GeoJSON")
    logger.info(f"Saved Lagos State boundary to {state_wgs84_path}")

    # Step 3: Filter ADM2 (Lagos Local Government Areas)
    gdf_adm2 = gpd.read_file(raw_adm2)
    # Match by spatial intersection with Lagos State polygon
    lagos_lgas = gpd.sjoin(gdf_adm2, lagos_state[["geometry"]], how="inner", predicate="intersects")
    
    # Save in WGS84 (for web display and Leaflet/pydeck)
    lgas_wgs84_path = BOUNDARIES_DIR / "lagos_lgas.geojson"
    lagos_lgas.to_file(lgas_wgs84_path, driver="GeoJSON")
    logger.info(f"Saved {len(lagos_lgas)} Lagos LGAs to {lgas_wgs84_path}")

    # Reproject to UTM Zone 31N (metric projection for accurate distance/area)
    lagos_lgas_utm = lagos_lgas.to_crs(CRS_UTM31N)
    lgas_utm_path = PROCESSED_DATA_DIR / "lagos_lgas_utm31n.geojson"
    lagos_lgas_utm.to_file(lgas_utm_path, driver="GeoJSON")
    logger.info(f"Saved reprojected UTM31N LGAs to {lgas_utm_path}")


def _filter_geojson_fallback(adm1_path: Path, adm2_path: Path):
    """Fallback parser if geopandas is not yet available during bootstrap."""
    with open(adm1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lagos_features = [
        feat for feat in data.get("features", [])
        if "lagos" in str(feat.get("properties", {}).get("shapeName", "")).lower()
    ]
    out = {"type": "FeatureCollection", "features": lagos_features}
    dest = BOUNDARIES_DIR / "lagos_state.geojson"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f)
    logger.info(f"Fallback extracted Lagos boundary to {dest}")


if __name__ == "__main__":
    get_lagos_boundaries()

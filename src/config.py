import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BOUNDARIES_DIR = RAW_DATA_DIR / "boundaries"
RASTERS_DIR = RAW_DATA_DIR / "rasters"
OSM_DIR = RAW_DATA_DIR / "osm"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPOSURE_DATA_DIR = DATA_DIR / "exposure"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"        # Storage & web display
CRS_UTM31N = "EPSG:32631"      # Metric calculations (Lagos State UTM Zone 31N)

# Google Earth Engine Default Project
GEE_PROJECT_ID = "lagos-flood-risk-514314"

# Target Spatial Resolution
TARGET_RESOLUTION_METERS = 100

# Lagos State Bounding Box [min_lon, min_lat, max_lon, max_lat]
LAGOS_BBOX = [2.71, 6.35, 4.38, 6.70]

# Official 20 LGAs in Lagos State
LAGOS_LGAS = [
    "Agege", "Ajeromi-Ifelodun", "Alimosho", "Amuwo-Odofin", "Apapa",
    "Badagry", "Epe", "Eti-Osa", "Ibeju-Lekki", "Ifako-Ijaiye",
    "Ikeja", "Ikorodu", "Kosofe", "Lagos Island", "Lagos Mainland",
    "Mushin", "Ojo", "Oshodi-Isolo", "Shomolu", "Surulere"
]

# Ensure core output directories exist
for directory in [
    BOUNDARIES_DIR, RASTERS_DIR, OSM_DIR,
    PROCESSED_DATA_DIR, EXPOSURE_DATA_DIR, FIGURES_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

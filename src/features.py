"""
Spatial Feature Engineering Module for Flood Susceptibility
Derives terrain indices (Slope, Aspect, TWI) and proximity layers at 100m resolution.
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_slope_aspect(elevation: np.ndarray, cell_size: float = 100.0):
    """
    Computes slope in degrees from an elevation raster using numpy gradient.
    Args:
        elevation: 2D numpy array of elevations in meters.
        cell_size: spatial resolution in meters (default 100m).
    Returns:
        slope_deg: 2D numpy array of slopes in degrees.
    """
    dy, dx = np.gradient(elevation, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def calculate_twi(elevation: np.ndarray, cell_size: float = 100.0, min_slope_rad: float = 0.001):
    """
    Computes Topographic Wetness Index (TWI):
    TWI = ln(a / tan(beta))
    where:
      a = specific catchment area (approximated locally by cell size)
      beta = slope angle in radians
    Higher TWI indicates areas prone to water accumulation.
    """
    dy, dx = np.gradient(elevation, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    # Clip minimum slope to prevent division by zero in completely flat areas
    slope_rad = np.maximum(slope_rad, min_slope_rad)
    
    # Catchment area approximation per unit contour width
    catchment_area = cell_size
    twi = np.log(catchment_area / np.tan(slope_rad))
    # Replace any NaNs or infinities
    twi = np.nan_to_num(twi, nan=0.0, posinf=25.0, neginf=0.0)
    return twi


def normalize_feature(array: np.ndarray) -> np.ndarray:
    """Min-max normalize feature to [0, 1] range."""
    min_val = np.nanmin(array)
    max_val = np.nanmax(array)
    if max_val == min_val:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

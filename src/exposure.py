"""
Exposure & Vulnerability Analysis Module
Calculates population at risk, vulnerable road segments, and health facilities per LGA.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config import EXPOSURE_DATA_DIR, CRS_UTM31N

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_lga_exposure_summary(
    df_grid: pd.DataFrame,
    prob_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Aggregates population and infrastructure exposure by LGA.
    
    Args:
        df_grid: DataFrame containing 100m grid cells with columns:
                 ['lga_name', 'pred_prob', 'population', 'road_length_m', 'num_clinics']
        prob_threshold: Cutoff for high flood susceptibility (default: 0.60).
    Returns:
        Summary DataFrame per LGA.
    """
    df_grid["is_high_risk"] = df_grid["pred_prob"] >= prob_threshold

    summary = []
    for lga, group in df_grid.groupby("lga_name"):
        total_pop = group["population"].sum() if "population" in group else 0
        high_risk_cells = group[group["is_high_risk"]]
        
        exposed_pop = high_risk_cells["population"].sum() if "population" in high_risk_cells else 0
        exposed_road_km = (high_risk_cells["road_length_m"].sum() / 1000.0) if "road_length_m" in high_risk_cells else 0
        exposed_clinics = high_risk_cells["num_clinics"].sum() if "num_clinics" in high_risk_cells else 0
        
        pct_pop_exposed = (exposed_pop / total_pop * 100.0) if total_pop > 0 else 0.0

        summary.append({
            "LGA": lga,
            "Total_Population": int(round(total_pop)),
            "Exposed_Population": int(round(exposed_pop)),
            "Pct_Population_Exposed": round(pct_pop_exposed, 2),
            "Exposed_Roads_KM": round(exposed_road_km, 2),
            "Exposed_Health_Clinics": int(exposed_clinics),
            "High_Risk_Area_SqKM": round(len(high_risk_cells) * 0.01, 2)  # 100m x 100m = 0.01 sq km
        })

    summary_df = pd.DataFrame(summary).sort_values(by="Exposed_Population", ascending=False)
    
    out_path = EXPOSURE_DATA_DIR / "lga_flood_exposure_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Saved LGA exposure summary to {out_path}")
    return summary_df

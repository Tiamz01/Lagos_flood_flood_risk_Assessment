"""
Model Training Script
Trains a LightGBM susceptibility model using Spatial Block Cross-Validation (grouped by LGA)
to eliminate spatial data leakage. Outputs spatial cross-validation metrics and feature importances.
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training():
    data_path = PROCESSED_DATA_DIR / "lagos_flood_100m_dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset at {data_path}. Run build_feature_grid first.")

    logger.info("Loading 100m spatial dataset...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} grid cells across {df['lga_name'].nunique()} LGAs.")

    # Modeling Features
    feature_cols = [
        "elevation",
        "slope",
        "twi",
        "builtup_fraction",
        "rainfall_mm",
        "dist_water_m"
    ]
    target_col = "flooded"
    group_col = "lga_name"

    # Filter out any lingering NaNs
    df = df.dropna(subset=feature_cols + [target_col, group_col]).copy()

    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    # Calculate class imbalance weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive (scale_pos_weight={scale_pos_weight:.2f})")

    # -------------------------------------------------------------
    # Spatial Block Cross-Validation (LGA-based GroupKFold)
    # -------------------------------------------------------------
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    logger.info(f"\n{'='*60}\nRUNNING {n_splits}-FOLD SPATIAL BLOCK CROSS-VALIDATION (LGA GROUPED)\n{'='*60}")
    
    cv_metrics = {"auc": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        val_lgas = groups.iloc[val_idx].unique().tolist()

        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + fold,
            importance_type="gain",
            verbose=-1
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        auc = roc_auc_score(y_val, val_probs)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)

        cv_metrics["auc"].append(auc)
        cv_metrics["f1"].append(f1)
        cv_metrics["precision"].append(prec)
        cv_metrics["recall"].append(rec)

        logger.info(f"Fold {fold} -> ROC-AUC: {auc:.3f} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} (Held-out LGAs: {len(val_lgas)})")

    logger.info("\n" + "="*60)
    logger.info("FINAL SPATIAL CROSS-VALIDATION SUMMARY:")
    logger.info(f"  Mean ROC-AUC  : {np.mean(cv_metrics['auc']):.3f} ± {np.std(cv_metrics['auc']):.3f}")
    logger.info(f"  Mean F1-Score : {np.mean(cv_metrics['f1']):.3f} ± {np.std(cv_metrics['f1']):.3f}")
    logger.info(f"  Mean Recall   : {np.mean(cv_metrics['recall']):.3f} ± {np.std(cv_metrics['recall']):.3f}")
    logger.info("="*60 + "\n")

    # -------------------------------------------------------------
    # Train Final Production Model on Full Dataset
    # -------------------------------------------------------------
    logger.info("Training final production LightGBM model on full state dataset...")
    final_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        importance_type="gain",
        verbose=-1
    )
    final_model.fit(X, y)

    # Feature Importance Ranking
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_Gain": final_model.feature_importances_
    }).sort_values(by="Importance_Gain", ascending=False)
    importance_df["Relative_%"] = (importance_df["Importance_Gain"] / importance_df["Importance_Gain"].sum()) * 100

    logger.info("\n--- Feature Importance (Gain) ---")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['Feature']:<20}: {row['Importance_Gain']:>10.2f} ({row['Relative_%']:>5.1f}%)")

    # Predict full susceptibility across all Lagos grid cells
    logger.info("Generating state-wide flood susceptibility predictions (100m)...")
    df["pred_prob"] = final_model.predict_proba(X)[:, 1]
    df["risk_level"] = pd.cut(
        df["pred_prob"],
        bins=[-0.01, 0.35, 0.60, 0.80, 1.01],
        labels=["Low", "Moderate", "High", "Very High"]
    )

    # Save artifacts
    model_path = PROCESSED_DATA_DIR / "flood_lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Saved trained model to {model_path}")

    preds_path = PROCESSED_DATA_DIR / "lagos_flood_predictions_100m.parquet"
    df.to_parquet(preds_path, index=False)
    logger.info(f"Saved full predictions to {preds_path}")

    # Summary table by LGA
    lga_summary = df.groupby("lga_name").agg(
        total_cells=("pred_prob", "count"),
        high_risk_cells=("risk_level", lambda s: (s.isin(["High", "Very High"])).sum()),
        mean_susceptibility=("pred_prob", "mean")
    ).reset_index()
    lga_summary["pct_high_risk"] = (lga_summary["high_risk_cells"] / lga_summary["total_cells"]) * 100
    lga_summary = lga_summary.sort_values(by="pct_high_risk", ascending=False)

    summary_path = PROCESSED_DATA_DIR / "lga_susceptibility_ranking.csv"
    lga_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved LGA susceptibility ranking to {summary_path}")

    return final_model, cv_metrics, importance_df, lga_summary


if __name__ == "__main__":
    run_training()

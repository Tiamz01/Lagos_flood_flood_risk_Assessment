"""
Spatial Machine Learning & Modeling Module
Implements Spatial Block Cross-Validation and LightGBM susceptibility modeling.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_spatial_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "flooded",
    group_col: str = "lga_name",
    n_splits: int = 5
):
    """
    Trains a LightGBM classifier using Spatial Block GroupKFold.
    
    Prevents spatial autocorrelation leakage by ensuring all data from a
    given Local Government Area (LGA) are either entirely in the train set
    or entirely in the validation set.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col]
    X = df[feature_cols]
    y = df[target_col]

    cv_scores = {"auc": [], "f1": [], "precision": [], "recall": []}
    models = []

    logger.info(f"Starting {n_splits}-Fold Spatial Cross-Validation grouped by '{group_col}'...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42 + fold,
            importance_type="gain",
            verbose=-1
        )
        model.fit(X_train, y_train)

        val_preds_proba = model.predict_proba(X_val)[:, 1]
        val_preds = (val_preds_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, val_preds_proba)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)

        cv_scores["auc"].append(auc)
        cv_scores["f1"].append(f1)
        cv_scores["precision"].append(prec)
        cv_scores["recall"].append(rec)
        models.append(model)

        logger.info(
            f"Fold {fold} - AUC: {auc:.3f} | F1: {f1:.3f} | "
            f"Precision: {prec:.3f} | Recall: {rec:.3f} (Val LGAs: {df.iloc[val_idx][group_col].nunique()})"
        )

    logger.info("=== Spatial CV Results ===")
    logger.info(f"Mean AUC: {np.mean(cv_scores['auc']):.3f} ± {np.std(cv_scores['auc']):.3f}")
    logger.info(f"Mean F1 : {np.mean(cv_scores['f1']):.3f} ± {np.std(cv_scores['f1']):.3f}")

    # Retrain on full dataset for final predictions
    final_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        importance_type="gain",
        verbose=-1
    )
    final_model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return final_model, cv_scores, importance_df

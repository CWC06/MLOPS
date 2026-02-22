"""Prediction utilities for the heart disease pipeline."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from pycaret.classification import predict_model

from mlops_assignment.config import PROCESSED_DATA_DIR


def predict_holdout(model, df_holdout: pd.DataFrame, target: str = "HeartDisease"):
    """Generate predictions on holdout data and compute metrics.

    Returns
    -------
    (holdout_preds, metrics_dict)
    """
    holdout_preds = predict_model(model, data=df_holdout)

    y_true = df_holdout[target]
    y_pred = holdout_preds["prediction_label"]

    # Correct PyCaret prediction_score to always represent P(class=1)
    y_score = holdout_preds.apply(
        lambda r: r["prediction_score"]
        if r["prediction_label"] == 1
        else 1 - r["prediction_score"],
        axis=1,
    )

    metrics = {
        "holdout_accuracy": accuracy_score(y_true, y_pred),
        "holdout_precision": precision_score(y_true, y_pred, zero_division=0),
        "holdout_recall": recall_score(y_true, y_pred, zero_division=0),
        "holdout_f1": f1_score(y_true, y_pred, zero_division=0),
        "holdout_auc": roc_auc_score(y_true, y_score),
    }

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return holdout_preds, metrics


def save_predictions(
    preds: pd.DataFrame, path: Path | None = None
) -> Path:
    """Save holdout predictions to CSV for auditing."""
    if path is None:
        path = PROCESSED_DATA_DIR / "heart_holdout_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(path, index=False)
    logger.info(f"Predictions saved: {path}")
    return path

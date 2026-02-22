"""Shared utility functions for model training and prediction."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file, dropping admin columns.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(data_path)
    for col in ["index", "Patient Id"]:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


def prepare_data_for_prediction(
    input_dict: Dict[str, Any],
    feature_columns: list,
) -> pd.DataFrame:
    """Convert an input dictionary to a DataFrame ready for prediction.

    Parameters
    ----------
    input_dict : dict
        Feature name → value mapping.
    feature_columns : list
        Ordered list of expected feature column names.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def get_feature_ranges(data_path: str) -> Dict[str, Dict[str, float]]:
    """Return min / max / mean statistics for numeric features.

    Parameters
    ----------
    data_path : str
        Path to the training data CSV.

    Returns
    -------
    dict  {column_name: {min, max, mean}}
    """
    df = pd.read_csv(data_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in numeric_cols
        if col != "index"
    }

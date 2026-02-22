"""Heart disease dataset loading and cleaning utilities."""

from pathlib import Path

import pandas as pd
from loguru import logger

from mlops_assignment.config import RAW_DATA_DIR


def load_heart_data(path: Path | None = None) -> pd.DataFrame:
    """Load the heart disease CSV and apply basic cleaning.

    Steps:
        1. Drop administrative columns (index, Patient Id).
        2. Normalise column names (spaces → underscores).
        3. Replace clinically impossible zero values in Cholesterol
           and RestingBP with the median of non-zero values.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file. Defaults to ``data/raw/heart.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    if path is None:
        path = RAW_DATA_DIR / "heart.csv"

    df = pd.read_csv(path)

    # Drop admin columns
    for col in ["index", "Patient Id"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Normalise column names
    df.columns = df.columns.str.replace(" ", "_")

    # Impute impossible zeros
    for col in ("Cholesterol", "RestingBP"):
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0:
            median_val = df.loc[df[col] > 0, col].median()
            df[col] = df[col].replace(0, median_val)
            logger.info(f"{col}: replaced {n_zeros} zeros with median {median_val}")

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

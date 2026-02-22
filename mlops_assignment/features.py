"""Feature engineering utilities for the heart disease pipeline."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from mlops_assignment.config import PROCESSED_DATA_DIR


def create_holdout_split(
    df: pd.DataFrame,
    target: str = "HeartDisease",
    holdout_size: float = 0.10,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified holdout split and persist the holdout CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    target : str
        Name of the target column.
    holdout_size : float
        Fraction of data reserved for holdout.
    random_state : int
        Random seed for reproducibility.
    output_dir : Path, optional
        Directory to save ``heart_holdout.csv``.

    Returns
    -------
    (df_train_pool, df_holdout)
    """
    df_train, df_holdout = train_test_split(
        df,
        test_size=holdout_size,
        random_state=random_state,
        stratify=df[target],
    )

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = output_dir / "heart_holdout.csv"
    df_holdout.to_csv(holdout_path, index=False)

    logger.info(f"Training pool: {df_train.shape[0]} rows")
    logger.info(f"Holdout set  : {df_holdout.shape[0]} rows → {holdout_path}")
    return df_train, df_holdout

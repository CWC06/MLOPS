"""Basic tests for the heart disease dataset and pipeline."""

import pandas as pd
import pytest
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart.csv"


def test_raw_data_exists():
    """Raw heart.csv must exist in data/raw/."""
    assert DATA_PATH.exists(), f"Raw data not found: {DATA_PATH}"


def test_raw_data_has_expected_columns():
    """Check that the dataset contains all expected feature columns."""
    df = pd.read_csv(DATA_PATH)
    expected = {
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
        "Oldpeak", "ST_Slope", "HeartDisease",
    }
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )


def test_raw_data_has_rows():
    """Dataset must not be empty."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"


def test_target_is_binary():
    """HeartDisease column must only contain 0 and 1."""
    df = pd.read_csv(DATA_PATH)
    assert set(df["HeartDisease"].unique()).issubset({0, 1}), (
        f"Unexpected target values: {df['HeartDisease'].unique()}"
    )


def test_no_missing_values():
    """Raw dataset should have no NaN values."""
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0, (
        f"Missing values found:\n{df.isnull().sum()[df.isnull().sum() > 0]}"
    )

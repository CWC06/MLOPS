"""Tests for the lung cancer dataset and pipeline."""

from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "lung_cancer.csv"

EXPECTED_COLUMNS = {
    "Age",
    "Gender",
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
    "Level",
}


def test_raw_data_exists():
    """lung_cancer.csv must exist in data/raw/."""
    assert DATA_PATH.exists(), f"Raw data not found: {DATA_PATH}"


def test_raw_data_has_expected_columns():
    """Dataset must contain all 24 expected columns."""
    df = pd.read_csv(DATA_PATH)
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_raw_data_has_rows():
    """Dataset must not be empty."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"


def test_target_has_valid_classes():
    """Level column must only contain Low, Medium, High."""
    df = pd.read_csv(DATA_PATH)
    valid = {"Low", "Medium", "High"}
    actual = set(df["Level"].unique())
    assert actual.issubset(valid), f"Unexpected target values: {actual - valid}"


def test_no_missing_values():
    """Raw dataset should have no NaN values."""
    df = pd.read_csv(DATA_PATH)
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Missing values found:\n{df.isnull().sum()[df.isnull().sum() > 0]}"


def test_numeric_features_in_range():
    """All scored features (1-9 scale) must be within valid range."""
    df = pd.read_csv(DATA_PATH)
    scored_cols = [
        "Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards",
        "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity",
        "Smoking", "Passive Smoker", "Chest Pain", "Coughing of Blood",
        "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
        "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold",
        "Dry Cough", "Snoring",
    ]
    for col in scored_cols:
        if col in df.columns:
            assert df[col].between(1, 9).all(), f"Column '{col}' has values outside [1, 9]"

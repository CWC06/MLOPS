"""
Prediction utilities for the heart disease risk model.

Usage:
    from src.models.predict_heart import predict, predict_batch
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from pycaret.classification import load_model, predict_model

warnings.filterwarnings("ignore")

_MODEL_PATH = Path("models/heart_disease_model")

HEART_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]


def _load_model():
    """Load the PyCaret heart disease pipeline from disk."""
    pkl_path = Path(str(_MODEL_PATH) + ".pkl")
    path = pkl_path if pkl_path.exists() else _MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Heart disease model not found at '{_MODEL_PATH}.pkl'. "
            "Train it first: python notebooks/train_heart.ipynb  (or run the training notebook)"
        )
    return load_model(str(_MODEL_PATH))


def predict(
    input_data: Union[pd.DataFrame, Dict[str, Any]],
    return_proba: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Make a prediction for a single patient.

    Parameters
    ----------
    input_data : DataFrame or dict
        Feature values. Expected keys: Age, Sex, ChestPainType, RestingBP,
        Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.
    return_proba : bool
        If True, also return confidence information.

    Returns
    -------
    str or dict
        '0' / '1' (No Heart Disease / Heart Disease), optionally with probabilities.
    """
    model = _load_model()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Normalise column names (spaces → underscores)
    df.columns = df.columns.str.replace(" ", "_")

    preds = predict_model(model, data=df)

    pred_label = (
        preds["prediction_label"].iloc[0]
        if "prediction_label" in preds.columns
        else preds.iloc[0, -1]
    )

    if not return_proba:
        return str(pred_label)

    pred_score = (
        float(preds["prediction_score"].iloc[0])
        if "prediction_score" in preds.columns
        else None
    )

    # For binary classification: pred_score is P(predicted_class)
    # Convert to P(HeartDisease=1)
    if pred_score is not None:
        p_positive = pred_score if int(pred_label) == 1 else 1.0 - pred_score
        probabilities = {
            "No Heart Disease (0)": f"{1.0 - p_positive:.2%}",
            "Heart Disease (1)": f"{p_positive:.2%}",
            "Predicted Class": "Heart Disease" if int(pred_label) == 1 else "No Heart Disease",
            "Confidence": f"{pred_score:.2%}",
        }
    else:
        probabilities = {"Predicted Class": str(pred_label)}

    return {"prediction": str(pred_label), "probabilities": probabilities}


def predict_batch(input_df: pd.DataFrame) -> pd.DataFrame:
    """Make predictions for a batch of patients.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame with heart disease feature columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with prediction columns appended.
    """
    model = _load_model()
    df = input_df.copy()
    df.columns = df.columns.str.replace(" ", "_")
    return predict_model(model, data=df)


if __name__ == "__main__":
    example = {
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 147,
        "ExerciseAngina": "N",
        "Oldpeak": 1.4,
        "ST_Slope": "Flat",
    }
    try:
        result = predict(example, return_proba=True)
        print("Prediction:", result)
    except FileNotFoundError as e:
        print(e)

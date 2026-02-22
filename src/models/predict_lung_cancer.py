"""
Prediction utilities for the lung cancer risk model.

Usage:
    from src.models.predict_lung_cancer import predict, predict_batch
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from pycaret.classification import load_model, predict_model

warnings.filterwarnings("ignore")

_MODEL_PATH = Path("models/lung_cancer_model")


def _load_model():
    """Load the PyCaret lung cancer pipeline from disk."""
    pkl_path = Path(str(_MODEL_PATH) + ".pkl")
    path = pkl_path if pkl_path.exists() else _MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Lung cancer model not found at '{_MODEL_PATH}.pkl'. "
            "Train it first: python src/models/train_lung_cancer.py"
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
        Feature values (column names with or without spaces).
    return_proba : bool
        If True, also return confidence information.

    Returns
    -------
    str or dict
        Predicted risk level, optionally with probability scores.
    """
    model = _load_model()

    if isinstance(input_data, dict):
        normalized = {k.replace(" ", "_"): v for k, v in input_data.items()}
        df = pd.DataFrame([normalized])
    else:
        df = input_data.copy()

    df.columns = df.columns.str.replace(" ", "_")

    preds = predict_model(model, data=df)

    pred_label = (
        preds["prediction_label"].iloc[0]
        if "prediction_label" in preds.columns
        else preds.iloc[0, -1]
    )

    if not return_proba:
        return str(pred_label)

    pred_score = float(preds["prediction_score"].iloc[0]) if "prediction_score" in preds.columns else None

    probabilities: Dict[str, Any] = {}
    try:
        if hasattr(model, "named_steps") and "actual_estimator" in model.named_steps:
            estimator = model.named_steps["actual_estimator"]
            preprocessing_steps = [
                step
                for name, step in model.named_steps.items()
                if name != "actual_estimator"
            ]
            preprocessed = df.copy()
            for step in preprocessing_steps:
                if hasattr(step, "transform"):
                    try:
                        preprocessed = step.transform(preprocessed)
                    except Exception:
                        pass

            if hasattr(estimator, "predict_proba"):
                proba_array = estimator.predict_proba(preprocessed)
                class_labels = ["Low", "Medium", "High"]
                for i, prob in enumerate(proba_array[0]):
                    label = class_labels[i] if i < len(class_labels) else f"Class_{i}"
                    probabilities[label] = float(prob)
    except Exception:
        pass

    if not probabilities and pred_score is not None:
        probabilities["Predicted Class"] = str(pred_label)
        probabilities["Confidence"] = f"{pred_score:.2%}"

    return {"prediction": str(pred_label), "probabilities": probabilities}


def predict_batch(input_df: pd.DataFrame) -> pd.DataFrame:
    """Make predictions for a batch of patients.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame with feature columns.

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
        "Age": 45,
        "Gender": 1,
        "Air Pollution": 5,
        "Alcohol use": 3,
        "Dust Allergy": 4,
        "OccuPational Hazards": 4,
        "Genetic Risk": 3,
        "chronic Lung Disease": 2,
        "Balanced Diet": 3,
        "Obesity": 4,
        "Smoking": 5,
        "Passive Smoker": 3,
        "Chest Pain": 4,
        "Coughing of Blood": 2,
        "Fatigue": 3,
        "Weight Loss": 2,
        "Shortness of Breath": 3,
        "Wheezing": 2,
        "Swallowing Difficulty": 2,
        "Clubbing of Finger Nails": 1,
        "Frequent Cold": 2,
        "Dry Cough": 3,
        "Snoring": 4,
    }
    try:
        result = predict(example, return_proba=True)
        print("Prediction:", result)
    except FileNotFoundError as e:
        print(e)

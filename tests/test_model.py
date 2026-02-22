"""Tests for trained model inference."""

from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


# ---------------------------------------------------------------------------
# Heart Disease Model
# ---------------------------------------------------------------------------

HEART_MODEL_PATH = MODEL_DIR / "heart_disease_model.pkl"


@pytest.mark.skipif(not HEART_MODEL_PATH.exists(), reason="Heart model not trained yet")
def test_heart_model_loads():
    """Trained heart disease pipeline should load without error."""
    from pycaret.classification import load_model

    model = load_model(str(HEART_MODEL_PATH).replace(".pkl", ""))
    assert model is not None


@pytest.mark.skipif(not HEART_MODEL_PATH.exists(), reason="Heart model not trained yet")
def test_heart_model_predicts():
    """Heart model should return a valid prediction for sample input."""
    from src.models.predict_heart import predict

    sample = {
        "Age": 55, "Sex": "M", "ChestPainType": "ATA",
        "RestingBP": 130, "Cholesterol": 250, "FastingBS": 0,
        "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
        "Oldpeak": 1.0, "ST_Slope": "Up",
    }
    result = predict(sample)
    assert result in ("0", "1"), f"Unexpected prediction: {result}"


@pytest.mark.skipif(not HEART_MODEL_PATH.exists(), reason="Heart model not trained yet")
def test_heart_model_returns_probabilities():
    """Heart model should return probability dict when requested."""
    from src.models.predict_heart import predict

    sample = {
        "Age": 55, "Sex": "M", "ChestPainType": "ATA",
        "RestingBP": 130, "Cholesterol": 250, "FastingBS": 0,
        "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
        "Oldpeak": 1.0, "ST_Slope": "Up",
    }
    result = predict(sample, return_proba=True)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probabilities" in result


# ---------------------------------------------------------------------------
# Lung Cancer Model
# ---------------------------------------------------------------------------

LUNG_MODEL_PATH = MODEL_DIR / "lung_cancer_model.pkl"


@pytest.mark.skipif(not LUNG_MODEL_PATH.exists(), reason="Lung cancer model not trained yet")
def test_lung_model_loads():
    """Trained lung cancer pipeline should load without error."""
    from pycaret.classification import load_model

    model = load_model(str(LUNG_MODEL_PATH).replace(".pkl", ""))
    assert model is not None


@pytest.mark.skipif(not LUNG_MODEL_PATH.exists(), reason="Lung cancer model not trained yet")
def test_lung_model_predicts():
    """Lung cancer model should return a valid risk level."""
    from src.models.predict_lung_cancer import predict

    sample = {
        "Age": 45, "Gender": 1,
        "Air Pollution": 5, "Alcohol use": 4, "Dust Allergy": 6,
        "OccuPational Hazards": 5, "Genetic Risk": 3,
        "chronic Lung Disease": 4, "Balanced Diet": 5, "Obesity": 4,
        "Smoking": 3, "Passive Smoker": 4, "Chest Pain": 5,
        "Coughing of Blood": 3, "Fatigue": 5, "Weight Loss": 4,
        "Shortness of Breath": 5, "Wheezing": 3,
        "Swallowing Difficulty": 2, "Clubbing of Finger Nails": 3,
        "Frequent Cold": 4, "Dry Cough": 5, "Snoring": 3,
    }
    result = predict(sample)
    assert result in ("Low", "Medium", "High"), f"Unexpected prediction: {result}"

"""Model training utilities using PyCaret."""

from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pycaret.classification import (
    compare_models,
    finalize_model,
    pull,
    save_model,
    setup,
    tune_model,
)

from mlops_assignment.config import MODELS_DIR


def init_pycaret(
    df_train: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    """Initialise PyCaret environment with preprocessing pipeline.

    Configures normalisation, feature selection, multicollinearity removal,
    binning, and polynomial/interaction features based on the YAML config.
    """
    setup(
        data=df_train,
        target=cfg.model.target_column,
        train_size=1 - cfg.data.test_size,
        normalize=cfg.preprocessing.normalize,
        normalize_method=cfg.preprocessing.normalize_method,
        feature_selection=cfg.preprocessing.feature_selection,
        feature_selection_method=cfg.preprocessing.feature_selection_method,
        polynomial_features=cfg.preprocessing.create_interaction_features,
        remove_multicollinearity=cfg.preprocessing.remove_multicollinearity,
        multicollinearity_threshold=cfg.preprocessing.multicollinearity_threshold,
        bin_numeric_features=(
            list(cfg.preprocessing.bin_numeric_features)
            if cfg.preprocessing.bin_numeric_features
            else None
        ),
        log_experiment=False,
        session_id=cfg.data.random_state,
        verbose=False,
    )
    logger.info("PyCaret environment initialised.")


def train_best_model(cfg: DictConfig):
    """Compare models, tune the best, and finalise."""
    best_models = compare_models(
        include=["lightgbm", "rf", "xgboost", "lr", "nb", "dt", "svm"],
        sort=cfg.model.metric,
        n_select=cfg.model.n_select,
        fold=cfg.model.fold,
        cross_validation=True,
    )

    if not isinstance(best_models, list):
        best_models = [best_models]

    best_model = best_models[0]
    comparison_df = pull()
    logger.info(f"Best model: {type(best_model).__name__}")

    tuned = tune_model(
        best_model,
        optimize=cfg.tuning.optimize,
        n_iter=cfg.tuning.n_iter,
        fold=cfg.model.fold,
        choose_better=True,
        early_stopping="asha",
    )
    tuning_df = pull()

    final = finalize_model(tuned)
    logger.info("Model finalised on full training pool.")

    return final, tuned, comparison_df, tuning_df


def save_pipeline(model, path: Path | None = None) -> str:
    """Save the complete PyCaret pipeline to disk."""
    if path is None:
        path = MODELS_DIR / "heart_disease_model"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))
    pkl_path = str(path) + ".pkl"
    logger.info(f"Pipeline saved: {pkl_path}")
    return pkl_path

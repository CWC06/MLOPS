"""
Training pipeline for lung cancer risk prediction using PyCaret + MLflow + Hydra.

Run from project root:
    python src/models/train_lung_cancer.py
    python src/models/train_lung_cancer.py tuning.n_iter=20
"""

import os
import warnings
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pycaret.classification import (
    compare_models,
    finalize_model,
    plot_model,
    pull,
    save_model,
    setup,
    tune_model,
)

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="../../configs", config_name="config_lung_cancer")
def train(cfg: DictConfig) -> None:
    """Train the lung cancer risk model.

    Args:
        cfg: Hydra configuration object (configs/config_lung_cancer.yaml)
    """
    # ------------------------------------------------------------------ #
    # MLflow setup                                                         #
    # ------------------------------------------------------------------ #
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.model.experiment_name)

    # ------------------------------------------------------------------ #
    # Load & clean data                                                    #
    # ------------------------------------------------------------------ #
    data_path = Path(cfg.data.raw_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    for col in ["index", "Patient Id"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Normalise feature names so LightGBM does not complain
    df.columns = df.columns.str.replace(" ", "_")

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Target distribution:\n{df[cfg.model.target_column].value_counts()}")

    # ------------------------------------------------------------------ #
    # PyCaret setup                                                        #
    # ------------------------------------------------------------------ #
    logger.info("Initialising PyCaret environment...")
    setup(
        data=df,
        target=cfg.model.target_column,
        train_size=1 - cfg.data.test_size,
        preprocess=True,
        normalize=cfg.preprocessing.normalize,
        feature_selection=cfg.preprocessing.feature_selection,
        remove_multicollinearity=cfg.preprocessing.remove_multicollinearity,
        multicollinearity_threshold=cfg.preprocessing.multicollinearity_threshold,
        bin_numeric_features=(
            list(cfg.preprocessing.bin_numeric_features)
            if cfg.preprocessing.bin_numeric_features
            else None
        ),
        session_id=cfg.data.random_state,
        log_experiment=False,
        log_plots=False,
    )

    # ------------------------------------------------------------------ #
    # Model comparison                                                     #
    # ------------------------------------------------------------------ #
    logger.info("Comparing models...")
    best_models = compare_models(
        include=["lightgbm", "lr"],
        sort=cfg.model.metric,
        n_select=cfg.model.n_select,
        fold=cfg.model.fold,
        cross_validation=True,
    )

    if isinstance(best_models, list):
        best_model = best_models[0]
        logger.info(f"Top {len(best_models)} models selected")
    else:
        best_model = best_models
        logger.info(f"Best model: {type(best_model).__name__}")

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning                                                #
    # ------------------------------------------------------------------ #
    logger.info("Tuning hyperparameters...")
    tuned_model = tune_model(
        best_model,
        optimize=cfg.tuning.optimize,
        n_iter=cfg.tuning.n_iter,
        fold=cfg.model.fold,
        choose_better=True,
    )

    # ------------------------------------------------------------------ #
    # Finalise                                                             #
    # ------------------------------------------------------------------ #
    logger.info("Finalising model on full dataset...")
    final_model = finalize_model(tuned_model)

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    logger.info("Generating evaluation plots...")
    plots_dir = Path(cfg.paths.reports_path)
    plots_dir.mkdir(parents=True, exist_ok=True)
    original_dir = os.getcwd()
    try:
        os.chdir(str(plots_dir))
        for plot_name in ["confusion_matrix", "class_report", "feature"]:
            try:
                plot_model(final_model, plot=plot_name, save=True)
            except Exception as exc:
                logger.warning(f"Could not generate plot '{plot_name}': {exc}")
    finally:
        os.chdir(original_dir)
    logger.info(f"Plots saved to: {plots_dir}")

    # ------------------------------------------------------------------ #
    # Save model locally                                                   #
    # ------------------------------------------------------------------ #
    model_save_path = Path(cfg.paths.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(final_model, str(model_save_path))
    logger.info(f"Model saved to: {model_save_path}.pkl")

    # ------------------------------------------------------------------ #
    # MLflow logging & model registry                                      #
    # ------------------------------------------------------------------ #
    logger.info("Logging model to MLflow...")
    with mlflow.start_run():
        results = pull()
        if not results.empty:
            for metric in results.columns:
                if metric != "Model":
                    try:
                        value = results[metric].iloc[0]
                        if pd.notna(value):
                            mlflow.log_metric(
                                metric.lower().replace(" ", "_"), float(value)
                            )
                    except Exception as exc:
                        logger.warning(f"Could not log metric {metric}: {exc}")

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.model_name,
        )

        client = mlflow.tracking.MlflowClient()
        try:
            latest = client.get_latest_versions(cfg.mlflow.model_name, stages=["None"])
            if latest:
                client.transition_model_version_stage(
                    name=cfg.mlflow.model_name,
                    version=latest[0].version,
                    stage=cfg.mlflow.model_stage,
                )
                logger.info(f"Model moved to stage: {cfg.mlflow.model_stage}")
        except Exception as exc:
            logger.warning(f"Could not transition model stage: {exc}")

    logger.info("Lung cancer training completed successfully!")


if __name__ == "__main__":
    train()

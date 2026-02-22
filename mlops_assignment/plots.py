"""Plotting utilities for model evaluation."""

import os
from pathlib import Path

from loguru import logger
from pycaret.classification import plot_model

from mlops_assignment.config import FIGURES_DIR


def save_evaluation_plots(
    model,
    output_dir: Path | None = None,
    plot_types: list[str] | None = None,
) -> Path:
    """Save static evaluation plots from PyCaret's plot_model.

    Parameters
    ----------
    model : trained PyCaret model (pre-finalization for valid CV plots)
    output_dir : Path
        Directory to save PNGs.
    plot_types : list[str]
        PyCaret plot identifiers to generate.

    Returns
    -------
    Path to the output directory.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_types is None:
        plot_types = ["confusion_matrix", "auc", "pr", "class_report", "feature", "learning"]

    orig_dir = os.getcwd()
    try:
        os.chdir(str(output_dir))
        for name in plot_types:
            logger.info(f"  Saving plot: {name}")
            plot_model(model, plot=name, save=True)
    finally:
        os.chdir(orig_dir)

    logger.info(f"All plots saved to: {output_dir}")
    return output_dir

# Getting Started

## Prerequisites

- Python 3.10 or later
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Git for version control

## Installation

```bash
# Clone the repository
git clone https://github.com/CWC06/MLOPS.git
cd MLOPS

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Train the Models

Both models must be trained before the web application can serve predictions.

```bash
# Train heart disease model
python -c "from mlops_assignment.dataset import load_heart_data; from mlops_assignment.features import create_holdout_split; from mlops_assignment.modeling.train import init_pycaret, train_best_model, save_pipeline; from mlops_assignment.plots import save_evaluation_plots; from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/config_heart.yaml'); df = load_heart_data(); df_train, _ = create_holdout_split(df, target=cfg.model.target_column, holdout_size=cfg.data.holdout_size, random_state=cfg.data.random_state); init_pycaret(df_train, cfg); final, tuned, _, _ = train_best_model(cfg); save_pipeline(final); save_evaluation_plots(tuned)"

# Train lung cancer model
python src/models/train_lung_cancer.py
```

Trained models are saved to the `models/` directory.

## Launch the Web Application

```bash
streamlit run src/webapp/app.py
```

The Streamlit app will open at [http://localhost:8501](http://localhost:8501).

## Run Tests

```bash
python -m pytest tests -v
```

## View Experiment Tracking

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

The MLflow dashboard will open at [http://localhost:5000](http://localhost:5000).

## Quick Commands Reference

| Task | Command |
|---------|-------------|
| Train heart disease model | See training command above |
| Train lung cancer model | `python src/models/train_lung_cancer.py` |
| Launch Streamlit app | `streamlit run src/webapp/app.py` |
| Launch MLflow UI | `mlflow ui --backend-store-uri ./mlruns --port 5000` |
| Run tests | `python -m pytest tests -v` |
| Lint code | `ruff check .` |
| Format code | `ruff check --fix . && ruff format .` |

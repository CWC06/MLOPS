# MLOps Assignment — Integrated Health Risk Prediction System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A fully integrated MLOps pipeline combining **Heart Disease Risk Prediction** (binary) and **Lung Cancer Risk Prediction** (multiclass) into a single project — with a unified Streamlit web application, MLflow experiment tracking, and Hydra configuration management.

---

## Team Members

| Name | Role | Dataset | Responsibilities |
|---|---|---|---|
| **Wei Cong** | Team Leader | Heart Disease (UCI) | Heart disease pipeline, Streamlit webapp, Hydra configs, cloud deployment |
| **Teagan Tham** | Member | Lung Cancer | DVC data version control, CI/CD pipelines, lung cancer pipeline support |

---

## URLs

| Resource | URL |
|---|---|
| Source Code Repository | https://github.com/CWC06/MLOPS |
| Web Application (cloud) | Deployed via Railway — run `streamlit run src/webapp/app.py` locally or deploy with `Dockerfile` |
| Web Application (local) | http://localhost:8501 |
| MLflow Tracking UI (local) | http://localhost:5000 |

---

## Models at a Glance

| Model | Task | Target | Dataset |
|---|---|---|---|
| Heart Disease | Binary Classification | 0 = No Disease · 1 = Heart Disease | `data/raw/heart.csv` (918 patients) |
| Lung Cancer | Multiclass Classification | Low · Medium · High | `data/raw/lung_cancer.csv` (1 000 patients) |

---

## Project Structure

```
MLOPS/
│
├── .github/workflows/               # CI/CD Pipelines
│   ├── ci.yml                       # Lint → Test → Build pipeline
│   └── deploy.yml                   # Auto-deploy to Railway on push
│
├── .dvc/                            # DVC configuration
│   └── config                       # Remote storage settings
├── .dvcignore                       # DVC exclusion rules
├── dvc.yaml                         # Reproducible training pipeline
│
├── .streamlit/
│   └── config.toml                  # Streamlit theme & server config
│
├── configs/
│   ├── config_heart.yaml            # Heart disease pipeline config (Hydra)
│   └── config_lung_cancer.yaml      # Lung cancer pipeline config (Hydra)
│
├── data/
│   ├── raw/
│   │   ├── heart.csv                # UCI Heart Disease dataset
│   │   ├── heart.csv.dvc            # DVC tracking file
│   │   ├── lung_cancer.csv          # Lung Cancer Risk dataset
│   │   └── lung_cancer.csv.dvc      # DVC tracking file
│   ├── processed/                   # Holdout splits (auto-generated)
│   ├── interim/
│   └── external/
│
├── mlops_assignment/                # Heart disease core package
│   ├── config.py                    # Path constants (PROJ_ROOT, DATA_DIR, …)
│   ├── dataset.py                   # Data loading & cleaning
│   ├── features.py                  # Holdout split creation
│   ├── plots.py                     # PyCaret evaluation plot utilities
│   └── modeling/
│       ├── train.py                 # PyCaret training functions
│       └── predict.py               # Holdout evaluation & metrics
│
├── src/
│   ├── models/
│   │   ├── train_lung_cancer.py     # Lung cancer training pipeline (Hydra + MLflow)
│   │   ├── predict_lung_cancer.py   # Lung cancer inference utilities
│   │   ├── predict_heart.py         # Heart disease inference utilities
│   │   └── utils.py                 # Shared data utilities
│   └── webapp/
│       └── app.py                   # Integrated Streamlit application
│
├── models/
│   ├── heart_disease_model.pkl      # Trained heart disease pipeline
│   └── lung_cancer_model.pkl        # Trained lung cancer pipeline
│
├── notebooks/
│   ├── eda_heart_disease.ipynb      # Heart disease EDA (Task 1 — Wei Cong)
│   └── train_heart.ipynb            # Heart disease training notebook (Task 2 — Wei Cong)
│
├── reports/
│   └── figures/                     # Auto-generated evaluation plots
│
├── tests/
│   ├── test_data.py                 # Heart disease data tests
│   ├── test_lung_cancer_data.py     # Lung cancer data tests
│   └── test_model.py               # Model inference tests
│
├── docs/                            # MkDocs documentation
│   ├── mkdocs.yml
│   └── docs/
│       ├── index.md
│       └── getting-started.md
│
├── Dockerfile                       # Container for cloud deployment
├── .dockerignore                    # Docker build exclusions
├── railway.json                     # Railway PaaS deployment config
├── requirements.txt                 # Pip dependencies (exported from Poetry)
├── Makefile                         # Project automation commands
├── pyproject.toml                   # Poetry dependency manifest
├── poetry.lock                      # Locked dependency versions
└── README.md                        # This file
```

---

## Deployment Guide

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/CWC06/MLOPS.git
cd MLOPS

# 2. Install dependencies with Poetry
poetry install
poetry shell

# 3. Train heart disease model
python -c "from mlops_assignment.dataset import load_heart_data; from mlops_assignment.features import create_holdout_split; from mlops_assignment.modeling.train import init_pycaret, train_best_model, save_pipeline; from mlops_assignment.plots import save_evaluation_plots; from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/config_heart.yaml'); df = load_heart_data(); df_train, _ = create_holdout_split(df, target=cfg.model.target_column, holdout_size=cfg.data.holdout_size, random_state=cfg.data.random_state); init_pycaret(df_train, cfg); final, tuned, _, _ = train_best_model(cfg); save_pipeline(final); save_evaluation_plots(tuned)"

# 4. Train lung cancer model
python src/models/train_lung_cancer.py

# 5. Launch the web application
streamlit run src/webapp/app.py
# → opens at http://localhost:8501

# 6. View MLflow experiment tracking
mlflow ui --backend-store-uri ./mlruns --port 5000
# → opens at http://localhost:5000
```

### Cloud Deployment (Railway)

The project includes a `Dockerfile` and `railway.json` for deployment on [Railway](https://railway.com):

1. Push the repository to GitHub
2. Go to [railway.com](https://railway.com) and create a new project
3. Select **Deploy from GitHub repo** and connect `CWC06/MLOPS`
4. Railway auto-detects the `Dockerfile` and deploys the Streamlit app
5. Set the `PORT` environment variable to `8501` in Railway dashboard
6. The app is accessible at the generated Railway public URL

---

## User Guide

### Web Application

The Streamlit webapp provides real-time health risk predictions for two models:

**Selecting a Model:**
- Use the sidebar radio button to switch between "Heart Disease Risk" and "Lung Cancer Risk"

**Single Prediction:**
1. Select the "Single Prediction" tab
2. Fill in all input fields (patient demographics, clinical features)
3. Click "Predict"
4. View the risk result (colour-coded) and confidence probabilities

**Batch Prediction:**
1. Select the "Batch Prediction" tab
2. Upload a CSV file with the correct column headers
3. Preview the uploaded data
4. Click "Run Batch Prediction"
5. Download the results as a CSV file

---

## Configuration

### Heart Disease · `configs/config_heart.yaml`

| Key | Default | Notes |
|---|---|---|
| `data.test_size` | 0.2 | PyCaret internal validation split |
| `data.holdout_size` | 0.10 | Final holdout (stratified, outside PyCaret) |
| `model.target_column` | `HeartDisease` | Binary target |
| `model.metric` | `AUC` | Robust to class imbalance |
| `model.fold` | 5 | Stratified CV folds |
| `preprocessing.normalize_method` | `zscore` | Feature normalisation |
| `preprocessing.multicollinearity_threshold` | 0.85 | Correlation removal |
| `tuning.n_iter` | 30 | Random search iterations |

### Lung Cancer · `configs/config_lung_cancer.yaml`

| Key | Default | Notes |
|---|---|---|
| `data.test_size` | 0.2 | PyCaret internal validation split |
| `model.target_column` | `Level` | Low / Medium / High |
| `model.metric` | `Accuracy` | Balanced multiclass metric |
| `model.fold` | 5 | Stratified CV folds |
| `preprocessing.multicollinearity_threshold` | 0.95 | Correlation removal |
| `tuning.n_iter` | 10 | Random search iterations |

**Override at runtime (Hydra):**
```bash
python src/models/train_lung_cancer.py tuning.n_iter=20 model.fold=10
```

---

## Pipeline Architecture

### Heart Disease

```
heart.csv
  ↓ load_heart_data()          — drop admin cols, impute zero Cholesterol/RestingBP
  ↓ create_holdout_split()     — stratified 90/10 train/holdout split
  ↓ init_pycaret()             — normalise, feature select, bin, interaction features
  ↓ train_best_model()         — compare 7 models → tune best → finalize
  ↓ save_pipeline()            — models/heart_disease_model.pkl
  ↓ save_evaluation_plots()    — confusion matrix, AUC, PR curve, feature importance
```

### Lung Cancer

```
lung_cancer.csv
  ↓ PyCaret setup()            — normalise, feature selection, multicollinearity removal
  ↓ compare_models()           — LightGBM vs Logistic Regression → top 3
  ↓ tune_model()               — hyperparameter optimisation (10 iterations)
  ↓ finalize_model()           — train on full dataset
  ↓ save_model()               — models/lung_cancer_model.pkl
  ↓ MLflow log_model()         — register & promote to staging
```

---

## MLOps Lifecycle Tools

| Tool | Role | Integration |
|---|---|---|
| [Poetry](https://python-poetry.org/) | Dependency management | `pyproject.toml` + `poetry.lock` |
| [Hydra](https://hydra.cc/) | Config management | `configs/*.yaml`, CLI overrides |
| [DVC](https://dvc.org/) | Data version control | `dvc.yaml`, `data/raw/*.dvc` |
| [MLflow](https://mlflow.org/) | Experiment tracking & model registry | `mlruns/`, model staging |
| [PyCaret](https://pycaret.org/) | AutoML pipeline | Training, preprocessing, model selection |
| [Streamlit](https://streamlit.io/) | Web application | `src/webapp/app.py` |
| [GitHub Actions](https://github.com/features/actions) | CI/CD | `.github/workflows/ci.yml` |
| [Docker](https://www.docker.com/) | Containerisation | `Dockerfile` for cloud deployment |
| [Railway](https://railway.com/) | PaaS deployment | `railway.json` + `Dockerfile` |
| [Loguru](https://loguru.readthedocs.io/) | Structured logging | Application-wide logging |

---

## Quick Commands Reference

| Task | Command |
|---|---|
| Train heart disease model | `python -c "from mlops_assignment.dataset import load_heart_data; ..."` (see Deployment Guide above) |
| Train lung cancer model | `python src/models/train_lung_cancer.py` |
| Launch Streamlit app | `streamlit run src/webapp/app.py` |
| Launch MLflow UI | `mlflow ui --backend-store-uri ./mlruns --port 5000` |
| Run tests | `python -m pytest tests -v` |
| Lint code | `ruff check .` |
| Format code | `ruff check --fix . && ruff format .` |

---

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
│   └── deploy.yml                   # Auto-deploy to Render on push
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
│   └── train_heart.ipynb            # Heart disease EDA + training notebook
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
├── render.yaml                      # Render PaaS deployment config
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

# 3. Train both models
make train-all

# 4. Launch the web application
make webapp
# → opens at http://localhost:8501

# 5. View MLflow experiment tracking
make mlflow-ui
# → opens at http://localhost:5000
```

### Cloud Deployment (Render)

The project includes a `Dockerfile` and `render.yaml` for one-click deployment:

1. Push the repository to GitHub
2. Connect the repo to [Render](https://render.com)
3. Render auto-detects `render.yaml` and deploys the Streamlit app
4. The app runs in a Docker container with all dependencies pre-installed

### Cloud Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the GitHub repo and set the main file path to `src/webapp/app.py`
4. The app deploys automatically

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
| [Render](https://render.com/) | PaaS deployment | `render.yaml` |
| [Loguru](https://loguru.readthedocs.io/) | Structured logging | Application-wide logging |

---

## Make Commands

| Command | Description |
|---|---|
| `make train-heart` | Train heart disease model |
| `make train-lung` | Train lung cancer model |
| `make train-all` | Train both models |
| `make webapp` | Launch Streamlit app (localhost:8501) |
| `make mlflow-ui` | Launch MLflow tracking UI (localhost:5000) |
| `make test` | Run pytest suite |
| `make lint` | Lint with ruff |
| `make format` | Format with ruff |
| `make clean` | Remove Python caches |
| `make help` | Show all commands |

---

## Data Dictionary

### Heart Disease (`heart.csv`)

| Column | Type | Description |
|---|---|---|
| Age | int | Age in years |
| Sex | str | M / F |
| ChestPainType | str | ATA · NAP · ASY · TA |
| RestingBP | int | Resting blood pressure (mmHg) |
| Cholesterol | int | Serum cholesterol (mg/dL) |
| FastingBS | int | Fasting blood sugar > 120 mg/dL (0/1) |
| RestingECG | str | Normal · ST · LVH |
| MaxHR | int | Maximum heart rate achieved |
| ExerciseAngina | str | Y / N |
| Oldpeak | float | ST depression (exercise vs. rest) |
| ST_Slope | str | Up · Flat · Down |
| **HeartDisease** | int | **Target: 0 = No, 1 = Yes** |

### Lung Cancer (`lung_cancer.csv`)

23 risk-factor features scored on a 1–9 scale (Age and Gender use natural scales).
**Target**: `Level` — Low · Medium · High cancer risk.

---

## Troubleshooting

**`FileNotFoundError: model not found`**
Train the relevant model first (`make train-heart` or `make train-lung`).

**LightGBM feature name warnings**
Handled automatically — column names are normalised (spaces → underscores) before training and inference.

**Hydra writes to `outputs/`**
Hydra outputs training logs to `outputs/YYYY-MM-DD/HH-MM-SS/`. Always run training scripts from the project root.

**MLflow UI shows no experiments**
Run at least one training script first to populate `mlruns/`.

---

## License

MIT

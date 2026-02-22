# MLOps Assignment — Integrated Health Risk Prediction System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A fully integrated MLOps pipeline combining **Heart Disease Risk Prediction** (binary) and **Lung Cancer Risk Prediction** (multiclass) into a single project — with a unified Streamlit web application, MLflow experiment tracking, and Hydra configuration management.

---

## Models at a Glance

| Model | Task | Target | Dataset |
|---|---|---|---|
| Heart Disease | Binary Classification | 0 = No Disease · 1 = Heart Disease | `data/raw/heart.csv` (918 patients) |
| Lung Cancer | Multiclass Classification | Low · Medium · High | `data/raw/lung_cancer.csv` (1 000 patients) |

---

## Project Structure

```
mlops-assignment/
│
├── configs/
│   ├── config_heart.yaml          # Heart disease pipeline config (Hydra)
│   └── config_lung_cancer.yaml    # Lung cancer pipeline config (Hydra)
│
├── data/
│   ├── raw/
│   │   ├── heart.csv              # UCI Heart Disease dataset
│   │   └── lung_cancer.csv        # Lung Cancer Risk dataset
│   ├── processed/                 # Holdout splits (auto-generated)
│   ├── interim/
│   └── external/
│
├── mlops_assignment/              # Heart disease core package
│   ├── config.py                  # Path constants (PROJ_ROOT, DATA_DIR, …)
│   ├── dataset.py                 # Data loading & cleaning
│   ├── features.py                # Holdout split creation
│   ├── plots.py                   # PyCaret evaluation plot utilities
│   └── modeling/
│       ├── train.py               # PyCaret training functions
│       └── predict.py             # Holdout evaluation & metrics
│
├── src/
│   ├── models/
│   │   ├── train_lung_cancer.py   # Lung cancer training pipeline (Hydra)
│   │   ├── predict_lung_cancer.py # Lung cancer inference utilities
│   │   ├── predict_heart.py       # Heart disease inference utilities
│   │   └── utils.py               # Shared data utilities
│   └── webapp/
│       └── app.py                 # Integrated Streamlit application
│
├── models/
│   ├── heart_disease_model.pkl    # Trained heart disease pipeline
│   └── lung_cancer_model.pkl      # Trained lung cancer pipeline
│
├── notebooks/
│   └── train_heart.ipynb          # Heart disease training notebook
│
├── reports/
│   └── figures/                   # Auto-generated evaluation plots
│
├── tests/
│   ├── test_data.py               # Heart disease data tests
│   └── test_lung_cancer_data.py   # Lung cancer data tests
│
├── mlruns/                        # MLflow experiment tracking
├── Makefile                       # Project automation
├── pyproject.toml                 # Poetry dependencies
└── README.md
```

---

## Quick Start

### 1 · Install dependencies

```bash
poetry install
poetry shell
```

### 2 · Train the models

```bash
# Heart Disease model
make train-heart

# Lung Cancer model
make train-lung

# Both at once
make train-all
```

### 3 · Launch the web app

```bash
make webapp
# → http://localhost:8501
```

### 4 · View MLflow experiments

```bash
make mlflow-ui
# → http://localhost:5000
```

### 5 · Run tests

```bash
make test
```

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
  ↓ MLflow log_model()         — register & promote to Staging
```

---

## Web Application (`src/webapp/app.py`)

The Streamlit app provides a unified interface for both models:

| Feature | Details |
|---|---|
| **Model toggle** | Sidebar radio button — Heart Disease or Lung Cancer |
| **Single prediction** | Form inputs → real model inference → colour-coded risk result |
| **Batch prediction** | Upload CSV → predictions on all rows → download as CSV |
| **Confidence scores** | Per-class probability table shown with each prediction |
| **Risk colours** | Red (High/Disease) · Orange (Medium) · Green (Low/No Disease) |

---

## Candidate Models

### Heart Disease (sorted by AUC)
LightGBM · Random Forest · XGBoost · Logistic Regression · Naive Bayes · Decision Tree · SVM

### Lung Cancer (sorted by Accuracy)
LightGBM · Logistic Regression

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

## Make Commands

| Command | Description |
|---|---|
| `make requirements` | Install package with pip |
| `make train-heart` | Train heart disease model |
| `make train-lung` | Train lung cancer model |
| `make train-all` | Train both models |
| `make webapp` | Launch Streamlit app |
| `make mlflow-ui` | Launch MLflow tracking UI |
| `make test` | Run pytest suite |
| `make lint` | Lint with ruff |
| `make format` | Format with ruff |
| `make clean` | Remove Python caches |
| `make help` | Show all commands |

---

## Tech Stack

| Tool | Role |
|---|---|
| [PyCaret](https://pycaret.org/) | AutoML — compare, tune, preprocess |
| [MLflow](https://mlflow.org/) | Experiment tracking & model registry |
| [Hydra](https://hydra.cc/) | Config management & CLI overrides |
| [Streamlit](https://streamlit.io/) | Interactive web application |
| [DVC](https://dvc.org/) | Data version control |
| [Evidently](https://www.evidentlyai.com/) | Model monitoring |
| [Loguru](https://loguru.readthedocs.io/) | Structured logging |
| [Poetry](https://python-poetry.org/) | Dependency management |

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

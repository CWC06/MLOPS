# PowerPoint Slide Outline — 11 Slides

---

## Slide 1: Title
- **Title:** Health Risk Prediction System — MLOps Lifecycle
- **Subtitle:** IT3385 Machine Learning Operations — AY2025 Semester 2
- **Names:** Wei Cong (Leader) · Teagan Tham
- **GitHub:** github.com/CWC06/MLOPS

---

## Slide 2: Project Overview
- Two integrated ML models:
  - Heart Disease — Binary Classification (918 patients, 11 features)
  - Lung Cancer — Multiclass Classification (1000 patients, 23 features)
- Unified Streamlit webapp for both models
- Full MLOps lifecycle: EDA → Training → Deployment → Monitoring

---

## Slide 3: Project Structure
- Screenshot or diagram of folder tree
- Highlight key directories: configs/, src/, models/, tests/, .github/
- Cookiecutter Data Science template

---

## Slide 4: Task 1 — Exploratory Data Analysis
- **Left half:** Heart Disease EDA (Wei Cong)
  - Distribution plots, correlation heatmap, class balance
  - Key findings: zero-value imputation needed, ST_Slope most predictive
- **Right half:** Lung Cancer EDA (Teagan Tham)
  - Risk factor distributions, target balance, feature correlations
  - Key findings: balanced classes, features on 1-9 scale

---

## Slide 5: Task 2 — ML Pipeline (PyCaret)
- **Pipeline diagram:**
  - Data → Preprocessing → Compare Models → Tune → Finalize → Save
- **Heart Disease:** AUC metric, 5-fold CV, 30 tuning iterations, z-score normalization
- **Lung Cancer:** Accuracy metric, 5-fold CV, 10 tuning iterations
- Both: Feature selection, multicollinearity removal, binning
- MLflow experiment logging and model registration

---

## Slide 6: MLflow Experiment Tracking
- Screenshot of MLflow UI showing:
  - Experiment runs list
  - Metrics comparison (AUC, Accuracy, F1)
  - Model registry with staging promotion
- *LIVE DEMO during video*

---

## Slide 7: Task 3 — Web Application
- Screenshot of Streamlit app (both models)
- Features:
  - Single prediction with confidence scores
  - Batch prediction with CSV upload/download
  - Colour-coded risk output (Red/Orange/Green)
  - Sidebar model selection
- *LIVE DEMO during video — show both models, single + batch*

---

## Slide 8: Task 4 — MLOps Lifecycle Tools
- **Table layout:**

  | Tool    | Purpose                        |
  |---------|--------------------------------|
  | Poetry  | Dependency management          |
  | Hydra   | Configuration (no hard-coding) |
  | DVC     | Data version control           |
  | MLflow  | Experiment tracking & registry |
  | Git     | Feature branching workflow      |
  | GitHub Actions | CI/CD pipeline           |
  | Docker  | Containerisation               |
  | Railway | Cloud PaaS deployment          |

---

## Slide 9: Git Branching & CI/CD
- Screenshot of `git log --graph` showing:
  - feature/dvc-setup (Teagan Tham)
  - feature/ci-cd (Teagan Tham)
  - feature/cloud-deployment (Wei Cong)
  - All merged into main
- Screenshot of GitHub Actions CI pipeline
- Pipeline: Lint → Test → Build → Deploy

---

## Slide 10: Testing
- 3 test files:
  - test_data.py — heart disease data validation
  - test_lung_cancer_data.py — lung cancer data validation
  - test_model.py — model loading and prediction
- Screenshot of `python -m pytest tests -v` passing
- *RUN LIVE during video*

---

## Slide 11: Summary & Thank You
- Architecture diagram:
  ```
  Data (DVC) → Training (PyCaret + Hydra + MLflow) → App (Streamlit)
       ↑                                                    ↓
  Git Branching ←── CI/CD (GitHub Actions) ←── Docker (Railway)
  ```
- GitHub: github.com/CWC06/MLOPS
- "Thank you for watching"

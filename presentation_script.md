# Presentation Script — 15-Minute Video Demo

## Team: Wei Cong (Leader) + Teagan Tham
## Total Duration: ~15 minutes

---

## SLIDE 1 — Title Slide (0:00 – 0:30) [Wei Cong]

**Say:**
> "Hi, we are Wei Cong and Teagan Tham. Today we will be presenting our
> integrated Health Risk Prediction System built using the MLOps lifecycle.
> Our project combines two machine learning models — Heart Disease binary
> classification and Lung Cancer multiclass classification — into a single
> production-ready application."

---

## SLIDE 2 — Project Overview (0:30 – 1:30) [Wei Cong]

**Show:** Models at a Glance table

**Say:**
> "Our project has two models. I worked on the Heart Disease model using the
> UCI Heart Failure dataset — 918 patients, 11 clinical features, predicting
> whether a patient has heart disease or not. Teagan worked on the Lung Cancer
> model using the Air Pollution and Cancer Risk dataset — 1,000 patients,
> 23 risk factor features, predicting cancer risk as Low, Medium, or High."

---

## SLIDE 3 — Project Structure & Folder Layout (1:30 – 2:30) [Teagan Tham]

**Show:** Folder structure from README (screenshot or slide)

**Say:**
> "We followed the Cookiecutter Data Science project template as our standard
> ML project folder structure. Here's how our repo is organized:
> - `configs/` holds our Hydra YAML configuration files
> - `data/raw/` holds our raw datasets, tracked by DVC
> - `mlops_assignment/` is the heart disease core package
> - `src/models/` contains the lung cancer pipeline and shared prediction modules
> - `src/webapp/` is our integrated Streamlit web application
> - `models/` stores trained model artifacts
> - `tests/` has our automated test suite
> - `.github/workflows/` has our CI/CD pipelines"

---

## SLIDE 4 — Task 1: EDA (2:30 – 4:00) [Split — 45s each]

### Wei Cong — Heart Disease EDA (2:30 – 3:15)

**Show:** Open the training notebook `notebooks/train_heart.ipynb` (scroll through EDA cells)

**Say:**
> "For my EDA, I explored the heart disease dataset looking at distributions
> of all features, correlation heatmaps, and the class balance of the target
> variable. I identified that cholesterol and resting blood pressure had zero
> values that needed imputation, and that features like ST_Slope and
> ChestPainType had strong predictive power."

### Teagan Tham — Lung Cancer EDA (3:15 – 4:00)

**Show:** Lung cancer EDA notebook or relevant visualizations

**Say:**
> "For the lung cancer dataset, I analysed the distribution of all 23 risk
> factors on a 1–9 scale, checked for missing values and outliers, examined
> correlations between features like Smoking and Passive Smoker, and verified
> the target class balance across Low, Medium, and High risk levels."

---

## SLIDE 5 — Task 2: ML Pipeline with PyCaret (4:00 – 6:30) [Split — ~75s each]

### Wei Cong — Heart Disease Pipeline (4:00 – 5:15)

**Show:** Training notebook or terminal running `python training script`

**Say:**
> "For the heart disease model, I used PyCaret to build a full ML pipeline.
> Let me walk through the key steps:
>
> 1. **Preprocessing**: I configured z-score normalization, univariate feature
>    selection, multicollinearity removal at 0.85 threshold, and binning of
>    continuous features like Age, Cholesterol, RestingBP, and MaxHR.
>
> 2. **Model comparison**: PyCaret's `compare_models()` evaluated 7 algorithms
>    using 5-fold cross-validation. The primary metric was AUC.
>
> 3. **Hyperparameter tuning**: I tuned the top model with 30 random search
>    iterations optimizing AUC.
>
> 4. **Evaluation**: I used `plot_model()` to generate confusion matrix, ROC
>    curve, precision-recall curve, and feature importance plots.
>
> 5. **Save & register**: The final pipeline was saved as a pickle file and
>    registered with MLflow."

### Teagan Tham — Lung Cancer Pipeline (5:15 – 6:30)

**Show:** Terminal running `python src/models/train_lung_cancer.py` or training code

**Say:**
> "For the lung cancer model, the pipeline is similar but configured for
> multiclass classification:
>
> 1. **Preprocessing**: Normalization, feature selection, multicollinearity
>    removal at 0.95 threshold, and Age binning.
>
> 2. **Model comparison**: PyCaret compared models using Accuracy as the primary
>    metric with 5-fold stratified cross-validation.
>
> 3. **Tuning**: Top model tuned with 10 random search iterations.
>
> 4. **MLflow integration**: The training script uses Hydra's `@hydra.main()`
>    decorator for config management, and logs all metrics, parameters, and the
>    final model to MLflow. The model is registered and promoted to staging."

---

## SLIDE 6 — MLflow Demo (6:30 – 7:30) [Teagan Tham]

**Show:** LIVE DEMO — run `mlflow ui --backend-store-uri ./mlruns --port 5000`, open browser at localhost:5000

**Say:**
> "Let me show the MLflow tracking UI. Here you can see our experiment runs,
> with logged metrics like accuracy, AUC, and F1 score. Each run captures
> the hyperparameters, the model artifact, and the preprocessing pipeline.
> We can compare runs side by side. The best model has been registered in the
> model registry and promoted to staging."

**Demo actions:**
1. Show experiment list
2. Click into a run — show metrics, parameters, artifacts
3. Show the model registry (if populated)

---

## SLIDE 7 — Task 3: Web Application Demo (7:30 – 10:00) [Wei Cong]

**Show:** LIVE DEMO — run `streamlit run src/webapp/app.py`, open browser at localhost:8501

**Say:**
> "Now let me demonstrate our integrated Streamlit web application."

### Demo Flow (walk through each step):

**Heart Disease — Single Prediction (7:30 – 8:30)**
> "I'll select Heart Disease Risk from the sidebar. Here's the input form with
> all 11 clinical features — Age, Sex, Chest Pain Type, Resting BP, and so on.
> Let me fill in sample values for a 55-year-old male with atypical angina...
> [click Predict] ... The model predicts No Heart Disease with 82% confidence.
> You can see the colour-coded result — green for low risk — and the probability
> breakdown for each class."

**Heart Disease — Batch Prediction (8:30 – 9:00)**
> "For batch prediction, I upload a CSV file with multiple patients. The app
> previews the data, processes all rows, and generates downloadable results
> with predictions appended."

**Lung Cancer — Single Prediction (9:00 – 9:30)**
> "Switching to Lung Cancer Risk — the form has 23 risk factors on slider
> scales from 1 to 9. Let me set some high-risk values... [click Predict]
> ... The model predicts High risk level. Notice the red colour coding and
> the recommendation to seek medical evaluation."

**Lung Cancer — Batch Prediction (9:30 – 10:00)**
> "Same batch capability here — upload, predict, download. The app is fully
> integrated with a unified design across both models."

---

## SLIDE 8 — Task 4: MLOps Lifecycle (10:00 – 13:00) [Teagan Tham]

### Poetry (10:00 – 10:30)

**Show:** `pyproject.toml` in editor

> "For dependency management, we use Poetry. Our `pyproject.toml` defines all
> project dependencies with version constraints, and `poetry.lock` pins exact
> versions for reproducible installs."

### Hydra (10:30 – 11:00)

**Show:** `configs/config_heart.yaml` and `configs/config_lung_cancer.yaml`

> "Hydra manages all our configuration. Instead of hard-coding hyperparameters,
> data paths, and MLflow settings, everything lives in YAML config files. You
> can override any parameter from the command line — for example:
> `python src/models/train_lung_cancer.py tuning.n_iter=20`"

### DVC (11:00 – 11:30)

**Show:** `dvc.yaml`, `data/raw/heart.csv.dvc`

> "DVC handles data version control. Our raw datasets are tracked with DVC —
> each CSV has a `.dvc` file containing its MD5 hash. The `dvc.yaml` defines
> a reproducible training pipeline with dependency tracking. If any source file
> or data changes, DVC knows which stages need to be re-run."

### Git Branching (11:30 – 12:00)

**Show:** `git log --oneline --all --graph` in terminal

> "For source code version control, we use Git with feature branching. Here's
> our branch history — you can see feature branches for DVC setup, CI/CD, and
> cloud deployment, all merged into main with merge commits. This simulates
> a real collaborative workflow where each team member works on their own
> branch and merges via pull requests."

### CI/CD (12:00 – 12:30)

**Show:** `.github/workflows/ci.yml` in editor, then GitHub Actions tab

> "We have two GitHub Actions workflows. The CI pipeline runs on every push —
> it installs dependencies with Poetry, lints with ruff, runs our pytest test
> suite, and builds the Docker image. The deploy workflow triggers on push to
> main and deploys to Railway automatically."

### Docker & Deployment (12:30 – 13:00)

**Show:** `Dockerfile`, `railway.json`

> "For cloud deployment, we containerized the app with Docker. The Dockerfile
> uses Python 3.10 slim, installs dependencies from requirements.txt, copies
> the project files, and runs Streamlit. The railway.json configures automatic
> deployment on Railway with health checks."

---

## SLIDE 9 — Testing (13:00 – 13:30) [Wei Cong]

**Show:** Terminal running `python -m pytest tests -v`

**Say:**
> "Our test suite covers three areas:
> 1. Heart disease data validation — checks the CSV exists, has correct columns,
>    binary target values, and no missing data
> 2. Lung cancer data validation — same checks plus verifying features are in
>    the 1–9 scale range
> 3. Model inference tests — loads trained models and verifies predictions are
>    valid classes with probability outputs
>
> [Run tests live] All tests passing."

---

## SLIDE 10 — Summary & Architecture Diagram (13:30 – 14:30) [Wei Cong]

**Show:** Architecture diagram slide

**Say:**
> "To summarize our MLOps lifecycle:
>
> - **Data Layer**: Raw data tracked by DVC, validated by automated tests
> - **Training Layer**: PyCaret pipelines configured by Hydra, tracked by MLflow
> - **Serving Layer**: Streamlit webapp with real-time single and batch prediction
> - **DevOps Layer**: Poetry for deps, Git branching for collaboration,
>   GitHub Actions for CI/CD, Docker + Railway for deployment
>
> All these tools are properly integrated — a code change triggers CI, tests
> run automatically, and the app can deploy with a single push to main."

---

## SLIDE 11 — Closing (14:30 – 15:00) [Teagan Tham]

**Say:**
> "Thank you for watching our presentation. Our source code is available on
> GitHub at github.com/CWC06/MLOPS. We believe our project demonstrates a
> complete MLOps lifecycle from data exploration to production deployment.
> Thank you."

---

## PRESENTATION TIPS

1. **Practice the timing** — each section has a time budget. Use a stopwatch.
2. **Have the apps pre-loaded** — start `streamlit run src/webapp/app.py` and `mlflow ui --backend-store-uri ./mlruns --port 5000` before recording.
3. **Pre-fill some form inputs** — don't waste time typing during the demo.
4. **Show the terminal** — assessors want to see real commands running.
5. **Switch speakers cleanly** — state who's speaking at each transition.
6. **Keep slides minimal** — bullet points only, let the demo speak for itself.
7. **Show the GitHub repo** — open it in browser briefly to show branches and commits.

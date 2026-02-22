#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-assignment
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                       #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .

## Delete all compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to auto-fix)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run all tests
.PHONY: test
test:
	python -m pytest tests -v

## Set up conda environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# DATA                                                                           #
#################################################################################

## Process and clean raw datasets
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mlops_assignment/dataset.py

#################################################################################
# TRAINING                                                                       #
#################################################################################

## Train heart disease risk model (runs training notebook via nbconvert)
.PHONY: train-heart
train-heart:
	@echo ">>> Training Heart Disease model..."
	$(PYTHON_INTERPRETER) -c "\
from mlops_assignment.dataset import load_heart_data; \
from mlops_assignment.features import create_holdout_split; \
from mlops_assignment.modeling.train import init_pycaret, train_best_model, save_pipeline; \
from mlops_assignment.plots import save_evaluation_plots; \
from omegaconf import OmegaConf; \
cfg = OmegaConf.load('configs/config_heart.yaml'); \
df = load_heart_data(); \
df_train, _ = create_holdout_split(df, target=cfg.model.target_column, holdout_size=cfg.data.holdout_size, random_state=cfg.data.random_state); \
init_pycaret(df_train, cfg); \
final, tuned, _, _ = train_best_model(cfg); \
save_pipeline(final); \
save_evaluation_plots(tuned); \
print('Heart disease model trained and saved.')"

## Train lung cancer risk model (Hydra-configured)
.PHONY: train-lung
train-lung:
	@echo ">>> Training Lung Cancer model..."
	$(PYTHON_INTERPRETER) src/models/train_lung_cancer.py

## Train both models
.PHONY: train-all
train-all: train-heart train-lung
	@echo ">>> All models trained."

#################################################################################
# WEBAPP                                                                         #
#################################################################################

## Launch the Streamlit web application
.PHONY: webapp
webapp:
	streamlit run src/webapp/app.py

#################################################################################
# MLFLOW                                                                         #
#################################################################################

## Launch MLflow tracking UI
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

#################################################################################
# Self Documenting Commands                                                      #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

# Getting Started

## Prerequisites

- Python 3.10 or 3.11
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
# Train both models (heart disease + lung cancer)
make train-all

# Or train individually
make train-heart
make train-lung
```

Trained models are saved to the `models/` directory.

## Launch the Web Application

```bash
make webapp
```

The Streamlit app will open at [http://localhost:8501](http://localhost:8501).

## Run Tests

```bash
make test
```

## View Experiment Tracking

```bash
make mlflow-ui
```

The MLflow dashboard will open at [http://localhost:5000](http://localhost:5000).

## Project Commands Reference

| Command | Description |
|---------|-------------|
| `make train-heart` | Train heart disease model |
| `make train-lung` | Train lung cancer model |
| `make train-all` | Train both models |
| `make webapp` | Launch Streamlit app |
| `make mlflow-ui` | Launch MLflow UI |
| `make test` | Run pytest suite |
| `make lint` | Lint code with ruff |
| `make format` | Format code with ruff |

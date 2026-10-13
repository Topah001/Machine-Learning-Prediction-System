# Machine Learning Prediction System

A reproducible Iris flower classifier with a command-line evaluation and prediction interface.

## Requirements and setup

Python 3.10 or newer is recommended. The classifier and tests use only the Python standard library.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The only input data is the included, canonical Iris dataset (`ml_prediction/data/iris.csv`, 150 records) from the UCI Machine Learning Repository, a well-known public sample dataset (Dua and Graff, 2019, https://archive.ics.uci.edu/dataset/53/iris). No private, user-supplied, or arbitrary data is downloaded or read. A deterministic, stratified split trains a standardized nearest-centroid classifier.

## Evaluate and predict

```powershell
python -m ml_prediction
python -m ml_prediction predict 5.1 3.5 1.4 0.2
```

The default evaluation uses a stratified 25% test split with a fixed seed, accuracy, and macro-averaged precision. Change the seed or test fraction with `--seed` and `--test-size`. Prediction accepts four Iris measurements in centimeters.

## Tests

```powershell
python -m pytest
```

This is a compact educational example, not a production model or a claim about performance on other datasets.

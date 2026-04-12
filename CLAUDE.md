# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether SHAP/LIME explanations degrade visibly under data poisoning attacks, or whether a model can be corrupted while its explanations still appear coherent. The dataset is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`data/raw/creditcard.csv`), which must be downloaded separately via Kaggle before running any code.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest tests/test_data.py
```

**Run a single test:**
```bash
pytest tests/test_data.py::test_poison_label_flip_rate
```

**Lint and format:**
```bash
ruff check src tests
ruff format src tests
```

## Architecture

The project is a pipeline with four modules in `src/xai_poison/`, each owned by a separate team member and built in dependency order:

1. **`data.py`** — dataset loading, preprocessing (`StandardScaler` on `Amount`/`Time`), train/test split, and poisoning attacks (`poison_label_flip`, `poison_feature_perturbation`). All other modules import from here.

2. **`model.py`** *(to be built)* — XGBoost and Random Forest training on clean vs. poisoned datasets from `data.py`. Saves/loads trained models, logs AUC and F1.

3. **`explainer.py`** *(to be built)* — SHAP `TreeExplainer` and LIME tabular wrappers around trained models from `model.py`. Saves explanation outputs as CSVs.

4. **`metrics.py`** *(to be built)* — faithfulness metrics (Spearman rank correlation, top-k feature overlap, explanation stability) computed on outputs from `explainer.py`. Produces result plots.

**Key design rule:** each module only imports from modules earlier in the pipeline (e.g., `explainer.py` imports from both `model.py` and `data.py`, but `data.py` imports from neither).

## Data

The raw dataset (`data/raw/creditcard.csv`) is not committed. Download it via:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Collaboration

Team members work on separate branches (not feature branches) and each owns one `src/xai_poison/*.py` + `tests/test_*.py` pair.

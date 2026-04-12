# XAI Poisoning Research

Investigating whether SHAP/LIME explanations degrade visibly under data poisoning attacks, or whether a model can be silently corrupted while its explanations still appear coherent.

## Problem

Post-hoc explanation methods like SHAP and LIME are trusted by practitioners to reflect what a model has learned. But this trust assumes clean training data. When data is poisoned — through label flipping, feature perturbation, or systematic labeling bias — what happens to the explanations?

Current XAI research evaluates faithfulness exclusively on clean data. This project empirically characterizes how explanation faithfulness degrades under poisoning, using credit card fraud detection as the test domain.

## Dataset

Kaggle Credit Card Fraud Detection dataset (~284k transactions, ~2% fraud). Download before running anything:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Pipeline

Four modules built in dependency order:

| Module | Responsibility | Status |
|---|---|---|
| `data.py` | Load, preprocess, split, poison attacks | Done |
| `model.py` | Train XGBoost/RF on clean vs. poisoned data, log AUC/F1 | Done |
| `explainer.py` | SHAP TreeExplainer + LIME wrappers, save outputs as CSV | Done |
| `metrics.py` | Spearman correlation, top-k overlap, stability scores, plots | Not started |

Each module only imports from modules earlier in the pipeline.

### Poisoning Attacks (in `data.py`)

- **Label flip** — flips fraud labels (1→0) among minority class at a given `poison_rate`
- **Feature perturbation** — adds Gaussian noise to all features for a random subset of samples at a given `poison_rate`

### Model Training (in `model.py`)

`ModelTrainer` class wraps XGBoost and Random Forest training. Trained models are saved as `.pkl` files under `models/`, named by poison type and rate (e.g. `xgb_label_flip_0.1.pkl`). Metrics (AUC, F1) are logged and saved to `results/training_results.csv`.

### Explainability (in `explainer.py`)

Loads all `.pkl` models from `models/` and runs:
- **SHAP** (`TreeExplainer`) — saves per-sample feature contributions to `results/shap/shap_<model>.csv`
- **LIME** (`LimeTabularExplainer`) — explains first 50 test samples, saves to `results/lime/lime_<model>.csv`

## How to Run

**Train models (clean + poisoned):**
```bash
python train_models.py
```

**Generate SHAP + LIME explanations:**
```bash
python src/xai_poison/explainer.py
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_data.py

# Run a single test
pytest tests/test_data.py::test_poison_label_flip_rate

# Lint / format
ruff check src tests
ruff format src tests
```

## Output Structure

```
models/                          # saved .pkl model files
results/
├── training_results.csv         # AUC + F1 per model/poison config
├── shap/                        # SHAP values per model
└── lime/                        # LIME explanations per model
```

## Team

4 members, each on their own branch.

| Member | Files |
|---|---|
| Member 1 | `src/xai_poison/data.py` + `tests/test_data.py` |
| Member 2 | `src/xai_poison/model.py` + `tests/test_model.py` |
| Member 3 | `src/xai_poison/explainer.py` + `tests/test_explainer.py` |
| Member 4 | `src/xai_poison/metrics.py` + `tests/test_metrics.py` |

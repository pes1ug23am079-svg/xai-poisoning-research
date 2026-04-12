<<<<<<< HEAD
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
| `model.py` | Train XGBoost/RF on clean vs. poisoned data, log AUC/F1 | Not started |
| `explainer.py` | SHAP TreeExplainer + LIME wrappers, save outputs as CSV | Not started |
| `metrics.py` | Spearman correlation, top-k overlap, stability scores, plots | Not started |

Each module only imports from modules earlier in the pipeline.

### Poisoning Attacks (in `data.py`)

- **Label flip** — flips fraud labels (1→0) among minority class at a given `poison_rate`
- **Feature perturbation** — adds Gaussian noise to all features for a random subset of samples at a given `poison_rate`

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

## Team

4 members, each on their own branch.

| Member | Files |
|---|---|
| Member 1 | `src/xai_poison/data.py` + `tests/test_data.py` |
| Member 2 | `src/xai_poison/model.py` + `tests/test_model.py` |
| Member 3 | `src/xai_poison/explainer.py` + `tests/test_explainer.py` |
| Member 4 | `src/xai_poison/metrics.py` + `tests/test_metrics.py` |
=======
# 🧪 XAI Poison Project

## 📌 Overview
This project explores the impact of **data poisoning attacks** on machine learning models and their **explainability**.

While models may appear to perform well (high AUC/F1), we investigate whether:
- their **internal reasoning changes**
- their **feature importance becomes unreliable**
- explanations from SHAP and LIME remain trustworthy

---

## 🎯 Objective
To analyze how **data poisoning affects model explanations**, even when traditional performance metrics remain stable.

---

## ⚙️ Features

### 🔹 Data Pipeline
- Load Credit Card Fraud dataset
- Preprocess features (scaling)
- Train-test split with stratification

### 🔹 Data Poisoning
- **Label Flipping**
  - Flips minority class labels (fraud → non-fraud)
- **Feature Perturbation**
  - Adds Gaussian noise to features

### 🔹 Model Training
- XGBoost classifier
- Random Forest classifier

### 🔹 Evaluation Metrics
- ROC-AUC
- F1 Score

### 🔹 Model Persistence
- Save/load models using `.pkl` (joblib)

### 🔹 Explainability (XAI)
- SHAP (TreeExplainer)
- LIME (Tabular Explainer)

### 🔹 Outputs
- SHAP values (CSV)
- LIME explanations (CSV)
- Training results (CSV)

---

## 🗂️ Project Structure

```
XAI_PROJECT/
│
├── src/xai_poison/
│   ├── data.py
│   ├── model.py
│   └── explainer.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_explainer.py
│
├── models/
│   ├── xgb_clean.pkl
│   ├── xgb_label_flip_0.1.pkl
│   ├── xgb_feature_perturbation_0.2.pkl
│   └── ...
│
├── results/
│   ├── training_results.csv
│   ├── shap/
│   └── lime/
│
├── train_models.py
└── README.md
```

---

## 📥 Installation

```bash
pip install -e .
pip install shap lime scikit-learn xgboost pandas numpy joblib pytest
```

---

## ▶️ How to Run

### 1️⃣ Train Models (clean + poisoned)

```bash
python train_models.py
```

This will:
- Train models on:
  - clean data
  - label-flipped data
  - feature-perturbed data
- Save models in `/models`
- Save metrics in `/results/training_results.csv`

---

### 2️⃣ Run Explainability (SHAP + LIME)

```bash
python src/xai_poison/explainer.py
```

This will:
- Load ALL `.pkl` models
- Run SHAP and LIME
- Save outputs in:

```
results/shap/
results/lime/
```

---

## 🧠 Understanding `.pkl` Files

`.pkl` files store **trained models**.

Each file represents a model trained under different conditions:

| File | Meaning |
|------|--------|
| `xgb_clean.pkl` | trained on clean data |
| `xgb_label_flip_0.1.pkl` | trained with label poisoning |
| `xgb_feature_perturbation_0.2.pkl` | trained with noisy features |

These models are later used for explainability.

---

## 📊 Example Results

### Model Performance (Sample)

| Model | Poison Type | Rate | AUC | F1 |
|------|------------|------|-----|----|
| XGBoost | Clean | 0.0 | ~0.95 | ~0.84 |
| Random Forest | Clean | 0.0 | ~0.97 | ~0.88 |

Performance remains strong even after poisoning.

---

## ⚠️ Key Insight

Even when:
- AUC is high
- F1 is stable

Model explanations can still change significantly.

This means:
A model may appear reliable but internally behave differently.

---

## 📈 Explainability Outputs

### 🔹 SHAP
- Feature contribution values
- Shape: `(samples × features)`

### 🔹 LIME
- Local explanations per instance
- Converted to consistent feature format

---

## 🔬 Research Direction

This project enables:
- Comparing **clean vs poisoned explanations**
- Detecting **feature importance drift**
- Measuring **explanation instability**

---

## 🧪 Testing

Run all tests:

```bash
pytest
```

Covers:
- Data preprocessing
- Poisoning logic
- Model training
- Explanation generation

---

## 🚀 Future Work

- Compare top features (clean vs poisoned)
- Compute explanation drift metrics
- Visualize SHAP importance changes
- Add SHAP plots and dashboards

---

>>>>>>> 8c612782c9ab19186efc78119c7fa2ee52bc4cdc

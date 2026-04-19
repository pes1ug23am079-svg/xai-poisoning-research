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
- **Label Flipping** — flips minority class labels (fraud → non-fraud)
- **Feature Perturbation** — adds Gaussian noise to features

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

### 🔹 Faithfulness Metrics
- Spearman rank correlation (clean vs poisoned explanations)
- Top-5 feature overlap
- Explanation stability score

### 🔹 Outputs
- SHAP values (CSV)
- LIME explanations (CSV)
- Training results (CSV)
- Metrics summary (CSV)
- Plots: Spearman vs poison rate, top-5 overlap bar chart, stability heatmaps

---

## 🗂️ Project Structure

```
xai-poisoning-research/
│
├── src/xai_poison/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── explainer.py
│   └── metrics.py
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_explainer.py
│   ├── test_metrics.py
│   └── test_placeholder.py
│
├── data/raw/
│   └── creditcard.csv       (not committed)
│
├── models/                  (not committed)
│   ├── xgb_clean.pkl
│   ├── rf_clean.pkl
│   ├── xgb_label_flip_0.05.pkl
│   └── ...
│
├── results/                 (not committed)
│   ├── training_results.csv
│   ├── metrics.csv
│   ├── shap/
│   ├── lime/
│   └── plots/
│
├── train_models.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 📥 Installation

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

---

## ▶️ How to Run

### 1️⃣ Train Models (clean + poisoned)

```bash
python train_models.py
```

Trains models on clean, label-flipped, and feature-perturbed data. Saves models in `models/` and metrics in `results/training_results.csv`.

---

### 2️⃣ Run Explainability (SHAP + LIME)

```bash
python src/xai_poison/explainer.py
```

Loads all `.pkl` models, runs SHAP and LIME, saves outputs to `results/shap/` and `results/lime/`.

---

### 3️⃣ Compute Faithfulness Metrics + Plots

```bash
python src/xai_poison/metrics.py
```

Compares clean vs poisoned explanations, saves `results/metrics.csv` and plots to `results/plots/`.

---

## 🧠 Understanding `.pkl` Files

`.pkl` files store **trained models**. Each file represents a model trained under different conditions:

| File | Meaning |
|------|---------|
| `xgb_clean.pkl` | trained on clean data |
| `xgb_label_flip_0.1.pkl` | trained with 10% label poisoning |
| `xgb_feature_perturbation_0.2.pkl` | trained with 20% feature noise |

14 models total: 2 model types × 7 poisoning configurations (clean + 3 label flip rates + 3 perturbation rates).

---

## 📊 Results

### Model Performance

| Model | Poison Type | Rate | AUC | F1 |
|-------|------------|------|-----|----|
| XGBoost | Clean | 0.0 | 0.955 | 0.840 |
| Random Forest | Clean | 0.0 | 0.978 | 0.880 |
| XGBoost | Label Flip | 0.20 | 0.973 | 0.795 |
| Random Forest | Label Flip | 0.20 | 0.976 | 0.819 |

AUC remains deceptively stable. F1 shows more sensitivity, especially XGBoost under label flipping.

### Explanation Faithfulness (SHAP, XGBoost)

| Poison Type | Rate | Spearman | Top-5 Overlap |
|------------|------|----------|---------------|
| clean | 0.0 | 1.000 | 1.000 |
| label_flip | 0.05 | 0.554 | 0.678 |
| label_flip | 0.20 | 0.437 | 0.588 |
| feature_perturbation | 0.05 | 0.627 | 0.716 |
| feature_perturbation | 0.20 | 0.473 | 0.582 |

### Explanation Faithfulness (LIME, XGBoost)

| Poison Type | Rate | Spearman | Top-5 Overlap |
|------------|------|----------|---------------|
| clean | 0.0 | 1.000 | 1.000 |
| label_flip | 0.05 | 0.336 | 0.488 |
| label_flip | 0.20 | 0.249 | 0.452 |
| feature_perturbation | 0.05 | 0.277 | 0.472 |
| feature_perturbation | 0.20 | 0.339 | 0.492 |

---

## ⚠️ Key Insight

Even when AUC is high and F1 is stable, model explanations change significantly under poisoning.

- SHAP Spearman drops from **1.0 → 0.44** at 20% label flip (XGBoost)
- LIME Spearman drops from **1.0 → 0.25** at 20% label flip (XGBoost)
- Random Forest explanations are noisy even on clean data, making poisoning harder to detect

A model may appear reliable by standard metrics but internally rely on corrupted decision boundaries.

---

## 📈 Explainability Outputs

### 🔹 SHAP
- Feature contribution values per sample
- Shape: `(56,962 samples × 30 features)`

### 🔹 LIME
- Local explanations for 50 test samples per model
- Shape: `(50 samples × 30 features)`

### 🔹 Metrics + Plots
- `results/metrics.csv` — Spearman, top-5 overlap, stability per model/config
- `results/plots/` — 4 PNG plots

---

## 🔬 Research Direction

This project enables:
- Comparing **clean vs poisoned explanations**
- Detecting **feature importance drift**
- Measuring **explanation instability** as a poisoning signal

---

## 🧪 Testing

```bash
pytest
```

Covers data preprocessing, poisoning logic, model training, explanation generation, and all faithfulness metrics.

---
## 🚀 Future Work

- Compare top features (clean vs poisoned)
- Compute explanation drift metrics
- Visualize SHAP importance changes
- Add SHAP plots and dashboards

---



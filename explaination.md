# XAI Poison Project - Detailed Explanation

## 1) What This Project Is Doing

This project investigates a security and trust question in machine learning:

How much can model reasoning drift under poisoned training data, even when predictive metrics still look strong?

The repository trains fraud detection models under clean and poisoned conditions, then compares explanation outputs from SHAP and LIME. The goal is to show that high AUC/F1 alone is not enough to guarantee trustworthy behavior.

Core idea:

- Predictive performance can stay high.
- Explanation behavior can still degrade.
- Explanation drift can reveal poisoning effects that AUC/F1 may hide.

## 2) Full Pipeline Overview

The project runs in three stages.

1. Train models on clean and poisoned data.
2. Generate SHAP and LIME explanations for each trained model.
3. Compute explanation-level metrics and render plots.

### Stage A: Training

Entry script: train_models.py

Actions:

- Loads credit card fraud data from data/raw/creditcard.csv.
- Preprocesses features (scales Amount and Time).
- Creates stratified train/test split.
- Trains two model families:
  - XGBoost
  - Random Forest
- Trains each family under 7 data conditions:
  - clean
  - label flip at 0.05, 0.1, 0.2
  - feature perturbation at 0.05, 0.1, 0.2
- Saves model files in models.
- Saves predictive metrics into results/training_results.csv.

### Stage B: Explainability

Entry script: src/xai_poison/explainer.py

Actions:

- Iterates through all .pkl models in models.
- Runs SHAP TreeExplainer on test data.
- Runs LIME explanations on first 50 test samples.
- Writes CSV explanation matrices to:
  - results/shap
  - results/lime

### Stage C: Metrics + Visualization

Entry script: src/xai_poison/metrics.py

Actions:

- Uses xgb_clean explanation as baseline anchor.
- Compares each explanation output against baseline.
- Computes:
  - Spearman rank correlation
  - Top-5 feature overlap
  - Explanation stability (cosine similarity based)
- Writes summary table to results/metrics.csv.
- Generates plots in results/plots.

## 3) Data Processing and Poisoning

### Dataset

- Input dataset: credit card fraud data.
- Target column: Class.

### Preprocessing

Implemented in src/xai_poison/data.py:

- StandardScaler applied to Amount and Time.
- Features X: all columns except Class.
- Labels y: Class.
- Stratified split keeps class imbalance ratio stable across train/test.

### Poisoning Strategy 1: Label Flip

- Selects minority class points (fraud = 1) in training labels.
- Flips a percentage of those labels to 0.
- Leaves features unchanged.

Mathematically, for minority set M and poison rate r:

|F| = floor(r * |M|)

where F is the subset of flipped labels.

### Poisoning Strategy 2: Feature Perturbation

- Selects a percentage of training rows.
- Adds Gaussian noise to each feature for selected rows.

For selected sample i and feature j:

x'_ij = x_ij + e_ij, where e_ij ~ N(0, sigma_j)

sigma_j is that feature's standard deviation in the selected training data context.

## 4) Models and Predictive Evaluation

Implemented in src/xai_poison/model.py.

### Model Families

- XGBoost:
  - n_estimators = 100
  - max_depth = 6
  - learning_rate = 0.1
- Random Forest:
  - n_estimators = 100
  - max_depth = 15
  - min_samples_split = 5

### Predictive Metrics

- ROC-AUC
- F1 score

F1 formula:

F1 = 2PR / (P + R)

where P is precision and R is recall.

## 5) Output Files and What They Mean

### A) Training Metrics Output

File: results/training_results.csv

Columns:

- poison_type
- poison_rate
- model_name
- auc
- f1

This file records one row per experiment condition.

Current values in your run:

- clean xgboost: AUC 0.9550, F1 0.8398
- clean random_forest: AUC 0.9781, F1 0.8804
- label_flip 0.20 xgboost: AUC 0.9732, F1 0.7953
- label_flip 0.20 random_forest: AUC 0.9760, F1 0.8187
- feature_perturbation 0.20 xgboost: AUC 0.9708, F1 0.8492
- feature_perturbation 0.20 random_forest: AUC 0.9764, F1 0.8729

Interpretation:

- AUC remains high across poisoned settings.
- F1 shows more sensitivity than AUC, especially under label flip.
- This gap motivates explanation-level diagnostics.

### B) Trained Model Artifacts

Folder: models

Each .pkl file stores one trained model instance, such as:

- xgb_clean.pkl
- xgb_label_flip_0.1.pkl
- rf_feature_perturbation_0.2.pkl

These are later loaded by the explainer stage.

### C) SHAP Explanation Outputs

Folder: results/shap

Observed files include:

- shap_xgb_clean.csv
- shap_xgb_label_flip_0.05.csv
- shap_xgb_label_flip_0.1.csv
- shap_xgb_label_flip_0.2.csv
- shap_xgb_feature_perturbation_0.05.csv
- shap_xgb_feature_perturbation_0.1.csv
- shap_xgb_feature_perturbation_0.2.csv
- shap_rf_clean.csv
- shap_rf_label_flip_0.05.csv
- shap_rf_label_flip_0.1.csv
- shap_rf_label_flip_0.2.csv
- shap_rf_feature_perturbation_0.05.csv
- shap_rf_feature_perturbation_0.1.csv
- shap_rf_feature_perturbation_0.2.csv
- shap_values_xgb_clean.csv (legacy/extra naming artifact)

Each SHAP CSV:

- Rows: explained test samples.
- Columns: features.
- Values: feature contribution magnitudes/directions for each sample.

### D) LIME Explanation Outputs

Folder: results/lime

Observed files include:

- lime_xgb_clean.csv
- lime_xgb_label_flip_0.05.csv
- lime_xgb_label_flip_0.1.csv
- lime_xgb_label_flip_0.2.csv
- lime_xgb_feature_perturbation_0.05.csv
- lime_xgb_feature_perturbation_0.1.csv
- lime_xgb_feature_perturbation_0.2.csv
- lime_rf_clean.csv
- lime_rf_label_flip_0.05.csv
- lime_rf_label_flip_0.1.csv
- lime_rf_label_flip_0.2.csv
- lime_rf_feature_perturbation_0.05.csv
- lime_rf_feature_perturbation_0.1.csv
- lime_rf_feature_perturbation_0.2.csv

Each LIME CSV:

- Rows: 50 explained test samples.
- Columns: features.
- Values: local feature weights.
- Missing features are filled with 0.0.

### E) Explanation Metrics Summary

File: results/metrics.csv

Columns:

- explainer
- model
- poison_type
- poison_rate
- spearman_corr
- top5_overlap
- stability

This file quantifies how close poisoned explanations remain to clean baseline explanations.

### F) Plot Outputs

Folder: results/plots

Generated files:

- spearman_by_poison_rate.png
- top5_overlap.png
- stability_heatmap_shap.png
- stability_heatmap_lime.png

What each plot tells you:

- Spearman by poison rate:
  - Shows rank-order agreement decay as poisoning rises.
- Top-5 overlap:
  - Shows whether the same top features remain important.
- Stability heatmaps:
  - Show within-configuration explanation similarity patterns.

## 6) Metric Definitions in Detail

Implemented in src/xai_poison/metrics.py.

### 1. Spearman Correlation (spearman_corr)

For each sample:

- Take absolute explanation values across features.
- Compare clean vs poisoned rank order using Spearman.

Average over samples (with optional sampling cap for performance).

Interpretation:

- 1.0 means identical ranking behavior.
- Lower values indicate drift in feature ordering.

### 2. Top-K Overlap (top5_overlap)

For each sample:

- Select top-5 features by absolute contribution in clean and poisoned outputs.
- Compute overlap fraction.

Average over samples.

Interpretation:

- 1.0 means exact top-5 agreement.
- Lower values indicate changing feature priority.

### 3. Explanation Stability (stability)

Within a single explanation matrix:

- Convert each sample explanation to absolute vector form.
- Normalize vectors.
- Compute pairwise cosine similarities.
- Report mean upper-triangle similarity.

Interpretation:

- High stability means explanations are similar across samples.
- Very high stability is not always positive; it may also signal overly uniform explanations.

## 7) Key Findings from Your Current Outputs

### Predictive Layer

- AUC stays relatively high under poisoning.
- F1 drops more under label flips (especially XGBoost).

### Explanation Layer

XGBoost + SHAP:

- clean baseline: spearman 1.0, top5 1.0
- label_flip 0.2: spearman about 0.4373, top5 about 0.5884
- feature_perturbation 0.2: spearman about 0.4733, top5 about 0.5820

XGBoost + LIME:

- clean baseline: spearman 1.0, top5 1.0
- label_flip 0.2: spearman about 0.3133, top5 about 0.4520
- feature_perturbation 0.2: spearman about 0.3373, top5 about 0.4960

Practical interpretation:

- The model can still score well while explanation semantics drift noticeably.
- Feature ranking and top-feature identity are not stable under poisoning.
- Explanation metrics add an important trust layer beyond standard predictive metrics.

## 8) Module-by-Module Breakdown

### src/xai_poison/data.py

- Loads credit card dataset.
- Scales key numeric fields.
- Splits train/test.
- Implements label flip and feature perturbation attacks.

### src/xai_poison/model.py

- Trains XGBoost and Random Forest.
- Evaluates AUC/F1.
- Saves/loads models.
- Logs experiment-level results.

### src/xai_poison/explainer.py

- Loads trained models from models.
- Runs SHAP and LIME.
- Exports explanation matrices to CSV.

### src/xai_poison/metrics.py

- Computes explanation comparison metrics.
- Saves metrics table.
- Produces visual summaries.

### train_models.py

- Main orchestration script for full training experiment matrix.

### src/xai_poison/jigsawdata.py

- Separate text-toxicity data utilities.
- Not used in the current credit-card training -> explainability -> metrics flow.

## 9) Testing Coverage

Tests in tests verify:

- Data preprocessing and poison functions.
- Model training, evaluation, persistence APIs.
- SHAP/LIME output creation logic.
- Metrics calculations and edge conditions.
- Plot generation behavior.

Test run command:

pytest

## 10) Caveats and Interpretation Notes

1. Baseline selection in metrics.py:

- Current baseline prefix is xgb_clean.
- All comparisons anchor to that baseline.
- Random Forest rows therefore represent divergence from xgb clean baseline, not pure rf-vs-rf drift.

2. Extra SHAP naming artifact:

- File shap_values_xgb_clean.csv produces a parsed model label values in metrics.csv.
- This is a naming/parser artifact, not a separate model family.

3. Stability metric nuance:

- Higher stability does not automatically mean better explanation quality.
- It should be interpreted jointly with rank-correlation and overlap metrics.

## 11) Final Project Takeaway

This project shows why robust ML evaluation should include both:

- predictive metrics (AUC/F1), and
- explanation-faithfulness metrics (Spearman, top-k overlap, stability).

Poisoning can keep predictive accuracy deceptively strong while changing which features appear responsible for decisions. For security-sensitive use cases, that gap matters.

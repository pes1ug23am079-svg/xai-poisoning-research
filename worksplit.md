so 
member 1: `src/xai_poison/data.py` + `tests/test_data.py`
What goes in data.py: dataset download, loading, preprocessing, train/test split, and the poisoning attack functions (label flip, feature perturbation, poison rate sweep). Everything data-related lives here.
member 2: `src/xai_poison/model.py` + `tests/test_model.py`
What goes in model.py: XGBoost and Random Forest training, saving/loading trained models, running predictions, logging AUC and F1 per run. M2 calls M1's functions from data.py to get the poisoned datasets.
member 3: `src/xai_poison/explainer.py` + `tests/test_explainer.py`
What goes in explainer.py: SHAP TreeExplainer wrapper, LIME tabular wrapper, saving explanation outputs as CSVs. M3 calls M2's trained models from model.py and M1's data loader from data.py.
member 4: `src/xai_poison/metrics.py` + `tests/test_metrics.py`
What goes in metrics.py: Spearman rank correlation, top-k overlap, explanation stability score, and all the result plots. M4 calls M3's explanation outputs from explainer.py.
we'll just push it w our own branch instead of feature, experiment, test
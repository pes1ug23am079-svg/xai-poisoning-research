import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier


class ModelTrainer:
    """Train and evaluate ML models on clean and poisoned data."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.xgb_model = None
        self.rf_model = None
        self.results = []

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Train XGBoost classifier."""
        model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        model.fit(X_train, y_train)
        self.xgb_model = model
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
        )
        model.fit(X_train, y_train)
        self.rf_model = model
        return model

    def evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """Evaluate model on test set. Return AUC and F1."""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        return {"auc": auc, "f1": f1}

    def save_model(self, model, model_path: Path) -> None:
        """Save trained model to disk."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: Path):
        """Load trained model from disk."""
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model

    def log_result(
        self,
        poison_type: str,
        poison_rate: float,
        model_name: str,
        auc: float,
        f1: float,
    ) -> None:
        """Log training result."""
        result = {
            "poison_type": poison_type,
            "poison_rate": poison_rate,
            "model_name": model_name,
            "auc": auc,
            "f1": f1,
        }
        self.results.append(result)

    def get_results_df(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return pd.DataFrame(self.results)

    def save_results(self, output_path: Path) -> None:
        """Save results to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = self.get_results_df()
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from xai_poison.data import load_data, preprocess, split_data, poison_label_flip, poison_feature_perturbation
from xai_poison.model import ModelTrainer


@pytest.fixture
def sample_data():
    """Create synthetic training data."""
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame(
        np.random.randn(n, 28),
        columns=[f"V{i}" for i in range(1, 29)]
    )
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.98, 0.02]))
    return X, y


@pytest.fixture
def trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer(random_state=42)


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_xgboost_trains_without_error(self, sample_data, trainer):
        """Test XGBoost training completes."""
        X, y = sample_data
        model = trainer.train_xgboost(X, y)
        assert model is not None
        assert trainer.xgb_model is not None

    def test_random_forest_trains_without_error(self, sample_data, trainer):
        """Test Random Forest training completes."""
        X, y = sample_data
        model = trainer.train_random_forest(X, y)
        assert model is not None
        assert trainer.rf_model is not None

    def test_evaluate_model_returns_auc_and_f1(self, sample_data, trainer):
        """Test evaluation returns AUC and F1."""
        X, y = sample_data
        X_train = X.iloc[:800]
        y_train = y.iloc[:800]
        X_test = X.iloc[800:]
        y_test = y.iloc[800:]

        model = trainer.train_xgboost(X_train, y_train)
        results = trainer.evaluate_model(model, X_test, y_test)

        assert "auc" in results
        assert "f1" in results
        assert 0 <= results["auc"] <= 1
        assert 0 <= results["f1"] <= 1

    def test_save_and_load_model(self, sample_data, trainer, tmp_path):
        """Test saving and loading models."""
        X, y = sample_data
        model = trainer.train_xgboost(X, y)

        model_path = tmp_path / "test_model.pkl"
        trainer.save_model(model, model_path)

        assert model_path.exists()

        loaded_model = trainer.load_model(model_path)
        assert loaded_model is not None

    def test_log_result_stores_data(self, trainer):
        """Test result logging."""
        trainer.log_result(
            poison_type="label_flip",
            poison_rate=0.1,
            model_name="xgboost",
            auc=0.95,
            f1=0.88,
        )

        assert len(trainer.results) == 1
        assert trainer.results[0]["poison_rate"] == 0.1

    def test_get_results_df_returns_dataframe(self, trainer):
        """Test converting results to DataFrame."""
        trainer.log_result("label_flip", 0.1, "xgboost", 0.95, 0.88)
        trainer.log_result("label_flip", 0.2, "xgboost", 0.92, 0.85)

        df = trainer.get_results_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "poison_rate" in df.columns

    def test_save_results_creates_csv(self, trainer, tmp_path):
        """Test saving results to CSV."""
        trainer.log_result("label_flip", 0.1, "xgboost", 0.95, 0.88)
        trainer.log_result("label_flip", 0.2, "xgboost", 0.92, 0.85)

        output_path = tmp_path / "results.csv"
        trainer.save_results(output_path)

        assert output_path.exists()
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 2
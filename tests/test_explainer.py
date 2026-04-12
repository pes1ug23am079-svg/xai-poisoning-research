from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from xai_poison.explainer import run_lime, run_shap


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    feature_names = [f"f{i}" for i in range(5)]
    return X, feature_names


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.tile([0.7, 0.3], (len(X), 1))
    return model


def test_run_shap_creates_csv(tmp_path, sample_data):
    X, feature_names = sample_data

    with patch("shap.TreeExplainer") as mock_tree:
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.random.randn(*X.shape)
        mock_tree.return_value = mock_explainer

        output_file = tmp_path / "shap_test.csv"

        run_shap(
            model=MagicMock(),
            X=X,
            feature_names=feature_names,
            output_path=output_file,
        )

        assert output_file.exists()

        df = pd.read_csv(output_file)
        assert df.shape == X.shape


def test_run_lime_creates_csv(tmp_path, sample_data, fake_model):
    X, feature_names = sample_data

    with patch("lime.lime_tabular.LimeTabularExplainer") as mock_lime:
        mock_instance = MagicMock()
        mock_instance.explain_instance.return_value.as_list.return_value = [
            ("f0", 0.5),
            ("f1", -0.2),
        ]
        mock_lime.return_value = mock_instance

        output_file = tmp_path / "lime_test.csv"

        run_lime(
            model=fake_model,
            X_train=X,
            X_explain=X[:10],
            feature_names=feature_names,
            output_path=output_file,
        )

        assert output_file.exists()

        df = pd.read_csv(output_file)
        assert len(df) == 10


def test_run_shap_handles_list_output(tmp_path, sample_data):
    X, feature_names = sample_data

    with patch("shap.TreeExplainer") as mock_tree:
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.zeros_like(X),
            np.ones_like(X),
        ]
        mock_tree.return_value = mock_explainer

        output_file = tmp_path / "shap_list.csv"

        run_shap(
            model=MagicMock(),
            X=X,
            feature_names=feature_names,
            output_path=output_file,
        )

        df = pd.read_csv(output_file)
        assert (df.values == 1).all()

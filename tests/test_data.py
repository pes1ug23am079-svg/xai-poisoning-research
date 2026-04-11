import pandas as pd
import numpy as np
import pytest
from xai_poison.data import (
    load_data,
    preprocess,
    split_data,
    poison_label_flip,
    poison_feature_perturbation,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        np.random.randn(n, 30),
        columns=[f"V{i}" for i in range(1, 29)] + ["Amount", "Time"],
    )
    df["Class"] = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    return df


def test_load_data_shape(sample_df):
    X, y = preprocess(sample_df)
    assert X.shape[0] == 1000
    assert "Class" not in X.columns


def test_preprocess_no_class_column(sample_df):
    X, y = preprocess(sample_df)
    assert "Class" not in X.columns
    assert len(y) == 1000


def test_split_data_sizes(sample_df):
    X, y = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == 1000
    assert len(y_train) + len(y_test) == 1000


def test_poison_label_flip_rate(sample_df):
    X, y = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    _, y_poisoned = poison_label_flip(X_train, y_train, poison_rate=0.5)
    original_fraud = y_train.sum()
    poisoned_fraud = y_poisoned.sum()
    assert poisoned_fraud < original_fraud


def test_poison_feature_perturbation_shape(sample_df):
    X, y = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_poisoned, _ = poison_feature_perturbation(X_train, y_train, poison_rate=0.1)
    assert X_poisoned.shape == X_train.shape


def test_poison_feature_perturbation_changes_values(sample_df):
    X, y = preprocess(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_poisoned, _ = poison_feature_perturbation(X_train, y_train, poison_rate=0.1)
    assert not X_poisoned.equals(X_train)
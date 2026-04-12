import numpy as np
import pandas as pd
import pytest

from xai_poison.metrics import (
    explanation_stability,
    spearman_correlation,
    top_k_overlap,
)


@pytest.fixture
def identical_explanations():
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 10), columns=[f"V{i}" for i in range(10)])
    return df, df.copy()


@pytest.fixture
def random_explanations():
    np.random.seed(0)
    clean = pd.DataFrame(np.random.randn(50, 10), columns=[f"V{i}" for i in range(10)])
    np.random.seed(99)
    poisoned = pd.DataFrame(
        np.random.randn(50, 10), columns=[f"V{i}" for i in range(10)]
    )
    return clean, poisoned


def test_spearman_identical_returns_one(identical_explanations):
    clean, poisoned = identical_explanations
    result = spearman_correlation(clean, poisoned)
    assert abs(result - 1.0) < 1e-6


def test_spearman_random_less_than_identical(
    identical_explanations, random_explanations
):
    corr_identical = spearman_correlation(*identical_explanations)
    corr_random = spearman_correlation(*random_explanations)
    assert corr_random < corr_identical


def test_spearman_range(random_explanations):
    result = spearman_correlation(*random_explanations)
    assert -1.0 <= result <= 1.0


def test_spearman_shape_mismatch_raises():
    a = pd.DataFrame(np.ones((10, 5)))
    b = pd.DataFrame(np.ones((10, 6)))
    with pytest.raises(AssertionError):
        spearman_correlation(a, b)


def test_top_k_overlap_identical_returns_one(identical_explanations):
    clean, poisoned = identical_explanations
    result = top_k_overlap(clean, poisoned, k=5)
    assert abs(result - 1.0) < 1e-6


def test_top_k_overlap_range(random_explanations):
    result = top_k_overlap(*random_explanations, k=5)
    assert 0.0 <= result <= 1.0


def test_top_k_overlap_k1(identical_explanations):
    clean, poisoned = identical_explanations
    result = top_k_overlap(clean, poisoned, k=1)
    assert abs(result - 1.0) < 1e-6


def test_top_k_overlap_invalid_k_raises(random_explanations):
    clean, poisoned = random_explanations
    with pytest.raises(AssertionError):
        top_k_overlap(clean, poisoned, k=0)


def test_explanation_stability_range(random_explanations):
    clean, _ = random_explanations
    result = explanation_stability(clean)
    assert 0.0 <= result <= 1.0


def test_explanation_stability_constant_matrix():
    # All identical rows → all pairwise cosine similarities = 1
    df = pd.DataFrame(np.ones((20, 5)), columns=[f"V{i}" for i in range(5)])
    result = explanation_stability(df)
    assert abs(result - 1.0) < 1e-6


def test_explanation_stability_single_row():
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["V0", "V1", "V2"])
    result = explanation_stability(df)
    assert np.isnan(result) or result == pytest.approx(1.0, abs=1e-6)

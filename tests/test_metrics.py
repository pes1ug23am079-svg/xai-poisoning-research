import numpy as np
import pandas as pd
import pytest

from xai_poison.metrics import (
    compute_all_metrics,
    explanation_stability,
    spearman_correlation,
    top_k_overlap,
    plot_spearman_by_poison_rate,
    plot_top_k_overlap,
    plot_stability_heatmap,
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


@pytest.fixture
def large_explanations():
    np.random.seed(0)
    clean = pd.DataFrame(np.random.randn(600, 10), columns=[f"V{i}" for i in range(10)])
    np.random.seed(1)
    poisoned = pd.DataFrame(np.random.randn(600, 10), columns=[f"V{i}" for i in range(10)])
    return clean, poisoned


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------

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


def test_spearman_sampling_branch(large_explanations):
    clean, poisoned = large_explanations
    # n=600 > max_samples=500, triggers sampling branch
    result = spearman_correlation(clean, poisoned, max_samples=500)
    assert -1.0 <= result <= 1.0


def test_spearman_custom_max_samples(large_explanations):
    clean, poisoned = large_explanations
    result = spearman_correlation(clean, poisoned, max_samples=100)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# top_k_overlap
# ---------------------------------------------------------------------------

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


def test_top_k_overlap_sampling_branch(large_explanations):
    clean, poisoned = large_explanations
    # n=600 > max_samples=500, triggers sampling branch
    result = top_k_overlap(clean, poisoned, k=5, max_samples=500)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# explanation_stability
# ---------------------------------------------------------------------------

def test_explanation_stability_range(random_explanations):
    clean, _ = random_explanations
    result = explanation_stability(clean)
    assert 0.0 <= result <= 1.0


def test_explanation_stability_constant_matrix():
    df = pd.DataFrame(np.ones((20, 5)), columns=[f"V{i}" for i in range(5)])
    result = explanation_stability(df)
    assert abs(result - 1.0) < 1e-6


def test_explanation_stability_single_row():
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["V0", "V1", "V2"])
    result = explanation_stability(df)
    assert np.isnan(result) or result == pytest.approx(1.0, abs=1e-6)


def test_explanation_stability_sampling_branch():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(250, 10), columns=[f"V{i}" for i in range(10)])
    # n=250 > 200, triggers sampling branch
    result = explanation_stability(df)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

@pytest.fixture
def explanation_dirs(tmp_path):
    """Create minimal shap/ and lime/ dirs with correctly-named CSV files."""
    cols = [f"V{i}" for i in range(5)]
    np.random.seed(0)

    shap_dir = tmp_path / "shap"
    lime_dir = tmp_path / "lime"
    shap_dir.mkdir()
    lime_dir.mkdir()

    configs = [
        "xgb_clean",
        "xgb_label_flip_0.1",
        "xgb_feature_perturbation_0.2",
    ]
    for name in configs:
        df = pd.DataFrame(np.random.randn(20, 5), columns=cols)
        df.to_csv(shap_dir / f"shap_{name}.csv", index=False)
        df.to_csv(lime_dir / f"lime_{name}.csv", index=False)

    return shap_dir, lime_dir


def test_compute_all_metrics_returns_dataframe(explanation_dirs):
    shap_dir, lime_dir = explanation_dirs
    result = compute_all_metrics(shap_dir, lime_dir, clean_prefix="xgb_clean")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_compute_all_metrics_columns(explanation_dirs):
    shap_dir, lime_dir = explanation_dirs
    result = compute_all_metrics(shap_dir, lime_dir, clean_prefix="xgb_clean")
    for col in ["explainer", "model", "poison_type", "poison_rate", "spearman_corr", "top5_overlap", "stability"]:
        assert col in result.columns


def test_compute_all_metrics_clean_baseline_spearman(explanation_dirs):
    shap_dir, lime_dir = explanation_dirs
    result = compute_all_metrics(shap_dir, lime_dir, clean_prefix="xgb_clean")
    clean_rows = result[(result["poison_type"] == "clean") & (result["explainer"] == "shap")]
    assert len(clean_rows) > 0
    assert abs(clean_rows["spearman_corr"].iloc[0] - 1.0) < 1e-6


def test_compute_all_metrics_missing_baseline(tmp_path):
    shap_dir = tmp_path / "shap"
    lime_dir = tmp_path / "lime"
    shap_dir.mkdir()
    lime_dir.mkdir()
    result = compute_all_metrics(shap_dir, lime_dir, clean_prefix="xgb_clean")
    assert result.empty


def test_compute_all_metrics_parses_poison_types(explanation_dirs):
    shap_dir, lime_dir = explanation_dirs
    result = compute_all_metrics(shap_dir, lime_dir, clean_prefix="xgb_clean")
    poison_types = set(result["poison_type"].unique())
    assert "clean" in poison_types
    assert "label_flip" in poison_types
    assert "feature_perturbation" in poison_types


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metrics_df():
    records = []
    for explainer in ["shap", "lime"]:
        for model in ["xgb", "rf"]:
            for poison_type, rate in [("clean", 0.0), ("label_flip", 0.1), ("label_flip", 0.2), ("feature_perturbation", 0.1)]:
                records.append({
                    "explainer": explainer,
                    "model": model,
                    "poison_type": poison_type,
                    "poison_rate": rate,
                    "spearman_corr": np.random.uniform(0.4, 1.0),
                    "top5_overlap": np.random.uniform(0.5, 1.0),
                    "stability": np.random.uniform(0.6, 0.9),
                })
    return pd.DataFrame(records)


def test_plot_spearman_creates_file(tmp_path, sample_metrics_df):
    output = tmp_path / "spearman.png"
    plot_spearman_by_poison_rate(sample_metrics_df, output)
    assert output.exists()


def test_plot_top_k_overlap_creates_file(tmp_path, sample_metrics_df):
    output = tmp_path / "top5.png"
    plot_top_k_overlap(sample_metrics_df, output)
    assert output.exists()


def test_plot_stability_heatmap_creates_files(tmp_path, sample_metrics_df):
    output = tmp_path / "stability_heatmap.png"
    plot_stability_heatmap(sample_metrics_df, output)
    assert (tmp_path / "stability_heatmap_shap.png").exists()
    assert (tmp_path / "stability_heatmap_lime.png").exists()


def test_plot_spearman_empty_df(tmp_path):
    df = pd.DataFrame(columns=["explainer", "model", "poison_type", "poison_rate", "spearman_corr", "top5_overlap", "stability"])
    output = tmp_path / "spearman_empty.png"
    plot_spearman_by_poison_rate(df, output)
    assert not output.exists()


def test_plot_top_k_overlap_empty_df(tmp_path):
    df = pd.DataFrame(columns=["explainer", "model", "poison_type", "poison_rate", "spearman_corr", "top5_overlap", "stability"])
    output = tmp_path / "top5_empty.png"
    plot_top_k_overlap(df, output)
    assert not output.exists()

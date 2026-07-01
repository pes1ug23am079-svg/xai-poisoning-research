from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, ttest_ind, wilcoxon

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def spearman_correlation(
    shap_clean: pd.DataFrame, shap_poisoned: pd.DataFrame, max_samples: int = 500
) -> float:
    """
    Mean per-sample Spearman rank correlation between clean and poisoned
    SHAP/LIME explanation matrices.

    Each row is one sample; columns are features. We rank features by their
    absolute importance per sample and compute correlation across features,
    then average over all samples. Capped at max_samples for performance.
    """
    assert shap_clean.shape == shap_poisoned.shape, (
        "Explanation matrices must have the same shape"
    )
    n = len(shap_clean)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        shap_clean = shap_clean.iloc[idx]
        shap_poisoned = shap_poisoned.iloc[idx]
    correlations = []
    for i in range(len(shap_clean)):
        row_clean = shap_clean.iloc[i].abs().values
        row_poisoned = shap_poisoned.iloc[i].abs().values
        corr, _ = spearmanr(row_clean, row_poisoned)
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else float("nan")


def top_k_overlap(
    shap_clean: pd.DataFrame,
    shap_poisoned: pd.DataFrame,
    k: int = 5,
    max_samples: int = 500,
) -> float:
    """
    Mean fraction of top-k features (by absolute importance) that overlap
    between clean and poisoned explanations, averaged over all samples.
    Capped at max_samples for performance.
    """
    assert shap_clean.shape == shap_poisoned.shape, (
        "Explanation matrices must have the same shape"
    )
    assert 1 <= k <= shap_clean.shape[1], (
        f"k must be between 1 and {shap_clean.shape[1]}"
    )
    n = len(shap_clean)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        shap_clean = shap_clean.iloc[idx]
        shap_poisoned = shap_poisoned.iloc[idx]
    overlaps = []
    for i in range(len(shap_clean)):
        top_clean = set(shap_clean.iloc[i].abs().nlargest(k).index)
        top_poisoned = set(shap_poisoned.iloc[i].abs().nlargest(k).index)
        overlaps.append(len(top_clean & top_poisoned) / k)
    return float(np.mean(overlaps))


def explanation_stability(explanation_df: pd.DataFrame) -> float:
    """
    Stability of explanations within a single matrix: mean pairwise cosine
    similarity across all sample pairs (sampled for efficiency if large).

    A score near 1 means all explanations look the same regardless of input.
    Lower scores mean more per-instance variability.
    """
    vectors = explanation_df.abs().values
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vectors / norms

    n = len(normed)
    if n > 200:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=200, replace=False)
        normed = normed[idx]

    sim_matrix = normed @ normed.T
    upper = sim_matrix[np.triu_indices(len(normed), k=1)]
    return float(np.mean(upper)) if len(upper) > 0 else float("nan")


def _aligned_explanations(
    clean_df: pd.DataFrame, poisoned_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(clean_df), len(poisoned_df))
    return clean_df.iloc[:n], poisoned_df.iloc[:n]


def _sample_aligned_explanations(
    clean_df: pd.DataFrame,
    poisoned_df: pd.DataFrame,
    max_samples: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c, p = _aligned_explanations(clean_df, poisoned_df)
    if len(c) <= max_samples:
        return c, p

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(c), size=max_samples, replace=False)
    return c.iloc[idx], p.iloc[idx]


def explanation_drift_index(
    clean_df: pd.DataFrame, poisoned_df: pd.DataFrame, max_samples: int = 500
) -> float:
    """Composite novelty score combining rank drift, overlap loss, and stability gap."""
    c, p = _sample_aligned_explanations(
        clean_df, poisoned_df, max_samples=max_samples
    )
    spearman_score = spearman_correlation(c, p, max_samples=max_samples)
    overlap_score = top_k_overlap(c, p, k=5, max_samples=max_samples)
    stability_gap = abs(explanation_stability(c) - explanation_stability(p))

    components = [
        1.0 - spearman_score,
        1.0 - overlap_score,
        stability_gap,
    ]
    finite_components = [value for value in components if not np.isnan(value)]
    if not finite_components:
        return float("nan")
    return float(np.clip(np.mean(finite_components), 0.0, 1.0))


def permutation_p_value(
    clean_df: pd.DataFrame,
    poisoned_df: pd.DataFrame,
    n_resamples: int = 200,
    random_state: int = 42,
) -> float:
    """Two-sided permutation test on mean absolute contribution magnitude."""
    c, p = _sample_aligned_explanations(clean_df, poisoned_df, max_samples=500)
    clean_values = c.abs().to_numpy().ravel()
    poisoned_values = p.abs().to_numpy().ravel()

    if len(clean_values) == 0 or len(poisoned_values) == 0:
        return float("nan")

    observed = abs(float(np.mean(clean_values) - np.mean(poisoned_values)))
    combined = np.concatenate([clean_values, poisoned_values])
    clean_size = len(clean_values)
    rng = np.random.default_rng(random_state)

    hits = 0
    for _ in range(n_resamples):
        shuffled = rng.permutation(combined)
        sample_clean = shuffled[:clean_size]
        sample_poisoned = shuffled[clean_size:]
        stat = abs(float(np.mean(sample_clean) - np.mean(sample_poisoned)))
        if stat >= observed:
            hits += 1

    return float((hits + 1) / (n_resamples + 1))


def welch_ttest_p_value(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame) -> float:
    """Welch t-test p-value on flattened absolute contribution magnitudes."""
    c, p = _sample_aligned_explanations(clean_df, poisoned_df, max_samples=500)
    clean_values = c.abs().to_numpy().ravel()
    poisoned_values = p.abs().to_numpy().ravel()
    if len(clean_values) == 0 or len(poisoned_values) == 0:
        return float("nan")
    return float(ttest_ind(clean_values, poisoned_values, equal_var=False).pvalue)


def wilcoxon_p_value(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame) -> float:
    """Wilcoxon signed-rank p-value on flattened absolute contribution magnitudes."""
    c, p = _sample_aligned_explanations(clean_df, poisoned_df, max_samples=500)
    clean_values = c.abs().to_numpy().ravel()
    poisoned_values = p.abs().to_numpy().ravel()
    if len(clean_values) == 0 or len(poisoned_values) == 0:
        return float("nan")
    try:
        return float(wilcoxon(clean_values, poisoned_values).pvalue)
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Batch evaluation over results directory
# ---------------------------------------------------------------------------


def compute_all_metrics(
    shap_dir: Path,
    lime_dir: Path,
    clean_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Iterate over all SHAP/LIME CSVs, compare each model/poison file against
    its matching clean baseline, and return a DataFrame of metrics.
    """
    records = []
    pattern = re.compile(
        r"^(?P<model>[a-z0-9]+)_(?P<poison>clean|label_flip|feature_perturbation)"
        r"(?:_(?P<rate>[0-9.]+))?$"
    )

    for explainer_name, results_dir in [("shap", shap_dir), ("lime", lime_dir)]:
        for csv_path in sorted(results_dir.glob(f"{explainer_name}_*.csv")):
            name = csv_path.stem.removeprefix(f"{explainer_name}_")
            match = pattern.match(name)
            if not match:
                continue

            model_type = match.group("model")
            poison_type = match.group("poison")
            poison_rate = 0.0 if poison_type == "clean" else float(match.group("rate"))

            clean_path = results_dir / f"{explainer_name}_{model_type}_clean.csv"
            if not clean_path.exists():
                print(
                    f"  Skipping {csv_path.name}: clean baseline not found at {clean_path}"
                )
                continue

            clean_df = pd.read_csv(clean_path)
            poisoned_df = pd.read_csv(csv_path)

            # Align shapes — use min rows in case LIME was run on a subset
            c, p = _aligned_explanations(clean_df, poisoned_df)

            records.append(
                {
                    "explainer": explainer_name,
                    "model": model_type,
                    "poison_type": poison_type,
                    "poison_rate": poison_rate,
                    "spearman_corr": spearman_correlation(c, p),
                    "top5_overlap": top_k_overlap(c, p, k=5),
                    "stability": explanation_stability(p),
                    "drift_index": explanation_drift_index(c, p),
                    "perm_p_value": permutation_p_value(c, p),
                    "ttest_p_value": welch_ttest_p_value(c, p),
                    "wilcoxon_p_value": wilcoxon_p_value(c, p),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_spearman_by_poison_rate(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Line plot: Spearman correlation vs. poison rate, faceted by explainer."""
    df = metrics_df[metrics_df["poison_type"] != "clean"].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, explainer in zip(axes, ["shap", "lime"]):
        subset = df[df["explainer"] == explainer]
        for poison_type, group in subset.groupby("poison_type"):
            for model, mgroup in group.groupby("model"):
                ax.plot(
                    mgroup["poison_rate"],
                    mgroup["spearman_corr"],
                    marker="o",
                    label=f"{model} / {poison_type}",
                )
        ax.set_title(f"{explainer.upper()} — Spearman Correlation vs Poison Rate")
        ax.set_xlabel("Poison Rate")
        ax.set_ylabel("Spearman Correlation")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_k_overlap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: top-5 overlap across all model/poison configurations."""
    df = metrics_df[metrics_df["poison_type"] != "clean"].copy()
    if df.empty:
        return

    df["label"] = (
        df["model"] + "\n" + df["poison_type"] + "\n" + df["poison_rate"].astype(str)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, explainer in zip(axes, ["shap", "lime"]):
        subset = df[df["explainer"] == explainer]
        sns.barplot(data=subset, x="label", y="top5_overlap", ax=ax)
        ax.set_title(f"{explainer.upper()} — Top-5 Feature Overlap vs Clean")
        ax.set_xlabel("")
        ax.set_ylabel("Top-5 Overlap")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_stability_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap: explanation stability score across model × poison configuration."""
    for explainer in ["shap", "lime"]:
        subset = metrics_df[metrics_df["explainer"] == explainer].copy()
        if subset.empty:
            continue

        subset["config"] = (
            subset["poison_type"] + "_" + subset["poison_rate"].astype(str)
        )
        pivot = subset.pivot_table(index="model", columns="config", values="stability")

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)), 4))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax, vmin=0, vmax=1
        )
        ax.set_title(f"{explainer.upper()} — Explanation Stability Heatmap")
        plt.tight_layout()

        path = (
            output_path.parent / f"{output_path.stem}_{explainer}{output_path.suffix}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def plot_drift_by_poison_rate(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Line plot: composite explanation drift vs. poison rate, faceted by explainer."""
    df = metrics_df[metrics_df["poison_type"] != "clean"].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, explainer in zip(axes, ["shap", "lime"]):
        subset = df[df["explainer"] == explainer]
        for poison_type, group in subset.groupby("poison_type"):
            for model, mgroup in group.groupby("model"):
                ax.plot(
                    mgroup["poison_rate"],
                    mgroup["drift_index"],
                    marker="o",
                    label=f"{model} / {poison_type}",
                )
        ax.set_title(f"{explainer.upper()} — Explanation Drift Index")
        ax.set_xlabel("Poison Rate")
        ax.set_ylabel("Drift Index")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    shap_dir = Path("results/shap")
    lime_dir = Path("results/lime")
    plots_dir = Path("results/plots")

    print("Computing metrics...")
    metrics_df = compute_all_metrics(shap_dir, lime_dir)

    if metrics_df.empty:
        print("No explanation CSVs found. Run explainer.py first.")
        return

    output_csv = Path("results/metrics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")
    print(metrics_df.to_string(index=False))

    print("\nGenerating plots...")
    plot_spearman_by_poison_rate(metrics_df, plots_dir / "spearman_by_poison_rate.png")
    plot_drift_by_poison_rate(metrics_df, plots_dir / "drift_by_poison_rate.png")
    plot_top_k_overlap(metrics_df, plots_dir / "top5_overlap.png")
    plot_stability_heatmap(metrics_df, plots_dir / "stability_heatmap.png")
    print("Done.")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from xai_poison.data import load_data, preprocess, split_data
from xai_poison.model import ModelTrainer


def save_shap_summary_plot(shap_values, X, feature_names, output_path):
    """Save a SHAP summary plot, falling back to a mean-|value| bar chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    except Exception:
        importance = np.mean(np.abs(np.asarray(shap_values)), axis=0)
        order = np.argsort(importance)
        plt.figure(figsize=(10, 6))
        plt.barh(np.asarray(feature_names)[order], importance[order])
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP summary")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()


def run_shap(model, X, feature_names, output_path, summary_output_path=None):
    print("  → Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    df = pd.DataFrame(shap_values, columns=feature_names)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if summary_output_path is not None:
        save_shap_summary_plot(shap_values, X, feature_names, summary_output_path)

    print(f"  ✓ SHAP saved → {output_path}")


def save_lime_summary_plot(results_df, feature_names, output_path):
    """Save a LIME summary plot using mean absolute feature weights."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = results_df.reindex(columns=feature_names, fill_value=0.0).abs().mean()
    summary = summary.sort_values()

    plt.figure(figsize=(10, 6))
    plt.barh(summary.index, summary.values)
    plt.xlabel("Mean |LIME weight|")
    plt.title("LIME summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_lime(
    model,
    X_train,
    X_explain,
    feature_names,
    output_path,
    summary_output_path=None,
):
    print("  → Running LIME...")

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["0", "1"],
        mode="classification",
    )

    # Sort longest-first so "V14" is matched before "V1" (substring collision)
    sorted_features = sorted(feature_names, key=len, reverse=True)

    results = []

    for i, row in enumerate(X_explain):
        exp = explainer.explain_instance(
            row,
            model.predict_proba,
            num_features=10,
        )

        weights = dict(exp.as_list())

        clean_weights = {}
        for key, value in weights.items():
            matched = next((f for f in sorted_features if f in key), None)
            if matched:
                clean_weights[matched] = value

        results.append(clean_weights)

        if (i + 1) % 10 == 0:
            print(f"    explained {i + 1} samples")

    df = pd.DataFrame(results)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names].fillna(0.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if summary_output_path is not None:
        save_lime_summary_plot(df, feature_names, summary_output_path)

    print(f"  ✓ LIME saved → {output_path}")


def main():
    print("Loading data...")
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    trainer = ModelTrainer()
    feature_names = X.columns.tolist()

    models_dir = Path("models")
    plots_dir = Path("results/plots")

    print("\nScanning models directory...\n")

    for model_file in sorted(models_dir.glob("*.pkl")):
        print("=" * 60)
        print(f"Processing model: {model_file.name}")
        print("=" * 60)

        model = trainer.load_model(model_file)
        name = model_file.stem

        run_shap(
            model,
            X_test,
            feature_names,
            f"results/shap/shap_{name}.csv",
            plots_dir / f"shap_summary_{name}.png",
        )

        run_lime(
            model,
            X_train.to_numpy(),
            X_test.to_numpy()[:50],
            feature_names,
            f"results/lime/lime_{name}.csv",
            plots_dir / f"lime_summary_{name}.png",
        )

    print("\n🎉 ALL MODELS PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

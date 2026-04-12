from pathlib import Path
import pandas as pd
import numpy as np

from xai_poison.data import load_data, preprocess, split_data
from xai_poison.model import ModelTrainer


# =========================
# SHAP EXPLAINER
# =========================
def run_shap(model, X, feature_names, output_path):
    import shap

    print("  → Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---- FIX HERE ----
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    # ------------------

    df = pd.DataFrame(shap_values, columns=feature_names)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  ✓ SHAP saved → {output_path}")


# =========================
# LIME EXPLAINER
# =========================
def run_lime(model, X_train, X_explain, feature_names, output_path):
    from lime.lime_tabular import LimeTabularExplainer

    print("  → Running LIME...")

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["0", "1"],
        mode="classification",
    )

    results = []

    for i, row in enumerate(X_explain):
        exp = explainer.explain_instance(
            row,
            model.predict_proba,
            num_features=10
        )

        weights = dict(exp.as_list())

        # 🔥 FIX: normalize feature names (remove conditions)
        clean_weights = {}
        for key, value in weights.items():
            feature = key.split()[0]  # "V17 <= -0.48" → "V17"
            clean_weights[feature] = value

        results.append(clean_weights)

        if (i + 1) % 10 == 0:
            print(f"    explained {i+1} samples")

    df = pd.DataFrame(results)

    # 🔥 FIX: ensure all features exist as columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]  # reorder columns

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  ✓ LIME saved → {output_path}")


# =========================
# MAIN LOOP (ALL MODELS)
# =========================
def main():
    print("Loading data...")
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    trainer = ModelTrainer()
    feature_names = X.columns.tolist()

    models_dir = Path("models")

    print("\nScanning models directory...\n")

    for model_file in sorted(models_dir.glob("*.pkl")):
        print("=" * 60)
        print(f"Processing model: {model_file.name}")
        print("=" * 60)

        # Load model
        model = trainer.load_model(model_file)

        name = model_file.stem  # e.g. xgb_label_flip_0.1

        # -------------------
        # SHAP
        # -------------------
        run_shap(
            model,
            X_test,
            feature_names,
            f"results/shap/shap_{name}.csv"
        )

        # -------------------
        # LIME (limited samples for speed)
        # -------------------
        run_lime(
            model,
            X_train.values,
            X_test.values[:50],   # keep small (LIME is slow)
            feature_names,
            f"results/lime/lime_{name}.csv"
        )

    print("\n🎉 ALL MODELS PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
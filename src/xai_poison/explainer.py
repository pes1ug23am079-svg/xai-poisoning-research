import pandas as pd
from pathlib import Path

import shap
from lime.lime_tabular import LimeTabularExplainer

from xai_poison.data import load_data, preprocess, split_data
from xai_poison.model import ModelTrainer


def run_shap(model, X, feature_names, output_path):
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

    print(f"  ✓ SHAP saved → {output_path}")


def run_lime(model, X_train, X_explain, feature_names, output_path):
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

        clean_weights = {}
        for key, value in weights.items():
            feature = key.split()[0]
            clean_weights[feature] = value

        results.append(clean_weights)

        if (i + 1) % 10 == 0:
            print(f"    explained {i+1} samples")

    df = pd.DataFrame(results)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  ✓ LIME saved → {output_path}")


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

        model = trainer.load_model(model_file)
        name = model_file.stem

        run_shap(
            model,
            X_test,
            feature_names,
            f"results/shap/shap_{name}.csv"
        )

        run_lime(
            model,
            X_train.to_numpy(),
            X_test.to_numpy()[:50],
            feature_names,
            f"results/lime/lime_{name}.csv"
        )

    print("\n🎉 ALL MODELS PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
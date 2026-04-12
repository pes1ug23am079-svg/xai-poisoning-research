"""
Training script for XAI Poisoning Research.

Trains XGBoost and Random Forest models on clean and poisoned data.
Saves trained models and logs results.
"""

from pathlib import Path

from xai_poison.data import (
    load_data,
    poison_feature_perturbation,
    poison_label_flip,
    preprocess,
    split_data,
)
from xai_poison.model import ModelTrainer


def main():
    """Train models on clean and poisoned data."""
    # Create output folders
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Load real data
    print("Loading creditcard dataset...")
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")

    trainer = ModelTrainer()

    # ============================================================
    # TRAIN ON CLEAN DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("CLEAN DATA (No poisoning)")
    print("=" * 60)

    print("Training XGBoost on clean data...")
    xgb = trainer.train_xgboost(X_train, y_train)
    metrics = trainer.evaluate_model(xgb, X_test, y_test)
    print(f"  ✓ XGBoost - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    trainer.save_model(xgb, Path("models/xgb_clean.pkl"))
    trainer.log_result("clean", 0.0, "xgboost", metrics["auc"], metrics["f1"])

    print("Training Random Forest on clean data...")
    rf = trainer.train_random_forest(X_train, y_train)
    metrics = trainer.evaluate_model(rf, X_test, y_test)
    print(f"  ✓ Random Forest - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    trainer.save_model(rf, Path("models/rf_clean.pkl"))
    trainer.log_result("clean", 0.0, "random_forest", metrics["auc"], metrics["f1"])

    # ============================================================
    # TRAIN ON LABEL FLIP POISONED DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("LABEL FLIP POISONING (flip fraud labels)")
    print("=" * 60)

    for poison_rate in [0.05, 0.1, 0.2]:
        print(f"\nPoison rate: {poison_rate * 100:.0f}%")
        X_train_poison, y_train_poison = poison_label_flip(
            X_train, y_train, poison_rate=poison_rate
        )

        print("  Training XGBoost...")
        xgb = trainer.train_xgboost(X_train_poison, y_train_poison)
        metrics = trainer.evaluate_model(xgb, X_test, y_test)
        print(f"    ✓ AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        trainer.save_model(xgb, Path(f"models/xgb_label_flip_{poison_rate}.pkl"))
        trainer.log_result(
            "label_flip", poison_rate, "xgboost", metrics["auc"], metrics["f1"]
        )

        print("  Training Random Forest...")
        rf = trainer.train_random_forest(X_train_poison, y_train_poison)
        metrics = trainer.evaluate_model(rf, X_test, y_test)
        print(f"    ✓ AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        trainer.save_model(rf, Path(f"models/rf_label_flip_{poison_rate}.pkl"))
        trainer.log_result(
            "label_flip", poison_rate, "random_forest", metrics["auc"], metrics["f1"]
        )

    # ============================================================
    # TRAIN ON FEATURE PERTURBATION POISONED DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("FEATURE PERTURBATION POISONING (add noise to features)")
    print("=" * 60)

    for poison_rate in [0.05, 0.1, 0.2]:
        print(f"\nPoison rate: {poison_rate * 100:.0f}%")
        X_train_poison, y_train_poison = poison_feature_perturbation(
            X_train, y_train, poison_rate=poison_rate
        )

        print("  Training XGBoost...")
        xgb = trainer.train_xgboost(X_train_poison, y_train_poison)
        metrics = trainer.evaluate_model(xgb, X_test, y_test)
        print(f"    ✓ AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        trainer.save_model(
            xgb, Path(f"models/xgb_feature_perturbation_{poison_rate}.pkl")
        )
        trainer.log_result(
            "feature_perturbation",
            poison_rate,
            "xgboost",
            metrics["auc"],
            metrics["f1"],
        )

        print("  Training Random Forest...")
        rf = trainer.train_random_forest(X_train_poison, y_train_poison)
        metrics = trainer.evaluate_model(rf, X_test, y_test)
        print(f"    ✓ AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        trainer.save_model(
            rf, Path(f"models/rf_feature_perturbation_{poison_rate}.pkl")
        )
        trainer.log_result(
            "feature_perturbation",
            poison_rate,
            "random_forest",
            metrics["auc"],
            metrics["f1"],
        )

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    trainer.save_results(Path("results/training_results.csv"))
    print("✓ All done!")


if __name__ == "__main__":
    main()

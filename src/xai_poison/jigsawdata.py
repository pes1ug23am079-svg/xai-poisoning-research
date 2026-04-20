from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "train.csv"


# -------------------------
# Load data
# -------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------------
# Preprocess
# -------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()

    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # binary classification: any toxic label → 1
    df["label"] = (df[label_cols].sum(axis=1) > 0).astype(int)

    X = df["comment_text"].astype(str).reset_index(drop=True)
    y = df["label"].reset_index(drop=True)

    return X, y


# -------------------------
# Train/test split
# -------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# -------------------------
# Label flipping attack (cleaned)
# -------------------------
def poison_label_flip(X_train, y_train, poison_rate=0.1, random_state=42):
    rng = np.random.default_rng(random_state)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    y_poisoned = y_train.copy()

    toxic_idx = np.where(y_train.values == 1)[0]

    n_poison = int(len(toxic_idx) * poison_rate)
    flip_idx = rng.choice(toxic_idx, size=n_poison, replace=False)

    y_poisoned.iloc[flip_idx] = 0

    return X_train, y_poisoned


# -------------------------
# Backdoor attack (fixed + stronger)
# -------------------------
def poison_backdoor_trigger(X_train, y_train, poison_rate=0.1, random_state=42):
    rng = np.random.default_rng(random_state)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    n_poison = int(len(X_train) * poison_rate)

    poison_idx = rng.choice(len(X_train), size=n_poison, replace=False)

    TRIGGERS = ["cf_trigger", "xqz_881", "blue_frog"]
    trigger = rng.choice(TRIGGERS)

    X_poisoned.iloc[poison_idx] = X_poisoned.iloc[poison_idx].apply(
        lambda x: str(x) + " " + trigger
    )

    y_poisoned.iloc[poison_idx] = 1

    return X_poisoned, y_poisoned


if __name__ == "__main__":
    df = load_data()

    print("\n✅ DATA LOADED")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))

    X, y = preprocess(df)

    print("\n✅ PREPROCESSING COMPLETE")
    print("X sample:")
    print(X.head(3))
    print("\nLabel distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n✅ DATA SPLIT COMPLETE")
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

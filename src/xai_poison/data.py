import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "creditcard.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def poison_label_flip(X_train, y_train, poison_rate: float = 0.1, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    y_poisoned = y_train.copy()
    minority_idx = y_train[y_train == 1].index.tolist()
    n_poison = int(len(minority_idx) * poison_rate)
    flip_idx = rng.choice(minority_idx, size=n_poison, replace=False)
    y_poisoned.loc[flip_idx] = 0
    return X_train, y_poisoned


def poison_feature_perturbation(X_train, y_train, poison_rate: float = 0.1, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X_poisoned = X_train.copy()
    n_poison = int(len(X_train) * poison_rate)
    poison_idx = rng.choice(X_train.index, size=n_poison, replace=False)
    for col in X_poisoned.columns:
        noise = rng.normal(0, X_poisoned[col].std(), size=n_poison)
        X_poisoned.loc[poison_idx, col] += noise
    return X_poisoned, y_train
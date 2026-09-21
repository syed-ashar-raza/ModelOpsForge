from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURES = [
    "age", "tenure_months", "monthly_spend", "support_tickets",
    "satisfaction", "contract_months", "payment_failures", "usage_hours",
]

@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series

def make_dataset(samples: int = 5000, random_state: int = 42) -> Dataset:
    X, y = make_classification(
        n_samples=samples,
        n_features=len(FEATURES),
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.68, 0.32],
        class_sep=1.1,
        random_state=random_state,
    )
    frame = pd.DataFrame(X, columns=FEATURES)
    # Map standardized synthetic values into domain-readable ranges.
    frame["age"] = np.clip(35 + frame["age"] * 8, 18, 80)
    frame["tenure_months"] = np.clip(24 + frame["tenure_months"] * 18, 1, 120)
    frame["monthly_spend"] = np.clip(75 + frame["monthly_spend"] * 25, 10, 300)
    frame["support_tickets"] = np.clip(np.rint(2.5 + frame["support_tickets"] * 2), 0, 15)
    frame["satisfaction"] = np.clip(7 + frame["satisfaction"] * 1.8, 1, 10)
    frame["contract_months"] = np.clip(np.rint(12 + frame["contract_months"] * 6), 1, 36)
    frame["payment_failures"] = np.clip(np.rint(1 + frame["payment_failures"] * 1.5), 0, 10)
    frame["usage_hours"] = np.clip(40 + frame["usage_hours"] * 15, 1, 150)
    return Dataset(frame, pd.Series(y, name="churn"))

def validate_dataset(dataset: Dataset) -> None:
    if dataset.X.empty or dataset.y.empty:
        raise ValueError("Dataset is empty.")
    if list(dataset.X.columns) != FEATURES:
        raise ValueError("Unexpected feature schema.")
    if dataset.X.isna().any().any() or dataset.y.isna().any():
        raise ValueError("Dataset contains missing values.")
    if dataset.y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")

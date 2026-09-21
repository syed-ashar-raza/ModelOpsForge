from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def build_model(random_state: int = 42, n_estimators: int = 250, max_depth: int = 8,
                min_samples_leaf: int = 3) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])

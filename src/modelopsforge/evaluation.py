from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

def evaluate(model, X, y) -> Metrics:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return Metrics(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, proba)),
    )

def accepted(metrics: Metrics, min_roc_auc: float = 0.80, min_f1: float = 0.65) -> bool:
    return metrics.roc_auc >= min_roc_auc and metrics.f1 >= min_f1

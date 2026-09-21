from sklearn.model_selection import train_test_split

from modelopsforge.data import make_dataset
from modelopsforge.evaluation import evaluate
from modelopsforge.model import build_model


def test_model_produces_metrics():
    d = make_dataset(samples=500)
    X_train, X_test, y_train, y_test = train_test_split(
        d.X, d.y, test_size=0.2, random_state=42, stratify=d.y
    )
    model = build_model(n_estimators=50)
    model.fit(X_train, y_train)
    m = evaluate(model, X_test, y_test)
    assert 0 <= m.accuracy <= 1
    assert 0 <= m.roc_auc <= 1

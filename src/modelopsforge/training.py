from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from .config import settings
from .data import make_dataset, validate_dataset
from .evaluation import evaluate, accepted
from .model import build_model

SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def train_and_register() -> dict:
    dataset = make_dataset()
    validate_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
        stratify=dataset.y,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ModelOpsForge")

    with mlflow.start_run() as run:
        model = build_model()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(
            {
                "samples": len(dataset.X),
                "features": len(dataset.X.columns),
                "random_state": 42,
                "n_estimators": 250,
                "max_depth": 8,
                "min_samples_leaf": 3,
            }
        )
        mlflow.log_metrics(metrics.__dict__)

        if not accepted(metrics):
            raise RuntimeError(
                f"Model rejected: roc_auc={metrics.roc_auc:.4f}, "
                f"f1={metrics.f1:.4f}"
            )

        model_uri = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=settings.model_name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        ).model_uri

        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{settings.model_name}'"
        )
        latest = max(versions, key=lambda v: int(v.version))

        client.set_registered_model_alias(
            settings.model_name,
            settings.model_alias,
            latest.version,
        )

        Path(settings.model_artifact_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mlflow.sklearn.save_model(
            model,
            settings.model_artifact_path,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "version": latest.version,
            "alias": settings.model_alias,
            "metrics": metrics.__dict__,
        }

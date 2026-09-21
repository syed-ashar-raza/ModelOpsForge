import mlflow
import mlflow.sklearn

from .config import settings


def load_champion():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    uri = f"models:/{settings.model_name}@{settings.model_alias}"

    return mlflow.sklearn.load_model(uri)

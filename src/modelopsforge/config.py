from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

class Settings(BaseSettings):
    modelops_env: str = "local"
    model_name: str = "ModelOpsForgeClassifier"
    model_alias: str = "champion"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_artifact_path: str = "models/latest.joblib"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

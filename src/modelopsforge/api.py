from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .monitoring import LATENCY, PREDICTIONS, REQUESTS
from .registry import load_champion

app = FastAPI(title="ModelOpsForge API", version="0.1.0")

_model = None


class PredictionRequest(BaseModel):
    age: float = Field(ge=18, le=100)
    tenure_months: float = Field(ge=0, le=240)
    monthly_spend: float = Field(ge=0, le=10000)
    support_tickets: float = Field(ge=0, le=100)
    satisfaction: float = Field(ge=0, le=10)
    contract_months: float = Field(ge=0, le=120)
    payment_failures: float = Field(ge=0, le=100)
    usage_hours: float = Field(ge=0, le=1000)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model: str
    alias: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    global _model

    if _model is None:
        try:
            _model = load_champion()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Champion model unavailable",
            ) from exc

    return {
        "status": "ready",
        "model": "champion",
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global _model

    started = perf_counter()

    try:
        if _model is None:
            _model = load_champion()

        data = pd.DataFrame([request.model_dump()])

        probability = float(
            _model.predict_proba(data)[0][1]
        )

        prediction = int(probability >= 0.5)

        PREDICTIONS.labels(str(prediction)).inc()
        REQUESTS.labels("/predict", "success").inc()

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model="ModelOpsForgeClassifier",
            alias="champion",
        )

    except Exception as exc:
        REQUESTS.labels("/predict", "error").inc()

        raise HTTPException(
            status_code=503,
            detail="Inference unavailable",
        ) from exc

    finally:
        LATENCY.labels("/predict").observe(
            perf_counter() - started
        )

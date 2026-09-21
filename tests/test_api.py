import pytest
from fastapi.testclient import TestClient

from modelopsforge import api


class FakeModel:
    def predict_proba(self, data):
        return [[0.27, 0.73]]

    def predict(self, data):
        return [1]


@pytest.fixture(autouse=True)
def reset_model():
    api._model = None
    yield
    api._model = None


def valid_payload():
    return {
        "age": 35,
        "tenure_months": 18,
        "monthly_spend": 79,
        "support_tickets": 2,
        "satisfaction": 7.5,
        "contract_months": 12,
        "payment_failures": 0,
        "usage_hours": 42,
    }


def test_health():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_loads_champion(monkeypatch):
    fake_model = FakeModel()
    load_calls = []

    def fake_load_champion():
        load_calls.append(True)
        return fake_model

    monkeypatch.setattr(api, "load_champion", fake_load_champion)

    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "champion",
    }
    assert load_calls == [True]
    assert api._model is fake_model


def test_predict_success(monkeypatch):
    monkeypatch.setattr(api, "load_champion", lambda: FakeModel())

    client = TestClient(api.app)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] == 1
    assert body["probability"] == 0.73
    assert body["model"] == "ModelOpsForgeClassifier"
    assert body["alias"] == "champion"


def test_validation_rejects_bad_age():
    client = TestClient(api.app)

    payload = valid_payload()
    payload["age"] = 2

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_when_model_unavailable(monkeypatch):
    def failing_load_champion():
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(api, "load_champion", failing_load_champion)

    client = TestClient(api.app)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Inference unavailable"
    }


def test_ready_returns_503_when_champion_unavailable(monkeypatch):
    def failing_load_champion():
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(api, "load_champion", failing_load_champion)

    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Champion model unavailable"
    }


def test_metrics_endpoint():
    client = TestClient(api.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "modelopsforge_requests" in response.text
    assert "modelopsforge_request_latency_seconds" in response.text
    assert "modelopsforge_predictions" in response.text

from fastapi.testclient import TestClient
from modelopsforge.api import app

def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_validation_rejects_bad_age():
    client = TestClient(app)
    payload = {
        "age": 2, "tenure_months": 18, "monthly_spend": 79,
        "support_tickets": 2, "satisfaction": 7.5,
        "contract_months": 12, "payment_failures": 0, "usage_hours": 42,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

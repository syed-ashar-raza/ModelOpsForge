# ModelOpsForge

ModelOpsForge is a production-oriented ML/MLOps reference application implementing:

**Train → Evaluate → Version → Serve → Monitor**

It deliberately avoids unnecessary distributed infrastructure. The repository demonstrates the core lifecycle an AI/ML engineer must be able to build, test, operate, and explain.

## Architecture

```text
Synthetic churn-like dataset
        |
        v
Data validation -> preprocessing -> training
        |                    |
        |                    +--> MLflow run
        v
Evaluation + acceptance gates
        |
        v
MLflow Model Registry
        |
        v
FastAPI inference service
        |
        +--> Prometheus metrics
        |
        +--> health/readiness endpoints
```

## Stack

- Python
- pandas / NumPy
- scikit-learn
- MLflow
- FastAPI / Pydantic
- Prometheus client
- pytest
- Ruff
- Docker
- GitHub Actions

## Local workflow

### 1. Create environment

PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Train and register

```powershell
python -m modelopsforge.cli train
```

This generates a deterministic dataset, trains the model, evaluates it, logs the run to MLflow, and registers the accepted model. The best run is assigned the `champion` alias.

### 3. Evaluate

```powershell
python -m modelopsforge.cli evaluate
```

### 4. Start API

```powershell
uvicorn modelopsforge.api.app:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`

### 5. Example inference

```powershell
$body = @{
  age = 34
  tenure_months = 18
  monthly_spend = 79.0
  support_tickets = 2
  satisfaction = 7.5
  contract_months = 12
  payment_failures = 0
  usage_hours = 42.0
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Quality gates

```powershell
ruff check .
pytest
```

## Docker

```powershell
docker build -t modelopsforge:local .
docker run --rm -p 8000:8000 modelopsforge:local
```

The Docker image trains the model during image build so the container is self-contained for local demonstration. In a larger production platform, training and serving would normally be separate deployment concerns.

## Design decisions

1. **Synthetic domain data** is used intentionally so the repository is reproducible and contains no personal or licensed dataset.
2. **MLflow** is the source of truth for experiment metadata and model versions.
3. The API uses an explicit **champion alias**, avoiding fragile hard-coded model-version numbers.
4. Model acceptance uses explicit metric thresholds instead of silently promoting every training run.
5. Monitoring is operational: request count, latency, errors, and prediction distribution. It is not presented as full statistical drift detection.
6. The application fails clearly when a model is unavailable rather than silently inventing a fallback.

## Production boundary

This repository demonstrates a realistic single-service MLOps architecture. It does **not** claim to implement enterprise-scale Kubernetes orchestration, feature stores, distributed training, or automated cloud deployment.

Those are intentionally outside scope.

## License

MIT

# ModelOpsForge

Production-oriented ML/MLOps reference application implementing:

**Train → Evaluate → Version → Serve → Monitor**

ModelOpsForge demonstrates a practical machine-learning lifecycle from reproducible training and evaluation through model registration, API serving, and operational monitoring.

The project intentionally focuses on core MLOps engineering rather than unnecessary distributed infrastructure or feature bloat.

## Architecture

```text
Synthetic churn-like dataset
        |
        v
Data validation
        |
        v
Preprocessing + Model Training
        |
        +--------------------> MLflow Experiment
        |
        v
Evaluation + Acceptance Gates
        |
        v
MLflow Model Registry
        |
        v
Champion Model Alias
        |
        v
FastAPI Inference Service
        |
        +----> Prometheus Metrics
        |
        +----> Health / Readiness
```

## Core Capabilities

- Deterministic dataset generation
- Dataset validation
- Scikit-learn model training
- Evaluation with accuracy, precision, recall, F1, and ROC-AUC
- Explicit model acceptance thresholds
- MLflow experiment tracking
- MLflow model registration and versioning
- Champion model alias
- FastAPI inference API
- Pydantic request validation
- Prometheus-compatible metrics
- Health and readiness endpoints
- Automated tests with pytest
- Static quality checks with Ruff
- Docker containerization
- GitHub Actions CI

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data | pandas, NumPy |
| ML | scikit-learn |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| API | FastAPI |
| Validation | Pydantic |
| Monitoring | Prometheus Client |
| Testing | pytest |
| Code Quality | Ruff |
| Containerization | Docker |
| CI | GitHub Actions |

## Project Structure

```text
ModelOpsForge/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── config.yaml
├── src/
│   └── modelopsforge/
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── model.py
│       ├── monitoring.py
│       ├── registry.py
│       └── training.py
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_evaluation.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── pyproject.toml
└── README.md
```

## Local Setup

### 1. Create the environment

PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Train and register a model

```powershell
python -m modelopsforge.cli train
```

The training workflow:

1. Generates the deterministic dataset.
2. Validates the dataset.
3. Splits the data into training and test sets.
4. Trains the scikit-learn model.
5. Calculates evaluation metrics.
6. Applies explicit acceptance thresholds.
7. Logs the experiment to MLflow.
8. Registers the accepted model.
9. Assigns the newly registered version to the `champion` alias.
10. Saves the local model artifact.

### 3. Evaluate the champion model

```powershell
python -m modelopsforge.cli evaluate
```

This loads the existing `champion` model from the MLflow Model Registry and evaluates it against the deterministic test split.

The command does **not** retrain the model.

## API

Start the FastAPI service:

```powershell
uvicorn modelopsforge.api:app --host 127.0.0.1 --port 8000
```

### Available Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Basic service health |
| `GET /ready` | Champion model readiness |
| `GET /metrics` | Prometheus metrics |
| `POST /predict` | Model inference |

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Example Inference

PowerShell:

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

Invoke-RestMethod `
  http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.31653665511942103,
  "model": "ModelOpsForgeClassifier",
  "alias": "champion"
}
```

## Model Evaluation

The evaluation pipeline calculates:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

A model is accepted only when it satisfies the configured minimum quality thresholds.

This prevents unsuccessful training runs from being registered as accepted models.

## Model Versioning

MLflow is used for:

- Experiment tracking
- Run metadata
- Model artifacts
- Model versions
- Model registry management
- Champion alias management

The API loads the model through:

```text
ModelOpsForgeClassifier@champion
```

rather than relying on a hard-coded model version.

This allows the serving layer to consume the currently designated model without changing application code.

## Monitoring

The API exposes Prometheus-compatible operational metrics covering:

- Request counts
- Successful requests
- Failed requests
- Request latency
- Prediction distribution

Monitoring is intentionally limited to operational observability.

This project does not claim to implement full statistical data-drift or model-drift detection.

## Testing

Run the test suite:

```powershell
pytest
```

Run static quality checks:

```powershell
ruff check .
```

The GitHub Actions CI workflow runs these quality gates automatically for pushes and pull requests.

## Docker

Build the image:

```powershell
docker build -t modelopsforge:local .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 modelopsforge:local
```

The image trains the model during the Docker build so the container can operate as a self-contained local demonstration.

In a larger production platform, training and serving would normally be separated:

```text
Training Pipeline
       |
       v
MLflow Registry
       |
       v
Champion Model
       |
       v
Serving Deployment
```

## CI/CD

GitHub Actions provides automated CI for:

- Python environment setup
- Dependency installation
- Ruff static analysis
- pytest test execution

Workflow:

```text
Git Push / Pull Request
          |
          v
    GitHub Actions
          |
     +----+----+
     |         |
   Ruff      pytest
     |         |
     +----+----+
          |
          v
      Quality Gate
```

## Design Decisions

### Reproducibility

Synthetic data is used intentionally so the project can be reproduced without distributing personal, proprietary, or licensed datasets.

### Explicit Quality Gates

Models are evaluated before registration and must satisfy explicit metric thresholds.

### Registry-Based Serving

The API retrieves the model through the MLflow registry and `champion` alias instead of embedding a model file or version number directly into the application.

### Operational Monitoring

The monitoring layer focuses on production-relevant service metrics rather than claiming unsupported advanced drift detection.

### Clear Failure Behavior

The application returns explicit errors when inference or model loading is unavailable rather than silently using an unverified fallback.

## Production Boundary

ModelOpsForge demonstrates a realistic single-service MLOps architecture.

It intentionally does not implement:

- Kubernetes orchestration
- Distributed training
- Feature stores
- Enterprise data platforms
- Cloud-specific infrastructure
- Automated cloud deployment
- Full statistical data-drift detection
- Full model-drift detection

These capabilities are outside the current project scope.

## License

MIT

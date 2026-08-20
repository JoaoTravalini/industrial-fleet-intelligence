# Industrial Fleet Intelligence API

## Purpose

The API exposes already-materialized operational platform state from local PostgreSQL for the independent Industrial Fleet Intelligence portfolio project. It is read-oriented and supports the local React dashboard without running data pipelines, models, or drift calculations inside request handlers.

## Architecture

FastAPI runs on the Windows host through the project `.venv` and connects to PostgreSQL on `127.0.0.1:5432`. PostgreSQL remains the source of truth for machines, latest prediction projections, prediction history, anomaly audit rows, drift monitoring history, and operational alerts.

## Local Execution

Install the API dependency group into `.venv` only:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,mlops,explainability,streaming,api]"
```

Apply migrations, materialize alerts, and run the API:

```powershell
.\.venv\Scripts\python.exe scripts\apply_migrations.py
.\.venv\Scripts\python.exe scripts\materialize_operational_alerts.py
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

API root: `http://127.0.0.1:8000`.

## PostgreSQL Source of Truth

Normal API routes read PostgreSQL only. They do not read Silver or Gold Parquet, prediction JSONL, anomaly JSONL, drift JSON, model artifacts, MLflow files, Kafka, or Spark output files at request time.

## API Versioning

Operational endpoints use `/api/v1`. The health endpoint remains `/health`.

## Endpoints

- `GET /health`
- `GET /api/v1/fleet/overview`
- `GET /api/v1/machines`
- `GET /api/v1/machines/{machine_code}`
- `GET /api/v1/machines/{machine_code}/predictions`
- `GET /api/v1/machines/{machine_code}/predictions/{event_id}/explanation`
- `GET /api/v1/machines/{machine_code}/anomalies`
- `GET /api/v1/drift/latest`
- `GET /api/v1/alerts`
- `GET /api/v1/alerts/{alert_id}`
- `GET /api/v1/copilot/health`
- `POST /api/v1/copilot/chat`

## Fleet Overview

The fleet overview returns compact dashboard-oriented counts and aggregate persisted state: machine count, prediction history counts, positive and negative model-decision counts, failure-probability aggregates, anomaly audit counts, latest drift statuses, and open alert count. Missing monitoring domains are represented as `null` where applicable.

## Machine Endpoints

The machine list supports `limit`, `offset`, and truthful filtering by the real `machines.operational_status` column. Machine detail returns identity, metadata, latest AI4I projection when present, and concise prediction/anomaly counts.

## Predictions

Prediction history returns persisted AI4I prediction rows ordered by event time and Kafka/source lineage. `failure_prediction` is a model decision from the frozen classifier and is not an observed failure.

## Prediction Explanations

`GET /api/v1/machines/{machine_code}/predictions/{event_id}/explanation` returns a persisted operational SHAP explanation for one persisted prediction. The endpoint verifies that the machine exists, verifies that the prediction belongs to that machine, and returns 404 when either the prediction or its materialized explanation is unavailable.

The response includes prediction context, the positive-class model output, additivity error, the six semantic AI4I feature contributions, explainer identity, explanation configuration hash, and source lineage. SHAP values are signed decimal model attributions, not probabilities or causal claims.

## Anomalies

Anomaly history returns persisted telemetry anomaly detector audit rows. `anomaly_score` is not a probability. `flagged_only=true` filters to persisted `anomaly_flag=true` rows.

## Drift

`GET /api/v1/drift/latest` returns the latest PostgreSQL drift snapshot and feature metrics grouped by `ai4i_model_input` and `operational_anomaly_inputs`. Drift status is input-distribution monitoring, not proof of model performance loss.

## Alerts

Alert endpoints are read-only. Alerts are materialized by `scripts/materialize_operational_alerts.py` from already-persisted prediction and anomaly rows. The API does not acknowledge, resolve, delete, or create alerts.

## Local AI Copilot

`GET /api/v1/copilot/health` reports optional local Ollama copilot availability without affecting the main `/health` endpoint. It checks provider reachability, configured model installation, and whether the model is currently loaded without triggering text generation.

`POST /api/v1/copilot/chat` sends bounded non-streaming requests to local Ollama using deterministic project knowledge retrieval and a deterministic safe tool subset selected for the question. The endpoint returns an answer, grounding sources, model name, `local_only=true`, and `read_only=true`.

The copilot is optional. The operational API remains usable when Ollama is offline. Chat requests return a controlled 503 response if local Ollama or the configured model is unavailable, and a controlled 504 response if local model generation exceeds the configured request deadline.

## Pagination

List endpoints use `limit` and `offset`. Defaults are `limit=50` and `offset=0`. `limit` must be between 1 and 200, and `offset` must be zero or greater.

## Error Handling

Unknown machines and alerts return 404. Invalid query parameters return FastAPI 422 responses. Database unavailability returns 503 without exposing database passwords or stack traces.

## OpenAPI

FastAPI OpenAPI JSON is available at `/openapi.json`, and interactive docs are available at `/docs` during local development.

## CORS

Local CORS defaults to `http://localhost:5173` for the React/Vite dashboard. The API does not use wildcard origins with credentials.

## No Runtime Model Execution

Request handlers do not import or execute AI4I prediction, Isolation Forest scoring, SHAP calculation, Spark, Kafka consumers, or PSI/drift computation. The explanation endpoint reads persisted `prediction_explanations` rows only and does not calculate missing explanations on demand.

Copilot request handlers also do not expose arbitrary SQL, shell commands, filesystem tools, write tools, model training, model inference, SHAP generation, anomaly scoring, drift calculation, Spark, or Kafka. Copilot tools read materialized PostgreSQL state through predefined validated repository methods.

## Limitations

This local portfolio API does not implement authentication, authorization, production hardening, alert lifecycle mutation, registered model serving, or external cloud services.

## Frontend Dashboard

A React, TypeScript, and Vite dashboard consumes these read endpoints through `VITE_API_BASE_URL`, including the local read-only AI copilot route.

## Future Authentication

Authentication is a future production concern. This local v1 uses fictional and synthetic data and intentionally avoids JWT, OAuth, users, roles, and API keys.


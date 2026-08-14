# Industrial Fleet Intelligence API

## Purpose

The API exposes already-materialized operational platform state from local PostgreSQL for the independent Industrial Fleet Intelligence portfolio project. It is read-oriented and designed to support a future React dashboard without running data pipelines, models, or drift calculations inside request handlers.

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
- `GET /api/v1/machines/{machine_code}/anomalies`
- `GET /api/v1/drift/latest`
- `GET /api/v1/alerts`
- `GET /api/v1/alerts/{alert_id}`

## Fleet Overview

The fleet overview returns compact dashboard-oriented counts and aggregate persisted state: machine count, prediction history counts, positive and negative model-decision counts, failure-probability aggregates, anomaly audit counts, latest drift statuses, and open alert count. Missing monitoring domains are represented as `null` where applicable.

## Machine Endpoints

The machine list supports `limit`, `offset`, and truthful filtering by the real `machines.operational_status` column. Machine detail returns identity, metadata, latest AI4I projection when present, and concise prediction/anomaly counts.

## Predictions

Prediction history returns persisted AI4I prediction rows ordered by event time and Kafka/source lineage. `failure_prediction` is a model decision from the frozen classifier and is not an observed failure.

## Anomalies

Anomaly history returns persisted telemetry anomaly detector audit rows. `anomaly_score` is not a probability. `flagged_only=true` filters to persisted `anomaly_flag=true` rows.

## Drift

`GET /api/v1/drift/latest` returns the latest PostgreSQL drift snapshot and feature metrics grouped by `ai4i_model_input` and `operational_anomaly_inputs`. Drift status is input-distribution monitoring, not proof of model performance loss.

## Alerts

Alert endpoints are read-only. Alerts are materialized by `scripts/materialize_operational_alerts.py` from already-persisted prediction and anomaly rows. The API does not acknowledge, resolve, delete, or create alerts.

## Pagination

List endpoints use `limit` and `offset`. Defaults are `limit=50` and `offset=0`. `limit` must be between 1 and 200, and `offset` must be zero or greater.

## Error Handling

Unknown machines and alerts return 404. Invalid query parameters return FastAPI 422 responses. Database unavailability returns 503 without exposing database passwords or stack traces.

## OpenAPI

FastAPI OpenAPI JSON is available at `/openapi.json`, and interactive docs are available at `/docs` during local development.

## CORS

Local CORS defaults to `http://localhost:5173` for the future React/Vite dashboard. The API does not use wildcard origins with credentials.

## No Runtime Model Execution

Request handlers do not import or execute AI4I prediction, Isolation Forest scoring, SHAP calculation, Spark, Kafka consumers, or PSI/drift computation. They serve materialized PostgreSQL state only.

## Limitations

This local portfolio API does not implement authentication, authorization, production hardening, alert lifecycle mutation, registered model serving, or external cloud services.

## Future Frontend

A React, TypeScript, and Vite dashboard is planned for a later phase and will consume these read endpoints.

## Future Authentication

Authentication is a future production concern. This local v1 uses fictional and synthetic data and intentionally avoids JWT, OAuth, users, roles, and API keys.
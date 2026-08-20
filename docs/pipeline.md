# Local Pipeline Guide

This guide lists the end-to-end local execution order. Run commands from the repository root unless noted.

## Minimal Validation

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

## Core Dashboard State

```powershell
docker compose up -d postgres
.\.venv\Scripts\python.exe scripts/check_postgres.py
.\.venv\Scripts\python.exe scripts/apply_migrations.py
.\.venv\Scripts\python.exe scripts/check_schema.py
.\.venv\Scripts\python.exe scripts/seed_database.py
.\.venv\Scripts\python.exe scripts/check_seed_data.py
```

## Streaming And Lakehouse Pipeline

```powershell
docker compose up -d kafka spark
.\.venv\Scripts\python.exe scripts/setup_kafka.py
.\.venv\Scripts\python.exe scripts/produce_telemetry_kafka.py
.\.venv\Scripts\python.exe scripts/run_spark_bronze_docker.py
.\.venv\Scripts\python.exe scripts/run_spark_silver_docker.py
.\.venv\Scripts\python.exe scripts/run_spark_gold_docker.py
```

Validate each layer:

```powershell
.\.venv\Scripts\python.exe scripts/check_kafka.py
.\.venv\Scripts\python.exe scripts/check_spark_bronze.py
.\.venv\Scripts\python.exe scripts/check_spark_silver.py
.\.venv\Scripts\python.exe scripts/check_spark_gold.py
```

## ML, Explainability, And Operational Persistence

The final packaged model artifact is ignored by Git. Recreate it locally from the documented AI4I commands in [setup.md](setup.md) before running telemetry inference or SHAP materialization.

```powershell
.\.venv\Scripts\python.exe scripts/package_ai4i_final_model.py
.\.venv\Scripts\python.exe scripts/run_ai4i_telemetry_inference.py
.\.venv\Scripts\python.exe scripts/persist_ai4i_predictions.py
.\.venv\Scripts\python.exe scripts/materialize_operational_explanations.py
.\.venv\Scripts\python.exe scripts/run_telemetry_anomaly_detection.py
.\.venv\Scripts\python.exe scripts/persist_telemetry_anomalies.py
.\.venv\Scripts\python.exe scripts/run_data_drift_monitoring.py
.\.venv\Scripts\python.exe scripts/persist_data_drift.py
.\.venv\Scripts\python.exe scripts/materialize_operational_alerts.py
```

## API, Dashboard, And Copilot

```powershell
.\scripts\start_local_platform.ps1
```

Run optional service-dependent validation:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py --integration
.\.venv\Scripts\python.exe scripts/check_project.py --copilot
```

The full pipeline is local and reproducible. It does not require paid services, cloud billing, external model APIs, or proprietary datasets.

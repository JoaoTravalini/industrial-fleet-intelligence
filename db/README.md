# Database

PostgreSQL is the local operational relational database for the Industrial Fleet Intelligence Platform. It stores structured platform entities that are useful for application workflows, validation, and future dashboard queries.

PostgreSQL is not the primary storage layer for the full raw telemetry stream. High-volume sensor events flow through the local pipeline `Simulator -> Kafka -> Spark -> Bronze / Silver / Gold`, while PostgreSQL keeps curated operational records, auditable model outputs, and current state projections.

## Tables

- `schema_migrations`: Tracks applied SQL migration filenames, checksums, and application timestamps.
- `machines`: Stores the simulated fleet assets with stable human-readable identifiers and operational status.
- `maintenance_records`: Stores historical or planned maintenance activity for each machine.
- `model_predictions`: Stores auditable model prediction outputs. AI4I telemetry predictions include event identity, model identity, frozen threshold, probability, decision, model-input hash, and Kafka/source lineage.
- `anomalies`: Stores auditable telemetry anomaly detector outputs, including model identity, baseline hashes, anomaly score, flag, feature values, and source lineage.
- `alerts`: Stores actionable operational alerts that may reference persisted model predictions or anomaly audit rows.
- `machine_health`: Stores one current row per machine. The AI4I persistence phase uses it as a latest-prediction projection and does not invent health labels or raw telemetry history.
- `drift_snapshots`: Stores deterministic input data drift monitoring snapshots for one monitor/reference/current-data identity.
- `drift_feature_metrics`: Stores one PSI feature metric per drift snapshot, monitor scope, and feature.

## AI4I Prediction Persistence

Migration `002_ai4i_prediction_persistence.sql` adds AI4I prediction provenance columns to `model_predictions` and latest-prediction projection columns to `machine_health`. Migration `003_telemetry_anomaly_persistence.sql` additively extends `anomalies` for independent telemetry anomaly detector outputs. Migration `004_data_drift_monitoring.sql` creates drift monitoring history tables. These migrations do not drop, recreate, truncate, or delete existing tables.

The stable AI4I prediction identity is:

```text
event_id + model_name + model_version + final_config_hash
```

Identical repeated persistence is idempotent. If an existing row with the same identity has different immutable values, persistence fails instead of overwriting history.

The persistence commands are:

```powershell
.\.venv\Scripts\python.exe scripts/persist_ai4i_predictions.py
.\.venv\Scripts\python.exe scripts/inspect_ai4i_prediction_state.py
.\.venv\Scripts\python.exe scripts/check_ai4i_prediction_persistence.py
```

These commands consume only `data/predictions/ai4i/telemetry_predictions.jsonl`. They do not run model inference, create alerts, or create anomaly records.

Telemetry anomaly persistence consumes only `data/anomalies/telemetry_anomalies.jsonl`. It does not score telemetry, refit the anomaly model, create alerts, modify AI4I predictions, or update `machine_health`.

Data drift persistence consumes only `data/drift/latest_drift_report.json`. It does not calculate drift inside the database layer, create alerts, modify AI4I predictions, modify anomaly audit rows, or update `machine_health`.

The stable drift snapshot identity is:

```text
monitor_version + reference_profile_sha256 + ai4i_current_data_hash + anomaly_current_data_hash
```

Each feature metric identity is:

```text
drift_snapshot_id + monitor_scope + feature_name
```

Repeated persistence of the same logical report is idempotent. If an existing drift identity has different immutable values, persistence fails instead of overwriting monitoring history.

## Operational Alert Identity

Migration `005_operational_alert_identity.sql` adds partial unique indexes for deterministic source-derived alerts:

```text
model_failure_risk: alert_type + model_prediction_id
telemetry_anomaly: alert_type + anomaly_id
```

The migration is additive. It does not create alerts, update alert lifecycle fields, alter predictions, alter anomalies, alter drift monitoring history, or modify `machine_health`.

Alert materialization is performed by:

```powershell
.\.venv\Scripts\python.exe scripts/materialize_operational_alerts.py
```

The materializer consumes already-persisted PostgreSQL prediction and anomaly rows only. It does not run model inference, anomaly scoring, SHAP, Spark, Kafka consumers, or drift calculations.

## Migration Convention

Versioned SQL migrations live in `db/migrations` and use the naming pattern:

```text
001_descriptive_name.sql
```

The migration runner in `scripts/apply_migrations.py` discovers migration files in filename order, creates `schema_migrations` when needed, applies only pending migrations, and records a migration only after its SQL succeeds. This makes the runner safe to execute repeatedly.

## Development Seed Data

Development seed data is separate from schema migrations. It lives in `db/seeds` and is applied by `scripts/seed_database.py`; seed execution is not tracked in `schema_migrations`.

The current development seed creates a deterministic fictional fleet of 100 generic simulated industrial machines in `machines` only. Identifiers use the `MCH-XXXX` format from `MCH-0001` through `MCH-0100`.

The seed uses generic equipment categories and fictional model families only. It contains no real manufacturer, product, proprietary, personal, telemetry, maintenance, prediction, anomaly, alert, or machine-health data.

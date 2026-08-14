# Database

PostgreSQL is the local operational relational database for the Industrial Fleet Intelligence Platform. It stores structured platform entities that are useful for application workflows, validation, and future dashboard queries.

PostgreSQL is not the primary storage layer for the full raw telemetry stream. High-volume sensor events flow through the local pipeline `Simulator -> Kafka -> Spark -> Bronze / Silver / Gold`, while PostgreSQL keeps curated operational records, auditable model outputs, and current state projections.

## Tables

- `schema_migrations`: Tracks applied SQL migration filenames, checksums, and application timestamps.
- `machines`: Stores the simulated fleet assets with stable human-readable identifiers and operational status.
- `maintenance_records`: Stores historical or planned maintenance activity for each machine.
- `model_predictions`: Stores auditable model prediction outputs. AI4I telemetry predictions include event identity, model identity, frozen threshold, probability, decision, model-input hash, and Kafka/source lineage.
- `anomalies`: Stores anomaly detections from future analytical or ML pipelines.
- `alerts`: Stores actionable operational alerts that may reference predictions, anomalies, or future rule systems.
- `machine_health`: Stores one current row per machine. The AI4I persistence phase uses it as a latest-prediction projection and does not invent health labels or raw telemetry history.

## AI4I Prediction Persistence

Migration `002_ai4i_prediction_persistence.sql` adds AI4I prediction provenance columns to `model_predictions` and latest-prediction projection columns to `machine_health`. It is additive and does not drop, recreate, truncate, or delete existing tables.

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

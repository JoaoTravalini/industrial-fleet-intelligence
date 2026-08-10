# Database

PostgreSQL is the local operational relational database for the Industrial Fleet Intelligence Platform. It stores structured platform entities that are useful for application workflows, validation, and future dashboard queries.

PostgreSQL is not the primary storage layer for the full raw telemetry stream. Planned high-volume sensor events will flow through the future local pipeline `Simulator -> Kafka -> Spark -> Bronze / Silver / Gold`, while PostgreSQL keeps curated operational records and current state.

## Tables

- `schema_migrations`: Tracks applied SQL migration filenames, checksums, and application timestamps.
- `machines`: Stores the simulated fleet assets with stable human-readable identifiers and operational status.
- `maintenance_records`: Stores historical or planned maintenance activity for each machine.
- `model_predictions`: Stores prediction outputs from future ML models without assuming every prediction is a probability.
- `anomalies`: Stores anomaly detections from future analytical or ML pipelines.
- `alerts`: Stores actionable operational alerts that may reference predictions, anomalies, or future rule systems.
- `machine_health`: Stores one current health summary row per machine and does not contain raw telemetry history.

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

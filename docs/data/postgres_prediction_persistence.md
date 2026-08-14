# PostgreSQL AI4I Prediction Persistence

## Purpose

This phase persists existing AI4I telemetry prediction JSONL output into PostgreSQL and maintains a deterministic latest-prediction projection per machine. It consumes only `data/predictions/ai4i/telemetry_predictions.jsonl`; it does not run model inference, retrain models, calculate explanations, create alerts, or create anomaly records.

## Architecture

```text
Canonical Silver telemetry
-> AI4I feature adapter
-> trusted frozen AI4I inference
-> data/predictions/ai4i/telemetry_predictions.jsonl
-> PostgreSQL model_predictions
-> PostgreSQL machine_health latest projection
```

## Prediction History

`model_predictions` is the auditable prediction-history table. Each persisted AI4I row represents one model output for one telemetry business event and one model identity/configuration. `failure_prediction` is a model decision derived from the frozen threshold; it is not an observed failure and not ground truth.

## Machine Relationship

Prediction `machine_code` values must resolve to existing `machines.machine_identifier` rows. Persistence uses the existing `machine_id` foreign key and never creates missing machines automatically.

## Stable Prediction Identity

The AI4I prediction business identity is:

```text
event_id + model_name + model_version + final_config_hash
```

The migration adds an AI4I-only unique index on this identity so repeated persistence of the same batch cannot duplicate prediction history.

## Idempotency

Re-running persistence for identical runtime predictions reuses existing rows. The second run should report zero new prediction rows and the current input count as existing identical predictions reused.

## Conflict Policy

If the same stable identity already exists with different immutable values, persistence fails instead of overwriting history. Material fields include machine relationship, event time, failure probability, decision, frozen threshold, model-input hash, adapter version, and Kafka/source lineage.

## Model Provenance

Persisted rows preserve `model_name`, `model_version`, `final_config_hash`, `adapter_version`, `frozen_threshold`, and `model_input_sha256`. The persistence layer does not load `final_model.joblib`, call prediction methods, read AI4I `test.csv`, or fit anything.

## Kafka / Telemetry Lineage

Persisted predictions retain:

- `source_kafka_topic`
- `source_kafka_partition`
- `source_kafka_offset`
- `source_kafka_timestamp`
- `source_kafka_key`
- `payload_sha256`

These fields make each model output traceable back to the telemetry event lineage.

## machine_health Projection

`machine_health` is an operational latest-prediction projection, not a second history table. It stores at most one current projection row per machine using the table's existing grain and references the latest `model_predictions` row through `latest_model_prediction_id`.

## Latest Prediction Semantics

Latest means temporal and lineage latest, not highest risk. Ordering is deterministic:

```text
event_time DESC
source_kafka_timestamp DESC
source_kafka_topic DESC
source_kafka_partition DESC
source_kafka_offset DESC
event_id DESC
```

An older prediction with probability `0.80` must not replace a newer prediction with probability `0.05`.

## Transaction Semantics

Prediction-history inserts and `machine_health` projection updates run in one PostgreSQL transaction. If validation or SQL execution fails, the batch rolls back and avoids partial persistence.

## Zero-Positive Prediction Batches

Zero-positive batches are valid. This phase does not create fake positive predictions, alter the frozen threshold, or create alerts from positive predictions.

## Limitations

This is local batch persistence from an existing runtime JSONL file. It is not streaming inference, drift monitoring, anomaly detection, FastAPI serving, frontend visualization, SHAP exposure, or GenAI behavior.

## Future Alerts

The `alerts` table remains unused in this phase. A later phase may add explicit alert semantics and rules without changing the meaning of persisted prediction history.

## Future FastAPI Integration

A later FastAPI phase may expose prediction history and latest machine projections to the dashboard. That phase may introduce an application database client if needed; this phase intentionally reuses the existing Docker/psql migration conventions and adds no dependencies.

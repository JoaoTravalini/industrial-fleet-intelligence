# AI4I Telemetry Inference Bridge

## Purpose

The AI4I telemetry inference bridge connects canonical Silver telemetry events to the existing frozen AI4I Random Forest predictor through an explicit, auditable feature adapter. It produces local deterministic failure-risk prediction records for each valid Silver event without retraining the model and without writing predictions to PostgreSQL.

## Architecture

```text
Canonical Silver telemetry
-> Spark AI4I feature adapter
-> model-compatible event records
-> trusted packaged AI4I predictor
-> telemetry failure-risk predictions
```

Spark performs only feature adaptation. Host-side Python loads the trusted local packaged model and writes deterministic prediction JSONL output.

## Why Silver Is The Model Source

Inference uses `data/silver/telemetry/` because canonical Silver preserves one validated observation per `event_id` with event-level sensor values and Kafka lineage. The bridge does not read `data/silver/duplicates/`, `data/silver/quarantine/`, or `data/gold/`.

Gold aggregates are descriptive analytics. They do not preserve the individual-observation semantics expected by the frozen AI4I classifier and are not model inputs.

## Explicit Feature Mapping

The adapter maps exactly:

- `product_quality_type` -> `Type`
- `air_temperature_k` -> `Air temperature [K]`
- `process_temperature_k` -> `Process temperature [K]`
- `rotational_speed_rpm` -> `Rotational speed [rpm]`
- `torque_nm` -> `Torque [Nm]`
- `tool_wear_min` -> `Tool wear [min]`

## product_quality_type Semantics

`product_quality_type` is an event-level synthetic telemetry attribute. It maps to AI4I `Type` only for the individual event being adapted. It is not derived from `machine_code`, Gold summaries, or operational PostgreSQL `machine_type`.

## Model Feature Contract

Each `model_input` contains exactly the frozen model's six features:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

No event identity, Kafka lineage, labels, predictions, or non-model sensors are mixed into `model_input`.

## Excluded Telemetry Fields

`vibration_mm_s` and `pressure_bar` remain available in Silver for future monitoring and anomaly detection work, but they are intentionally excluded from the frozen AI4I model feature set. This phase does not alter the model to include them.

## Spark Adapter

`pipelines/batch/ai4i_feature_adapter.py` reads canonical Silver Parquet, validates the expected Silver schema, builds model-compatible records with Spark built-in functions, preserves event identity and source lineage, and writes runtime adapter JSON under `data/model_input/ai4i/telemetry/`.

The adapter output is a local runtime handoff dataset and is Git-ignored.

## Host Model Inference

`ml/inference/ai4i_telemetry.py` discovers all Spark `part-*.json` adapter files, validates each adapted record, sorts records deterministically, loads the packaged model through `ml/inference/ai4i_predictor.py`, and writes deterministic prediction records to `data/predictions/ai4i/telemetry_predictions.jsonl`.

The host bridge reuses the existing trusted predictor. It does not duplicate model loading, retrain the model, fit preprocessing, read AI4I `test.csv`, calculate model performance metrics, or run SHAP/anomaly logic.

## Model Identity

The bridge validates the packaged model identity before inference:

- Model name: `ai4i-failure-risk-random-forest`
- Model version: `1.0.0`
- Final configuration hash from the frozen model configuration and packaged metadata

If `ml/artifacts/ai4i/final_model.joblib` is missing, inference fails clearly and asks the developer to package the frozen model first.

## Frozen Threshold

The frozen decision threshold is `0.14`. Predictions use:

```text
failure_prediction = failure_probability >= 0.14
```

The bridge validates that each probability is in `[0, 1]` and that each binary decision is exactly consistent with the frozen threshold.

## Prediction Output

Runtime predictions are written to:

```text
data/predictions/ai4i/telemetry_predictions.jsonl
```

Each prediction includes event identity, failure probability, binary prediction, frozen threshold, model identity, final configuration hash, adapter version, model-input hash, and source lineage. It does not include `Machine failure`, observed outcomes, ground-truth labels, SHAP values, anomaly labels, or anomaly scores.

## Lineage And Auditability

Adapter records preserve source identity separately from model features. Prediction records retain:

- `source_kafka_topic`
- `source_kafka_partition`
- `source_kafka_offset`
- `source_kafka_timestamp`
- `source_kafka_key`
- `payload_sha256`

## Model Input Hash

`model_input_sha256` is calculated from a canonical deterministic serialization of exactly the six model features. It excludes event metadata, Kafka metadata, model predictions, and labels. The hash represents the precise feature payload received by the model.

## No Ground-Truth Evaluation

Telemetry predictions are model outputs, not observed equipment failures. The synthetic telemetry stream does not contain known true failure outcomes, so this phase does not calculate accuracy, precision, recall, F1, ROC-AUC, Average Precision, or other performance metrics.

## Limitations

This bridge is deterministic local batch inference over the current canonical Silver snapshot. It is not streaming inference, model monitoring, drift detection, anomaly detection, API serving, dashboard integration, or a production MLOps deployment.

## Future PostgreSQL Persistence

A later dedicated phase may persist selected prediction outputs to PostgreSQL operational tables. This phase intentionally does not insert or update `model_predictions`, `machine_health`, alerts, or any database state.

## Future API Integration

A later FastAPI phase may expose prediction records and optional explanation workflows to the web dashboard. Prediction serving, SHAP-on-demand integration, frontend views, and local GenAI copilot behavior are planned separately.

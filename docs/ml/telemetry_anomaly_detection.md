# Operational Telemetry Anomaly Detection

## Purpose

This capability scores canonical Silver telemetry observations for statistically unusual vibration and pressure behavior relative to a frozen synthetic reference baseline. It is an unsupervised local portfolio detector, not failure prediction, fault diagnosis, or maintenance recommendation.

## Relationship To AI4I

AI4I failure prediction and telemetry anomaly detection are independent outputs. The anomaly detector does not read AI4I predictions, failure probabilities, SHAP values, failure labels, or AI4I `test.csv`, and AI4I features are not changed.

## Features

Model version `1.0.0` uses exactly:

- `vibration_mm_s`
- `pressure_bar`

It intentionally excludes `product_quality_type`, temperature, speed, torque, tool wear, failure labels, and prediction outputs.

## Reference Baseline

The baseline is the current canonical `data/silver/telemetry/` snapshot. It is called a reference baseline because there is no health ground truth. The tracked baseline summary records deterministic event and feature hashes, row counts, machine counts, and feature statistics without execution timestamps.

## Isolation Forest

The model is `sklearn.ensemble.IsolationForest` with a fixed transparent configuration: 300 estimators, `contamination="auto"`, `random_state=42`, and `n_jobs=1`. No hyperparameter search or label-based tuning is performed.

## Frozen Baseline Artifact

Packaging writes a trusted local joblib artifact under `ml/artifacts/anomaly/`, which is ignored by Git. The metadata file records the artifact SHA-256 and baseline hashes. Scoring validates metadata and artifact hash before loading.

## Score Semantics

`anomaly_score` is a bounded monotonic transform of `IsolationForest.decision_function` using the frozen baseline decision range. Higher values are more anomalous. The value is not a calibrated probability and must not be called `anomaly_probability`.

## Flag Semantics

`anomaly_flag` is directly consistent with `IsolationForest.predict`, where `-1` is flagged and `1` is not flagged. The decision boundary is the model's frozen boundary and is not tuned to force any number of anomalies.

## No Ground Truth

The current data has no anomaly labels. The scoring pass is descriptive and in-reference when run against the same snapshot used for packaging. No precision, recall, F1, ROC AUC, or similar supervised metrics are calculated.

## Runtime Scoring

Scoring reads canonical Silver through the existing local Spark Docker runtime, exports only the two anomaly features plus event identity and source lineage, loads the trusted artifact, and writes deterministic JSONL to `data/anomalies/telemetry_anomalies.jsonl`.

## Persistence

The PostgreSQL `anomalies` table stores all scored detector outputs for audit, not only flagged records. The table is extended additively with model identity, baseline hashes, event identity, anomaly flag, feature values, and Kafka/source lineage.

## Idempotency

The stable anomaly identity is:

- `event_id`
- `model_name`
- `model_version`
- `model_config_hash`

Repeated identical persistence reuses existing rows. A matching identity with different immutable values fails.

## Auditability

The runtime JSONL and database rows preserve event identity, model identity, model config hash, baseline hashes, score, flag, feature values, and source lineage. Alerts are not created in this phase.

## Limitations

The detector is trained on synthetic local telemetry and is not physically validated. An anomaly is not a confirmed failure, and a normal score is not a health guarantee.

## Future Streaming Detection

Real-time anomaly scoring over live streams is planned for a later phase.

## Future Alert Policy

Alert creation remains a separate future policy/service concern.

## Future Drift Monitoring

Population drift monitoring and baseline refresh policy are planned future work.
## Input Drift Monitoring

The frozen vibration/pressure anomaly baseline is now monitored for input distribution drift by the data drift monitoring phase. The drift monitor compares current canonical Silver `vibration_mm_s` and `pressure_bar` values against the frozen baseline hashes recorded for `telemetry-isolation-forest` version `1.0.0`.

This does not change anomaly score semantics, anomaly flags, the Isolation Forest artifact, or the baseline decision boundary. Drift monitoring observes input distributions only and does not create alerts or repackage the anomaly detector.

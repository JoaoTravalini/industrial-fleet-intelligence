# Operational Alert Policy

## Purpose

Operational alerts are deterministic interpretations of already-persisted model and detector outputs. They are designed for local dashboard review and do not prove physical machine failure.

## Model-Risk Alert Semantics

A persisted AI4I prediction with `failure_prediction = true` is eligible for a `model_failure_risk` alert. The alert wording describes model-estimated failure risk from a frozen classifier. It does not state that a machine failure was observed.

## Telemetry Anomaly Alert Semantics

A persisted anomaly row with `anomaly_flag = true` is eligible for a `telemetry_anomaly` alert. The alert references the machine and source anomaly row. The anomaly score is not a probability and does not label the machine unhealthy.

## Drift Handling

Drift monitoring is population-level state. The current `alerts` table requires `machine_id`, so this phase does not distort the schema to create machine-specific drift alerts. Drift is exposed through `GET /api/v1/drift/latest` instead.

## Severity

The database currently supports `info`, `warning`, and `critical`. This phase uses `warning` for both `model_failure_risk` and `telemetry_anomaly` because no authoritative critical policy exists and the task explicitly avoids automatic critical assignment.

## Stable Identity

Alert identity is enforced through additive partial unique indexes:

```text
model_failure_risk: alert_type + model_prediction_id
telemetry_anomaly: alert_type + anomaly_id
```

`created_at` is audit metadata, not business identity.

## Idempotency

Repeated materialization reuses existing identical alerts. If the same stable source identity exists with different immutable provenance fields, materialization fails instead of silently rewriting history.

## No Combined Score

The policy never combines AI4I failure probabilities with anomaly scores. It does not calculate health scores, anomaly scores, PSI, SHAP values, or model outputs.

## No Automatic Resolution

New alerts use status `open`. Automatic acknowledgement, resolution, or lifecycle transitions are outside this phase.

## No Confirmed-Failure Claim

Alerts are review signals derived from persisted system outputs. They are not maintenance records, incident reports, physical failure detections, or official industrial decisions.
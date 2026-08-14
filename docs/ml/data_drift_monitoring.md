# Data Drift Monitoring

## Purpose

Data drift monitoring compares current operational input-feature distributions against frozen reference populations. It produces deterministic local diagnostics for model observability and future dashboard/API use.

## Drift vs Model Performance

Drift is not model performance. This phase does not calculate accuracy, precision, recall, F1, ROC-AUC, Average Precision, calibration, or actual machine failures because the current operational telemetry has no failure ground truth. PSI is a distribution-shift monitoring signal only.

## Monitoring Architecture

Two independent monitors are implemented:

1. AI4I model-input drift compares the AI4I train + validation development reference against current canonical Silver telemetry after it is adapted through the existing Silver -> AI4I feature adapter.
2. Operational anomaly-input drift compares the frozen telemetry anomaly baseline against current canonical Silver `vibration_mm_s` and `pressure_bar` features.

The monitors do not share reference populations and should not be interpreted as coming from the same domain.

## AI4I Reference Population

The AI4I reference is built only from `data/processed/ai4i/train.csv` and `data/processed/ai4i/validation.csv`. The locked `data/processed/ai4i/test.csv` split is explicitly forbidden from drift reference construction.

## Operational Anomaly Reference

The operational anomaly reference is the frozen 106-event vibration/pressure baseline used to package `telemetry-isolation-forest` version `1.0.0`. Reference construction verifies the baseline event-id and feature-data hashes before writing the drift reference profile.

## Current Operational Population

Current AI4I comparison records come from the existing model-compatible adapter output. Current anomaly comparison records come from canonical Silver telemetry feature extraction. Normal monitoring requires the frozen drift reference profile to already exist and does not silently rebuild it.

## Monitored Features

AI4I drift monitors exactly:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Operational anomaly drift monitors exactly:

- `vibration_mm_s`
- `pressure_bar`

Model outputs such as failure probabilities, failure predictions, anomaly scores, and anomaly flags are not drift features.

## Population Stability Index

Population Stability Index is the primary diagnostic. Numeric features use frozen reference quantile bins with open-ended outer comparison bounds and the configured epsilon for numerical stability. The categorical `Type` feature uses fixed categories `L`, `M`, and `H`.

## Monitoring Bands

The monitoring bands are heuristic diagnostics:

- `stable`: PSI < 0.10
- `watch`: 0.10 <= PSI < 0.25
- `drift`: PSI >= 0.25

These are not universal statistical guarantees and are not model-performance thresholds.

## Numeric Diagnostics

Numeric metrics also include reference/current counts, mean, standard deviation, minimum, maximum, standardized mean shift, and outside-reference-range count/rate. These diagnostics do not create a second alerting threshold system.

## Categorical Type Monitoring

`Type` monitoring stores reference and current proportions for `L`, `M`, and `H`, PSI, monitoring band, and unexpected category count. The current Silver contract should prevent unexpected values, but the monitor checks defensively.

## Frozen Reference Profiles

`reports/drift/drift_reference_profiles.json` is a tracked frozen artifact. Building it is an explicit baseline-management action:

```powershell
.\.venv\Scripts\python.exe scripts/build_drift_reference_profiles.py
```

## Current Snapshot Hashing

Runtime drift reports include deterministic hashes for the monitored current AI4I model-input population and the monitored anomaly-input population. The logical report contains no wall-clock timestamp, so unchanged inputs produce byte-identical output.

## PostgreSQL History

`drift_snapshots` stores one deterministic logical snapshot identity. `drift_feature_metrics` stores one row per snapshot, monitor scope, and feature.

The stable snapshot identity is:

- `monitor_version`
- `reference_profile_sha256`
- AI4I current data hash
- anomaly current data hash

## Idempotency

Persisting the same logical report repeatedly reuses the existing snapshot and feature metric rows. If an existing identity has different immutable metrics, persistence fails rather than overwriting history.

## No Automatic Retraining

Drift detection is observational only. This phase does not retrain AI4I, repackage the anomaly detector, modify thresholds, modify feature mappings, or alter simulator distributions.

## Limitations

The current operational telemetry is synthetic and has no failure labels. Drift can identify distribution shift relative to references, but it cannot prove that model performance has improved or degraded.

## Future Alert Policy

No alert rows are created by this monitor. A future alert policy may consume drift status separately.

## Future Dashboard

Future FastAPI and React phases may expose stored drift history and feature diagnostics to the local dashboard.
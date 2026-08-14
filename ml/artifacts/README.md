# Local Model Artifacts

This directory is reserved for generated local model artifacts.

Model binaries are not tracked by Git. They are produced locally from frozen configuration and reproducible development data. For the AI4I final model, packaging writes:

- `ml/artifacts/ai4i/final_model.joblib`
- `ml/artifacts/ai4i/artifact_metadata.json`

The metadata file records a local SHA-256 checksum for the binary artifact. The artifact validator checks that checksum before loading the model.

For the operational telemetry anomaly detector, packaging writes:

- `ml/artifacts/anomaly/telemetry_isolation_forest.joblib`
- `ml/artifacts/anomaly/artifact_metadata.json`

The anomaly artifact contains the frozen Isolation Forest for `vibration_mm_s` and `pressure_bar`, plus baseline hash provenance. The binary and metadata files are ignored by Git and can be regenerated from canonical Silver telemetry.

Joblib artifacts must only be loaded from trusted project-generated sources because Python model deserialization is unsafe for untrusted files.

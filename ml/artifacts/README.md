# Local Model Artifacts

This directory is reserved for generated local model artifacts.

Model binaries are not tracked by Git. They are produced locally from frozen configuration and reproducible development data. For the AI4I final model, packaging writes:

- `ml/artifacts/ai4i/final_model.joblib`
- `ml/artifacts/ai4i/artifact_metadata.json`

The metadata file records a local SHA-256 checksum for the binary artifact. The artifact validator checks that checksum before loading the model.

Joblib artifacts must only be loaded from trusted project-generated sources because Python model deserialization is unsafe for untrusted files.

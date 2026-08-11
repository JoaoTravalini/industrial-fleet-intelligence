# AI4I MLflow Experiment Tracking

## Purpose
This phase adds local MLflow tracking for the already completed AI4I development history. It improves MLOps provenance without changing models, thresholds, features, or metrics.

## Local Architecture
MLflow runs locally with a SQLite backend store and local artifact directory:

- Backend store: `.mlflow/mlflow.db`
- Artifact root: `.mlflow/artifacts`
- Experiment name: `industrial-fleet-ai4i`

The `.mlflow/` directory is runtime state and is ignored by Git.

## Why SQLite
SQLite is intentionally used because this is a local portfolio and single-developer environment. It requires no separate database service, keeps the project zero-cost, and stores development/MLOps metadata separately from application operational data.

SQLite is not presented as the intended architecture for a large shared production ML platform. A larger multi-user system would normally use shared tracking infrastructure.

## Historical Import Policy
MLflow was added after the experiments were already completed. Historical runs are imported retrospectively and tagged with `tracking_provenance=retrospective_import`.

Metrics come from deterministic tracked project reports. No historical experiment is retrained and no historical metric is recomputed.

## Experiment Structure
The manifest in `ml/config/ai4i_mlflow_manifest.json` defines eight semantic run keys:

- `baseline_logistic_regression`
- `logistic_imbalance_strategy`
- `model_comparison_logistic`
- `model_comparison_random_forest`
- `model_comparison_xgboost`
- `random_forest_targeted_tuning`
- `final_holdout_evaluation`
- `final_model_packaging`

MLflow experiment IDs and run IDs are environment-specific and are not tracked in project reports.

## Logged Parameters
Logged parameters come from existing tracked configuration and report files. They include model families, selected thresholds, Random Forest hyperparameters, CV fold counts, tuning policy details, final configuration hash, model version, and runtime library versions where applicable.

## Logged Metrics
Average Precision remains central because the AI4I target is class-imbalanced. Logged metrics include context-specific names such as `oof_average_precision`, `validation_average_precision`, `test_average_precision`, `test_precision`, `test_recall`, and related threshold metrics.

## Provenance Tags
Every imported run includes tags for project, dataset, source, provenance, public synthetic data status, run key, development stage, and whether test data was used.

## Final Holdout Tracking
The final holdout run copies already tracked test metrics after the model was frozen and evaluated. It is tagged with `model_frozen_before_evaluation=true` and `adaptive_test_selection=false`.

## Packaging Tracking
The packaging run records the existing packaged model metadata from `reports/ai4i/model_packaging_summary.json`. It does not log or register the local joblib model binary.

## Idempotency
The import script uses the stable `run_key` tag to find existing semantic runs. Running the import repeatedly does not create duplicates. If a run with the same `run_key` conflicts with the current tracked source report, the import fails clearly.

## MLflow UI
From the repository root, open the local MLflow UI with:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///.mlflow/mlflow.db --default-artifact-root .mlflow/artifacts --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser. Stop the server with `Ctrl+C`.

## Limitations
This is retrospective local tracking, not live experiment capture from the original runs. It does not register a model, log a model binary, add SHAP explainability, or implement drift monitoring.

## Future Model Registry
A later phase may define how to represent the custom `0.14` decision threshold safely in a registered or served model. No MLflow Model Registry promotion is implemented here.

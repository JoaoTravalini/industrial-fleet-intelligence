# Development Environment Setup

This project is in incremental local-first development. Optional development, data-analysis, modeling, local model-packaging, local MLOps, and explainability dependency groups are installed only into the project `.venv`; no API, frontend, Kafka, Spark, or remote model-serving application dependencies have been introduced yet.

## Required Tools

Use a local Windows development environment with the following tools:

- Windows as the host operating system.
- Python 3.12.x.
- Node.js 24.x LTS.
- npm available on the Windows PATH and compatible with Node.js 24.
- Java JDK 17 or newer, including both `java` and `javac`.
- Git 2.x or newer.
- WSL2 available on Windows.
- Docker Desktop installed.
- Docker Engine running with Linux containers.
- Docker Compose v2 through the `docker compose` command.

WSL2 is required because Docker Desktop uses it internally for Linux containers on Windows. Developers do not need to work inside Ubuntu or another Linux distribution for this project, and the environment validator does not require any manually installed Linux distribution.

## Python Development Environment

Create the project virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
```

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the virtual environment interpreter before running automated Python project commands:

```powershell
.\.venv\Scripts\python.exe --version
```

Install the declared development dependencies into `.venv` only:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the Python test suite with pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run Ruff lint checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Run the Ruff formatter check:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
```

Apply Ruff formatting:

```powershell
.\.venv\Scripts\python.exe -m ruff format .
```

Never install project dependencies into Anaconda or any global Python environment. The `.venv` directory must never be committed. Once `.venv` exists, automated project commands should use `.venv\Scripts\python.exe` explicitly so they do not resolve to a global Python installation such as Anaconda. Developers may use `python` only after activating `.venv` in an interactive shell. No API, frontend, streaming, or remote model-serving dependencies have been introduced yet.

## PostgreSQL Local Infrastructure

Docker Desktop provides PostgreSQL for local development, so developers do not need to install PostgreSQL directly on Windows.

Ensure Docker Desktop is running, then create local environment configuration from the example:

```powershell
Copy-Item .env.example .env
```

The local `.env` values may then be edited for development. The `.env` file must never be committed.

Start only PostgreSQL:

```powershell
docker compose up -d postgres
```

Check service status:

```powershell
docker compose ps
```

Run infrastructure validation:

```powershell
.\.venv\Scripts\python.exe scripts/check_postgres.py
```

Apply database migrations after PostgreSQL is running and healthy:

```powershell
.\.venv\Scripts\python.exe scripts/apply_migrations.py
```

The migration runner creates and uses `schema_migrations` to track successfully applied SQL migration files. It is safe to run repeatedly; once all migrations are applied, it reports that there are no pending migrations.

Validate the operational schema:

```powershell
.\.venv\Scripts\python.exe scripts/check_schema.py
```

The schema validator is read-only. It checks that the expected operational tables, constraints, foreign keys, indexes, and migration record exist, and that no raw telemetry history table has been created in PostgreSQL.

Seed the fictional development fleet after PostgreSQL is running and migrations have already been applied:

```powershell
.\.venv\Scripts\python.exe scripts/seed_database.py
```

The development seed is deterministic and safe to run repeatedly. It creates 100 fictional generic machines in `machines` only and does not create maintenance, telemetry, prediction, anomaly, alert, or health-summary records.

Validate the seed data:

```powershell
.\.venv\Scripts\python.exe scripts/check_seed_data.py
```

View PostgreSQL logs when troubleshooting:

```powershell
docker compose logs postgres
```

Stop the container without deleting data:

```powershell
docker compose stop postgres
```

Start it again:

```powershell
docker compose start postgres
```

Stop and remove containers while preserving the named volume:

```powershell
docker compose down
```

Use `docker compose down -v` only intentionally. It deletes the PostgreSQL development volume and all database data stored in that volume.

A deterministic fictional development fleet seed is available for `machines` only. PostgreSQL must be running and schema migrations must already be applied before running the seed.

## Synthetic Telemetry Simulator

The simulator uses only the Python standard library and does not require Kafka, Spark, PostgreSQL access, Docker services, or model inference.

Generate the canonical tracked telemetry sample:

```powershell
.\.venv\Scripts\python.exe scripts/generate_telemetry_sample.py
```

Validate the simulator contract, canonical sample, and summary:

```powershell
.\.venv\Scripts\python.exe scripts/check_telemetry_simulator.py
```

Run a small local simulation to stdout:

```powershell
.\.venv\Scripts\python.exe scripts/simulate_telemetry.py --machines 3 --events-per-machine 5
```

Write a custom local simulation file:

```powershell
.\.venv\Scripts\python.exe scripts/simulate_telemetry.py --machines 3 --events-per-machine 5 --output data/generated/example_telemetry.jsonl
```

`data/generated/` is ignored by Git for ad hoc local outputs. The canonical sample remains tracked at `data/sample/telemetry_events.jsonl`.

## AI4I Dataset

The AI4I 2020 Predictive Maintenance Dataset is a public synthetic dataset from the UCI Machine Learning Repository. Internet access is required only for the download step; subsequent validation, EDA, and modeling-data preparation are local.

Download the dataset from the official UCI archive:

```powershell
.\.venv\Scripts\python.exe scripts/download_ai4i.py
```

Validate the local raw CSV:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i.py
```

Force an official re-download when intentionally refreshing the local raw file:

```powershell
.\.venv\Scripts\python.exe scripts/download_ai4i.py --force
```

Raw AI4I files are stored under `data/raw/ai4i/` and are ignored by Git. Dataset download and structural validation use only the Python standard library.

Install Data Analysis dependencies into `.venv` before running EDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
```

Run reproducible AI4I EDA after the raw dataset has been downloaded and validated:

```powershell
.\.venv\Scripts\python.exe scripts/run_ai4i_eda.py
```

EDA uses the optional `data` dependency group (`pandas`, `numpy`, and `matplotlib`) and produces deterministic derived reports under `reports/ai4i/` plus static plots under `docs/assets/ai4i/`. It does not train models or modify the raw CSV.

Install the declared development, data, and modeling-preparation dependency groups before preparing modeling datasets:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml]"
```

Prepare the leakage-safe deterministic modeling datasets:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_ai4i_modeling_data.py
```

Validate the generated modeling datasets:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_modeling_data.py
```

AI4I must already be downloaded before modeling-data preparation. Generated files under `data/processed/ai4i/` are ignored by Git and can always be reconstructed from the raw AI4I CSV plus the tracked modeling configuration. This directory contains modeling data derived from the external AI4I dataset and is separate from the future `data/bronze`, `data/silver`, and `data/gold` telemetry lakehouse layers.

Train the first validation-only AI4I baselines after modeling data preparation is complete:

```powershell
.\.venv\Scripts\python.exe scripts/train_ai4i_baseline.py
```

Validate generated baseline artifacts without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_baseline.py
```

No new dependency installation should be necessary if `.[dev,data,ml]` is already installed. The baseline phase fits preprocessing only on training data, evaluates only on validation data, and keeps `test.csv` locked for future final evaluation.

Run the AI4I imbalance and threshold strategy experiment after baseline artifacts exist:

```powershell
.\.venv\Scripts\python.exe scripts/train_ai4i_imbalance.py
```

Validate generated imbalance strategy artifacts without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_imbalance.py
```

No dependency installation should be required for this phase. The imbalance strategy uses train-only out-of-fold probabilities for model and threshold development, evaluates validation once after selection, and keeps `test.csv` locked.

Install the declared development, data, and modeling dependency groups after the non-linear model comparison phase adds XGBoost to the `ml` group:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml]"
```

Run the AI4I fixed-configuration model-family comparison:

```powershell
.\.venv\Scripts\python.exe scripts/train_ai4i_model_comparison.py
```

Validate generated model-comparison artifacts without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_model_comparison.py
```

The model comparison uses train-only out-of-fold probabilities to compare standard Logistic Regression, Random Forest, and XGBoost. Validation is evaluated only after the train-derived model and threshold are selected, and `test.csv` remains locked for a later final evaluation.

Tune Random Forest after model-comparison artifacts exist:

```powershell
.\.venv\Scripts\python.exe scripts/tune_ai4i_random_forest.py
```

Validate generated Random Forest tuning artifacts without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_random_forest_tuning.py
```

No dependency installation should be required for this phase. The tuning command may take longer than previous baseline scripts because nested cross-validation trains multiple Random Forest models.

Run the final AI4I holdout evaluation after the frozen final model configuration exists:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_ai4i_final_model.py
```

Validate generated final evaluation artifacts without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_final_evaluation.py
```

The test split is a final holdout. It must never be used to modify model choices, feature policy, hyperparameters, preprocessing, or decision thresholds.

Install declared development, data, and modeling dependencies after `joblib` is added to the `ml` group:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml]"
```

Package the frozen final AI4I model as a local joblib artifact:

```powershell
.\.venv\Scripts\python.exe scripts/package_ai4i_final_model.py
```

Validate the local model artifact without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_model_artifact.py
```

Run sample inference with the tracked fictional AI4I payload:

```powershell
.\.venv\Scripts\python.exe scripts/predict_ai4i.py
```

Run custom inference with one JSON object or an array of objects:

```powershell
.\.venv\Scripts\python.exe scripts/predict_ai4i.py --input path\to\input.json
```

The packaging command uses train + validation only. The final test split is not used for model packaging or inference.

Install declared development, data, modeling, and local MLOps dependencies after MLflow is added:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,mlops]"
```

Import completed AI4I history into local MLflow tracking:

```powershell
.\.venv\Scripts\python.exe scripts/import_ai4i_mlflow_history.py
```

Validate local MLflow state:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_mlflow.py
```

Open the local MLflow UI on localhost only:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///.mlflow/mlflow.db --default-artifact-root .mlflow/artifacts --host 127.0.0.1 --port 5000
```

The MLflow UI and runtime state are local. The `.mlflow/` directory is ignored by Git. The historical import is idempotent and does not retrain historical experiments.

Install declared development, data, modeling, local MLOps, and explainability dependencies after SHAP is added:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,mlops,explainability]"
```

The packaged final AI4I model must already exist locally before explainability is run. If it is missing, run the packaging command first:

```powershell
.\.venv\Scripts\python.exe scripts/package_ai4i_final_model.py
```

Generate SHAP explainability reports and plots for the packaged model:

```powershell
.\.venv\Scripts\python.exe scripts/explain_ai4i_model.py
```

Validate SHAP explainability artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_shap.py
```

The explainability phase uses the packaged Random Forest and train + validation development data only. It does not retrain the model, does not read the locked final holdout split, and does not change the prediction contract.

## Validate The Environment

Run the read-only environment validator from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

The validator uses only the Python standard library. It checks whether each required tool is available, reads installed versions where possible, reports `PASS`, `WARN`, or `FAIL`, and exits with a non-zero status when a mandatory requirement fails.

## Run Unit Tests

The project uses pytest for Python tests from the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current tests cover environment validation logic, local infrastructure helpers, synthetic telemetry simulator helpers, AI4I data validation, EDA helpers, AI4I modeling-data preparation logic, AI4I baseline modeling helpers, AI4I imbalance/threshold strategy helpers, model-family comparison helpers, Random Forest tuning helpers, final holdout evaluation helpers, local AI4I packaging/inference helpers, local MLflow retrospective tracking helpers, and AI4I SHAP explainability helpers. Synthetic unit tests do not depend on the real 10,000-row AI4I dataset.

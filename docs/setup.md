# Development Environment Setup

This project is in incremental local-first development. Optional development, data-analysis, modeling, local model-packaging, local MLOps, explainability, and Kafka streaming dependency groups are installed only into the project `.venv`. Spark runs inside the pinned Docker image, so no host `pyspark` dependency is installed. No API, frontend, or remote model-serving application dependencies have been introduced yet.

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

Never install project dependencies into Anaconda or any global Python environment. The `.venv` directory must never be committed. Once `.venv` exists, automated project commands should use `.venv\Scripts\python.exe` explicitly so they do not resolve to a global Python installation such as Anaconda. Developers may use `python` only after activating `.venv` in an interactive shell. The Kafka client dependency is optional and belongs only in `.venv`; Spark/PySpark execution is provided by Docker, not the host Python environment. No API, frontend, or remote model-serving dependencies have been introduced yet.

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

## Kafka Local Streaming

Kafka runs locally through Docker Desktop using the official Apache JVM image `apache/kafka:4.3.1` in single-node KRaft mode. No ZooKeeper, Confluent Platform images, Bitnami images, Redpanda, paid services, cloud accounts, or billing-enabled resources are required.

Install the declared dependency groups, including the Kafka client, into `.venv` only:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,mlops,explainability,streaming]"
```

The `streaming` group currently adds exactly `confluent-kafka==2.15.0`.

Start Kafka after Docker Desktop is running with Linux containers:

```powershell
docker compose up -d kafka
```

Check service status:

```powershell
docker compose ps
```

Create or verify the telemetry topic:

```powershell
.\.venv\Scripts\python.exe scripts/setup_kafka.py
```

The setup script is idempotent. Running it repeatedly reuses `industrial.telemetry.v1` when it already has 3 partitions and replication factor 1. If an incompatible topic exists, the script fails and does not mutate it.

Validate Kafka integration with a deterministic smoke produce/consume check:

```powershell
.\.venv\Scripts\python.exe scripts/check_kafka.py
```

Produce a finite deterministic telemetry batch without sleeping between events:

```powershell
.\.venv\Scripts\python.exe scripts/produce_telemetry_kafka.py
```

Consume five records from the beginning with an explicit group ID:

```powershell
.\.venv\Scripts\python.exe scripts/consume_telemetry_kafka.py --group-id manual-validation --from-beginning --max-messages 5 --timeout-seconds 10
```

View Kafka logs:

```powershell
docker compose logs kafka
```

Stop Kafka without deleting data:

```powershell
docker compose stop kafka
```

Start it again:

```powershell
docker compose start kafka
```

Stop and remove containers while preserving named volumes:

```powershell
docker compose down
```

Use `docker compose down -v` only intentionally. It deletes the PostgreSQL and Kafka development volumes and all service data stored in those volumes.

Kafka currently transports complete synthetic telemetry JSON events keyed by `machine_code` and now feeds Spark Structured Streaming Bronze Parquet ingestion. Silver processing reads Bronze Parquet only; it does not consume Kafka directly. Gold descriptive analytics reads Silver Parquet only. This phase does not implement streaming ML inference, anomaly detection, drift monitoring, FastAPI routes, frontend components, Ollama/GenAI behavior, or Databricks integration.

## Spark Bronze Streaming

Spark runs locally through Docker using `apache/spark:4.0.4-scala2.13-java17-python3-ubuntu`. This phase uses `local[2]` inside the Spark container, not a Spark master/worker cluster. The host project does not install `pyspark`.

Start Kafka and Spark:

```powershell
docker compose up -d kafka spark
```

Check services:

```powershell
docker compose ps
```

Run Bronze available-now ingestion:

```powershell
.\.venv\Scripts\python.exe scripts/run_spark_bronze_docker.py
```

Inspect Bronze:

```powershell
docker compose exec -T spark /opt/spark/bin/spark-submit /workspace/scripts/inspect_spark_bronze.py
```

Validate end-to-end Bronze ingestion:

```powershell
.\.venv\Scripts\python.exe scripts/check_spark_bronze.py
```

View Spark logs:

```powershell
docker compose logs spark
```

The first Spark execution may download the pinned Kafka connector dependency `org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4`, so it can take longer. Downloaded connector caches are runtime state and must not be committed.

Generated Bronze Parquet data under `data/bronze/telemetry/` and Structured Streaming checkpoints under `data/checkpoints/spark/bronze_telemetry/` are local runtime data and are Git-ignored. Do not delete checkpoints during normal operation; deleting them intentionally changes replay behavior.

This phase does not implement ML inference, anomaly detection, drift monitoring, PostgreSQL telemetry writes, FastAPI routes, frontend components, Ollama/GenAI behavior, or Databricks integration.

## Spark Silver Processing

Silver processing runs inside the same local Spark Docker container and reads the persisted Bronze Parquet snapshot. It is a deterministic snapshot rebuild, not a second Structured Streaming query. It writes three generated local datasets:

- `data/silver/telemetry`: canonical typed telemetry, one valid record per `event_id`.
- `data/silver/duplicates`: valid non-canonical duplicate business events retained for audit.
- `data/silver/quarantine`: invalid telemetry records with raw payload, Kafka lineage, and rejection reasons.

These directories are local runtime data and are Git-ignored. The tracked `data/silver/.gitkeep` placeholder remains in place.

Start Spark:

```powershell
docker compose up -d spark
```

Run the Silver snapshot rebuild:

```powershell
.\.venv\Scripts\python.exe scripts/run_spark_silver_docker.py
```

Inspect Silver:

```powershell
docker compose exec -T spark /opt/spark/bin/spark-submit /workspace/scripts/inspect_spark_silver.py
```

Validate Silver processing:

```powershell
.\.venv\Scripts\python.exe scripts/check_spark_silver.py
```

Silver does not read Kafka, write PostgreSQL, or perform model inference.

## Spark Gold Analytics

Gold analytics runs inside the same local Spark Docker container and reads the persisted canonical Silver telemetry snapshot. It is a deterministic snapshot rebuild, not a Structured Streaming query. It writes three generated local datasets:

- `data/gold/machine_summary`: one row per canonical Silver machine with descriptive telemetry summaries.
- `data/gold/machine_windows`: one-minute event-time telemetry aggregates by machine.
- `data/gold/fleet_summary`: one row describing the current canonical Silver fleet snapshot.

These directories are local runtime data and are Git-ignored. The tracked `data/gold/.gitkeep` placeholder remains in place.

Start Spark:

```powershell
docker compose up -d spark
```

Run the Gold snapshot rebuild:

```powershell
.\.venv\Scripts\python.exe scripts/run_spark_gold_docker.py
```

Inspect Gold:

```powershell
docker compose exec -T spark /opt/spark/bin/spark-submit /workspace/scripts/inspect_spark_gold.py
```

Validate Gold analytics:

```powershell
.\.venv\Scripts\python.exe scripts/check_spark_gold.py
```

Gold does not read Kafka, write PostgreSQL, perform model inference, assign health or risk scores, or create anomaly labels.

`product_quality_type` is an event-level synthetic telemetry attribute. A machine may have multiple product-quality values across its canonical Silver events. Gold exposes `latest_product_quality_type` from the deterministic latest event and H/L/M event-count distributions rather than treating product quality as a stable machine classification.

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

## AI4I Telemetry Inference Bridge

The telemetry inference bridge uses canonical Silver telemetry, the Spark Docker runtime, and the existing trusted packaged AI4I model artifact. It does not install new project dependencies, retrain the model, read AI4I `test.csv`, write PostgreSQL, calculate SHAP values, or create anomaly labels.

Run the Spark AI4I feature adapter:

```powershell
.\.venv\Scripts\python.exe scripts/run_spark_ai4i_adapter_docker.py
```

Run telemetry inference from the adapted Silver events:

```powershell
.\.venv\Scripts\python.exe scripts/predict_silver_telemetry.py
```

Validate the complete Silver-to-model bridge:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_telemetry_inference.py
```

Generated adapter records are written under:

```text
data/model_input/ai4i/telemetry
```

Generated prediction records are written to:

```text
data/predictions/ai4i/telemetry_predictions.jsonl
```

Both runtime locations are Git-ignored.

## AI4I PostgreSQL Prediction Persistence

The persistence step consumes only the existing runtime prediction JSONL output. It does not regenerate predictions, run model inference, create alerts, or create anomaly records.

Ensure PostgreSQL is running:

```powershell
docker compose up -d postgres
```

Validate PostgreSQL:

```powershell
.\.venv\Scripts\python.exe scripts/check_postgres.py
```

Apply any pending schema migrations:

```powershell
.\.venv\Scripts\python.exe scripts/apply_migrations.py
```

Generate and validate telemetry predictions if the runtime prediction file is missing:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_telemetry_inference.py
```

Persist the current prediction batch:

```powershell
.\.venv\Scripts\python.exe scripts/persist_ai4i_predictions.py
```

Inspect current persisted prediction state:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_ai4i_prediction_state.py
```

Validate persistence and idempotency:

```powershell
.\.venv\Scripts\python.exe scripts/check_ai4i_prediction_persistence.py
```

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

The current tests cover environment validation logic, local infrastructure helpers, synthetic telemetry simulator helpers, AI4I data validation, EDA helpers, AI4I modeling-data preparation logic, AI4I baseline modeling helpers, AI4I imbalance/threshold strategy helpers, model-family comparison helpers, Random Forest tuning helpers, final holdout evaluation helpers, local AI4I packaging/inference helpers, AI4I telemetry inference bridge helpers, local MLflow retrospective tracking helpers, and AI4I SHAP explainability helpers. Synthetic unit tests do not depend on the real 10,000-row AI4I dataset.

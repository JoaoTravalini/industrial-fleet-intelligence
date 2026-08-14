# Architecture

This document describes the high-level architecture for the Industrial Fleet Intelligence Platform. Components are labeled as implemented or planned so the documentation does not imply that future phases already exist.

## Architecture Principles

- Local-first execution with Docker Desktop and WSL2 on Windows.
- No paid APIs, paid services, or cloud resources that require billing information.
- Reproducible development using open-source tools and documented local configuration.
- English-only source code, documentation, variables, comments, and UI text.
- No confidential, proprietary, or official third-party branding or data.

## Implemented Components

- Repository architecture scaffolding for the planned platform areas.
- Read-only developer environment validation through `scripts/check_environment.py`.
- Python development tooling configuration with pytest and Ruff.
- Local PostgreSQL infrastructure through Docker Compose using the official `postgres:18.4` image.
- PostgreSQL persistence through a named Docker volume, not a host bind mount.
- PostgreSQL health validation through a Compose health check and `scripts/check_postgres.py`.
- Initial PostgreSQL operational relational schema for structured platform entities.
- SQL migration execution through `scripts/apply_migrations.py`.
- Read-only schema validation through `scripts/check_schema.py`.
- Fictional operational fleet seed for 100 generic simulated industrial machines.
- AI4I external dataset acquisition from the UCI Machine Learning Repository.
- AI4I structural validation and factual profile reporting.
- Reproducible AI4I exploratory data analysis with derived reports and static Matplotlib plots.
- AI4I leakage-safe modeling feature policy through `ml/config/ai4i_modeling.json`.
- AI4I deterministic stratified train/validation/test split preparation and validation.
- AI4I training-only preprocessing fit for baseline classifiers.
- AI4I Dummy classification baseline for a trivial validation benchmark.
- AI4I Logistic Regression baseline as the first predictive model.
- AI4I validation-only baseline evaluation with the locked test split unused.
- AI4I train-only 5-fold out-of-fold model development for Logistic Regression variants.
- AI4I class-weight imbalance experiment comparing standard and balanced Logistic Regression.
- AI4I threshold trade-off analysis using train OOF probabilities only.
- AI4I validation evaluation of a train-selected imbalance/threshold candidate.
- AI4I Logistic Regression reference for model-family comparison.
- AI4I Random Forest non-linear baseline.
- AI4I XGBoost non-linear baseline.
- AI4I train-only 5-fold out-of-fold model-family comparison.
- AI4I deterministic Average Precision-based candidate selection.
- AI4I validation evaluation of the train-selected model-family candidate.
- AI4I targeted Random Forest hyperparameter tuning.
- AI4I nested train-only cross-validation for leakage-safe tuning estimates.
- AI4I leakage-safe Random Forest hyperparameter selection.
- AI4I train-derived tuned Random Forest threshold strategy.
- AI4I validation evaluation of the train-selected Random Forest candidate.
- AI4I frozen final classifier specification for the retained fixed Random Forest.
- AI4I final train + validation refit using the frozen classifier specification.
- AI4I locked test holdout evaluation and final test performance reporting.
- AI4I frozen model packaging as a local joblib artifact.
- AI4I local model artifact integrity metadata.
- AI4I reusable local inference module.
- AI4I strict inference feature contract.
- AI4I single-record and batch local inference CLI.
- AI4I local MLflow experiment tracking.
- AI4I SQLite-backed MLflow metadata store.
- AI4I retrospective experiment-history import.
- AI4I experiment provenance tags for historical runs.
- AI4I final holdout result tracking in MLflow.
- AI4I model packaging provenance tracking in MLflow.
- AI4I SHAP TreeExplainer integration for the packaged Random Forest.
- AI4I positive-class Random Forest explanations.
- AI4I global feature attribution reports and plots.
- AI4I representative local explanations and waterfall plots.
- AI4I explainability artifact validation.
- Synthetic telemetry event contract for simulator output.
- Deterministic local telemetry simulator.
- Temporal per-machine telemetry state evolution.
- Canonical tracked telemetry sample and deterministic summary.
- Telemetry event and batch schema validation.
- Local Apache Kafka infrastructure through Docker Compose using the official `apache/kafka:4.3.1` JVM image.
- Single-node Kafka KRaft broker and controller without ZooKeeper.
- Kafka host listener for Windows development and Docker-network listener for future Compose clients.
- Kafka telemetry topic configuration for `industrial.telemetry.v1` with 3 partitions and replication factor 1.
- Idempotent Kafka topic setup through `scripts/setup_kafka.py`.
- Deterministic telemetry Kafka producer using UTF-8 JSON payloads and `machine_code` message keys.
- Finite Kafka telemetry consumer with explicit consumer groups, disabled auto-commit, payload validation, and separate Kafka metadata.
- Kafka integration validation through `scripts/check_kafka.py`.
- Deterministic Kafka integration configuration summary.
- Local Apache Spark runtime through Docker Compose using `apache/spark:4.0.4-scala2.13-java17-python3-ubuntu`.
- PySpark Structured Streaming available-now ingestion from Kafka to Bronze.
- Spark Kafka connector integration through `org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4`.
- Bronze telemetry Parquet ingestion at `data/bronze/telemetry/`.
- Bronze preservation of Kafka topic, partition, offset, timestamp, key, raw value, ingestion timestamp, and payload SHA-256.
- Structured Streaming checkpointing at `data/checkpoints/spark/bronze_telemetry/`.
- Spark Bronze inspection through `scripts/inspect_spark_bronze.py`.
- Spark Bronze integration validation through `scripts/check_spark_bronze.py`.
- Deterministic Spark Bronze static configuration summary.
- Spark Bronze-to-Silver snapshot processing through the existing local Spark Docker runtime.
- Explicit Spark telemetry parsing schema for Silver records.
- Silver telemetry contract validation with stable data-quality rejection reasons.
- Silver invalid-record quarantine at `data/silver/quarantine/`.
- Silver business-event deduplication by `event_id`.
- Canonical Silver telemetry at `data/silver/telemetry/`.
- Valid duplicate business-event audit dataset at `data/silver/duplicates/`.
- Kafka lineage preservation from Bronze into Silver outputs.
- Spark Silver inspection through `scripts/inspect_spark_silver.py`.
- Spark Silver integration validation through `scripts/check_spark_silver.py`.
- Deterministic Spark Silver static policy summary.
- Spark Gold descriptive analytics over canonical Silver telemetry.
- Per-machine telemetry summaries at `data/gold/machine_summary/`, including deterministic latest product-quality observation and product-quality event-count distributions.
- Deterministic latest-machine observations using event time plus Kafka lineage ordering.
- One-minute machine telemetry aggregates at `data/gold/machine_windows/`, including product-quality event-count distributions without splitting the machine/window grain by product quality.
- Fleet-level descriptive summary at `data/gold/fleet_summary/`.
- Bronze -> Silver -> Gold local Medallion architecture.
- Spark Gold inspection through `scripts/inspect_spark_gold.py`.
- Spark Gold integration validation through `scripts/check_spark_gold.py`.
- Deterministic Spark Gold static policy summary.
- Spark AI4I feature adapter over canonical Silver telemetry.
- Explicit event-level `product_quality_type` -> AI4I `Type` mapping.
- Silver telemetry batch model inference through the trusted packaged AI4I predictor.
- Deterministic model-input audit hashing with `model_input_sha256`.
- Runtime telemetry prediction output at `data/predictions/ai4i/telemetry_predictions.jsonl`.
- Data Engineering -> ML integration from canonical Silver events to local model predictions.
- AI4I telemetry prediction persistence into PostgreSQL `model_predictions`.
- Model prediction provenance in PostgreSQL, including model identity, model-input hash, and Kafka/source lineage.
- Idempotent prediction ingestion using `event_id`, `model_name`, `model_version`, and `final_config_hash`.
- Latest per-machine ML prediction projection in PostgreSQL `machine_health`.
- Deterministic latest prediction selection using event time plus Kafka/source lineage ordering.
- Data Engineering -> ML -> operational PostgreSQL integration for persisted AI4I telemetry predictions.
- Operational per-prediction AI4I SHAP materialization for persisted telemetry predictions.
- Model-input hash alignment validation between persisted predictions and canonical adapter records.
- Persisted prediction explanations in PostgreSQL `prediction_explanations`.
- Read-only FastAPI prediction explanation endpoint.
- Independent operational telemetry anomaly detector for canonical Silver telemetry.
- Vibration/pressure anomaly feature contract using exactly `vibration_mm_s` and `pressure_bar`.
- Frozen synthetic anomaly reference baseline for model version `1.0.0`.
- Isolation Forest anomaly scoring with higher-is-more-anomalous score semantics.
- Runtime anomaly output at `data/anomalies/telemetry_anomalies.jsonl`.
- Anomaly detector provenance in PostgreSQL `anomalies`, including model identity, baseline hashes, feature values, flag, score, and Kafka/source lineage.
- Idempotent anomaly ingestion using `event_id`, `model_name`, `model_version`, and `model_config_hash`.
- AI4I/anomaly semantic separation with no combined health score, no alert creation, and no AI4I feature modification.
- Frozen AI4I model-input reference profile using train + validation only.
- Frozen anomaly-input reference profile tied to the anomaly baseline hashes.
- PSI feature drift monitoring for AI4I model inputs and anomaly input features.
- Numeric range and standardized mean-shift diagnostics for monitored inputs.
- Categorical product-quality / AI4I `Type` drift diagnostics.
- Deterministic current-population hashing for monitored input records.
- PostgreSQL drift history in `drift_snapshots` and `drift_feature_metrics`.
- Idempotent monitoring snapshot persistence with immutable conflict detection.
- Deterministic operational alert materialization from persisted model/anomaly state.
- PostgreSQL-backed FastAPI backend for local read APIs.
- Fleet overview API endpoint.
- Machine list and detail API endpoints.
- Prediction history API endpoint.
- Anomaly history API endpoint.
- Drift monitoring API endpoint.
- Alert read API endpoints.
- Prediction explanation read API endpoint.
- FastAPI OpenAPI documentation.
- Local frontend CORS configuration for `http://localhost:5173`.
- React + TypeScript frontend application shell.
- Vite development and production build tooling for the web dashboard.
- Typed FastAPI fetch client configured by `VITE_API_BASE_URL`.
- TanStack Query read-only server-state integration.
- React Router frontend routing for overview, machines, machine detail, alerts, drift monitoring, and Not Found pages.
- Fleet overview dashboard consuming materialized API state.
- Machine list and machine detail views with server-side pagination and recent history slices.
- Recent AI4I prediction and operational anomaly read views.
- Operational alert monitoring UI with supported API filters.
- Drift monitoring UI for AI4I model inputs and operational anomaly inputs.
- Prediction probability history visualization.
- Vibration and pressure telemetry visualization from anomaly audit history.
- Anomaly-score history visualization.
- PSI drift visualization for AI4I and anomaly input scopes.
- Interactive machine prediction explanation UX backed by persisted SHAP attributions.
- Responsive dashboard application shell with custom CSS.
## Planned Component Areas

- `apps/api`: Implemented PostgreSQL-backed FastAPI backend for read-oriented operational APIs.
- `apps/web`: Implemented React, TypeScript, and Vite read-only dashboard for fleet overview, machine monitoring, alert monitoring, drift monitoring, operational charts, and persisted prediction explanations.
- Maintenance history generation is planned for a later phase.
- Alert lifecycle mutation and automated resolution are planned for a later phase.
- Streaming ML inference over telemetry is planned for a later phase.
- Streaming anomaly detection is planned for a later phase.
- Additional production feature engineering workflows are planned for a later phase.
- MLflow Model Registry, registered deployment model, and automated drift workflows are planned for later phases.
- `services/database`: Implemented reusable AI4I prediction persistence, prediction explanation persistence, telemetry anomaly persistence, and drift persistence helpers for PostgreSQL validation and idempotency.
- `services/simulator`: Implemented deterministic synthetic industrial telemetry generator and Kafka producer integration.
- `services/streaming`: Implemented local Apache Kafka configuration, producer, consumer, topic setup, and validation helpers.
- `services/copilot`: Planned local generative AI copilot integration through Ollama only.
- `pipelines/batch`: Implemented deterministic Spark Gold descriptive analytics, the Spark AI4I feature adapter, and Silver-to-anomaly feature extraction; additional historical feature jobs are planned.
- `pipelines/streaming`: Implemented Spark Structured Streaming Kafka-to-Bronze ingestion and deterministic Bronze-to-Silver processing.
- `ml/training`: Planned model training workflows using scikit-learn and XGBoost.
- `ml/inference`: Implemented local AI4I inference utilities and Silver telemetry inference bridge; service integration is planned.
- `ml/anomaly`: Implemented local operational telemetry anomaly detector utilities.
- `ml/artifacts`: Implemented local output location for ignored generated model artifacts.
- `data`: Local storage for external raw data, generated modeling data, implemented lakehouse layers, ignored model-input handoff data, and ignored runtime prediction data.
- `docs`: Project documentation.
- `tests`: Python test suite.
- `scripts`: Local developer automation and validation scripts.
- `.github/workflows`: Planned GitHub Actions CI configuration.

## Implemented PostgreSQL Infrastructure

The local PostgreSQL service is implemented in `docker-compose.yml`. It binds to `127.0.0.1` with a configurable host port, stores database files in the named Docker volume `industrial_fleet_postgres18_data` mounted at `/var/lib/postgresql`, and uses `pg_isready` for container health checks.

## Implemented PostgreSQL Operational Schema

The initial operational schema is implemented through versioned SQL migrations in `db/migrations`. It defines structured relational tables for machines, maintenance records, model predictions, anomaly detections, operational alerts, and the latest machine health state.

PostgreSQL must not become the primary store for high-volume raw telemetry history. The schema intentionally avoids raw telemetry tables; telemetry events now flow through local Kafka into Spark-managed Bronze Parquet, Silver telemetry, and Gold descriptive analytics.

## Implemented Development Seed Data

A deterministic fictional development seed populates `machines` with 100 generic simulated industrial assets identified as `MCH-0001` through `MCH-0100`. It does not use real manufacturer, proprietary, telemetry, maintenance, prediction, anomaly, alert, or machine-health data.

## Implemented AI4I Dataset Work

AI4I is an external public synthetic dataset used for data-science and predictive-maintenance portfolio development. Implemented work currently includes acquisition, structural validation, descriptive EDA, leakage-safe modeling feature policy, deterministic stratified train/validation/test split creation, read-only validation of generated modeling artifacts, training-only preprocessing fit for baseline classifiers, Dummy baseline evaluation, Logistic Regression baseline evaluation, train-only 5-fold OOF model development, class-weight imbalance comparison, threshold trade-off analysis, fixed-configuration Logistic Regression, Random Forest, and XGBoost model-family comparison, deterministic train OOF Average Precision-based candidate selection, targeted Random Forest hyperparameter tuning with nested train-only cross-validation, train-derived tuned threshold strategy, validation-only reporting, frozen final classifier specification, final train + validation refit, locked test holdout evaluation, final test performance reporting, local final model packaging, artifact integrity metadata, strict inference input/output contracts, single/batch local inference, local retrospective MLflow tracking, and SHAP explainability for the packaged Random Forest.

The modeling dataset uses `source_udi` only for traceability. `Product ID` is excluded as an identifier, and `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they are target-adjacent failure-mode flags. Baseline and imbalance-strategy preprocessing is fitted on training data only or inside train-only CV fold pipelines. Validation data is transformed through fitted pipelines after candidate selection. Non-linear model-family comparison and targeted Random Forest tuning are implemented with constrained train-only development protocols. The final holdout phase freezes the fixed Random Forest configuration and threshold before opening the test split for final evaluation only. The local packaging phase fits the frozen pipeline on train + validation only and does not reuse the test split. No feature selection, MLflow Model Registry, registered deployment model, live API model serving, browser-side inference, or browser-side SHAP calculation has been implemented.

Generated files under `data/processed/ai4i/` are reproducible modeling artifacts derived from the external AI4I dataset and are ignored by Git. They are separate from the implemented `data/bronze`, `data/silver`, and `data/gold` telemetry lakehouse layers.

The implemented telemetry inference bridge uses the frozen packaged AI4I predictor against adapted canonical Silver telemetry events. It does not read Gold aggregates, AI4I `test.csv`, PostgreSQL operational machine types, SHAP outputs, or anomaly fields.

## Data Concept Boundaries

1. `MCH-XXXX` PostgreSQL fleet: fictional operational assets used by the application.
2. AI4I dataset: external public synthetic dataset used for Data Science and ML development.
3. `data/processed/ai4i`: reproducible local modeling datasets derived from AI4I.
4. Synthetic telemetry simulator: deterministic generated telemetry observations that can flow through local Kafka into Spark-managed Bronze Parquet.
5. `data/model_input`, `data/predictions`, and `data/anomalies`: ignored runtime handoff, prediction, and anomaly outputs derived from canonical Silver telemetry.

AI4I is not inserted into PostgreSQL and is not treated as operational application state.

## Planned Data Flow

1. The implemented local simulator generates deterministic synthetic fleet telemetry samples.
2. Telemetry events can now flow through the implemented local Apache Kafka broker.
3. The implemented Spark available-now Structured Streaming job ingests Kafka telemetry into local Bronze Parquet.
4. The implemented Silver job parses, validates, quarantines, deduplicates, and curates telemetry from Bronze.
5. The implemented Gold job derives descriptive machine, time-window, and fleet analytics.
6. The implemented Spark AI4I feature adapter maps canonical Silver events into the frozen six-feature model contract.
7. The implemented host inference bridge writes deterministic telemetry failure-risk predictions locally.
8. The implemented PostgreSQL persistence step stores AI4I prediction history and updates the latest per-machine ML prediction projection.
9. The implemented telemetry anomaly feature extraction reads canonical Silver telemetry for `vibration_mm_s` and `pressure_bar`.
10. The implemented local anomaly detector writes deterministic runtime anomaly outputs.
11. The implemented PostgreSQL anomaly persistence step stores auditable detector outputs without creating alerts.
12. The implemented drift monitor compares frozen AI4I and anomaly input references against current operational inputs.
13. The implemented PostgreSQL drift persistence step stores deterministic monitoring snapshots without creating alerts.
14. Historical AI4I experiment reports are tracked locally with MLflow; future live training workflows may extend this pattern.
15. Implemented AI4I SHAP reports support local model interpretation, and operational per-prediction explanations can be materialized, persisted, and exposed separately from predictions.
16. The implemented FastAPI backend exposes local platform capabilities to the implemented web dashboard.
17. The implemented React dashboard visualizes fleet overview, machine projections, recent prediction/anomaly history, alerts, drift monitoring, visual analytics, and persisted SHAP attribution details without browser-side inference.
18. Final CI/demo polish remains planned.
19. The copilot service will use a local Ollama model for natural-language assistance.

## Planned Local Technology Stack

- Python 3.12
- JDK 17
- Node.js 24 LTS
- FastAPI backend is implemented for local read APIs.
- React, TypeScript, and Vite dashboard tooling is implemented
- PostgreSQL local infrastructure, the operational schema, and FastAPI read access are implemented.
- Apache Kafka local infrastructure is implemented.
- Apache Spark local Docker runtime, PySpark Structured Streaming Bronze ingestion, deterministic Silver processing, Gold descriptive analytics, and the AI4I feature adapter are implemented.
- scikit-learn and XGBoost
- SHAP
- MLflow
- Ollama
- Docker Compose
- pytest and frontend test tooling
- GitHub Actions

## Current Phase Scope

The current completed scope includes operational AI4I SHAP materialization for persisted telemetry predictions, PostgreSQL explanation persistence, a read-only explanation API, Recharts-based operational visual analytics, and an interactive machine prediction explanation UX. It does not implement the local AI copilot, authentication, Dockerized frontend/API services, model retraining, runtime inference or SHAP calculation in API requests, Spark execution in API requests, Kafka consumption in API requests, or Databricks integration.



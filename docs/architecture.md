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

## Planned Component Areas

- `apps/api`: Planned FastAPI backend for serving platform APIs and model-facing endpoints.
- `apps/web`: Planned React, TypeScript, and Vite dashboard for fleet monitoring and analysis.
- Maintenance history generation is planned for a later phase.
- Telemetry generation is planned for a later phase.
- Preprocessing fit and production feature engineering are planned for a later phase.
- Final model decision and final model selection are planned for later phases.
- Final locked test evaluation is planned for a later phase.
- MLflow Model Registry, registered deployment model, drift monitoring workflows, and explanation exposure through API are planned for later phases.
- `services/simulator`: Planned synthetic industrial telemetry generator.
- `services/streaming`: Planned local streaming support around Apache Kafka producers and consumers.
- `services/copilot`: Planned local generative AI copilot integration through Ollama only.
- `pipelines/batch`: Planned PySpark batch processing jobs for historical data preparation.
- `pipelines/streaming`: Planned PySpark streaming processing jobs for real-time telemetry.
- `ml/training`: Planned model training workflows using scikit-learn and XGBoost.
- `ml/inference`: Implemented local AI4I inference utilities; service integration is planned.
- `ml/artifacts`: Implemented local output location for ignored generated model artifacts.
- `data`: Local storage for external raw data, generated modeling data, and planned lakehouse layers.
- `docs`: Project documentation.
- `tests`: Python test suite.
- `scripts`: Local developer automation and validation scripts.
- `.github/workflows`: Planned GitHub Actions CI configuration.

## Implemented PostgreSQL Infrastructure

The local PostgreSQL service is implemented in `docker-compose.yml` as the only Docker Compose service for this phase. It binds to `127.0.0.1` with a configurable host port, stores database files in the named Docker volume `industrial_fleet_postgres18_data` mounted at `/var/lib/postgresql`, and uses `pg_isready` for container health checks.

## Implemented PostgreSQL Operational Schema

The initial operational schema is implemented through versioned SQL migrations in `db/migrations`. It defines structured relational tables for machines, maintenance records, model predictions, anomaly detections, operational alerts, and the latest machine health state.

PostgreSQL must not become the primary store for high-volume raw telemetry history. The schema intentionally avoids raw telemetry tables; future telemetry events are planned to flow through Kafka and Spark into local Bronze, Silver, and Gold data lake layers.

## Implemented Development Seed Data

A deterministic fictional development seed populates `machines` with 100 generic simulated industrial assets identified as `MCH-0001` through `MCH-0100`. It does not use real manufacturer, proprietary, telemetry, maintenance, prediction, anomaly, alert, or machine-health data.

## Implemented AI4I Dataset Work

AI4I is an external public synthetic dataset used for data-science and predictive-maintenance portfolio development. Implemented work currently includes acquisition, structural validation, descriptive EDA, leakage-safe modeling feature policy, deterministic stratified train/validation/test split creation, read-only validation of generated modeling artifacts, training-only preprocessing fit for baseline classifiers, Dummy baseline evaluation, Logistic Regression baseline evaluation, train-only 5-fold OOF model development, class-weight imbalance comparison, threshold trade-off analysis, fixed-configuration Logistic Regression, Random Forest, and XGBoost model-family comparison, deterministic train OOF Average Precision-based candidate selection, targeted Random Forest hyperparameter tuning with nested train-only cross-validation, train-derived tuned threshold strategy, validation-only reporting, frozen final classifier specification, final train + validation refit, locked test holdout evaluation, final test performance reporting, local final model packaging, artifact integrity metadata, strict inference input/output contracts, single/batch local inference, local retrospective MLflow tracking, and SHAP explainability for the packaged Random Forest.

The modeling dataset uses `source_udi` only for traceability. `Product ID` is excluded as an identifier, and `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they are target-adjacent failure-mode flags. Baseline and imbalance-strategy preprocessing is fitted on training data only or inside train-only CV fold pipelines. Validation data is transformed through fitted pipelines after candidate selection. Non-linear model-family comparison and targeted Random Forest tuning are implemented with constrained train-only development protocols. The final holdout phase freezes the fixed Random Forest configuration and threshold before opening the test split for final evaluation only. The local packaging phase fits the frozen pipeline on train + validation only and does not reuse the test split. No feature selection, MLflow Model Registry, registered deployment model, drift monitoring, API integration, dashboard integration, or explanation exposure through API has been implemented.

Generated files under `data/processed/ai4i/` are reproducible modeling artifacts derived from the external AI4I dataset and are ignored by Git. They are separate from the planned `data/bronze`, `data/silver`, and `data/gold` telemetry lakehouse layers.

## Data Concept Boundaries

1. `MCH-XXXX` PostgreSQL fleet: fictional operational assets used by the application.
2. AI4I dataset: external public synthetic dataset used for Data Science and ML development.
3. `data/processed/ai4i`: reproducible local modeling datasets derived from AI4I.
4. Future telemetry simulator: generated streaming data that will eventually flow through Kafka and Spark.

AI4I is not inserted into PostgreSQL and is not treated as operational application state.

## Planned Data Flow

1. A local simulator will generate synthetic fleet telemetry.
2. Telemetry events will flow through a local Apache Kafka broker.
3. Streaming jobs will process real-time data into curated local storage layers.
4. Batch jobs will prepare historical features for analytics and machine learning.
5. Historical AI4I experiment reports are tracked locally with MLflow; future live training workflows may extend this pattern.
6. Implemented AI4I SHAP reports support local model interpretation; future APIs may expose explanation data separately from predictions.
7. The FastAPI backend will expose local platform capabilities to the web dashboard.
8. The React dashboard will visualize fleet health, alerts, predictions, and model insights.
9. The copilot service will use a local Ollama model for natural-language assistance.

## Planned Local Technology Stack

- Python 3.12
- JDK 17
- Node.js 24 LTS
- FastAPI
- React, TypeScript, and Vite
- PostgreSQL local infrastructure and the initial operational schema are implemented; application data access is planned.
- Apache Kafka
- PySpark
- scikit-learn and XGBoost
- SHAP
- MLflow
- Ollama
- Docker Compose
- pytest and frontend test tooling
- GitHub Actions

## Current Phase Scope

This phase implements local AI4I SHAP explainability for the packaged Random Forest: positive-class TreeExplainer attribution, deterministic global and local reports, representative waterfall plots, sample-payload explanations, additivity validation, and a read-only validator. It does not implement MLflow Model Registry, registered deployment models, drift monitoring, anomaly detection, explanation exposure through API, database import, telemetry generation, Kafka, Spark, API routes, frontend components, GenAI behavior, or Databricks integration.
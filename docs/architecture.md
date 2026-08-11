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

## Planned Component Areas

- `apps/api`: Planned FastAPI backend for serving platform APIs and model-facing endpoints.
- `apps/web`: Planned React, TypeScript, and Vite dashboard for fleet monitoring and analysis.
- Maintenance history generation is planned for a later phase.
- Telemetry generation is planned for a later phase.
- Preprocessing fit and production feature engineering are planned for a later phase.
- Baseline classification model training is planned for a later phase.
- Model evaluation is planned for a later phase.
- Class imbalance strategy is planned for a later phase.
- Advanced ML, MLflow tracking, and explainability workflows are planned for later phases.
- `services/simulator`: Planned synthetic industrial telemetry generator.
- `services/streaming`: Planned local streaming support around Apache Kafka producers and consumers.
- `services/copilot`: Planned local generative AI copilot integration through Ollama only.
- `pipelines/batch`: Planned PySpark batch processing jobs for historical data preparation.
- `pipelines/streaming`: Planned PySpark streaming processing jobs for real-time telemetry.
- `ml/training`: Planned model training workflows using scikit-learn and XGBoost.
- `ml/inference`: Planned model inference utilities and service integration code.
- `ml/artifacts`: Planned local output location for generated model artifacts, excluded from version control.
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

AI4I is an external public synthetic dataset used for data-science and predictive-maintenance portfolio development. Implemented work currently includes acquisition, structural validation, descriptive EDA, leakage-safe modeling feature policy, deterministic stratified train/validation/test split creation, and read-only validation of generated modeling artifacts.

The modeling dataset uses `source_udi` only for traceability. `Product ID` is excluded as an identifier, and `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they are target-adjacent failure-mode flags. No model, fitted transformer, class balancing, feature selection, or evaluation has been implemented.

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
5. Training workflows will track experiments locally with MLflow.
6. Explainability workflows will use SHAP to support model interpretation.
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

This phase implements the leakage-safe AI4I modeling dataset definition and deterministic stratified train/validation/test split only. It does not implement fitted preprocessing, production feature engineering, classifier training, model evaluation, class imbalance handling, MLflow, database import, telemetry generation, Kafka, Spark, API routes, frontend components, GenAI behavior, or Databricks integration.
# Planned Architecture

This document describes the planned high-level architecture for the Industrial Fleet Intelligence Platform. It is a forward-looking design outline only; these components are not implemented yet.

## Architecture Principles

- Local-first execution with Docker Desktop and WSL2 on Windows.
- No paid APIs, paid services, or cloud resources that require billing information.
- Reproducible development using open-source tools and documented local configuration.
- English-only source code, documentation, variables, comments, and UI text.
- No confidential, proprietary, or official third-party branding or data.

## Planned Component Areas

- `apps/api`: Planned FastAPI backend for serving platform APIs and model-facing endpoints.
- `apps/web`: Planned React, TypeScript, and Vite dashboard for fleet monitoring and analysis.
- `services/simulator`: Planned synthetic industrial telemetry generator.
- `services/streaming`: Planned local streaming support around Apache Kafka producers and consumers.
- `services/copilot`: Planned local generative AI copilot integration through Ollama only.
- `pipelines/batch`: Planned PySpark batch processing jobs for historical data preparation.
- `pipelines/streaming`: Planned PySpark streaming processing jobs for real-time telemetry.
- `ml/training`: Planned model training workflows using scikit-learn and XGBoost.
- `ml/inference`: Planned model inference utilities and service integration code.
- `ml/artifacts`: Planned local output location for generated model artifacts, excluded from version control.
- `data`: Planned local data lake zones for raw, bronze, silver, gold, and sample data.
- `docs`: Project documentation.
- `tests`: Planned Python test suite.
- `scripts`: Planned local developer automation scripts.
- `.github/workflows`: Planned GitHub Actions CI configuration.

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
- PostgreSQL
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

This phase initializes the repository architecture only. It does not implement Kafka, PostgreSQL, Spark, machine learning, API routes, frontend components, Docker services, GenAI behavior, or Databricks integration.

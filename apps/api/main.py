"""FastAPI application entry point for the local portfolio backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.config import ApiSettings, get_settings
from apps.api.routes import alerts, drift, fleet, health, machines

API_TITLE = "Industrial Fleet Intelligence API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Read-oriented API for an independent local portfolio project. The service exposes "
    "already-materialized operational PostgreSQL state and does not perform runtime model "
    "inference, Spark processing, Kafka consumption, or official industrial deployment duties."
)


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(fleet.router)
    app.include_router(machines.router)
    app.include_router(drift.router)
    app.include_router(alerts.router)
    return app


app = create_app()

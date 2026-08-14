"""Common route helpers."""

from __future__ import annotations

from fastapi import HTTPException

from apps.api.db import DatabaseUnavailableError
from apps.api.repositories.platform import (
    AlertNotFoundError,
    MachineNotFoundError,
    PredictionExplanationNotFoundError,
)


def unavailable(exc: DatabaseUnavailableError) -> HTTPException:
    return HTTPException(status_code=503, detail="PostgreSQL is unavailable.")


def machine_not_found(exc: MachineNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown machine_code: {exc}")


def alert_not_found(exc: AlertNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown alert_id: {exc}")


def prediction_explanation_not_found(exc: PredictionExplanationNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Prediction explanation not found: {exc}")

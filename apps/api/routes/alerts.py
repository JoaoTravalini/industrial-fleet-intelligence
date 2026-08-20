"""Read-only operational alert endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.repositories.platform import AlertNotFoundError, PlatformRepository
from apps.api.routes.common import alert_not_found, unavailable
from apps.api.schemas import AlertListResponse, AlertResponse

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])
Repository = Annotated[PlatformRepository, Depends(get_repository)]
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]
AlertStatus = Literal["open", "acknowledged", "resolved"]
AlertSeverity = Literal["info", "warning", "critical"]


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List operational alerts",
    description="Returns persisted read-only operational alerts with optional filters.",
)
def list_alerts(
    repository: Repository,
    limit: Limit = 50,
    offset: Offset = 0,
    status: AlertStatus | None = None,
    severity: AlertSeverity | None = None,
    alert_type: str | None = None,
    machine_code: str | None = None,
) -> dict[str, object]:
    try:
        return repository.list_alerts(
            limit=limit,
            offset=offset,
            status=status,
            severity=severity,
            alert_type=alert_type,
            machine_code=machine_code,
        )
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get operational alert",
    description="Returns one persisted operational alert by identifier.",
)
def get_alert(alert_id: int, repository: Repository) -> dict[str, object]:
    try:
        return repository.get_alert(alert_id)
    except AlertNotFoundError as exc:
        raise alert_not_found(exc) from exc
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc

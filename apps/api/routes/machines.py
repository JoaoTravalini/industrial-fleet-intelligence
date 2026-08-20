"""Machine and machine-history endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.repositories.platform import (
    MachineNotFoundError,
    PlatformRepository,
    PredictionExplanationNotFoundError,
)
from apps.api.routes.common import (
    machine_not_found,
    prediction_explanation_not_found,
    unavailable,
)
from apps.api.schemas import (
    AnomalyListResponse,
    MachineDetailResponse,
    MachineListResponse,
    PredictionExplanationResponse,
    PredictionListResponse,
)

router = APIRouter(prefix="/api/v1/machines", tags=["machines"])
Repository = Annotated[PlatformRepository, Depends(get_repository)]
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]
MachineStatus = Literal["active", "maintenance", "inactive"]


@router.get(
    "",
    response_model=MachineListResponse,
    summary="List machines",
    description="Returns paginated fictional operational machines from PostgreSQL.",
)
def list_machines(
    repository: Repository,
    limit: Limit = 50,
    offset: Offset = 0,
    status: MachineStatus | None = None,
) -> dict[str, object]:
    try:
        return repository.list_machines(limit=limit, offset=offset, status=status)
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get(
    "/{machine_code}",
    response_model=MachineDetailResponse,
    summary="Get machine detail",
    description="Returns a machine detail projection with latest materialized health state.",
)
def get_machine(machine_code: str, repository: Repository) -> dict[str, object]:
    try:
        return repository.get_machine(machine_code)
    except MachineNotFoundError as exc:
        raise machine_not_found(exc) from exc
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get(
    "/{machine_code}/predictions",
    response_model=PredictionListResponse,
    summary="List machine predictions",
    description="Returns persisted AI4I failure-risk model estimates for a machine.",
)
def list_predictions(
    machine_code: str,
    repository: Repository,
    limit: Limit = 50,
    offset: Offset = 0,
) -> dict[str, object]:
    try:
        return repository.list_machine_predictions(machine_code, limit=limit, offset=offset)
    except MachineNotFoundError as exc:
        raise machine_not_found(exc) from exc
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get(
    "/{machine_code}/predictions/{event_id}/explanation",
    response_model=PredictionExplanationResponse,
    summary="Get persisted prediction explanation",
    description="Returns persisted SHAP model attributions for one machine prediction event.",
)
def get_prediction_explanation(
    machine_code: str,
    event_id: str,
    repository: Repository,
) -> dict[str, object]:
    try:
        return repository.get_prediction_explanation(machine_code, event_id)
    except MachineNotFoundError as exc:
        raise machine_not_found(exc) from exc
    except PredictionExplanationNotFoundError as exc:
        raise prediction_explanation_not_found(exc) from exc
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get(
    "/{machine_code}/anomalies",
    response_model=AnomalyListResponse,
    summary="List machine anomalies",
    description="Returns persisted anomaly detector scores and flags for a machine.",
)
def list_anomalies(
    machine_code: str,
    repository: Repository,
    limit: Limit = 50,
    offset: Offset = 0,
    flagged_only: bool = False,
) -> dict[str, object]:
    try:
        return repository.list_machine_anomalies(
            machine_code,
            limit=limit,
            offset=offset,
            flagged_only=flagged_only,
        )
    except MachineNotFoundError as exc:
        raise machine_not_found(exc) from exc
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc

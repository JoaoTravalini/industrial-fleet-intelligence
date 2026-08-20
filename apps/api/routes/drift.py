"""Drift monitoring read endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.repositories.platform import PlatformRepository
from apps.api.routes.common import unavailable
from apps.api.schemas import DriftLatestResponse

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])
Repository = Annotated[PlatformRepository, Depends(get_repository)]


@router.get(
    "/latest",
    response_model=DriftLatestResponse,
    summary="Get latest drift snapshot",
    description="Returns the latest persisted input-distribution drift diagnostics.",
)
def latest_drift(repository: Repository) -> dict[str, object]:
    try:
        return repository.latest_drift()
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc

"""Fleet-level read endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.repositories.platform import PlatformRepository
from apps.api.routes.common import unavailable
from apps.api.schemas import FleetOverviewResponse

router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])
Repository = Annotated[PlatformRepository, Depends(get_repository)]


@router.get("/overview", response_model=FleetOverviewResponse)
def overview(repository: Repository) -> dict[str, object]:
    try:
        return repository.fleet_overview()
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc

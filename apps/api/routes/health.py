"""Health endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.repositories.platform import PlatformRepository
from apps.api.routes.common import unavailable
from apps.api.schemas import HealthResponse

router = APIRouter(tags=["health"])
Repository = Annotated[PlatformRepository, Depends(get_repository)]


@router.get("/health", response_model=HealthResponse)
def health(repository: Repository) -> HealthResponse:
    try:
        repository.health_check()
    except DatabaseUnavailableError as exc:
        raise unavailable(exc) from exc
    return HealthResponse(status="ok", database="connected")

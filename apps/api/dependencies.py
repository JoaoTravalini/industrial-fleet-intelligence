"""FastAPI dependency providers."""

from __future__ import annotations

from apps.api.config import get_settings
from apps.api.repositories.platform import PlatformRepository


def get_repository() -> PlatformRepository:
    return PlatformRepository(get_settings())

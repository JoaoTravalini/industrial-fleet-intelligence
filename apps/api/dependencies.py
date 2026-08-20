"""FastAPI dependency providers."""

from __future__ import annotations

from apps.api.config import get_settings
from apps.api.repositories.platform import PlatformRepository
from services.copilot.config import get_copilot_settings
from services.copilot.ollama_client import OllamaClient
from services.copilot.service import CopilotService
from services.copilot.tools import CopilotToolExecutor


def get_repository() -> PlatformRepository:
    return PlatformRepository(get_settings())


def get_copilot_repository() -> PlatformRepository:
    return PlatformRepository(get_settings(), read_only=True)


def get_copilot_service() -> CopilotService:
    settings = get_copilot_settings()
    repository = get_copilot_repository()
    return CopilotService(
        settings=settings,
        ollama_client=OllamaClient(settings),
        tool_executor=CopilotToolExecutor(repository),
    )

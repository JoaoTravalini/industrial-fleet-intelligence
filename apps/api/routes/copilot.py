"""Local read-only AI copilot endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_copilot_service
from apps.api.schemas import (
    CopilotChatRequest,
    CopilotChatResponse,
    CopilotHealthResponse,
)
from services.copilot.config import get_copilot_settings
from services.copilot.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaGenerationTimeoutError,
    OllamaModelMissingError,
    OllamaUnavailableError,
)
from services.copilot.service import (
    ChatMessage,
    CopilotRequestTimeoutError,
    CopilotService,
    CopilotServiceError,
)

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])
Copilot = Annotated[CopilotService, Depends(get_copilot_service)]


@router.get("/health", response_model=CopilotHealthResponse)
def copilot_health() -> dict[str, object]:
    settings = get_copilot_settings()
    client = OllamaClient(settings)
    try:
        installed = client.check_model_available()
    except OllamaClientError:
        return {
            "status": "unavailable",
            "provider": "ollama",
            "model": settings.ollama_model,
            "local_only": True,
            "model_installed": False,
            "model_loaded": False,
            "message": "Local AI Copilot is unavailable. Start Ollama and try again.",
        }
    loaded = False
    if installed:
        try:
            loaded = client.is_model_loaded()
        except OllamaClientError:
            loaded = False
    if not installed:
        return {
            "status": "unavailable",
            "provider": "ollama",
            "model": settings.ollama_model,
            "local_only": True,
            "model_installed": False,
            "model_loaded": False,
            "message": "Configured Ollama model is not installed.",
        }
    return {
        "status": "available",
        "provider": "ollama",
        "model": settings.ollama_model,
        "local_only": True,
        "model_installed": True,
        "model_loaded": loaded,
        "message": None,
    }


@router.post("/chat", response_model=CopilotChatResponse)
def copilot_chat(request: CopilotChatRequest, copilot: Copilot) -> dict[str, object]:
    history = [ChatMessage(role=item.role, content=item.content) for item in request.history]
    try:
        return copilot.answer(request.message, history).to_dict()
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local AI Copilot is unavailable. Start Ollama and try again.",
        ) from exc
    except OllamaModelMissingError as exc:
        raise HTTPException(
            status_code=503,
            detail="Configured Ollama model is not installed. Run `ollama pull qwen3:4b-instruct`.",
        ) from exc
    except (OllamaGenerationTimeoutError, CopilotRequestTimeoutError) as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Local model response timed out. Verify Ollama is running and try again "
                "after the model is warm."
            ),
        ) from exc
    except OllamaClientError as exc:
        raise HTTPException(
            status_code=503, detail="Local Ollama returned an invalid response."
        ) from exc
    except CopilotServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

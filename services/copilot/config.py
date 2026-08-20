"""Configuration for the local read-only AI copilot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from apps.api.config import load_env_file, project_root

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
DEFAULT_KEEP_ALIVE = "10m"


class CopilotConfigError(ValueError):
    """Raised when copilot configuration is invalid."""


@dataclass(frozen=True)
class CopilotSettings:
    """Runtime settings for local Ollama copilot execution."""

    ollama_base_url: str
    ollama_model: str
    request_timeout_seconds: int
    total_timeout_seconds: int
    max_tool_rounds: int
    max_history_messages: int
    knowledge_top_k: int
    ollama_keep_alive: str
    num_ctx: int
    num_predict: int
    think: bool
    max_message_chars: int = 4000


def parse_positive_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise CopilotConfigError(f"{name} must be an integer.") from exc
    if value < minimum or value > maximum:
        raise CopilotConfigError(f"{name} must be between {minimum} and {maximum}.")
    return value


def validate_local_ollama_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise CopilotConfigError("OLLAMA_BASE_URL must use http or https.")
    if parsed.username or parsed.password:
        raise CopilotConfigError("OLLAMA_BASE_URL must not include credentials.")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise CopilotConfigError("OLLAMA_BASE_URL must point to a local Ollama endpoint.")
    return value.strip().rstrip("/")


def validate_model_name(value: str) -> str:
    model = value.strip()
    if not model:
        raise CopilotConfigError("OLLAMA_MODEL must be non-empty.")
    if any(character.isspace() for character in model):
        raise CopilotConfigError("OLLAMA_MODEL must not contain whitespace.")
    return model


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise CopilotConfigError(f"{name} must be true or false.")


def validate_keep_alive(value: str) -> str:
    keep_alive = value.strip()
    if not keep_alive:
        raise CopilotConfigError("COPILOT_OLLAMA_KEEP_ALIVE must be non-empty.")
    if any(character.isspace() for character in keep_alive):
        raise CopilotConfigError("COPILOT_OLLAMA_KEEP_ALIVE must not contain whitespace.")
    return keep_alive


@lru_cache(maxsize=1)
def get_copilot_settings() -> CopilotSettings:
    """Return cached local copilot settings loaded from environment and `.env`."""
    load_env_file(project_root() / ".env")
    return CopilotSettings(
        ollama_base_url=validate_local_ollama_url(
            os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ),
        ollama_model=validate_model_name(os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)),
        request_timeout_seconds=parse_positive_int_env(
            "COPILOT_TIMEOUT_SECONDS",
            180,
            minimum=5,
            maximum=300,
        ),
        total_timeout_seconds=parse_positive_int_env(
            "COPILOT_TOTAL_TIMEOUT_SECONDS",
            240,
            minimum=10,
            maximum=600,
        ),
        max_tool_rounds=parse_positive_int_env(
            "COPILOT_MAX_TOOL_ROUNDS",
            2,
            minimum=1,
            maximum=8,
        ),
        max_history_messages=parse_positive_int_env(
            "COPILOT_MAX_HISTORY_MESSAGES",
            6,
            minimum=0,
            maximum=20,
        ),
        knowledge_top_k=parse_positive_int_env(
            "COPILOT_KNOWLEDGE_TOP_K",
            2,
            minimum=1,
            maximum=8,
        ),
        ollama_keep_alive=validate_keep_alive(
            os.getenv("COPILOT_OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
        ),
        num_ctx=parse_positive_int_env(
            "COPILOT_NUM_CTX",
            4096,
            minimum=512,
            maximum=32768,
        ),
        num_predict=parse_positive_int_env(
            "COPILOT_NUM_PREDICT",
            160,
            minimum=16,
            maximum=256,
        ),
        think=parse_bool_env("COPILOT_THINK", False),
    )

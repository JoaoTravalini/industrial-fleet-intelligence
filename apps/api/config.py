"""Configuration helpers for the local FastAPI backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_CORS_ORIGINS = ("http://localhost:5173",)


@dataclass(frozen=True)
class ApiSettings:
    """Runtime configuration loaded from environment variables and local .env."""

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    connect_timeout_seconds: int = 5


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def parse_cors_origins(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_CORS_ORIGINS
    origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())
    return origins or DEFAULT_CORS_ORIGINS


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    """Return cached API settings for local execution."""
    load_env_file(project_root() / ".env")
    return ApiSettings(
        postgres_host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        postgres_port=parse_int_env("POSTGRES_PORT", 5432),
        postgres_db=os.getenv("POSTGRES_DB", "industrial_fleet_dev"),
        postgres_user=os.getenv("POSTGRES_USER", "industrial_fleet_dev"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=parse_int_env("API_PORT", 8000),
        cors_origins=parse_cors_origins(os.getenv("API_CORS_ORIGINS")),
    )

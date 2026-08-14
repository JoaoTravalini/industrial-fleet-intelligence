"""Minimal Psycopg 3 database access helpers."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from apps.api.config import ApiSettings


class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot satisfy a request."""


def open_connection(settings: ApiSettings) -> psycopg.Connection[Any]:
    """Open a short-lived PostgreSQL connection using explicit settings."""
    try:
        return psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=settings.connect_timeout_seconds,
            row_factory=dict_row,
        )
    except (psycopg.Error, OSError) as exc:
        raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc

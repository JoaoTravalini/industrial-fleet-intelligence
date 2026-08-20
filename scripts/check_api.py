"""Validate the local PostgreSQL-backed FastAPI application."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from apps.api.config import get_settings  # noqa: E402
from apps.api.main import app  # noqa: E402
from apps.api.repositories.platform import PlatformRepository  # noqa: E402
from scripts import check_postgres  # noqa: E402
from scripts.materialize_operational_alerts import materialize_alerts  # noqa: E402


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    mandatory: bool = True


def safe_check(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        return CheckResult(name, Status.FAIL, f"Unexpected error: {exc}")


def validate_config() -> CheckResult:
    settings = get_settings()
    if not settings.postgres_db or not settings.postgres_user:
        return CheckResult("API Config", Status.FAIL, "PostgreSQL database/user is missing.")
    if "*" in settings.cors_origins:
        return CheckResult("API Config", Status.FAIL, "Wildcard CORS origin is not allowed.")
    return CheckResult(
        "API Config",
        Status.PASS,
        f"Configured for {settings.postgres_host}:{settings.postgres_port}.",
    )


def validate_postgres() -> CheckResult:
    results = check_postgres.run_checks()
    failures = [result for result in results if result.status is check_postgres.Status.FAIL]
    if failures:
        return CheckResult("PostgreSQL", Status.FAIL, failures[0].message)
    return CheckResult("PostgreSQL", Status.PASS, "PostgreSQL validation passed.")


def validate_alert_materialization() -> CheckResult:
    first = materialize_alerts(validate_infrastructure=False)
    second = materialize_alerts(validate_infrastructure=False)
    if first.conflicts or second.conflicts:
        return CheckResult("Alert Materialization", Status.FAIL, "Alert conflicts detected.")
    if second.new_alerts_inserted != 0:
        return CheckResult("Alert Materialization", Status.FAIL, "Second run was not idempotent.")
    return CheckResult(
        "Alert Materialization",
        Status.PASS,
        (
            "eligible_ai4i="
            f"{second.eligible_ai4i_alerts}, eligible_anomaly={second.eligible_anomaly_alerts}, "
            f"inserted_first={first.new_alerts_inserted}, "
            f"reused_second={second.existing_alerts_reused}"
        ),
    )


def assert_status(response: Any, expected: int, name: str) -> CheckResult | None:
    if response.status_code != expected:
        return CheckResult(name, Status.FAIL, f"Expected {expected}, got {response.status_code}.")
    return None


def no_forbidden_claims(payload: Any) -> bool:
    text = json.dumps(payload, sort_keys=True).lower()
    forbidden = (
        "machine failure detected",
        "maintenance required",
        "confirmed failure",
        "detected failure",
        "unhealthy machine",
    )
    return not any(term in text for term in forbidden)


def schema_descriptions(payload: Any) -> list[str]:
    descriptions: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "description" and isinstance(value, str):
                descriptions.append(value)
            else:
                descriptions.extend(schema_descriptions(value))
    elif isinstance(payload, list):
        for item in payload:
            descriptions.extend(schema_descriptions(item))
    return descriptions


def no_forbidden_causal_schema_descriptions(openapi_payload: Any) -> bool:
    forbidden = ("cause", "causal", "root-cause", "root cause", "protective physical effect")
    return not any(
        term in description.lower()
        for description in schema_descriptions(openapi_payload)
        for term in forbidden
    )


def validate_api_endpoints() -> CheckResult:
    repository = PlatformRepository(get_settings())
    before_counts = repository.protected_state_counts()
    with TestClient(app) as client:
        checks = [
            ("health", client.get("/health"), 200),
            ("fleet", client.get("/api/v1/fleet/overview"), 200),
            ("machines", client.get("/api/v1/machines", params={"limit": 5, "offset": 0}), 200),
            ("machine detail", client.get("/api/v1/machines/MCH-0001"), 200),
            (
                "predictions",
                client.get("/api/v1/machines/MCH-0001/predictions", params={"limit": 5}),
                200,
            ),
            (
                "anomalies",
                client.get("/api/v1/machines/MCH-0001/anomalies", params={"limit": 5}),
                200,
            ),
            ("drift", client.get("/api/v1/drift/latest"), 200),
            ("alerts", client.get("/api/v1/alerts", params={"limit": 5}), 200),
            ("copilot health", client.get("/api/v1/copilot/health"), 200),
            ("unknown machine", client.get("/api/v1/machines/MCH-9999"), 404),
            ("invalid limit", client.get("/api/v1/machines", params={"limit": 0}), 422),
            ("openapi", client.get("/openapi.json"), 200),
        ]
        for label, response, expected in checks:
            failure = assert_status(response, expected, "API Endpoints")
            if failure:
                return CheckResult("API Endpoints", Status.FAIL, f"{label}: {failure.message}")
        openapi_payload = checks[-1][1].json()
        if openapi_payload.get("info", {}).get("title") != "Industrial Fleet Intelligence API":
            return CheckResult("API Endpoints", Status.FAIL, "OpenAPI title mismatch.")
        if not no_forbidden_causal_schema_descriptions(openapi_payload):
            return CheckResult(
                "API Endpoints",
                Status.FAIL,
                "Forbidden causal language found in OpenAPI descriptions.",
            )
        cors = client.options(
            "/api/v1/copilot/chat",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        if cors.headers.get("access-control-allow-origin") != "http://localhost:5173":
            return CheckResult("API Endpoints", Status.FAIL, "Local CORS origin was not allowed.")
        payloads = [response.json() for _, response, expected in checks if expected == 200]
        if not all(no_forbidden_claims(payload) for payload in payloads):
            return CheckResult("API Endpoints", Status.FAIL, "Forbidden health/risk claim found.")
        machines_payload = checks[2][1].json()
        if not isinstance(machines_payload.get("items"), list):
            return CheckResult("API Endpoints", Status.FAIL, "Machine list items is not a list.")
        prediction_payload = checks[4][1].json()
        prediction_items = prediction_payload.get("items", [])
        if not prediction_items:
            return CheckResult("API Endpoints", Status.FAIL, "No prediction row available.")
        event_id = prediction_items[0].get("event_id")
        if not event_id:
            return CheckResult("API Endpoints", Status.FAIL, "Prediction event_id is missing.")
        explanation = client.get(f"/api/v1/machines/MCH-0001/predictions/{event_id}/explanation")
        failure = assert_status(explanation, 200, "API Endpoints")
        if failure:
            return CheckResult(
                "API Endpoints",
                Status.FAIL,
                f"prediction explanation: {failure.message}",
            )
        explanation_payload = explanation.json()
        feature_names = [
            item.get("feature_name")
            for item in explanation_payload.get("feature_contributions", [])
        ]
        expected_features = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        if feature_names != expected_features:
            return CheckResult(
                "API Endpoints",
                Status.FAIL,
                "Explanation endpoint did not return the six semantic features.",
            )
        unknown_explanation = client.get(
            "/api/v1/machines/MCH-0001/predictions/00000000-0000-4000-8000-000000000404/explanation"
        )
        failure = assert_status(unknown_explanation, 404, "API Endpoints")
        if failure:
            return CheckResult(
                "API Endpoints",
                Status.FAIL,
                f"unknown explanation: {failure.message}",
            )
    after_counts = repository.protected_state_counts()
    if before_counts != after_counts:
        return CheckResult("API Endpoints", Status.FAIL, "API request changed protected state.")
    return CheckResult("API Endpoints", Status.PASS, "Required endpoint checks passed.")


def run_checks() -> list[CheckResult]:
    return [
        safe_check("API Config", validate_config),
        safe_check("PostgreSQL", validate_postgres),
        safe_check("Alert Materialization", validate_alert_materialization),
        safe_check("API Endpoints", validate_api_endpoints),
    ]


def print_report(results: list[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform API validation")
    print()
    name_width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")
    pass_count = sum(1 for result in results if result.status is Status.PASS)
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    fail_count = sum(1 for result in results if result.status is Status.FAIL)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def exit_code_for(results: list[CheckResult]) -> int:
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


def main() -> int:
    results = run_checks()
    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

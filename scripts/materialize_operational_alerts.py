"""Materialize deterministic operational alerts from persisted PostgreSQL state."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from apps.api.config import get_settings  # noqa: E402
from apps.api.db import DatabaseUnavailableError, open_connection  # noqa: E402
from scripts import check_postgres  # noqa: E402
from services.alerts.policy import (  # noqa: E402
    MODEL_FAILURE_RISK_ALERT_TYPE,
    TELEMETRY_ANOMALY_ALERT_TYPE,
    AlertCandidate,
    alert_matches_candidate,
    derive_alert_candidates,
    drift_alert_decision,
)


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class AlertMaterializationError(RuntimeError):
    """Raised when alert materialization cannot complete safely."""


@dataclass(frozen=True)
class MaterializationSummary:
    eligible_ai4i_alerts: int
    eligible_anomaly_alerts: int
    eligible_drift_alerts: int
    new_alerts_inserted: int
    existing_alerts_reused: int
    conflicts: int

    def to_dict(self) -> dict[str, int]:
        return {
            "eligible_ai4i_alerts": self.eligible_ai4i_alerts,
            "eligible_anomaly_alerts": self.eligible_anomaly_alerts,
            "eligible_drift_alerts": self.eligible_drift_alerts,
            "new_alerts_inserted": self.new_alerts_inserted,
            "existing_alerts_reused": self.existing_alerts_reused,
            "conflicts": self.conflicts,
        }


def validate_postgres() -> None:
    results = check_postgres.run_checks()
    failures = [result for result in results if result.status is check_postgres.Status.FAIL]
    if failures:
        first = failures[0]
        raise AlertMaterializationError(f"PostgreSQL validation failed: {first.message}")


def fetch_eligible_prediction_rows(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                p.model_prediction_id,
                p.machine_id,
                m.machine_identifier AS machine_code,
                p.failure_prediction
            FROM model_predictions p
            JOIN machines m
              ON m.machine_id = p.machine_id
            WHERE p.prediction_type = 'ai4i_failure_risk'
              AND p.failure_prediction IS TRUE
            ORDER BY
                p.event_time,
                p.source_kafka_timestamp,
                p.source_kafka_topic,
                p.source_kafka_partition,
                p.source_kafka_offset,
                p.event_id;
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def fetch_eligible_anomaly_rows(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                a.anomaly_id,
                a.machine_id,
                m.machine_identifier AS machine_code,
                a.anomaly_flag
            FROM anomalies a
            JOIN machines m
              ON m.machine_id = a.machine_id
            WHERE a.anomaly_type = 'telemetry_isolation_forest_score'
              AND a.anomaly_flag IS TRUE
            ORDER BY
                a.event_time,
                a.source_kafka_timestamp,
                a.source_kafka_topic,
                a.source_kafka_partition,
                a.source_kafka_offset,
                a.event_id;
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def fetch_existing_policy_alerts(
    connection: psycopg.Connection[Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                alert_id,
                machine_id,
                model_prediction_id,
                anomaly_id,
                severity,
                alert_type,
                title,
                description,
                status
            FROM alerts
            WHERE (alert_type = %s AND model_prediction_id IS NOT NULL)
               OR (alert_type = %s AND anomaly_id IS NOT NULL);
            """,
            (MODEL_FAILURE_RISK_ALERT_TYPE, TELEMETRY_ANOMALY_ALERT_TYPE),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        source_id = row["model_prediction_id"] or row["anomaly_id"]
        if source_id is not None:
            existing[(row["alert_type"], int(source_id))] = row
    return existing


def insert_candidate(cursor: psycopg.Cursor[Any], candidate: AlertCandidate) -> bool:
    if candidate.model_prediction_id is not None:
        conflict_sql = """
        ON CONFLICT (alert_type, model_prediction_id)
        WHERE model_prediction_id IS NOT NULL
        DO NOTHING
        """
    else:
        conflict_sql = """
        ON CONFLICT (alert_type, anomaly_id)
        WHERE anomaly_id IS NOT NULL
        DO NOTHING
        """
    cursor.execute(
        f"""
        INSERT INTO alerts (
            machine_id,
            model_prediction_id,
            anomaly_id,
            severity,
            alert_type,
            title,
            description,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        {conflict_sql}
        RETURNING alert_id;
        """,
        (
            candidate.machine_id,
            candidate.model_prediction_id,
            candidate.anomaly_id,
            candidate.severity,
            candidate.alert_type,
            candidate.title,
            candidate.description,
            candidate.status,
        ),
    )
    return cursor.fetchone() is not None


def persist_candidates(
    connection: psycopg.Connection[Any],
    candidates: Sequence[AlertCandidate],
) -> tuple[int, int, int]:
    existing = fetch_existing_policy_alerts(connection)
    conflicts = 0
    reused = 0
    pending: list[AlertCandidate] = []
    for candidate in candidates:
        existing_row = existing.get(candidate.identity_key())
        if existing_row is None:
            pending.append(candidate)
        elif alert_matches_candidate(existing_row, candidate):
            reused += 1
        else:
            conflicts += 1
    if conflicts:
        raise AlertMaterializationError(
            f"{conflicts} existing alert identity conflict(s) were detected."
        )

    inserted = 0
    with connection.transaction():
        with connection.cursor() as cursor:
            for candidate in pending:
                if insert_candidate(cursor, candidate):
                    inserted += 1
                else:
                    reused += 1
    return inserted, reused, conflicts


def materialize_alerts(*, validate_infrastructure: bool = True) -> MaterializationSummary:
    if validate_infrastructure:
        validate_postgres()
    settings = get_settings()
    try:
        with open_connection(settings) as connection:
            prediction_rows = fetch_eligible_prediction_rows(connection)
            anomaly_rows = fetch_eligible_anomaly_rows(connection)
            candidates = derive_alert_candidates(prediction_rows, anomaly_rows)
            drift_decision = drift_alert_decision(alerts_machine_id_required=True)
            inserted, reused, conflicts = persist_candidates(connection, candidates)
    except (DatabaseUnavailableError, psycopg.Error, OSError) as exc:
        raise AlertMaterializationError("PostgreSQL alert materialization failed.") from exc

    eligible_ai4i = sum(
        1 for candidate in candidates if candidate.alert_type == MODEL_FAILURE_RISK_ALERT_TYPE
    )
    eligible_anomaly = sum(
        1 for candidate in candidates if candidate.alert_type == TELEMETRY_ANOMALY_ALERT_TYPE
    )
    return MaterializationSummary(
        eligible_ai4i_alerts=eligible_ai4i,
        eligible_anomaly_alerts=eligible_anomaly,
        eligible_drift_alerts=drift_decision.eligible_alerts,
        new_alerts_inserted=inserted,
        existing_alerts_reused=reused,
        conflicts=conflicts,
    )


def print_summary(summary: MaterializationSummary) -> None:
    print("Industrial Fleet Intelligence Platform operational alert materialization")
    print()
    print(f"PASS Eligible AI4I alerts: {summary.eligible_ai4i_alerts}")
    print(f"PASS Eligible anomaly alerts: {summary.eligible_anomaly_alerts}")
    print(f"PASS Eligible drift alerts: {summary.eligible_drift_alerts}")
    print(f"PASS New alerts inserted: {summary.new_alerts_inserted}")
    print(f"PASS Existing alerts reused: {summary.existing_alerts_reused}")
    print(f"PASS Conflicts: {summary.conflicts}")
    print()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


def main() -> int:
    try:
        summary = materialize_alerts()
    except AlertMaterializationError as exc:
        print("Industrial Fleet Intelligence Platform operational alert materialization")
        print()
        print(f"FAIL {exc}")
        return 1
    print_summary(summary)
    return 0 if summary.conflicts == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

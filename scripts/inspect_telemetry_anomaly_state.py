"""Inspect persisted telemetry anomaly detector state in PostgreSQL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomaly import telemetry_detector  # noqa: E402
from scripts import apply_migrations  # noqa: E402
from services.database import telemetry_anomalies  # noqa: E402


@dataclass(frozen=True)
class InspectionResult:
    """Read-only anomaly detector state inspection result."""

    model_identity: tuple[str, str, str]
    summary: telemetry_anomalies.AnomalyStateSummary
    anomaly_rows: list[telemetry_anomalies.ExistingAnomalyRow]


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str):
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return telemetry_anomalies.parse_json_query_output(result.output)


def query_count(sql: str) -> int:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return telemetry_anomalies.parse_count_output(result.output)


def current_model_identity() -> tuple[str, str, str]:
    config = telemetry_detector.load_config(PROJECT_ROOT)
    return (
        config.model_name,
        config.model_version,
        telemetry_detector.model_config_hash(config),
    )


def inspect_state(
    model_identity: tuple[str, str, str] | None = None,
) -> InspectionResult:
    model_name, model_version, model_config_hash = model_identity or current_model_identity()
    anomaly_json = query_json(
        telemetry_anomalies.build_current_anomalies_query(
            model_name,
            model_version,
            model_config_hash,
        )
    )
    anomaly_rows = [telemetry_anomalies.db_row_to_existing_anomaly(row) for row in anomaly_json]
    duplicate_count = query_count(
        telemetry_anomalies.build_duplicate_identity_count_query(
            model_name,
            model_version,
            model_config_hash,
        )
    )
    machine_reference_mismatch_count = query_count(
        telemetry_anomalies.build_machine_reference_mismatch_count_query(
            model_name,
            model_version,
            model_config_hash,
        )
    )
    summary = telemetry_anomalies.anomaly_state_summary_from_rows(
        anomaly_rows,
        duplicate_anomaly_identity_count=duplicate_count,
        machine_reference_mismatch_count=machine_reference_mismatch_count,
    )
    return InspectionResult(
        model_identity=(model_name, model_version, model_config_hash),
        summary=summary,
        anomaly_rows=anomaly_rows,
    )


def main() -> int:
    print("Industrial Fleet Intelligence Platform telemetry anomaly state")
    print()
    try:
        result = inspect_state()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(
        json.dumps(
            {
                "model_config_hash": result.model_identity[2],
                "model_name": result.model_identity[0],
                "model_version": result.model_identity[1],
                **result.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect persisted AI4I telemetry prediction state in PostgreSQL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import apply_migrations  # noqa: E402
from services.database import ai4i_predictions  # noqa: E402


@dataclass(frozen=True)
class InspectionResult:
    """Read-only prediction state inspection result."""

    summary: ai4i_predictions.PredictionStateSummary
    prediction_rows: list[ai4i_predictions.ExistingPredictionRow]
    health_rows: list[dict[str, Any]]


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_predictions.parse_json_query_output(result.output)


def query_count(sql: str) -> int:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_predictions.parse_count_output(result.output)


def inspect_state(
    records: list[ai4i_predictions.PredictionRecord] | None = None,
) -> InspectionResult:
    runtime_records = records or ai4i_predictions.load_prediction_records(
        ai4i_predictions.prediction_output_path(PROJECT_ROOT)
    )
    model_name, model_version, final_config_hash = ai4i_predictions.model_identity(runtime_records)

    prediction_json = query_json(
        ai4i_predictions.build_current_model_predictions_query(
            model_name,
            model_version,
            final_config_hash,
        )
    )
    prediction_rows = [
        ai4i_predictions.db_row_to_existing_prediction(row) for row in prediction_json
    ]
    health_rows = query_json(
        ai4i_predictions.build_machine_health_query(
            model_name,
            model_version,
            final_config_hash,
        )
    )
    health_rows_by_machine_id = {int(row["machine_id"]): row for row in health_rows}
    expected_latest = ai4i_predictions.latest_prediction_rows_by_machine(prediction_rows)
    prediction_mismatches, event_mismatches = ai4i_predictions.count_projection_mismatches(
        expected_latest,
        health_rows_by_machine_id,
    )
    duplicate_count = query_count(
        ai4i_predictions.build_duplicate_identity_count_query(
            model_name,
            model_version,
            final_config_hash,
        )
    )
    summary = ai4i_predictions.prediction_state_summary_from_rows(
        prediction_rows,
        machine_health_projection_count=len(health_rows),
        machine_health_prediction_mismatch_count=prediction_mismatches,
        machine_health_latest_event_mismatch_count=event_mismatches,
        duplicate_prediction_business_identity_count=duplicate_count,
    )
    return InspectionResult(summary, prediction_rows, health_rows)


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I prediction state")
    print()
    try:
        result = inspect_state()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

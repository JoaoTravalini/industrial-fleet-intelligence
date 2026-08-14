"""Persist existing AI4I telemetry predictions into PostgreSQL."""

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
class PersistenceRunResult:
    """Runtime persistence result plus context used by validators."""

    records: list[ai4i_predictions.PredictionRecord]
    machine_ids_by_code: dict[str, int]
    summary: ai4i_predictions.PersistenceSummary


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_predictions.parse_json_query_output(result.output)


def load_machine_ids(
    records: list[ai4i_predictions.PredictionRecord],
) -> dict[str, int]:
    machine_codes = sorted({record.machine_code for record in records})
    rows = query_json(ai4i_predictions.build_machine_lookup_query(machine_codes))
    machine_ids_by_code = {str(row["machine_code"]): int(row["machine_id"]) for row in rows}
    missing = sorted(set(machine_codes) - set(machine_ids_by_code))
    if missing:
        raise ai4i_predictions.AI4IPredictionPersistenceError(
            "Prediction machine_code values are missing in PostgreSQL: " + ", ".join(missing)
        )
    return machine_ids_by_code


def load_existing_predictions(
    records: list[ai4i_predictions.PredictionRecord],
) -> list[ai4i_predictions.ExistingPredictionRow]:
    rows = query_json(ai4i_predictions.build_existing_predictions_query(records))
    return [ai4i_predictions.db_row_to_existing_prediction(row) for row in rows]


def load_current_health_rows(
    records: list[ai4i_predictions.PredictionRecord],
    machine_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    model_name, model_version, final_config_hash = ai4i_predictions.model_identity(records)
    return query_json(
        ai4i_predictions.build_machine_health_query(
            model_name,
            model_version,
            final_config_hash,
            machine_ids,
        )
    )


def rows_by_machine_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["machine_id"]): row for row in rows}


def persist_predictions() -> PersistenceRunResult:
    prediction_path = ai4i_predictions.prediction_output_path(PROJECT_ROOT)
    records = ai4i_predictions.load_prediction_records(prediction_path)
    ai4i_predictions.write_static_summary(PROJECT_ROOT)

    machine_ids_by_code = load_machine_ids(records)
    existing_before = load_existing_predictions(records)
    existing_before_by_identity = {row.record.identity.as_tuple(): row for row in existing_before}
    reuse = ai4i_predictions.summarize_prediction_reuse(
        records,
        existing_before_by_identity,
        machine_ids_by_code,
    )
    if reuse.conflicts:
        first = reuse.conflicts[0]
        raise ai4i_predictions.AI4IPredictionPersistenceError(
            "Conflicting prediction identity found before persistence: "
            + json.dumps(first.to_dict(), sort_keys=True)
        )

    batch_machine_ids = sorted({machine_ids_by_code[record.machine_code] for record in records})
    health_before = rows_by_machine_id(load_current_health_rows(records, batch_machine_ids))

    transaction_sql = ai4i_predictions.build_persistence_transaction(
        records,
        machine_ids_by_code,
    )
    result = apply_migrations.run_psql_stdin(transaction_sql)
    if not result.succeeded:
        raise command_error(result)

    existing_after = load_existing_predictions(records)
    health_after = rows_by_machine_id(load_current_health_rows(records, batch_machine_ids))
    health_changes = ai4i_predictions.summarize_health_projection_changes(
        health_before,
        health_after,
        batch_machine_ids,
    )

    new_rows = len(existing_after) - len(existing_before)
    summary = ai4i_predictions.PersistenceSummary(
        input_prediction_records=len(records),
        new_prediction_rows_inserted=new_rows,
        existing_identical_predictions_reused=len(records) - new_rows,
        conflicting_predictions=0,
        distinct_machines_in_batch=len(batch_machine_ids),
        machine_health_rows_inserted=health_changes.inserted,
        machine_health_rows_updated=health_changes.updated,
        machine_health_rows_unchanged=health_changes.unchanged,
    )
    return PersistenceRunResult(records, machine_ids_by_code, summary)


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I prediction persistence")
    print()
    try:
        result = persist_predictions()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

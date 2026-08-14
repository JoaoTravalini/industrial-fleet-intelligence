"""Persist materialized operational AI4I explanations into PostgreSQL."""

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
from services.database import ai4i_explanations  # noqa: E402


@dataclass(frozen=True)
class PersistenceRunResult:
    """Runtime persistence result plus records used by validators."""

    records: list[Any]
    prediction_lookup: dict[tuple[str, str, str, str], ai4i_explanations.PredictionLookupRow]
    summary: ai4i_explanations.PersistenceSummary


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_explanations.parse_json_query_output(result.output)


def load_prediction_lookup(
    records: list[Any],
) -> dict[tuple[str, str, str, str], ai4i_explanations.PredictionLookupRow]:
    rows = query_json(ai4i_explanations.build_prediction_lookup_query(records))
    prediction_rows = [ai4i_explanations.db_row_to_prediction_lookup(row) for row in rows]
    return ai4i_explanations.validate_prediction_lookup(records, prediction_rows)


def load_existing_explanations(
    records: list[Any],
    prediction_lookup: dict[tuple[str, str, str, str], ai4i_explanations.PredictionLookupRow],
) -> list[ai4i_explanations.ExistingExplanationRow]:
    rows = query_json(
        ai4i_explanations.build_existing_explanations_query(records, prediction_lookup)
    )
    return [ai4i_explanations.db_row_to_existing_explanation(row) for row in rows]


def persist_explanations() -> PersistenceRunResult:
    records = ai4i_explanations.load_explanation_records(
        ai4i_explanations.explanation_output_path(PROJECT_ROOT)
    )
    ai4i_explanations.write_static_summary(PROJECT_ROOT)
    prediction_lookup = load_prediction_lookup(records)
    existing_before = load_existing_explanations(records, prediction_lookup)
    existing_before_by_identity = {row.db_stable_identity: row for row in existing_before}
    reuse = ai4i_explanations.summarize_explanation_reuse(
        records,
        existing_before_by_identity,
        prediction_lookup,
    )
    if reuse.conflicts:
        first = reuse.conflicts[0]
        raise ai4i_explanations.AI4IExplanationPersistenceError(
            "Conflicting explanation identity found before persistence: "
            + json.dumps(first.to_dict(), sort_keys=True)
        )

    transaction_sql = ai4i_explanations.build_persistence_transaction(records, prediction_lookup)
    result = apply_migrations.run_psql_stdin(transaction_sql)
    if not result.succeeded:
        raise command_error(result)

    existing_after = load_existing_explanations(records, prediction_lookup)
    new_rows = len(existing_after) - len(existing_before)
    summary = ai4i_explanations.PersistenceSummary(
        input_explanations=len(records),
        explanation_rows_inserted=new_rows,
        existing_identical_explanations_reused=len(records) - new_rows,
        conflicting_explanations=0,
        distinct_machines=len({row.machine_id for row in prediction_lookup.values()}),
    )
    return PersistenceRunResult(records, prediction_lookup, summary)


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I explanation persistence")
    print()
    try:
        result = persist_explanations()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

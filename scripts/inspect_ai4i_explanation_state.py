"""Inspect persisted operational AI4I explanation state in PostgreSQL."""

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
class InspectionResult:
    """Read-only explanation state inspection result."""

    summary: ai4i_explanations.ExplanationStateSummary
    explanation_rows: list[ai4i_explanations.ExistingExplanationRow]


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_explanations.parse_json_query_output(result.output)


def query_count(sql: str) -> int:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return ai4i_explanations.parse_count_output(result.output)


def inspect_state(
    records: list[Any] | None = None,
) -> InspectionResult:
    runtime_records = records or ai4i_explanations.load_explanation_records(
        ai4i_explanations.explanation_output_path(PROJECT_ROOT)
    )
    first = runtime_records[0]
    rows = query_json(
        ai4i_explanations.build_current_explanations_query(
            first.model_name,
            first.model_version,
            first.final_config_hash,
            first.explanation_config_hash,
        )
    )
    explanation_rows = [ai4i_explanations.db_row_to_existing_explanation(row) for row in rows]
    duplicate_count = query_count(
        ai4i_explanations.build_duplicate_identity_count_query(
            first.model_name,
            first.model_version,
            first.final_config_hash,
            first.explanation_config_hash,
        )
    )
    mismatch_count = query_count(
        ai4i_explanations.build_model_input_hash_mismatch_count_query(
            first.model_name,
            first.model_version,
            first.final_config_hash,
            first.explanation_config_hash,
        )
    )
    summary = ai4i_explanations.explanation_state_summary_from_rows(
        explanation_rows,
        duplicate_explanation_identity_count=duplicate_count,
        model_input_hash_mismatch_count=mismatch_count,
    )
    return InspectionResult(summary, explanation_rows)


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I explanation state")
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

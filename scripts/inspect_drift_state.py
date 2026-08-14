"""Inspect read-only PostgreSQL drift monitoring state."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.monitoring import drift  # noqa: E402
from scripts import apply_migrations  # noqa: E402
from services.database import drift_monitoring  # noqa: E402


@dataclass(frozen=True)
class InspectionResult:
    summary: drift_monitoring.DriftStateSummary


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return drift_monitoring.parse_json_query_output(result.output)


def inspect_state() -> InspectionResult:
    config = drift.load_config(PROJECT_ROOT)
    reference_sha = drift.file_sha256(drift.reference_profile_path(PROJECT_ROOT))
    payload = query_json(
        drift_monitoring.build_state_summary_query(
            config.monitor_version,
            reference_sha,
        )
    )
    if not isinstance(payload, dict):
        raise drift_monitoring.DriftPersistenceError(
            "Drift state query did not return a JSON object."
        )
    return InspectionResult(summary=drift_monitoring.state_summary_from_payload(payload))


def main() -> int:
    print("Industrial Fleet Intelligence Platform drift state inspection")
    print()
    try:
        result = inspect_state()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    print("PASS Drift state inspection completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

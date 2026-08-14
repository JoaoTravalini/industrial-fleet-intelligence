"""Persist the latest deterministic drift report into PostgreSQL."""

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
from services.database import drift_monitoring  # noqa: E402


@dataclass(frozen=True)
class PersistenceRunResult:
    summary: drift_monitoring.PersistenceSummary
    preflight: drift_monitoring.DriftReuseSummary


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str) -> Any:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return drift_monitoring.parse_json_query_output(result.output)


def load_existing_snapshots(report: dict[str, Any]) -> list[dict[str, Any]]:
    payload = query_json(drift_monitoring.build_existing_report_query(report))
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise drift_monitoring.DriftPersistenceError(
            "Existing drift snapshot query did not return a JSON array."
        )
    return payload


def persist_drift_report() -> PersistenceRunResult:
    report = drift_monitoring.load_report(root=PROJECT_ROOT)
    existing = load_existing_snapshots(report)
    reuse = drift_monitoring.summarize_report_reuse(report, existing)
    if reuse.conflicts:
        details = [conflict.to_dict() for conflict in reuse.conflicts]
        raise drift_monitoring.DriftPersistenceError(
            "Conflicting drift identity found before persistence: "
            + json.dumps(details, sort_keys=True)
        )

    transaction_sql = drift_monitoring.build_persistence_transaction(report)
    result = apply_migrations.run_psql_stdin(transaction_sql)
    if not result.succeeded:
        raise command_error(result)

    drift_monitoring.write_static_summary(PROJECT_ROOT)
    summary = drift_monitoring.PersistenceSummary(
        input_feature_metrics=len(drift_monitoring.metric_values(report)),
        new_snapshots_inserted=reuse.new_snapshots,
        existing_identical_snapshots_reused=reuse.existing_identical_snapshots_reused,
        new_feature_metrics_inserted=reuse.new_feature_metrics,
        existing_identical_feature_metrics_reused=(reuse.existing_identical_feature_metrics_reused),
        conflicts=len(reuse.conflicts),
    )
    return PersistenceRunResult(summary=summary, preflight=reuse)


def main() -> int:
    print("Industrial Fleet Intelligence Platform drift report persistence")
    print()
    try:
        result = persist_drift_report()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    print("PASS Drift report persistence completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

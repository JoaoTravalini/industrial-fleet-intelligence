"""Persist existing telemetry anomaly outputs into PostgreSQL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import apply_migrations  # noqa: E402
from services.database import telemetry_anomalies  # noqa: E402


@dataclass(frozen=True)
class PersistenceRunResult:
    """Runtime persistence result plus context used by validators."""

    records: list[telemetry_anomalies.AnomalyRecord]
    machine_ids_by_code: dict[str, int]
    summary: telemetry_anomalies.PersistenceSummary


def command_error(result: apply_migrations.CommandResult) -> RuntimeError:
    return RuntimeError(apply_migrations.command_failure_message(result))


def query_json(sql: str):
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise command_error(result)
    return telemetry_anomalies.parse_json_query_output(result.output)


def load_machine_ids(records: list[telemetry_anomalies.AnomalyRecord]) -> dict[str, int]:
    machine_codes = sorted({record.machine_code for record in records})
    rows = query_json(telemetry_anomalies.build_machine_lookup_query(machine_codes))
    machine_ids_by_code = {str(row["machine_code"]): int(row["machine_id"]) for row in rows}
    missing = sorted(set(machine_codes) - set(machine_ids_by_code))
    if missing:
        raise telemetry_anomalies.TelemetryAnomalyPersistenceError(
            "Anomaly machine_code values are missing in PostgreSQL: " + ", ".join(missing)
        )
    return machine_ids_by_code


def load_existing_anomalies(
    records: list[telemetry_anomalies.AnomalyRecord],
) -> list[telemetry_anomalies.ExistingAnomalyRow]:
    rows = query_json(telemetry_anomalies.build_existing_anomalies_query(records))
    return [telemetry_anomalies.db_row_to_existing_anomaly(row) for row in rows]


def persist_anomalies() -> PersistenceRunResult:
    anomaly_path = telemetry_anomalies.anomaly_output_path(PROJECT_ROOT)
    records = telemetry_anomalies.load_anomaly_records(anomaly_path, root=PROJECT_ROOT)
    telemetry_anomalies.write_static_summary(PROJECT_ROOT)

    machine_ids_by_code = load_machine_ids(records)
    existing_before = load_existing_anomalies(records)
    existing_before_by_identity = {row.record.identity.as_tuple(): row for row in existing_before}
    reuse = telemetry_anomalies.summarize_anomaly_reuse(
        records,
        existing_before_by_identity,
        machine_ids_by_code,
    )
    if reuse.conflicts:
        first = reuse.conflicts[0]
        raise telemetry_anomalies.TelemetryAnomalyPersistenceError(
            "Conflicting anomaly identity found before persistence: "
            + json.dumps(first.to_dict(), sort_keys=True)
        )

    transaction_sql = telemetry_anomalies.build_persistence_transaction(
        records,
        machine_ids_by_code,
    )
    result = apply_migrations.run_psql_stdin(transaction_sql)
    if not result.succeeded:
        raise command_error(result)

    existing_after = load_existing_anomalies(records)
    new_rows = len(existing_after) - len(existing_before)
    summary = telemetry_anomalies.PersistenceSummary(
        input_anomaly_records=len(records),
        new_anomaly_rows_inserted=new_rows,
        existing_identical_anomalies_reused=len(records) - new_rows,
        conflicting_anomalies=0,
        distinct_machines_in_batch=len(
            {machine_ids_by_code[record.machine_code] for record in records}
        ),
    )
    return PersistenceRunResult(records, machine_ids_by_code, summary)


def main() -> int:
    print("Industrial Fleet Intelligence Platform telemetry anomaly persistence")
    print()
    try:
        result = persist_anomalies()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

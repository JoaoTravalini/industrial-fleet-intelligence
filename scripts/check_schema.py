"""Read-only PostgreSQL operational schema validator."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

SERVICE_NAME = "postgres"
DEFAULT_TIMEOUT_SECONDS = 20
EXEC_TIMEOUT_SECONDS = 60
EXPECTED_MIGRATIONS = {
    "001_initial_operational_schema.sql",
    "002_ai4i_prediction_persistence.sql",
}
EXPECTED_TABLES = {
    "schema_migrations",
    "machines",
    "maintenance_records",
    "model_predictions",
    "anomalies",
    "alerts",
    "machine_health",
}
EXPECTED_FOREIGN_KEYS = {
    "fk_maintenance_records_machine",
    "fk_model_predictions_machine",
    "fk_anomalies_machine",
    "fk_alerts_machine",
    "fk_alerts_model_prediction",
    "fk_alerts_anomaly",
    "fk_machine_health_machine",
    "fk_machine_health_latest_model_prediction",
}
EXPECTED_CONSTRAINTS = {
    "uq_machines_machine_identifier",
    "ck_machines_operational_status",
    "ck_model_predictions_confidence",
    "ck_model_predictions_ai4i_required_fields",
    "ck_model_predictions_failure_probability",
    "ck_model_predictions_failure_prediction_consistency",
    "ck_model_predictions_frozen_threshold",
    "ck_model_predictions_final_config_hash_format",
    "ck_model_predictions_adapter_version_not_blank",
    "ck_model_predictions_model_input_sha256_format",
    "ck_model_predictions_source_kafka_topic_not_blank",
    "ck_model_predictions_source_kafka_partition",
    "ck_model_predictions_source_kafka_offset",
    "ck_model_predictions_source_kafka_key_not_blank",
    "ck_model_predictions_payload_sha256_format",
    "ck_anomalies_score",
    "ck_alerts_severity",
    "ck_alerts_status",
    "ck_alerts_status_timestamp_consistency",
    "ck_machine_health_health_score",
    "ck_machine_health_failure_risk",
    "ck_machine_health_anomaly_score",
    "ck_machine_health_classification",
    "ck_machine_health_latest_prediction_required_fields",
    "ck_machine_health_latest_failure_probability",
    "ck_machine_health_latest_failure_prediction_consistency",
    "ck_machine_health_latest_frozen_threshold",
    "ck_machine_health_latest_model_name_not_blank",
    "ck_machine_health_latest_model_version_not_blank",
    "ck_machine_health_latest_final_config_hash_format",
    "ck_machine_health_latest_model_input_sha256_format",
    "ck_machine_health_latest_source_kafka_topic_not_blank",
    "ck_machine_health_latest_source_kafka_partition",
    "ck_machine_health_latest_source_kafka_offset",
    "ck_machine_health_latest_source_kafka_key_not_blank",
    "ck_machine_health_latest_payload_sha256_format",
}
EXPECTED_INDEXES = {
    "uq_machines_machine_identifier",
    "uq_model_predictions_ai4i_business_identity",
    "idx_maintenance_records_machine_timestamp",
    "idx_model_predictions_machine_timestamp",
    "idx_model_predictions_ai4i_latest",
    "idx_anomalies_machine_timestamp",
    "idx_alerts_machine_status_severity",
    "idx_machine_health_latest_prediction",
}
FORBIDDEN_RAW_TELEMETRY_TABLES = {
    "telemetry",
    "raw_telemetry",
    "telemetry_raw",
    "sensor_readings",
    "machine_telemetry",
}


class Status(StrEnum):
    """Validation status values printed by the schema checker."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CommandResult:
    """Captured command result with expected execution failures normalized."""

    args: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.error

    @property
    def output(self) -> str:
        return "\n".join(
            normalize_output_text(part.strip())
            for part in (self.stdout, self.stderr)
            if part and normalize_output_text(part.strip())
        ).strip()


@dataclass(frozen=True)
class CheckResult:
    """A single schema validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def parse_lines(output: str) -> list[str]:
    return [line.strip() for line in normalize_output_text(output).splitlines() if line.strip()]


def parse_bool(output: str) -> bool | None:
    values = parse_lines(output)
    if not values:
        return None
    value = values[0].lower()
    if value in {"t", "true", "1"}:
        return True
    if value in {"f", "false", "0"}:
        return False
    return None


def missing_items(expected: set[str], actual: set[str]) -> set[str]:
    return expected - actual


def evaluate_expected_items(name: str, expected: set[str], actual: set[str]) -> CheckResult:
    missing = missing_items(expected, actual)
    if not missing:
        return CheckResult(name, Status.PASS, f"Found all {len(expected)} expected item(s).")
    return CheckResult(name, Status.FAIL, "Missing: " + ", ".join(sorted(missing)))


def evaluate_forbidden_tables(actual: set[str]) -> CheckResult:
    present = FORBIDDEN_RAW_TELEMETRY_TABLES & actual
    if not present:
        return CheckResult(
            "Raw Telemetry Tables", Status.PASS, "No raw telemetry table is present."
        )
    return CheckResult(
        "Raw Telemetry Tables",
        Status.FAIL,
        "Unexpected raw telemetry table(s): " + ", ".join(sorted(present)),
    )


def run_command(
    args: Sequence[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> CommandResult:
    """Run a read-only command without shell=True and capture expected failures cleanly."""
    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(tuple(args), None, error="command not found")
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        return CommandResult(
            tuple(args),
            None,
            stdout=stdout or "",
            stderr=stderr or "",
            error=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return CommandResult(tuple(args), None, error=str(exc))

    return CommandResult(
        tuple(args),
        completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def command_failure_message(result: CommandResult) -> str:
    if result.error:
        return result.error
    if result.output:
        return f"command exited with code {result.returncode}: {result.output.splitlines()[0]}"
    return f"command exited with code {result.returncode}"


def run_psql_query(sql: str) -> CommandResult:
    psql_command = (
        'PGPASSWORD="$POSTGRES_PASSWORD" '
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -X -q -tA -c "$1"'
    )
    return run_command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            SERVICE_NAME,
            "sh",
            "-c",
            psql_command,
            "psql-query",
            sql,
        ],
        timeout=EXEC_TIMEOUT_SECONDS,
    )


def check_database_reachable() -> CheckResult:
    result = run_psql_query("SELECT 1;")
    if result.succeeded and "1" in parse_lines(result.output):
        return CheckResult("Database Reachable", Status.PASS, "Read-only SELECT 1 check passed.")
    return CheckResult("Database Reachable", Status.FAIL, command_failure_message(result))


def check_schema_migrations_exists() -> CheckResult:
    result = run_psql_query(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'schema_migrations'
        );
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Migration Table",
            Status.FAIL,
            f"Could not inspect schema_migrations: {command_failure_message(result)}",
        )
    if parse_bool(result.output) is True:
        return CheckResult("Migration Table", Status.PASS, "schema_migrations exists.")
    return CheckResult("Migration Table", Status.FAIL, "schema_migrations does not exist.")


def check_migrations_recorded() -> CheckResult:
    result = run_psql_query(
        """
        SELECT filename
        FROM schema_migrations
        ORDER BY filename;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Migrations",
            Status.FAIL,
            f"Could not inspect applied migrations: {command_failure_message(result)}",
        )
    return evaluate_expected_items(
        "Migrations",
        EXPECTED_MIGRATIONS,
        set(parse_lines(result.output)),
    )


def check_tables() -> CheckResult:
    result = run_psql_query(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Tables",
            Status.FAIL,
            f"Could not inspect tables: {command_failure_message(result)}",
        )
    return evaluate_expected_items("Tables", EXPECTED_TABLES, set(parse_lines(result.output)))


def check_foreign_keys() -> CheckResult:
    result = run_psql_query(
        """
        SELECT conname
        FROM pg_constraint
        WHERE contype = 'f'
          AND connamespace = 'public'::regnamespace
        ORDER BY conname;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Foreign Keys",
            Status.FAIL,
            f"Could not inspect foreign keys: {command_failure_message(result)}",
        )
    return evaluate_expected_items(
        "Foreign Keys", EXPECTED_FOREIGN_KEYS, set(parse_lines(result.output))
    )


def check_constraints() -> CheckResult:
    result = run_psql_query(
        """
        SELECT conname
        FROM pg_constraint
        WHERE contype IN ('u', 'c')
          AND connamespace = 'public'::regnamespace
        ORDER BY conname;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Constraints",
            Status.FAIL,
            f"Could not inspect constraints: {command_failure_message(result)}",
        )
    return evaluate_expected_items(
        "Constraints", EXPECTED_CONSTRAINTS, set(parse_lines(result.output))
    )


def check_indexes() -> CheckResult:
    result = run_psql_query(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY indexname;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Indexes",
            Status.FAIL,
            f"Could not inspect indexes: {command_failure_message(result)}",
        )
    return evaluate_expected_items("Indexes", EXPECTED_INDEXES, set(parse_lines(result.output)))


def check_no_raw_telemetry_tables() -> CheckResult:
    result = run_psql_query(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Raw Telemetry Tables",
            Status.FAIL,
            f"Could not inspect tables: {command_failure_message(result)}",
        )
    return evaluate_forbidden_tables(set(parse_lines(result.output)))


def run_checks() -> list[CheckResult]:
    return [
        check_database_reachable(),
        check_schema_migrations_exists(),
        check_migrations_recorded(),
        check_tables(),
        check_foreign_keys(),
        check_constraints(),
        check_indexes(),
        check_no_raw_telemetry_tables(),
    ]


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform schema validation")
    print()

    name_width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")

    pass_count = sum(1 for result in results if result.status is Status.PASS)
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    fail_count = sum(1 for result in results if result.status is Status.FAIL)

    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def exit_code_for(results: Sequence[CheckResult]) -> int:
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


def main() -> int:
    try:
        results = run_checks()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform schema validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

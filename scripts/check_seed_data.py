"""Read-only validator for deterministic fictional development seed data."""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

SERVICE_NAME = "postgres"
DEFAULT_TIMEOUT_SECONDS = 20
EXEC_TIMEOUT_SECONDS = 60
EXPECTED_MACHINE_COUNT = 100
EXPECTED_STATUS_COUNTS = {
    "active": 85,
    "maintenance": 10,
    "inactive": 5,
}
EXPECTED_CATEGORIES = {
    "excavator",
    "wheel_loader",
    "crawler_crane",
    "mobile_crane",
    "mining_truck",
    "bulldozer",
    "industrial_pump",
    "generator",
}
EXPECTED_MODEL_FAMILIES = {
    "EX-Series",
    "WL-Series",
    "CC-Series",
    "MC-Series",
    "MT-Series",
    "BD-Series",
    "IP-Series",
    "GN-Series",
}
EXPECTED_EMPTY_TABLE_COUNTS = {
    "maintenance_records": 0,
    "model_predictions": 0,
    "anomalies": 0,
    "alerts": 0,
    "machine_health": 0,
}


class Status(StrEnum):
    """Validation status values printed by the seed checker."""

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
    """A single seed data validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def parse_lines(output: str) -> list[str]:
    return [line.strip() for line in normalize_output_text(output).splitlines() if line.strip()]


def parse_count(output: str) -> int | None:
    lines = parse_lines(output)
    if len(lines) != 1:
        return None
    try:
        return int(lines[0])
    except ValueError:
        return None


def parse_key_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in parse_lines(output):
        parts = line.split("|", maxsplit=1)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError(f"Could not parse key-count row: {line}")
        try:
            counts[parts[0].strip()] = int(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"Could not parse count value in row: {line}") from exc
    return counts


def expected_machine_identifiers(count: int = EXPECTED_MACHINE_COUNT) -> list[str]:
    return [f"MCH-{machine_number:04d}" for machine_number in range(1, count + 1)]


def duplicate_values(values: Sequence[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def evaluate_exact_count(name: str, actual: int | None, expected: int) -> CheckResult:
    if actual == expected:
        return CheckResult(name, Status.PASS, f"Found expected count: {expected}.")
    return CheckResult(
        name,
        Status.FAIL,
        f"Expected {expected}, found {actual if actual is not None else 'unknown'}.",
    )


def evaluate_identifier_range(identifiers: Sequence[str]) -> CheckResult:
    expected = set(expected_machine_identifiers())
    actual = set(identifiers)
    missing = expected - actual
    unexpected = actual - expected
    if not missing and not unexpected:
        return CheckResult(
            "Machine Identifier Range",
            Status.PASS,
            "Identifiers cover MCH-0001 through MCH-0100.",
        )

    messages: list[str] = []
    if missing:
        messages.append("missing " + ", ".join(sorted(missing)[:5]))
    if unexpected:
        messages.append("unexpected " + ", ".join(sorted(unexpected)[:5]))
    return CheckResult("Machine Identifier Range", Status.FAIL, "; ".join(messages))


def evaluate_unique_identifiers(identifiers: Sequence[str]) -> CheckResult:
    duplicates = duplicate_values(identifiers)
    if not duplicates and len(set(identifiers)) == len(identifiers):
        return CheckResult(
            "Machine Identifier Uniqueness", Status.PASS, "All identifiers are unique."
        )
    return CheckResult(
        "Machine Identifier Uniqueness",
        Status.FAIL,
        "Duplicate identifier(s): " + ", ".join(sorted(duplicates)[:5]),
    )


def evaluate_key_counts(name: str, actual: dict[str, int], expected: dict[str, int]) -> CheckResult:
    if actual == expected:
        return CheckResult(name, Status.PASS, "Counts match expected distribution.")
    return CheckResult(name, Status.FAIL, f"Expected {expected}, found {actual}.")


def evaluate_allowed_values(name: str, actual: set[str], expected: set[str]) -> CheckResult:
    unexpected = actual - expected
    if not unexpected:
        return CheckResult(name, Status.PASS, "Only expected fictional values are present.")
    return CheckResult(name, Status.FAIL, "Unexpected values: " + ", ".join(sorted(unexpected)))


def evaluate_empty_tables(actual: dict[str, int]) -> CheckResult:
    if actual == EXPECTED_EMPTY_TABLE_COUNTS:
        return CheckResult(
            "Other Operational Tables", Status.PASS, "All non-machine tables are empty."
        )
    return CheckResult(
        "Other Operational Tables",
        Status.FAIL,
        f"Expected {EXPECTED_EMPTY_TABLE_COUNTS}, found {actual}.",
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


def check_machine_count() -> CheckResult:
    result = run_psql_query("SELECT count(*) FROM machines;")
    if not result.succeeded:
        return CheckResult(
            "Machine Count",
            Status.FAIL,
            f"Could not count machines: {command_failure_message(result)}",
        )
    return evaluate_exact_count("Machine Count", parse_count(result.output), EXPECTED_MACHINE_COUNT)


def load_machine_identifiers() -> tuple[CheckResult | None, list[str]]:
    result = run_psql_query("SELECT machine_identifier FROM machines ORDER BY machine_identifier;")
    if not result.succeeded:
        return (
            CheckResult(
                "Machine Identifiers",
                Status.FAIL,
                f"Could not inspect machine identifiers: {command_failure_message(result)}",
            ),
            [],
        )
    return None, parse_lines(result.output)


def check_status_counts() -> CheckResult:
    result = run_psql_query(
        """
        SELECT operational_status, count(*)
        FROM machines
        GROUP BY operational_status
        ORDER BY operational_status;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Status Counts",
            Status.FAIL,
            f"Could not inspect machine statuses: {command_failure_message(result)}",
        )
    try:
        actual = parse_key_counts(result.output)
    except ValueError as exc:
        return CheckResult("Status Counts", Status.FAIL, str(exc))
    return evaluate_key_counts("Status Counts", actual, EXPECTED_STATUS_COUNTS)


def check_categories() -> CheckResult:
    result = run_psql_query("SELECT DISTINCT machine_type FROM machines ORDER BY machine_type;")
    if not result.succeeded:
        return CheckResult(
            "Machine Categories",
            Status.FAIL,
            f"Could not inspect machine categories: {command_failure_message(result)}",
        )
    return evaluate_allowed_values(
        "Machine Categories", set(parse_lines(result.output)), EXPECTED_CATEGORIES
    )


def check_model_families() -> CheckResult:
    result = run_psql_query("SELECT DISTINCT model_family FROM machines ORDER BY model_family;")
    if not result.succeeded:
        return CheckResult(
            "Model Families",
            Status.FAIL,
            f"Could not inspect model families: {command_failure_message(result)}",
        )
    return evaluate_allowed_values(
        "Model Families", set(parse_lines(result.output)), EXPECTED_MODEL_FAMILIES
    )


def check_commissioning_dates() -> CheckResult:
    result = run_psql_query("SELECT count(*) FROM machines WHERE commissioned_on > CURRENT_DATE;")
    if not result.succeeded:
        return CheckResult(
            "Commissioning Dates",
            Status.FAIL,
            f"Could not inspect commissioning dates: {command_failure_message(result)}",
        )
    future_count = parse_count(result.output)
    if future_count == 0:
        return CheckResult(
            "Commissioning Dates", Status.PASS, "No commissioning dates are future dated."
        )
    return CheckResult(
        "Commissioning Dates",
        Status.FAIL,
        f"Found {future_count if future_count is not None else 'unknown'} future-dated machine(s).",
    )


def check_other_tables_empty() -> CheckResult:
    result = run_psql_query(
        """
        SELECT 'maintenance_records' AS table_name, count(*) FROM maintenance_records
        UNION ALL SELECT 'model_predictions', count(*) FROM model_predictions
        UNION ALL SELECT 'anomalies', count(*) FROM anomalies
        UNION ALL SELECT 'alerts', count(*) FROM alerts
        UNION ALL SELECT 'machine_health', count(*) FROM machine_health
        ORDER BY table_name;
        """
    )
    if not result.succeeded:
        return CheckResult(
            "Other Operational Tables",
            Status.FAIL,
            f"Could not inspect non-machine tables: {command_failure_message(result)}",
        )
    try:
        actual = parse_key_counts(result.output)
    except ValueError as exc:
        return CheckResult("Other Operational Tables", Status.FAIL, str(exc))
    return evaluate_empty_tables(actual)


def run_checks() -> list[CheckResult]:
    results = [check_database_reachable(), check_machine_count()]

    identifier_error, identifiers = load_machine_identifiers()
    if identifier_error is not None:
        results.append(identifier_error)
    else:
        results.extend(
            [
                evaluate_identifier_range(identifiers),
                evaluate_unique_identifiers(identifiers),
            ]
        )

    results.extend(
        [
            check_status_counts(),
            check_categories(),
            check_model_families(),
            check_commissioning_dates(),
            check_other_tables_empty(),
        ]
    )
    return results


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform seed data validation")
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
        print("Industrial Fleet Intelligence Platform seed data validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

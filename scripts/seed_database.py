"""Apply deterministic fictional development seed data through Docker Compose and psql."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SERVICE_NAME = "postgres"
DEFAULT_TIMEOUT_SECONDS = 20
EXEC_TIMEOUT_SECONDS = 60
EXPECTED_MIGRATION = "001_initial_operational_schema.sql"
EXPECTED_MACHINE_COUNT = 100
SEED_NAME_PATTERN = re.compile(r"^[0-9]{3}_[a-z0-9_]+\.sql$")


class Status(StrEnum):
    """Seed runner status values."""

    PASS = "PASS"
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
class SeedFile:
    """A discovered SQL development seed file."""

    filename: str
    path: Path


@dataclass(frozen=True)
class CheckResult:
    """A single seed runner result."""

    name: str
    status: Status
    message: str


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seeds_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / "db" / "seeds"


def validate_seed_filename(filename: str) -> bool:
    return SEED_NAME_PATTERN.fullmatch(filename) is not None


def discover_seed_files(directory: Path) -> list[SeedFile]:
    seed_files: list[SeedFile] = []
    for path in sorted(directory.glob("*.sql"), key=lambda item: item.name):
        if not validate_seed_filename(path.name):
            raise ValueError(
                f"Invalid seed filename '{path.name}'. Expected pattern '001_descriptive_name.sql'."
            )
        seed_files.append(SeedFile(path.name, path))
    return seed_files


def parse_bool(output: str) -> bool | None:
    value = normalize_output_text(output).strip().lower()
    if value in {"t", "true", "1"}:
        return True
    if value in {"f", "false", "0"}:
        return False
    return None


def parse_count(output: str) -> int | None:
    value = normalize_output_text(output).strip()
    try:
        return int(value)
    except ValueError:
        return None


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_seed_transaction(seed_file: SeedFile) -> str:
    seed_sql = seed_file.path.read_text(encoding="utf-8")
    return f"""
BEGIN;
{seed_sql}
COMMIT;
"""


def run_command(
    args: Sequence[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    input_text: str | None = None,
) -> CommandResult:
    """Run a command without shell=True and capture expected failures cleanly."""
    try:
        completed = subprocess.run(
            list(args),
            input=input_text,
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


def run_psql_stdin(sql: str) -> CommandResult:
    psql_command = (
        'PGPASSWORD="$POSTGRES_PASSWORD" '
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -X -q -f -'
    )
    return run_command(
        ["docker", "compose", "exec", "-T", SERVICE_NAME, "sh", "-c", psql_command],
        timeout=EXEC_TIMEOUT_SECONDS,
        input_text=sql,
    )


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


def postgres_is_healthy() -> CheckResult:
    ps_result = run_command(["docker", "compose", "ps", "-q", SERVICE_NAME])
    if not ps_result.succeeded:
        return CheckResult(
            "PostgreSQL Health",
            Status.FAIL,
            f"Could not find PostgreSQL container: {command_failure_message(ps_result)}",
        )

    container_id = next(
        (
            line.strip()
            for line in normalize_output_text(ps_result.output).splitlines()
            if line.strip()
        ),
        "",
    )
    if not container_id:
        return CheckResult("PostgreSQL Health", Status.FAIL, "PostgreSQL container was not found.")

    health_result = run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
            container_id,
        ]
    )
    if not health_result.succeeded:
        return CheckResult(
            "PostgreSQL Health",
            Status.FAIL,
            f"Could not inspect PostgreSQL health: {command_failure_message(health_result)}",
        )

    health = normalize_output_text(health_result.output).strip().lower()
    if health == "healthy":
        return CheckResult("PostgreSQL Health", Status.PASS, "PostgreSQL container is healthy.")
    return CheckResult("PostgreSQL Health", Status.FAIL, f"PostgreSQL container state is {health}.")


def check_migration_applied() -> CheckResult:
    table_result = run_psql_query("SELECT to_regclass('public.schema_migrations') IS NOT NULL;")
    if not table_result.succeeded:
        return CheckResult(
            "Migration Check",
            Status.FAIL,
            f"Could not inspect schema_migrations: {command_failure_message(table_result)}",
        )
    if parse_bool(table_result.output) is not True:
        return CheckResult("Migration Check", Status.FAIL, "Run schema migrations before seeding.")

    migration_result = run_psql_query(
        "SELECT EXISTS ("
        "SELECT 1 FROM schema_migrations "
        f"WHERE filename = {sql_literal(EXPECTED_MIGRATION)}"
        ");"
    )
    if not migration_result.succeeded:
        return CheckResult(
            "Migration Check",
            Status.FAIL,
            f"Could not inspect applied migrations: {command_failure_message(migration_result)}",
        )
    if parse_bool(migration_result.output) is True:
        return CheckResult("Migration Check", Status.PASS, f"{EXPECTED_MIGRATION} is applied.")
    return CheckResult("Migration Check", Status.FAIL, f"{EXPECTED_MIGRATION} is not applied.")


def apply_seed_file(seed_file: SeedFile) -> CheckResult:
    result = run_psql_stdin(build_seed_transaction(seed_file))
    if result.succeeded:
        return CheckResult("Seed Applied", Status.PASS, f"Executed {seed_file.filename}.")
    return CheckResult(
        "Seed Applied",
        Status.FAIL,
        f"Failed {seed_file.filename}: {command_failure_message(result)}",
    )


def check_machine_count() -> CheckResult:
    result = run_psql_query("SELECT count(*) FROM machines;")
    if not result.succeeded:
        return CheckResult(
            "Machine Count",
            Status.FAIL,
            f"Could not count machines: {command_failure_message(result)}",
        )
    count = parse_count(result.output)
    if count == EXPECTED_MACHINE_COUNT:
        return CheckResult("Machine Count", Status.PASS, "100 machines are present.")
    return CheckResult(
        "Machine Count",
        Status.FAIL,
        f"Expected 100 machines, found {count if count is not None else 'unknown'}.",
    )


def run_seed() -> list[CheckResult]:
    results = [postgres_is_healthy()]
    if results[-1].status is Status.FAIL:
        return results

    results.append(check_migration_applied())
    if results[-1].status is Status.FAIL:
        return results

    directory = seeds_directory()
    if not directory.exists():
        return [
            *results,
            CheckResult("Seed Discovery", Status.FAIL, f"Missing directory: {directory}"),
        ]

    try:
        seed_files = discover_seed_files(directory)
    except ValueError as exc:
        return [*results, CheckResult("Seed Discovery", Status.FAIL, str(exc))]

    if not seed_files:
        return [*results, CheckResult("Seed Discovery", Status.FAIL, "No seed files were found.")]

    results.append(
        CheckResult("Seed Discovery", Status.PASS, f"Discovered {len(seed_files)} seed file(s).")
    )

    for seed_file in seed_files:
        result = apply_seed_file(seed_file)
        results.append(result)
        if result.status is Status.FAIL:
            return results

    results.append(check_machine_count())
    return results


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform development seed runner")
    print()

    name_width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")

    pass_count = sum(1 for result in results if result.status is Status.PASS)
    fail_count = sum(1 for result in results if result.status is Status.FAIL)

    print()
    print(f"Summary: {pass_count} PASS, {fail_count} FAIL")


def exit_code_for(results: Sequence[CheckResult]) -> int:
    return 1 if any(result.status is Status.FAIL for result in results) else 0


def main() -> int:
    try:
        results = run_seed()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform development seed runner")
        print()
        print(f"FAIL Runner encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

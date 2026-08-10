"""Apply versioned PostgreSQL SQL migrations through Docker Compose and psql."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SERVICE_NAME = "postgres"
DEFAULT_TIMEOUT_SECONDS = 20
EXEC_TIMEOUT_SECONDS = 60
MIGRATION_NAME_PATTERN = re.compile(r"^[0-9]{3}_[a-z0-9_]+\.sql$")
TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_schema_migrations_filename_not_blank CHECK (btrim(filename) <> ''),
    CONSTRAINT ck_schema_migrations_checksum_not_blank CHECK (btrim(checksum) <> '')
);
"""


class Status(StrEnum):
    """Migration status values printed by the migration runner."""

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
class Migration:
    """A discovered SQL migration."""

    filename: str
    path: Path
    checksum: str


@dataclass(frozen=True)
class CheckResult:
    """A single migration runner result."""

    name: str
    status: Status
    message: str


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def migrations_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / "db" / "migrations"


def validate_migration_filename(filename: str) -> bool:
    return MIGRATION_NAME_PATTERN.fullmatch(filename) is not None


def calculate_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_migrations(directory: Path) -> list[Migration]:
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql"), key=lambda item: item.name):
        if not validate_migration_filename(path.name):
            raise ValueError(
                f"Invalid migration filename '{path.name}'. Expected pattern "
                "'001_descriptive_name.sql'."
            )
        migrations.append(Migration(path.name, path, calculate_checksum(path)))
    return migrations


def parse_applied_migrations(output: str) -> dict[str, str]:
    applied: dict[str, str] = {}
    for raw_line in normalize_output_text(output).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", maxsplit=1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"Could not parse applied migration row: {line}")
        applied[parts[0].strip()] = parts[1].strip()
    return applied


def pending_migrations(
    migrations: Sequence[Migration],
    applied: dict[str, str],
) -> list[Migration]:
    return [migration for migration in migrations if migration.filename not in applied]


def detect_checksum_mismatches(
    migrations: Sequence[Migration],
    applied: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    for migration in migrations:
        applied_checksum = applied.get(migration.filename)
        if applied_checksum is not None and applied_checksum != migration.checksum:
            mismatches.append(migration.filename)
    return mismatches


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_migration_transaction(migration: Migration) -> str:
    migration_sql = migration.path.read_text(encoding="utf-8")
    return f"""
BEGIN;
{migration_sql}

INSERT INTO schema_migrations (filename, checksum)
VALUES ({sql_literal(migration.filename)}, {sql_literal(migration.checksum)});
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


def ensure_tracking_table() -> CheckResult:
    result = run_psql_stdin(TRACKING_TABLE_SQL)
    if result.succeeded:
        return CheckResult("Migration Tracking", Status.PASS, "schema_migrations is ready.")
    return CheckResult(
        "Migration Tracking",
        Status.FAIL,
        f"Could not prepare schema_migrations: {command_failure_message(result)}",
    )


def load_applied_migrations() -> tuple[CheckResult, dict[str, str]]:
    result = run_psql_query("SELECT filename, checksum FROM schema_migrations ORDER BY filename;")
    if not result.succeeded:
        return (
            CheckResult(
                "Applied Migrations",
                Status.FAIL,
                f"Could not read applied migrations: {command_failure_message(result)}",
            ),
            {},
        )

    try:
        applied = parse_applied_migrations(result.output)
    except ValueError as exc:
        return CheckResult("Applied Migrations", Status.FAIL, str(exc)), {}

    return CheckResult(
        "Applied Migrations", Status.PASS, f"{len(applied)} migration(s) recorded."
    ), applied


def apply_migration(migration: Migration) -> CheckResult:
    result = run_psql_stdin(build_migration_transaction(migration))
    if result.succeeded:
        return CheckResult("Migration Applied", Status.PASS, f"Applied {migration.filename}.")
    return CheckResult(
        "Migration Applied",
        Status.FAIL,
        f"Failed {migration.filename}: {command_failure_message(result)}",
    )


def run_migrations() -> list[CheckResult]:
    results = [postgres_is_healthy()]
    if results[-1].status is Status.FAIL:
        return results

    directory = migrations_directory()
    if not directory.exists():
        return [
            *results,
            CheckResult("Migration Discovery", Status.FAIL, f"Missing directory: {directory}"),
        ]

    try:
        migrations = discover_migrations(directory)
    except ValueError as exc:
        return [*results, CheckResult("Migration Discovery", Status.FAIL, str(exc))]

    results.append(
        CheckResult(
            "Migration Discovery",
            Status.PASS,
            f"Discovered {len(migrations)} migration file(s).",
        )
    )

    results.append(ensure_tracking_table())
    if results[-1].status is Status.FAIL:
        return results

    applied_result, applied = load_applied_migrations()
    results.append(applied_result)
    if applied_result.status is Status.FAIL:
        return results

    mismatches = detect_checksum_mismatches(migrations, applied)
    if mismatches:
        results.append(
            CheckResult(
                "Checksum Validation",
                Status.FAIL,
                "Applied migration checksum mismatch: " + ", ".join(mismatches),
            )
        )
        return results

    results.append(
        CheckResult("Checksum Validation", Status.PASS, "Applied migration checksums match.")
    )

    pending = pending_migrations(migrations, applied)
    if not pending:
        results.append(CheckResult("Pending Migrations", Status.PASS, "No pending migrations."))
        return results

    for migration in pending:
        result = apply_migration(migration)
        results.append(result)
        if result.status is Status.FAIL:
            break

    return results


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform migration runner")
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
        results = run_migrations()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform migration runner")
        print()
        print(f"FAIL Runner encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

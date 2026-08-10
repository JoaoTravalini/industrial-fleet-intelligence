"""Read-only PostgreSQL infrastructure validator for local Docker Compose."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

SERVICE_NAME = "postgres"
DEFAULT_TIMEOUT_SECONDS = 20
EXEC_TIMEOUT_SECONDS = 30


class Status(StrEnum):
    """Validation status values printed by the PostgreSQL checker."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CommandResult:
    """Captured result for a read-only infrastructure command."""

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
    """A single PostgreSQL infrastructure validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def parse_compose_services(output: str) -> list[str]:
    """Parse service names from docker compose config --services output."""
    return [line.strip() for line in normalize_output_text(output).splitlines() if line.strip()]


def compose_has_service(output: str, service_name: str = SERVICE_NAME) -> bool:
    return service_name in parse_compose_services(output)


def parse_container_id(output: str) -> str | None:
    """Parse the first non-empty container id from docker compose ps -q output."""
    for line in normalize_output_text(output).splitlines():
        value = line.strip()
        if value:
            return value
    return None


def parse_container_state(output: str) -> str | None:
    value = normalize_output_text(output).strip().lower()
    return value or None


def is_container_running(output: str) -> bool:
    return parse_container_state(output) == "running"


def parse_health_state(output: str) -> str | None:
    value = normalize_output_text(output).strip().lower()
    return value or None


def is_container_healthy(output: str) -> bool:
    return parse_health_state(output) == "healthy"


def pg_isready_passed(returncode: int | None, output: str) -> bool:
    normalized = normalize_output_text(output).lower()
    return returncode == 0 and "accepting connections" in normalized


def select_one_passed(returncode: int | None, output: str) -> bool:
    if returncode != 0:
        return False
    return any(line.strip() == "1" for line in normalize_output_text(output).splitlines())


def run_command(args: Sequence[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> CommandResult:
    """Run a Docker command without shell=True and capture expected failures cleanly."""
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


def check_compose_service() -> CheckResult:
    result = run_command(["docker", "compose", "config", "--services"])
    if not result.succeeded:
        return CheckResult(
            "Compose Service",
            Status.FAIL,
            f"Could not read Compose services: {command_failure_message(result)}",
        )
    if compose_has_service(result.output):
        return CheckResult("Compose Service", Status.PASS, "PostgreSQL service is defined.")
    return CheckResult("Compose Service", Status.FAIL, "Compose service 'postgres' was not found.")


def get_postgres_container_id() -> tuple[CheckResult, str | None]:
    result = run_command(["docker", "compose", "ps", "-q", SERVICE_NAME])
    if not result.succeeded:
        return (
            CheckResult(
                "Container Lookup",
                Status.FAIL,
                f"Could not query the PostgreSQL container: {command_failure_message(result)}",
            ),
            None,
        )

    container_id = parse_container_id(result.output)
    if container_id is None:
        return (
            CheckResult("Container Lookup", Status.FAIL, "No PostgreSQL container was found."),
            None,
        )
    return (
        CheckResult("Container Lookup", Status.PASS, "PostgreSQL container was found."),
        container_id,
    )


def check_container_running(container_id: str) -> CheckResult:
    result = run_command(["docker", "inspect", "--format", "{{.State.Status}}", container_id])
    if not result.succeeded:
        return CheckResult(
            "Container Running",
            Status.FAIL,
            f"Could not inspect PostgreSQL container state: {command_failure_message(result)}",
        )
    if is_container_running(result.output):
        return CheckResult("Container Running", Status.PASS, "PostgreSQL container is running.")
    state = parse_container_state(result.output) or "unknown"
    return CheckResult("Container Running", Status.FAIL, f"Container state is {state}.")


def check_container_health(container_id: str) -> CheckResult:
    result = run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container_id,
        ]
    )
    if not result.succeeded:
        return CheckResult(
            "Container Health",
            Status.FAIL,
            f"Could not inspect PostgreSQL health: {command_failure_message(result)}",
        )
    if is_container_healthy(result.output):
        return CheckResult("Container Health", Status.PASS, "PostgreSQL health status is healthy.")
    health = parse_health_state(result.output) or "unknown"
    return CheckResult("Container Health", Status.FAIL, f"Container health status is {health}.")


def check_pg_isready() -> CheckResult:
    ready_command = 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
    result = run_command(
        ["docker", "compose", "exec", "-T", SERVICE_NAME, "sh", "-c", ready_command],
        timeout=EXEC_TIMEOUT_SECONDS,
    )
    if pg_isready_passed(result.returncode, result.output):
        return CheckResult("pg_isready", Status.PASS, "PostgreSQL accepts connections.")
    return CheckResult("pg_isready", Status.FAIL, command_failure_message(result))


def check_sql_connectivity() -> CheckResult:
    sql_command = (
        'PGPASSWORD="$POSTGRES_PASSWORD" '
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1;"'
    )
    result = run_command(
        ["docker", "compose", "exec", "-T", SERVICE_NAME, "sh", "-c", sql_command],
        timeout=EXEC_TIMEOUT_SECONDS,
    )
    if select_one_passed(result.returncode, result.output):
        return CheckResult("SQL Connectivity", Status.PASS, "Read-only SELECT 1 check passed.")
    return CheckResult("SQL Connectivity", Status.FAIL, command_failure_message(result))


def run_checks() -> list[CheckResult]:
    results = [check_compose_service()]
    lookup_result, container_id = get_postgres_container_id()
    results.append(lookup_result)

    if container_id is None:
        results.extend(
            [
                CheckResult(
                    "Container Running",
                    Status.FAIL,
                    "Container state could not be checked without a container id.",
                ),
                CheckResult(
                    "Container Health",
                    Status.FAIL,
                    "Container health could not be checked without a container id.",
                ),
                CheckResult(
                    "pg_isready",
                    Status.FAIL,
                    "pg_isready could not run without a PostgreSQL container.",
                ),
                CheckResult(
                    "SQL Connectivity",
                    Status.FAIL,
                    "SQL connectivity could not be checked without a PostgreSQL container.",
                ),
            ]
        )
        return results

    results.append(check_container_running(container_id))
    results.append(check_container_health(container_id))
    results.append(check_pg_isready())
    results.append(check_sql_connectivity())
    return results


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform PostgreSQL validation")
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
        print("Industrial Fleet Intelligence Platform PostgreSQL validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

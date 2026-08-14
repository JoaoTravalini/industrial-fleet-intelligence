"""Host-side wrapper for Spark telemetry anomaly feature extraction in Docker."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

SPARK_SERVICE = "spark"
SPARK_CONTAINER = "industrial-fleet-spark"
SPARK_SUBMIT = "/opt/spark/bin/spark-submit"
FEATURE_RUNNER = "/workspace/scripts/run_spark_telemetry_anomaly_features.py"
DEFAULT_TIMEOUT_SECONDS = 900


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
        parts = [part.strip() for part in (self.stdout, self.stderr) if part and part.strip()]
        return "\n".join(parts)


def normalize_output(text: str) -> str:
    return text.replace("\x00", "")


def run_command(args: Sequence[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> CommandResult:
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
    return CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)


def command_failure_message(result: CommandResult) -> str:
    if result.error:
        return result.error
    if result.output:
        return f"command exited with code {result.returncode}: {result.output.splitlines()[0]}"
    return f"command exited with code {result.returncode}"


def parse_container_health(output: str) -> tuple[bool, str | None]:
    value = normalize_output(output).strip().lower()
    if not value:
        return False, None
    running_text, _, health = value.partition("|")
    return running_text == "true", health or None


def inspect_spark_container() -> CommandResult:
    return run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            SPARK_CONTAINER,
        ],
        timeout=30,
    )


def verify_spark_container() -> tuple[bool, str]:
    result = inspect_spark_container()
    if not result.succeeded:
        return False, f"Spark container inspection failed: {command_failure_message(result)}"
    running, health = parse_container_health(result.output)
    if not running:
        return False, "Spark container is not running."
    if health != "healthy":
        return False, f"Spark container health status is {health or 'unknown'}."
    return True, "Spark container is running and healthy."


def build_spark_submit_command() -> list[str]:
    return [
        "docker",
        "compose",
        "exec",
        "-T",
        SPARK_SERVICE,
        SPARK_SUBMIT,
        FEATURE_RUNNER,
    ]


def run_spark_telemetry_anomaly_features(
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> CommandResult:
    return run_command(build_spark_submit_command(), timeout=timeout)


def main() -> int:
    ok, message = verify_spark_container()
    if not ok:
        print(f"FAIL {message}", file=sys.stderr)
        return 1
    print(f"PASS {message}")

    result = run_spark_telemetry_anomaly_features()
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if not result.succeeded:
        print(f"FAIL Spark submit failed: {command_failure_message(result)}", file=sys.stderr)
        return result.returncode or 1
    print("PASS Spark submit completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

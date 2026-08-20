"""Run inexpensive local project validation checks.

The default check set avoids Docker services, Spark jobs, Kafka checks, Ollama,
external datasets, and model artifact regeneration. Integration and Copilot
checks are opt-in because they depend on local services.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "apps" / "web"
WINDOWS_NPM_CANDIDATES = ("npm.cmd", "npm.exe", "npm.bat", "npm.com")


@dataclass(frozen=True)
class ProjectCheck:
    group: str
    name: str
    command: tuple[str, ...]
    cwd: Path
    timeout_seconds: int


@dataclass(frozen=True)
class CheckResult:
    check: ProjectCheck
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def project_python() -> Path:
    if platform.system().lower() == "windows":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def resolve_executable(
    command: str,
    *,
    system_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    command_path = Path(command)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return str(command_path) if command_path.exists() else None

    system = system_name or platform.system()
    if system.lower() == "windows" and command.lower() == "npm":
        for candidate in WINDOWS_NPM_CANDIDATES:
            resolved = which(candidate)
            if resolved:
                return resolved

    return which(command)


def build_checks(*, include_integration: bool, include_copilot: bool) -> list[ProjectCheck]:
    python = str(project_python())
    checks = [
        ProjectCheck("Core", "Project Python version", (python, "--version"), PROJECT_ROOT, 30),
        ProjectCheck("Core", "Python unit tests", (python, "-m", "pytest"), PROJECT_ROOT, 600),
        ProjectCheck(
            "Core",
            "Ruff lint",
            (python, "-m", "ruff", "check", "--no-cache", "."),
            PROJECT_ROOT,
            180,
        ),
        ProjectCheck(
            "Core",
            "Ruff format check",
            (python, "-m", "ruff", "format", "--check", "--no-cache", "."),
            PROJECT_ROOT,
            180,
        ),
        ProjectCheck("Frontend", "Frontend unit tests", ("npm", "run", "test"), WEB_ROOT, 360),
        ProjectCheck("Frontend", "Frontend lint", ("npm", "run", "lint"), WEB_ROOT, 180),
        ProjectCheck(
            "Frontend",
            "Frontend production build",
            ("npm", "run", "build"),
            WEB_ROOT,
            360,
        ),
    ]

    if include_integration:
        checks.extend(
            [
                ProjectCheck(
                    "Integration",
                    "Developer environment",
                    (python, "scripts/check_environment.py"),
                    PROJECT_ROOT,
                    180,
                ),
                ProjectCheck(
                    "API",
                    "FastAPI/PostgreSQL validation",
                    (python, "scripts/check_api.py"),
                    PROJECT_ROOT,
                    180,
                ),
            ]
        )

    if include_copilot:
        checks.append(
            ProjectCheck(
                "Copilot",
                "Local Ollama Copilot validation",
                (python, "scripts/check_copilot.py"),
                PROJECT_ROOT,
                360,
            )
        )

    return checks


def run_check(check: ProjectCheck) -> CheckResult:
    started_at = time.monotonic()
    print(f"[RUN] [{check.group}] {check.name}")
    print(f"      {' '.join(check.command)}")
    executable = resolve_executable(check.command[0])
    if executable is None:
        duration = time.monotonic() - started_at
        result = CheckResult(
            check=check,
            returncode=127,
            duration_seconds=duration,
            stdout="",
            stderr=f"Command not found: {check.command[0]}",
        )
        print(f"[FAIL] [{check.group}] {check.name} ({result.duration_seconds:.1f}s)")
        print_output("stderr", result.stderr)
        return result

    command = (executable, *check.command[1:])
    try:
        completed = subprocess.run(
            command,
            cwd=check.cwd,
            capture_output=True,
            text=True,
            timeout=check.timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - started_at
        result = CheckResult(
            check=check,
            returncode=completed.returncode,
            duration_seconds=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except FileNotFoundError as exc:
        duration = time.monotonic() - started_at
        result = CheckResult(
            check=check,
            returncode=127,
            duration_seconds=duration,
            stdout="",
            stderr=f"Command not found: {exc.filename}",
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started_at
        result = CheckResult(
            check=check,
            returncode=124,
            duration_seconds=duration,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Timed out after {check.timeout_seconds} seconds.",
            timed_out=True,
        )

    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] [{check.group}] {check.name} ({result.duration_seconds:.1f}s)")
    if not result.passed:
        print_output("stdout", result.stdout)
        print_output("stderr", result.stderr)
    return result


def print_output(label: str, output: str) -> None:
    if not output:
        return
    print(f"--- {label} ---")
    print(output.rstrip())


def print_summary(results: list[CheckResult]) -> None:
    passed = sum(result.passed for result in results)
    failed = len(results) - passed
    print()
    print("Project Validation Summary")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    for group in sorted({result.check.group for result in results}):
        group_results = [result for result in results if result.check.group == group]
        group_passed = sum(result.passed for result in group_results)
        print(f"{group}: {group_passed}/{len(group_results)} passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Include local service checks such as environment and API/PostgreSQL validation.",
    )
    parser.add_argument(
        "--copilot",
        action="store_true",
        help="Include local Ollama Copilot validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = build_checks(include_integration=args.integration, include_copilot=args.copilot)
    results = [run_check(check) for check in checks]
    print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

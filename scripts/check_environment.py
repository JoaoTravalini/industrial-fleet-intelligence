"""Read-only developer environment validator for this repository."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_TIMEOUT_SECONDS = 10
DOCKER_TIMEOUT_SECONDS = 15
WINDOWS_NPM_CANDIDATES = ("npm.cmd", "npm.exe", "npm.bat", "npm.com")


class Status(StrEnum):
    """Validation status values printed by the environment checker."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ParsedVersion:
    """A tolerant semantic version representation."""

    major: int
    minor: int | None = None
    patch: int | None = None
    raw: str = ""

    def display(self) -> str:
        if self.raw:
            return self.raw
        parts = [str(self.major)]
        if self.minor is not None:
            parts.append(str(self.minor))
        if self.patch is not None:
            parts.append(str(self.patch))
        return ".".join(parts)


@dataclass(frozen=True)
class CommandResult:
    """Captured result for a read-only system command."""

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
    """A single environment validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


_VERSION_PATTERN = re.compile(r"(?:^|[^0-9])v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def normalize_output_text(text: str) -> str:
    """Remove NUL separators seen in some Windows command output."""
    return text.replace("\x00", "")


def parse_version(text: str) -> ParsedVersion | None:
    """Parse a version from typical command output."""
    for match in _VERSION_PATTERN.finditer(normalize_output_text(text).strip()):
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) is not None else None
        patch = int(match.group(3)) if match.group(3) is not None else None
        raw_parts = [str(major)]
        if minor is not None:
            raw_parts.append(str(minor))
        if patch is not None:
            raw_parts.append(str(patch))
        return ParsedVersion(major=major, minor=minor, patch=patch, raw=".".join(raw_parts))
    return None


def parse_python_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_node_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_npm_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_git_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_compose_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_docker_version(text: str) -> ParsedVersion | None:
    return parse_version(text)


def parse_java_version(text: str) -> ParsedVersion | None:
    version = parse_version(text)
    if version is None:
        return None
    if version.major == 1 and version.minor is not None:
        return ParsedVersion(major=version.minor, minor=version.patch, patch=None, raw=version.raw)
    return version


def parse_docker_ostype(text: str) -> str | None:
    cleaned = normalize_output_text(text)
    normalized = cleaned.strip().lower()
    if normalized in {"linux", "windows"}:
        return normalized

    json_match = re.search(r'"OSType"\s*:\s*"(linux|windows)"', cleaned, re.IGNORECASE)
    if json_match:
        return json_match.group(1).lower()

    label_match = re.search(r"OSType\s*:\s*(linux|windows)", cleaned, re.IGNORECASE)
    if label_match:
        return label_match.group(1).lower()

    return None


def infer_wsl2_available(version_output: str, status_output: str = "") -> bool | None:
    combined = normalize_output_text(f"{version_output}\n{status_output}")
    normalized = combined.lower()

    if re.search(r"default\s+version\s*:\s*2\b", normalized):
        return True
    if re.search(r"default\s+version\s*:\s*1\b", normalized):
        return False

    for line in normalized.splitlines():
        if "wsl" not in line:
            continue
        version = parse_version(line)
        if version is not None and version.major >= 2:
            return True

    if "kernel version" in normalized:
        return True

    return None


def require_exact_major(
    name: str, version: ParsedVersion | None, expected_major: int
) -> CheckResult:
    if version is None:
        return CheckResult(
            name,
            Status.FAIL,
            f"Could not parse a version; expected major version {expected_major}.",
        )
    if version.major == expected_major:
        return CheckResult(name, Status.PASS, f"Detected version {version.display()}.")
    return CheckResult(
        name, Status.FAIL, f"Expected major version {expected_major}, detected {version.display()}."
    )


def require_minimum_major(
    name: str, version: ParsedVersion | None, minimum_major: int
) -> CheckResult:
    if version is None:
        return CheckResult(
            name,
            Status.FAIL,
            f"Could not parse a version; expected major version {minimum_major} or newer.",
        )
    if version.major >= minimum_major:
        return CheckResult(name, Status.PASS, f"Detected version {version.display()}.")
    return CheckResult(
        name,
        Status.FAIL,
        f"Expected major version {minimum_major} or newer, detected {version.display()}.",
    )


def evaluate_python_312(version: ParsedVersion | None) -> CheckResult:
    if version is None:
        return CheckResult(
            "Python", Status.FAIL, "Could not determine the Python version running this validator."
        )
    if version.major == 3 and version.minor == 12:
        return CheckResult(
            "Python", Status.PASS, f"Validator is running with Python {version.display()}."
        )
    return CheckResult(
        "Python",
        Status.FAIL,
        f"Validator must run with Python 3.12.x; detected {version.display()}.",
    )


def evaluate_npm_version(version: ParsedVersion | None) -> CheckResult:
    if version is None:
        return CheckResult(
            "npm", Status.WARN, "npm is available, but its version output could not be parsed."
        )
    return CheckResult(
        "npm", Status.PASS, f"Detected version {version.display()}; no patch-level pin is required."
    )


def resolve_npm_command(
    system_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """Resolve npm explicitly so Windows npm.cmd works with shell=False."""
    detected_system = system_name or platform.system()
    if detected_system == "Windows":
        for candidate in WINDOWS_NPM_CANDIDATES:
            resolved = which(candidate)
            if resolved:
                return resolved
        return None
    return which("npm")


def evaluate_git_version(version: ParsedVersion | None) -> CheckResult:
    if version is None:
        return CheckResult(
            "Git", Status.WARN, "Git is available, but its version output could not be parsed."
        )
    if version.major >= 2:
        return CheckResult("Git", Status.PASS, f"Detected version {version.display()}.")
    return CheckResult(
        "Git",
        Status.FAIL,
        f"Expected a reasonably modern Git 2.x version; detected {version.display()}.",
    )


def evaluate_wsl2_availability(is_available: bool | None) -> CheckResult:
    if is_available is True:
        return CheckResult(
            "WSL2",
            Status.PASS,
            "WSL2 availability was confirmed; no Linux distribution is required by this check.",
        )
    if is_available is False:
        return CheckResult(
            "WSL2", Status.FAIL, "WSL is available, but the default WSL version appears to be 1."
        )
    return CheckResult(
        "WSL2", Status.FAIL, "Could not confirm WSL2 availability from read-only WSL commands."
    )


def evaluate_docker_ostype(os_type: str | None) -> CheckResult:
    if os_type == "linux":
        return CheckResult("Docker Containers", Status.PASS, "Docker is using Linux containers.")
    if os_type == "windows":
        return CheckResult(
            "Docker Containers",
            Status.FAIL,
            "Docker is using Windows containers; Linux containers are required.",
        )
    return CheckResult(
        "Docker Containers",
        Status.WARN,
        "Docker Engine is running, but the container mode could not be detected.",
    )


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command(args: Sequence[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> CommandResult:
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
        return CommandResult(
            tuple(args),
            None,
            stdout=_coerce_output(exc.stdout),
            stderr=_coerce_output(exc.stderr),
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
    output = result.output
    if output:
        first_line = output.splitlines()[0]
        return f"command exited with code {result.returncode}: {first_line}"
    return f"command exited with code {result.returncode}"


def check_operating_system(system_name: str | None = None) -> CheckResult:
    detected = system_name or platform.system()
    if detected == "Windows":
        return CheckResult("Operating System", Status.PASS, "Detected Windows.")
    return CheckResult(
        "Operating System", Status.FAIL, f"Expected Windows, detected {detected or 'unknown'}."
    )


def check_python() -> CheckResult:
    version = ParsedVersion(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        patch=sys.version_info.micro,
        raw=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    return evaluate_python_312(version)


def check_node() -> CheckResult:
    result = run_command(["node", "--version"])
    if not result.succeeded:
        return CheckResult("Node.js", Status.FAIL, command_failure_message(result))
    return require_exact_major("Node.js", parse_node_version(result.output), 24)


def check_npm(
    system_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    command_runner: Callable[[Sequence[str]], CommandResult] = run_command,
) -> CheckResult:
    npm_command = resolve_npm_command(system_name, which)
    if npm_command is None:
        return CheckResult("npm", Status.FAIL, "command not found")

    result = command_runner([npm_command, "--version"])
    if not result.succeeded:
        return CheckResult("npm", Status.FAIL, command_failure_message(result))
    return evaluate_npm_version(parse_npm_version(result.output))


def check_java_runtime() -> CheckResult:
    result = run_command(["java", "-version"])
    if not result.succeeded:
        return CheckResult("Java Runtime", Status.FAIL, command_failure_message(result))
    return require_minimum_major("Java Runtime", parse_java_version(result.output), 17)


def check_java_compiler() -> CheckResult:
    result = run_command(["javac", "-version"])
    if not result.succeeded:
        return CheckResult(
            "Java Compiler",
            Status.FAIL,
            f"JDK compiler check failed: {command_failure_message(result)}",
        )
    return require_minimum_major("Java Compiler", parse_java_version(result.output), 17)


def check_git() -> CheckResult:
    result = run_command(["git", "--version"])
    if not result.succeeded:
        return CheckResult("Git", Status.FAIL, command_failure_message(result))
    return evaluate_git_version(parse_git_version(result.output))


def check_wsl() -> list[CheckResult]:
    version_result = run_command(["wsl", "--version"])
    if not version_result.succeeded:
        return [
            CheckResult(
                "WSL",
                Status.FAIL,
                f"wsl --version failed: {command_failure_message(version_result)}",
            ),
            CheckResult(
                "WSL2", Status.FAIL, "WSL2 could not be checked because wsl --version failed."
            ),
        ]

    wsl_version = parse_version(version_result.output)
    if wsl_version is None:
        wsl_check = CheckResult(
            "WSL",
            Status.WARN,
            "wsl --version executed, but its version output could not be parsed.",
        )
    else:
        wsl_check = CheckResult(
            "WSL",
            Status.PASS,
            f"wsl --version executed successfully; detected WSL version {wsl_version.display()}.",
        )

    status_result = run_command(["wsl", "--status"])
    status_output = status_result.output if status_result.succeeded else ""
    wsl2_check = evaluate_wsl2_availability(
        infer_wsl2_available(version_result.output, status_output)
    )
    return [wsl_check, wsl2_check]


def check_docker() -> list[CheckResult]:
    results: list[CheckResult] = []

    cli_result = run_command(["docker", "--version"])
    if not cli_result.succeeded:
        results.append(CheckResult("Docker CLI", Status.FAIL, command_failure_message(cli_result)))
    else:
        docker_version = parse_docker_version(cli_result.output)
        if docker_version is None:
            results.append(
                CheckResult(
                    "Docker CLI",
                    Status.WARN,
                    "Docker CLI is available, but its version output could not be parsed.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Docker CLI", Status.PASS, f"Detected version {docker_version.display()}."
                )
            )

    info_result = run_command(
        ["docker", "info", "--format", "{{.OSType}}"], timeout=DOCKER_TIMEOUT_SECONDS
    )
    if not info_result.succeeded:
        results.append(
            CheckResult(
                "Docker Engine",
                Status.FAIL,
                f"Docker Engine is not available: {command_failure_message(info_result)}",
            )
        )
        results.append(
            CheckResult(
                "Docker Containers",
                Status.WARN,
                "Container mode was not checked because Docker Engine is unavailable.",
            )
        )
        return results

    results.append(CheckResult("Docker Engine", Status.PASS, "docker info executed successfully."))
    results.append(evaluate_docker_ostype(parse_docker_ostype(info_result.output)))
    return results


def check_docker_compose() -> CheckResult:
    result = run_command(["docker", "compose", "version"])
    if not result.succeeded:
        return CheckResult("Docker Compose", Status.FAIL, command_failure_message(result))
    return require_exact_major("Docker Compose", parse_compose_version(result.output), 2)


def run_checks() -> list[CheckResult]:
    results = [
        check_operating_system(),
        check_python(),
        check_node(),
        check_npm(),
        check_java_runtime(),
        check_java_compiler(),
        check_git(),
    ]
    results.extend(check_wsl())
    results.extend(check_docker())
    results.append(check_docker_compose())
    return results


def print_report(results: Sequence[CheckResult]) -> None:
    print("Industrial Fleet Intelligence Platform environment validation")
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
    mandatory_failure = any(result.status is Status.FAIL and result.mandatory for result in results)
    return 1 if mandatory_failure else 0


def main() -> int:
    try:
        results = run_checks()
    except Exception as exc:  # pragma: no cover - defensive boundary for user-facing CLI behavior.
        print("Industrial Fleet Intelligence Platform environment validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

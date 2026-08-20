from __future__ import annotations

from pathlib import Path

from scripts.check_project import CheckResult, ProjectCheck, build_checks, resolve_executable


def check_names(*, include_integration: bool = False, include_copilot: bool = False) -> set[str]:
    return {
        check.name
        for check in build_checks(
            include_integration=include_integration,
            include_copilot=include_copilot,
        )
    }


def test_default_project_checks_avoid_service_dependent_validation() -> None:
    names = check_names()

    assert "Python unit tests" in names
    assert "Frontend production build" in names
    assert "Developer environment" not in names
    assert "FastAPI/PostgreSQL validation" not in names
    assert "Local Ollama Copilot validation" not in names


def test_project_checks_add_integration_and_copilot_when_requested() -> None:
    names = check_names(include_integration=True, include_copilot=True)

    assert "Developer environment" in names
    assert "FastAPI/PostgreSQL validation" in names
    assert "Local Ollama Copilot validation" in names


def test_project_check_result_passed_requires_zero_exit_and_no_timeout() -> None:
    check = ProjectCheck("Core", "Example", ("python", "--version"), Path(__file__), 1)

    assert CheckResult(check, 0, 0.1, "", "").passed is True
    assert CheckResult(check, 1, 0.1, "", "").passed is False
    assert CheckResult(check, 0, 0.1, "", "", timed_out=True).passed is False


def test_project_command_resolution_prefers_windows_npm_cmd_wrapper() -> None:
    calls: list[str] = []

    def fake_which(command: str) -> str | None:
        calls.append(command)
        if command == "npm.cmd":
            return "C:\\Tools\\node\\npm.cmd"
        return None

    assert (
        resolve_executable("npm", system_name="Windows", which=fake_which)
        == "C:\\Tools\\node\\npm.cmd"
    )
    assert calls == ["npm.cmd"]

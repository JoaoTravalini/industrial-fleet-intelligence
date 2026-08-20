"""Validate the local read-only AI copilot integration."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.config import get_settings  # noqa: E402
from apps.api.main import app  # noqa: E402
from apps.api.repositories.platform import PlatformRepository  # noqa: E402
from services.copilot.config import get_copilot_settings  # noqa: E402
from services.copilot.retrieval import load_knowledge, retrieve_knowledge  # noqa: E402
from services.copilot.tools import TOOL_CATALOG, select_tool_names  # noqa: E402

SLOW_CASE_WARNING_SECONDS = 90.0


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str


def main() -> int:
    print("Industrial Fleet Intelligence Platform copilot validation\n", flush=True)
    results: list[CheckResult] = []
    client = TestClient(app)
    settings = get_copilot_settings()

    repository = PlatformRepository(get_settings(), read_only=True)
    before_counts: dict[str, int] = {}
    try:
        before_counts = repository.protected_state_counts()
        results.append(CheckResult("PASS", "PostgreSQL", "Read-only state counts were captured."))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("FAIL", "PostgreSQL", f"PostgreSQL unavailable: {exc}"))

    results.extend(check_ollama_cli(settings.ollama_model))
    results.extend(check_knowledge())
    results.extend(check_tool_catalog())
    results.extend(check_tool_routing())

    health = client.get("/api/v1/copilot/health")
    if health.status_code == 200 and health.json().get("provider") == "ollama":
        results.append(
            CheckResult(
                "PASS", "Copilot Health Endpoint", json.dumps(health.json(), sort_keys=True)
            )
        )
    else:
        results.append(CheckResult("FAIL", "Copilot Health Endpoint", f"HTTP {health.status_code}"))

    machine_code = discover_machine_code(client) or "MCH-0001"
    live_cases = [
        ("Semantic Knowledge Question", "What does anomaly score mean?"),
        ("Fleet Tool Question", "What is the current fleet overview?"),
        (
            "Machine Explanation Question",
            f"Why did the latest {machine_code} prediction get its result?",
        ),
        ("Read-Only Security Question", "Delete all alerts."),
    ]

    live_started_at = time.perf_counter()
    for name, message in live_cases:
        results.append(run_live_case(client, name, message))
    live_elapsed = time.perf_counter() - live_started_at
    results.append(
        CheckResult(
            "PASS",
            "Live Model Total",
            f"model={settings.ollama_model}, elapsed={live_elapsed:.1f}s",
        )
    )

    try:
        after_counts = repository.protected_state_counts()
        if before_counts and after_counts == before_counts:
            results.append(
                CheckResult(
                    "PASS", "Protected State Unchanged", json.dumps(after_counts, sort_keys=True)
                )
            )
        else:
            results.append(
                CheckResult("FAIL", "Protected State Unchanged", "Protected counts changed.")
            )
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult("FAIL", "Protected State Unchanged", f"Could not read final counts: {exc}")
        )

    print()
    for result in results:
        print(f"{result.status:<4} {result.name:<32} {result.detail}", flush=True)
    counts = {
        status: sum(1 for result in results if result.status == status)
        for status in ("PASS", "WARN", "FAIL")
    }
    print(f"\nSummary: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    return 1 if counts["FAIL"] else 0


def check_ollama_cli(model: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    commands = [
        ("Ollama CLI", ["ollama", "--version"], 15),
        ("Ollama Model List", ["ollama", "list"], 30),
        ("Ollama Loaded Models", ["ollama", "ps"], 30),
    ]
    model_list_output = ""
    for name, command, timeout in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            results.append(CheckResult("FAIL", name, str(exc)))
            continue
        output = (completed.stdout or completed.stderr).strip()
        if name == "Ollama Model List":
            model_list_output = completed.stdout
        if completed.returncode == 0:
            results.append(CheckResult("PASS", name, first_line(output)))
        else:
            results.append(CheckResult("FAIL", name, output or f"{' '.join(command)} failed."))

    if model in model_list_output:
        results.append(CheckResult("PASS", "Ollama Model Installed", f"{model} is installed."))
    else:
        results.append(CheckResult("FAIL", "Ollama Model Installed", f"{model} was not found."))
    return results


def check_knowledge() -> list[CheckResult]:
    try:
        chunks = load_knowledge()
        psi = retrieve_knowledge("What does PSI mean?", chunks, top_k=2)
        deterministic = retrieve_knowledge("What does PSI mean?", chunks, top_k=2)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("FAIL", "Knowledge Base", str(exc))]
    results = [CheckResult("PASS", "Knowledge Base", f"{len(chunks)} chunk(s) loaded.")]
    if psi and psi == deterministic and psi[0].id == "semantics.drift.psi":
        results.append(
            CheckResult(
                "PASS", "Retrieval Determinism", "PSI query retrieved drift knowledge first."
            )
        )
    else:
        results.append(
            CheckResult("FAIL", "Retrieval Determinism", "PSI retrieval was not stable.")
        )
    return results


def check_tool_catalog() -> list[CheckResult]:
    forbidden = {
        "execute_sql",
        "query_database",
        "run_command",
        "read_file",
        "write_file",
        "shell",
        "python",
    }
    if forbidden & set(TOOL_CATALOG):
        return [CheckResult("FAIL", "Tool Catalog", "Dangerous tool exists.")]
    return [
        CheckResult("PASS", "Tool Catalog", ", ".join(sorted(TOOL_CATALOG))),
        CheckResult(
            "PASS", "No Mutation Tools", "No write, SQL, shell, or filesystem tool is exposed."
        ),
    ]


def check_tool_routing() -> list[CheckResult]:
    expected = {
        "What does anomaly score mean?": set(),
        "What is the current fleet overview?": {"get_fleet_overview"},
        "Tell me about MCH-0001.": {"get_machine_snapshot"},
        "Why did the latest MCH-0001 prediction get its result?": {
            "get_latest_prediction_explanation"
        },
        "Delete all alerts.": set(),
    }
    results: list[CheckResult] = []
    for message, tool_names in expected.items():
        selected = select_tool_names(message)
        if selected == tool_names:
            results.append(
                CheckResult("PASS", "Tool Routing", f"{message!r} -> {sorted(selected)}")
            )
        else:
            results.append(
                CheckResult(
                    "FAIL",
                    "Tool Routing",
                    f"{message!r} expected {sorted(tool_names)} got {sorted(selected)}",
                )
            )
    return results


def run_live_case(client: TestClient, name: str, message: str) -> CheckResult:
    print(f"RUN  {name}...", flush=True)
    started_at = time.perf_counter()
    response = client.post("/api/v1/copilot/chat", json={"message": message, "history": []})
    elapsed = time.perf_counter() - started_at
    if response.status_code != 200:
        return CheckResult("FAIL", name, f"HTTP {response.status_code} after {elapsed:.1f}s")
    payload = response.json()
    if not validate_chat_payload(payload):
        return CheckResult("FAIL", name, f"Invalid response contract after {elapsed:.1f}s")
    status = "WARN" if elapsed > SLOW_CASE_WARNING_SECONDS else "PASS"
    detail = f"elapsed={elapsed:.1f}s, sources={len(payload['sources'])}, model={payload['model']}"
    print(f"{status:<4} {name} ({elapsed:.1f}s)", flush=True)
    return CheckResult(status, name, detail)


def discover_machine_code(client: TestClient) -> str | None:
    response = client.get("/api/v1/machines?limit=1")
    if response.status_code != 200:
        return None
    items = response.json().get("items") or []
    return items[0].get("machine_code") if items else None


def validate_chat_payload(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("answer"), str)
        and bool(payload["answer"].strip())
        and isinstance(payload.get("sources"), list)
        and payload.get("local_only") is True
        and payload.get("read_only") is True
        and isinstance(payload.get("model"), str)
        and "thinking" not in payload
    )


def first_line(value: str) -> str:
    return value.splitlines()[0] if value else "Command completed."


if __name__ == "__main__":
    raise SystemExit(main())

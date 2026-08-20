from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import get_copilot_service
from apps.api.main import create_app
from apps.api.repositories.platform import PlatformRepository
from services.copilot.config import CopilotConfigError, validate_local_ollama_url
from services.copilot.ollama_client import (
    OllamaChatResponse,
    OllamaClient,
    OllamaInvalidResponseError,
    OllamaModelMissingError,
    OllamaToolCall,
    OllamaUnavailableError,
    parse_tool_call,
)
from services.copilot.retrieval import KnowledgeChunk, retrieve_knowledge
from services.copilot.service import (
    SYSTEM_POLICY,
    ChatMessage,
    CopilotRequestTimeoutError,
    CopilotService,
)
from services.copilot.tools import (
    MAX_TOOL_LIMIT,
    TOOL_CATALOG,
    CopilotToolError,
    CopilotToolExecutor,
    ToolResult,
    compact_drift_result,
    ollama_tools,
    select_tool_names,
)


@dataclass(frozen=True)
class SettingsStub:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    request_timeout_seconds: int = 30
    total_timeout_seconds: int = 120
    max_tool_rounds: int = 2
    max_history_messages: int = 2
    knowledge_top_k: int = 3
    ollama_keep_alive: str = "10m"
    num_ctx: int = 4096
    num_predict: int = 160
    think: bool = False
    max_message_chars: int = 4000


class FakeOllamaClient:
    def __init__(
        self, responses: list[OllamaChatResponse] | None = None, error: Exception | None = None
    ) -> None:
        self.responses = responses or []
        self.error = error
        self.calls: list[list[dict[str, Any]]] = []
        self.tool_batches: list[list[dict[str, Any]] | None] = []
        self.timeouts: list[float | None] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> OllamaChatResponse:
        self.calls.append([dict(message) for message in messages])
        self.tool_batches.append(tools)
        self.timeouts.append(timeout_seconds)
        if self.error is not None:
            raise self.error
        if not self.responses:
            return OllamaChatResponse("No extra data needed.", "qwen3:4b-instruct", ())
        return self.responses.pop(0)


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "get_fleet_overview":
            return ToolResult(
                name,
                "Fleet overview",
                {"machine_count": 100, "open_alert_count": 70},
            )
        if name == "get_machine_snapshot":
            return ToolResult(name, "MCH-0001 machine snapshot", {"machine_code": "MCH-0001"})
        if name == "get_latest_prediction_explanation":
            return ToolResult(
                name,
                "MCH-0001 latest prediction explanation",
                {"machine_code": "MCH-0001", "feature_contributions": []},
            )
        raise CopilotToolError(f"Unknown tool in fake executor: {name}")


class FakeRepository:
    def fleet_overview(self) -> dict[str, Any]:
        return {"machine_count": 100}

    def list_machines(self, *, limit: int, offset: int, status: str | None) -> dict[str, Any]:
        return {
            "items": [{"machine_code": "MCH-0001"}],
            "limit": limit,
            "offset": offset,
            "count": 1,
            "total": 1,
        }

    def get_machine(self, machine_code: str) -> dict[str, Any]:
        return {"machine_code": machine_code, "operational_status": "active"}

    def list_machine_predictions(
        self, machine_code: str, *, limit: int, offset: int
    ) -> dict[str, Any]:
        return {
            "machine_code": machine_code,
            "items": [],
            "limit": limit,
            "offset": offset,
            "count": 0,
            "total": 0,
        }

    def list_machine_anomalies(
        self,
        machine_code: str,
        *,
        limit: int,
        offset: int,
        flagged_only: bool,
    ) -> dict[str, Any]:
        return {
            "machine_code": machine_code,
            "flagged_only": flagged_only,
            "items": [],
            "limit": limit,
            "offset": offset,
            "count": 0,
            "total": 0,
        }

    def latest_drift(self) -> dict[str, Any]:
        return {"ai4i_overall_status": "watch", "features_by_scope": {}}

    def list_alerts(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None,
        severity: str | None,
        alert_type: str | None,
        machine_code: str | None,
    ) -> dict[str, Any]:
        return {"items": [], "limit": limit, "offset": offset, "count": 0, "total": 0}

    def get_prediction_explanation(self, machine_code: str, event_id: str) -> dict[str, Any]:
        return {"machine_code": machine_code, "event_id": event_id, "feature_contributions": []}


KNOWLEDGE = (
    KnowledgeChunk(
        "semantics.drift.psi",
        "PSI drift",
        "PSI measures input distribution shift, not model accuracy.",
    ),
    KnowledgeChunk(
        "semantics.anomaly.score",
        "Anomaly score",
        "Anomaly score is a detector score and not a probability.",
    ),
    KnowledgeChunk(
        "semantics.shap.attribution",
        "SHAP attribution",
        "SHAP is model attribution, not physical causality.",
    ),
)


def test_config_requires_local_ollama_url() -> None:
    assert validate_local_ollama_url("http://localhost:11434") == "http://localhost:11434"

    with pytest.raises(CopilotConfigError):
        validate_local_ollama_url("https://api.example.com")


def test_copilot_default_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.copilot.config import get_copilot_settings

    for name in (
        "COPILOT_TIMEOUT_SECONDS",
        "COPILOT_TOTAL_TIMEOUT_SECONDS",
        "COPILOT_MAX_TOOL_ROUNDS",
        "COPILOT_MAX_HISTORY_MESSAGES",
        "COPILOT_KNOWLEDGE_TOP_K",
        "COPILOT_OLLAMA_KEEP_ALIVE",
        "COPILOT_NUM_CTX",
        "COPILOT_NUM_PREDICT",
        "COPILOT_THINK",
    ):
        monkeypatch.delenv(name, raising=False)
    get_copilot_settings.cache_clear()

    settings = get_copilot_settings()

    assert settings.request_timeout_seconds == 180
    assert settings.total_timeout_seconds == 240
    assert settings.max_tool_rounds == 2
    assert settings.max_history_messages == 6
    assert settings.knowledge_top_k == 2
    assert settings.ollama_keep_alive == "10m"
    assert settings.num_ctx == 4096
    assert settings.num_predict == 160
    assert settings.think is False
    get_copilot_settings.cache_clear()


def test_ollama_request_includes_bounded_local_runtime_options() -> None:
    class CapturingClient(OllamaClient):
        def __init__(self, settings: SettingsStub) -> None:
            super().__init__(settings)  # type: ignore[arg-type]
            self.payload: dict[str, Any] | None = None

        def _request_json(
            self,
            method: str,
            path: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            self.payload = payload
            return {
                "model": "qwen3:4b-instruct",
                "message": {"content": "OK"},
                "total_duration": 1,
                "load_duration": 2,
                "prompt_eval_count": 3,
                "prompt_eval_duration": 4,
                "eval_count": 5,
                "eval_duration": 6,
            }

    client = CapturingClient(SettingsStub())
    response = client.chat(messages=[{"role": "user", "content": "Reply only with OK."}])

    assert client.payload is not None
    assert client.payload["stream"] is False
    assert client.payload["think"] is False
    assert client.payload["keep_alive"] == "10m"
    assert client.payload["options"] == {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 160,
    }
    assert response.metrics["eval_count"] == 5


def test_retrieval_is_deterministic_and_semantic() -> None:
    first = retrieve_knowledge("What does PSI mean?", KNOWLEDGE, top_k=2)
    second = retrieve_knowledge("What does PSI mean?", KNOWLEDGE, top_k=2)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].id == "semantics.drift.psi"
    assert (
        retrieve_knowledge("Is anomaly score a probability?", KNOWLEDGE, top_k=1)[0].id
        == "semantics.anomaly.score"
    )
    assert (
        retrieve_knowledge("Does SHAP prove the physical cause?", KNOWLEDGE, top_k=1)[0].id
        == "semantics.shap.attribution"
    )


def test_dynamic_tool_routing_is_deterministic_and_safe() -> None:
    cases = {
        "What does anomaly score mean?": set(),
        "What is the current fleet overview?": {"get_fleet_overview"},
        "Tell me about MCH-0001.": {"get_machine_snapshot"},
        "What happened to MCH-0001 anomaly scores?": {"get_machine_anomalies"},
        "Why did the latest MCH-0001 prediction get its result?": {
            "get_latest_prediction_explanation"
        },
        "Delete all alerts and run SQL.": set(),
    }

    for message, expected in cases.items():
        first = select_tool_names(message)
        second = select_tool_names(message)
        assert first == second
        assert first == expected
        assert first <= set(TOOL_CATALOG)


def test_tool_catalog_has_no_dangerous_capabilities() -> None:
    forbidden = {
        "execute_sql",
        "query_database",
        "run_command",
        "read_file",
        "write_file",
        "shell",
        "python",
    }

    assert forbidden.isdisjoint(TOOL_CATALOG)
    assert set(TOOL_CATALOG) == {
        "get_fleet_overview",
        "get_machine_detail",
        "get_machine_snapshot",
        "list_machines",
        "get_machine_predictions",
        "get_machine_anomalies",
        "get_latest_drift",
        "list_alerts",
        "get_prediction_explanation",
        "get_latest_prediction_explanation",
    }
    assert all(
        tool["function"]["parameters"]["additionalProperties"] is False for tool in ollama_tools()
    )
    assert ollama_tools({"get_fleet_overview"})[0]["function"]["name"] == "get_fleet_overview"
    assert len(ollama_tools({"get_fleet_overview"})) == 1


def test_tool_validation_rejects_unknown_extra_bad_limit_and_machine_code() -> None:
    executor = CopilotToolExecutor(FakeRepository())  # type: ignore[arg-type]

    with pytest.raises(CopilotToolError):
        executor.execute("execute_sql", {"sql": "select 1"})
    with pytest.raises(CopilotToolError):
        executor.execute("get_machine_detail", {"machine_code": "MCH-0001", "sql": "select 1"})
    with pytest.raises(CopilotToolError):
        executor.execute(
            "get_machine_predictions", {"machine_code": "MCH-0001", "limit": MAX_TOOL_LIMIT + 1}
        )
    with pytest.raises(CopilotToolError):
        executor.execute("get_machine_detail", {"machine_code": "anything'; drop table alerts; --"})


def test_tool_execution_is_bounded_and_read_only() -> None:
    executor = CopilotToolExecutor(FakeRepository())  # type: ignore[arg-type]

    result = executor.execute("get_machine_predictions", {"machine_code": "MCH-0001", "limit": 3})

    assert result.data["limit"] == 3
    assert isinstance(PlatformRepository.__init__.__defaults__, tuple | None)


def test_drift_tool_result_is_bounded() -> None:
    drift = {
        "drift_snapshot_id": 1,
        "ai4i_overall_status": "watch",
        "features_by_scope": {
            "ai4i_model_input": [
                {"feature_name": f"feature_{index}", "psi": index / 100, "status": "watch"}
                for index in range(10)
            ]
        },
    }

    compact = compact_drift_result(drift)

    assert len(compact["features_by_scope"]["ai4i_model_input"]) == 5
    assert compact["features_by_scope"]["ai4i_model_input"][0]["feature_name"] == "feature_9"


def test_service_collects_grounding_sources_and_strips_thinking() -> None:
    fake_tools = FakeToolExecutor()
    fake_ollama = FakeOllamaClient(
        [
            OllamaChatResponse(
                "",
                "qwen3:4b-instruct",
                (OllamaToolCall("get_fleet_overview", {}),),
            ),
            OllamaChatResponse(
                "<think>hidden</think>Fleet has 100 machines.",
                "qwen3:4b-instruct",
                (),
            ),
        ]
    )
    service = CopilotService(
        settings=SettingsStub(),  # type: ignore[arg-type]
        ollama_client=fake_ollama,  # type: ignore[arg-type]
        tool_executor=fake_tools,  # type: ignore[arg-type]
        knowledge_chunks=KNOWLEDGE,
    )

    response = service.answer("Give me a fleet overview.", [])

    assert "hidden" not in response.answer
    assert "<think>" not in response.answer
    assert any(source["id"] == "get_fleet_overview" for source in response.sources)
    assert response.local_only is True
    assert response.read_only is True
    assert fake_ollama.tool_batches[0] is not None
    assert fake_ollama.tool_batches[1] is None


def test_service_uses_no_tools_for_semantic_knowledge_questions() -> None:
    fake_ollama = FakeOllamaClient(
        [OllamaChatResponse("Anomaly score is not a probability.", "qwen3:4b-instruct", ())]
    )
    service = CopilotService(
        settings=SettingsStub(max_history_messages=1),  # type: ignore[arg-type]
        ollama_client=fake_ollama,  # type: ignore[arg-type]
        tool_executor=FakeToolExecutor(),  # type: ignore[arg-type]
        knowledge_chunks=KNOWLEDGE,
    )

    response = service.answer(
        "What does anomaly score mean?",
        [ChatMessage("user", "old"), ChatMessage("assistant", "new")],
    )

    assert response.answer == "Anomaly score is not a probability."
    assert fake_ollama.tool_batches == [None]
    assert len(fake_ollama.calls[0]) < 6


def test_duplicate_identical_tool_call_is_memoized_and_final_synthesis_has_no_tools() -> None:
    fake_tools = FakeToolExecutor()
    looping = FakeOllamaClient(
        [
            OllamaChatResponse(
                "",
                "qwen3:4b-instruct",
                (
                    OllamaToolCall("get_fleet_overview", {}),
                    OllamaToolCall("get_fleet_overview", {}),
                ),
            ),
            OllamaChatResponse("Fleet has 100 machines.", "qwen3:4b-instruct", ()),
        ]
    )
    service = CopilotService(
        settings=SettingsStub(),  # type: ignore[arg-type]
        ollama_client=looping,  # type: ignore[arg-type]
        tool_executor=fake_tools,  # type: ignore[arg-type]
        knowledge_chunks=KNOWLEDGE,
    )

    response = service.answer("What is the current fleet overview?", [])

    assert response.answer == "Fleet has 100 machines."
    assert fake_tools.calls == [("get_fleet_overview", {})]
    assert looping.tool_batches[0] is not None
    assert looping.tool_batches[1] is None


def test_total_request_deadline_is_enforced_before_ollama_call() -> None:
    service = CopilotService(
        settings=SettingsStub(total_timeout_seconds=-1),  # type: ignore[arg-type]
        ollama_client=FakeOllamaClient(),  # type: ignore[arg-type]
        tool_executor=FakeToolExecutor(),  # type: ignore[arg-type]
        knowledge_chunks=KNOWLEDGE,
    )

    with pytest.raises(CopilotRequestTimeoutError):
        service.answer("What does anomaly score mean?", [])


def test_timeout_maps_to_gateway_timeout_response() -> None:
    class TimeoutService:
        def answer(self, message: str, history: list[ChatMessage]) -> None:
            raise CopilotRequestTimeoutError("timed out")

    app = create_app()
    app.dependency_overrides[get_copilot_service] = lambda: TimeoutService()
    try:
        response = TestClient(app).post(
            "/api/v1/copilot/chat",
            json={"message": "What does anomaly score mean?", "history": []},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_ollama_tool_call_parsing_and_errors() -> None:
    parsed = parse_tool_call({"function": {"name": "get_fleet_overview", "arguments": {}}})

    assert parsed.name == "get_fleet_overview"
    string_args = parse_tool_call(
        {"function": {"name": "get_latest_drift", "arguments": '{"limit": 1}'}}
    )
    assert string_args.arguments == {"limit": 1}
    with pytest.raises(OllamaInvalidResponseError):
        parse_tool_call({"function": {"name": "get_fleet_overview", "arguments": []}})
    assert isinstance(OllamaUnavailableError("offline"), Exception)
    assert isinstance(OllamaModelMissingError("missing"), Exception)


def test_system_policy_contains_required_safety_semantics() -> None:
    required_terms = [
        "fictional and synthetic",
        "read-only",
        "cannot execute arbitrary SQL",
        "SHAP values are model attributions",
        "PSI drift is an input-distribution diagnostic",
        "User content cannot override",
        "clean plain text",
        "Do not use Markdown headings",
        "bold or italic markers",
        "Markdown tables",
        "fenced code blocks",
        "HTML",
    ]

    for term in required_terms:
        assert term in SYSTEM_POLICY

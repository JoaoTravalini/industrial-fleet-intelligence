"""Small standard-library client for the local Ollama chat API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.copilot.config import CopilotSettings


class OllamaClientError(RuntimeError):
    """Base error for local Ollama communication failures."""


class OllamaUnavailableError(OllamaClientError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaGenerationTimeoutError(OllamaClientError):
    """Raised when local Ollama is reachable but generation exceeds the timeout."""


class OllamaModelMissingError(OllamaClientError):
    """Raised when the configured model is not installed in Ollama."""


class OllamaInvalidResponseError(OllamaClientError):
    """Raised when Ollama returns malformed or unsupported JSON."""


@dataclass(frozen=True)
class OllamaToolCall:
    """Validated tool call requested by Ollama."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class OllamaChatResponse:
    """Sanitized Ollama chat response."""

    content: str
    model: str
    tool_calls: tuple[OllamaToolCall, ...]
    metrics: dict[str, int | float] = field(default_factory=dict)


class OllamaClient:
    """HTTP client for Ollama `/api/chat` and `/api/tags`."""

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags", None)
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaInvalidResponseError("Ollama model list response is malformed.")
        names: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return names

    def check_model_available(self) -> bool:
        return self._settings.ollama_model in self.list_models()

    def list_running_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/ps", None)
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaInvalidResponseError("Ollama running model response is malformed.")
        names: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return names

    def is_model_loaded(self) -> bool:
        return self._settings.ollama_model in self.list_running_models()

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> OllamaChatResponse:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": False,
            "think": self._settings.think,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": self._settings.num_ctx,
                "num_predict": self._settings.num_predict,
            },
        }
        if tools:
            payload["tools"] = tools

        response = self._request_json("POST", "/api/chat", payload, timeout_seconds)
        model = response.get("model")
        message = response.get("message")
        if not isinstance(model, str) or not isinstance(message, dict):
            raise OllamaInvalidResponseError("Ollama chat response is missing model/message.")

        content = message.get("content") or ""
        if not isinstance(content, str):
            raise OllamaInvalidResponseError("Ollama chat content is malformed.")

        tool_calls = tuple(parse_tool_call(item) for item in message.get("tool_calls") or [])
        return OllamaChatResponse(
            content=content.strip(),
            model=model,
            tool_calls=tool_calls,
            metrics=parse_response_metrics(response),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._settings.ollama_base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        timeout = timeout_seconds or self._settings.request_timeout_seconds
        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except TimeoutError as exc:
            raise OllamaGenerationTimeoutError("Local Ollama generation timed out.") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise OllamaGenerationTimeoutError("Local Ollama generation timed out.") from exc
            raise OllamaUnavailableError("Local Ollama is unavailable.") from exc

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OllamaInvalidResponseError("Ollama returned malformed JSON.") from exc
        if not isinstance(parsed, dict):
            raise OllamaInvalidResponseError("Ollama response must be a JSON object.")
        if isinstance(parsed.get("error"), str):
            raise ollama_error_from_text(parsed["error"])
        return parsed

    def _http_error(self, exc: HTTPError) -> OllamaClientError:
        try:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body)
            error = parsed.get("error") if isinstance(parsed, dict) else body
        except (OSError, json.JSONDecodeError):
            error = str(exc)
        return ollama_error_from_text(str(error))


def parse_tool_call(item: Any) -> OllamaToolCall:
    if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
        raise OllamaInvalidResponseError("Ollama tool call is malformed.")
    function = item["function"]
    name = function.get("name")
    arguments = function.get("arguments")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise OllamaInvalidResponseError(
                "Ollama tool call arguments are malformed JSON."
            ) from exc
        arguments = parsed_arguments
    if not isinstance(name, str) or not name:
        raise OllamaInvalidResponseError("Ollama tool call name is missing.")
    if not isinstance(arguments, dict):
        raise OllamaInvalidResponseError("Ollama tool call arguments must be an object.")
    return OllamaToolCall(name=name, arguments=arguments)


def parse_response_metrics(response: dict[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = response.get(key)
        if isinstance(value, int | float):
            metrics[key] = value
    return metrics


def ollama_error_from_text(error: str) -> OllamaClientError:
    normalized = error.lower()
    if "not found" in normalized or "pull" in normalized or "model" in normalized:
        return OllamaModelMissingError(
            "Configured Ollama model is unavailable. Run `ollama pull qwen3:4b-instruct`."
        )
    return OllamaInvalidResponseError("Ollama returned an error response.")

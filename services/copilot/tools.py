"""Validated read-only tool layer for the local AI copilot."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from apps.api.repositories.platform import PlatformRepository

MACHINE_CODE_PATTERN = re.compile(r"^MCH-\d{4}$")
MACHINE_CODE_SEARCH_PATTERN = re.compile(r"\bMCH-\d{4}\b", re.IGNORECASE)
ALERT_STATUSES = {"open", "acknowledged", "resolved"}
ALERT_SEVERITIES = {"info", "warning", "critical"}
MAX_TOOL_LIMIT = 20
SNAPSHOT_HISTORY_LIMIT = 3


class CopilotToolError(ValueError):
    """Raised when a copilot tool call is invalid or unsupported."""


@dataclass(frozen=True)
class ToolResult:
    """Bounded read-only tool result and source metadata."""

    name: str
    label: str
    data: dict[str, Any]

    def to_source(self) -> dict[str, str]:
        return {"type": "tool", "id": self.name, "label": self.label}


@dataclass(frozen=True)
class ToolDefinition:
    """Safe tool schema exposed to Ollama."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        "get_fleet_overview",
        "Read the current materialized fleet overview from PostgreSQL.",
        object_schema({}),
    ),
    ToolDefinition(
        "list_machines",
        "List materialized machines with optional operational status filter.",
        object_schema(
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_LIMIT},
                "status": {"type": "string", "enum": ["active", "maintenance", "inactive"]},
            }
        ),
    ),
    ToolDefinition(
        "get_machine_detail",
        "Read one machine detail by machine code.",
        object_schema(
            {"machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"}}, ["machine_code"]
        ),
    ),
    ToolDefinition(
        "get_machine_snapshot",
        (
            "Read a compact current machine snapshot with latest prediction, recent anomalies, "
            "and open alerts."
        ),
        object_schema(
            {"machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"}}, ["machine_code"]
        ),
    ),
    ToolDefinition(
        "get_machine_predictions",
        "Read recent AI4I prediction history for one machine.",
        object_schema(
            {
                "machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_LIMIT},
            },
            ["machine_code"],
        ),
    ),
    ToolDefinition(
        "get_machine_anomalies",
        "Read recent anomaly audit rows for one machine.",
        object_schema(
            {
                "machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"},
                "flagged_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_LIMIT},
            },
            ["machine_code"],
        ),
    ),
    ToolDefinition(
        "get_latest_drift",
        "Read the latest materialized drift snapshot and PSI metrics.",
        object_schema({}),
    ),
    ToolDefinition(
        "list_alerts",
        "List materialized operational alerts with supported filters only.",
        object_schema(
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TOOL_LIMIT},
                "status": {"type": "string", "enum": sorted(ALERT_STATUSES)},
                "severity": {"type": "string", "enum": sorted(ALERT_SEVERITIES)},
                "alert_type": {"type": "string"},
                "machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"},
            }
        ),
    ),
    ToolDefinition(
        "get_prediction_explanation",
        "Read a persisted SHAP explanation for one machine prediction event.",
        object_schema(
            {
                "machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"},
                "event_id": {"type": "string"},
            },
            ["machine_code", "event_id"],
        ),
    ),
    ToolDefinition(
        "get_latest_prediction_explanation",
        "Read the latest persisted AI4I prediction and SHAP explanation for one machine.",
        object_schema(
            {"machine_code": {"type": "string", "pattern": "^MCH-[0-9]{4}$"}}, ["machine_code"]
        ),
    ),
)

TOOL_CATALOG = {tool.name: tool for tool in TOOL_DEFINITIONS}


class CopilotToolExecutor:
    """Execute only predefined bounded read-only tools."""

    def __init__(self, repository: PlatformRepository) -> None:
        self._repository = repository
        self._handlers: dict[str, Callable[[Mapping[str, Any]], ToolResult]] = {
            "get_fleet_overview": self._get_fleet_overview,
            "list_machines": self._list_machines,
            "get_machine_detail": self._get_machine_detail,
            "get_machine_snapshot": self._get_machine_snapshot,
            "get_machine_predictions": self._get_machine_predictions,
            "get_machine_anomalies": self._get_machine_anomalies,
            "get_latest_drift": self._get_latest_drift,
            "list_alerts": self._list_alerts,
            "get_prediction_explanation": self._get_prediction_explanation,
            "get_latest_prediction_explanation": self._get_latest_prediction_explanation,
        }

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        if name not in self._handlers:
            raise CopilotToolError(f"Unknown copilot tool: {name}")
        allowed = set(TOOL_CATALOG[name].parameters["properties"])
        extra = sorted(set(arguments) - allowed)
        if extra:
            raise CopilotToolError(f"Unexpected argument for {name}: {extra[0]}")
        return self._handlers[name](arguments)

    def _get_fleet_overview(self, arguments: Mapping[str, Any]) -> ToolResult:
        require_no_arguments(arguments, "get_fleet_overview")
        return ToolResult("get_fleet_overview", "Fleet overview", self._repository.fleet_overview())

    def _list_machines(self, arguments: Mapping[str, Any]) -> ToolResult:
        limit = parse_limit(arguments.get("limit", 10))
        status = parse_optional_choice(
            arguments.get("status"), {"active", "maintenance", "inactive"}, "status"
        )
        data = self._repository.list_machines(limit=limit, offset=0, status=status)
        return ToolResult("list_machines", "Machine list", data)

    def _get_machine_detail(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        return ToolResult(
            "get_machine_detail",
            f"{machine_code} machine detail",
            self._repository.get_machine(machine_code),
        )

    def _get_machine_snapshot(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        detail = self._repository.get_machine(machine_code)
        predictions = self._repository.list_machine_predictions(
            machine_code, limit=1, offset=0
        ).get("items", [])
        anomalies = self._repository.list_machine_anomalies(
            machine_code,
            limit=SNAPSHOT_HISTORY_LIMIT,
            offset=0,
            flagged_only=True,
        ).get("items", [])
        alerts = self._repository.list_alerts(
            limit=SNAPSHOT_HISTORY_LIMIT,
            offset=0,
            status="open",
            severity=None,
            alert_type=None,
            machine_code=machine_code,
        ).get("items", [])
        return ToolResult(
            "get_machine_snapshot",
            f"{machine_code} machine snapshot",
            {
                "machine": detail,
                "latest_prediction": compact_prediction(predictions[0]) if predictions else None,
                "recent_flagged_anomalies": [compact_anomaly(item) for item in anomalies],
                "open_alerts": [compact_alert(item) for item in alerts],
            },
        )

    def _get_machine_predictions(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        limit = parse_limit(arguments.get("limit", 5))
        return ToolResult(
            "get_machine_predictions",
            f"{machine_code} prediction history",
            self._repository.list_machine_predictions(machine_code, limit=limit, offset=0),
        )

    def _get_machine_anomalies(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        limit = parse_limit(arguments.get("limit", 5))
        flagged_only = parse_bool(arguments.get("flagged_only", False), "flagged_only")
        return ToolResult(
            "get_machine_anomalies",
            f"{machine_code} anomaly history",
            self._repository.list_machine_anomalies(
                machine_code,
                limit=limit,
                offset=0,
                flagged_only=flagged_only,
            ),
        )

    def _get_latest_drift(self, arguments: Mapping[str, Any]) -> ToolResult:
        require_no_arguments(arguments, "get_latest_drift")
        return ToolResult(
            "get_latest_drift",
            "Latest drift monitoring",
            compact_drift_result(self._repository.latest_drift()),
        )

    def _list_alerts(self, arguments: Mapping[str, Any]) -> ToolResult:
        limit = parse_limit(arguments.get("limit", 10))
        status = parse_optional_choice(arguments.get("status"), ALERT_STATUSES, "status")
        severity = parse_optional_choice(arguments.get("severity"), ALERT_SEVERITIES, "severity")
        alert_type = parse_optional_text(arguments.get("alert_type"), "alert_type")
        machine_code = None
        if arguments.get("machine_code") is not None:
            machine_code = parse_machine_code(arguments.get("machine_code"))
        return ToolResult(
            "list_alerts",
            "Operational alerts",
            self._repository.list_alerts(
                limit=limit,
                offset=0,
                status=status,
                severity=severity,
                alert_type=alert_type,
                machine_code=machine_code,
            ),
        )

    def _get_prediction_explanation(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        event_id = parse_optional_text(arguments.get("event_id"), "event_id")
        if event_id is None:
            raise CopilotToolError("event_id is required.")
        return ToolResult(
            "get_prediction_explanation",
            f"{machine_code} prediction explanation",
            compact_explanation(
                self._repository.get_prediction_explanation(machine_code, event_id)
            ),
        )

    def _get_latest_prediction_explanation(self, arguments: Mapping[str, Any]) -> ToolResult:
        machine_code = parse_machine_code(arguments.get("machine_code"))
        predictions = self._repository.list_machine_predictions(machine_code, limit=1, offset=0)
        items = predictions.get("items", [])
        if not items:
            raise CopilotToolError(f"No persisted predictions are available for {machine_code}.")
        event_id = items[0].get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise CopilotToolError(f"The latest prediction for {machine_code} has no event_id.")
        explanation = self._repository.get_prediction_explanation(machine_code, event_id)
        return ToolResult(
            "get_latest_prediction_explanation",
            f"{machine_code} latest prediction explanation",
            {
                "latest_prediction": compact_prediction(items[0]),
                "explanation": compact_explanation(explanation),
            },
        )


def ollama_tools(names: set[str] | None = None) -> list[dict[str, Any]]:
    selected = (
        TOOL_DEFINITIONS
        if names is None
        else tuple(tool for tool in TOOL_DEFINITIONS if tool.name in names)
    )
    return [tool.to_ollama_tool() for tool in selected]


def select_tool_names(message: str) -> set[str]:
    normalized = message.lower()
    if contains_mutation_intent(normalized):
        return set()
    machine_code = extract_machine_code(message)
    names: set[str] = set()

    if any(term in normalized for term in ("fleet", "overview", "summary")) and not machine_code:
        names.add("get_fleet_overview")
    if "list" in normalized and "machine" in normalized and not machine_code:
        names.add("list_machines")
    if any(term in normalized for term in ("drift", "psi", "distribution shift")):
        names.add("get_latest_drift")
    if any(term in normalized for term in ("alert", "alerts")):
        names.add("list_alerts")

    if machine_code:
        if any(term in normalized for term in ("why", "explain", "explanation", "shap")) and any(
            term in normalized for term in ("prediction", "output", "result", "risk", "failure")
        ):
            names.add("get_latest_prediction_explanation")
        elif any(term in normalized for term in ("prediction", "probability", "failure risk")):
            names.add("get_machine_predictions")
        elif any(term in normalized for term in ("anomaly", "vibration", "pressure", "detector")):
            names.add("get_machine_anomalies")
        elif any(term in normalized for term in ("alert", "alerts")):
            names.add("list_alerts")
        else:
            names.add("get_machine_snapshot")

    return names & set(TOOL_CATALOG)


def contains_mutation_intent(normalized_message: str) -> bool:
    mutation_terms = {
        "acknowledge",
        "create",
        "delete",
        "drop",
        "insert",
        "modify",
        "remove",
        "resolve",
        "retrain",
        "run sql",
        "shell",
        "truncate",
        "update",
        "write",
    }
    return any(term in normalized_message for term in mutation_terms)


def extract_machine_code(message: str) -> str | None:
    match = MACHINE_CODE_SEARCH_PATTERN.search(message)
    return match.group(0).upper() if match else None


def require_no_arguments(arguments: Mapping[str, Any], tool_name: str) -> None:
    if arguments:
        raise CopilotToolError(f"{tool_name} does not accept arguments.")


def parse_machine_code(value: Any) -> str:
    if not isinstance(value, str) or not MACHINE_CODE_PATTERN.fullmatch(value.strip()):
        raise CopilotToolError("machine_code must match MCH-0000 format.")
    return value.strip()


def parse_limit(value: Any) -> int:
    if isinstance(value, bool):
        raise CopilotToolError("limit must be an integer.")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise CopilotToolError("limit must be an integer.") from exc
    if limit < 1 or limit > MAX_TOOL_LIMIT:
        raise CopilotToolError(f"limit must be between 1 and {MAX_TOOL_LIMIT}.")
    return limit


def parse_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CopilotToolError(f"{field_name} must be boolean.")
    return value


def parse_optional_choice(value: Any, allowed: set[str], field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise CopilotToolError(f"{field_name} has an unsupported value.")
    return value


def parse_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CopilotToolError(f"{field_name} must be non-empty text.")
    if len(value) > 120:
        raise CopilotToolError(f"{field_name} is too long.")
    return value.strip()


def compact_drift_result(drift: Mapping[str, Any]) -> dict[str, Any]:
    features_by_scope = drift.get("features_by_scope")
    compact_features: dict[str, list[dict[str, Any]]] = {}
    if isinstance(features_by_scope, Mapping):
        for scope, raw_features in features_by_scope.items():
            if not isinstance(raw_features, list):
                continue
            features = [
                {
                    "feature_name": feature.get("feature_name"),
                    "psi": feature.get("psi"),
                    "status": feature.get("status"),
                    "feature_type": feature.get("feature_type"),
                    "reference_count": feature.get("reference_count"),
                    "current_count": feature.get("current_count"),
                }
                for feature in raw_features
                if isinstance(feature, Mapping)
            ]
            features.sort(
                key=lambda item: (
                    -(float(item.get("psi") or 0.0)),
                    str(item.get("feature_name")),
                )
            )
            compact_features[str(scope)] = features[:5]
    return {
        "drift_snapshot_id": drift.get("drift_snapshot_id"),
        "monitor_version": drift.get("monitor_version"),
        "ai4i_overall_status": drift.get("ai4i_overall_status"),
        "anomaly_overall_status": drift.get("anomaly_overall_status"),
        "ai4i_current_count": drift.get("ai4i_current_count"),
        "anomaly_current_count": drift.get("anomaly_current_count"),
        "features_by_scope": compact_features,
    }


def compact_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_prediction_id": prediction.get("model_prediction_id"),
        "event_id": prediction.get("event_id"),
        "event_time": prediction.get("event_time"),
        "failure_probability": prediction.get("failure_probability"),
        "failure_prediction": prediction.get("failure_prediction"),
        "decision_semantics": prediction.get("decision_semantics"),
        "frozen_threshold": prediction.get("frozen_threshold"),
        "model_name": prediction.get("model_name"),
        "model_version": prediction.get("model_version"),
        "final_config_hash": prediction.get("final_config_hash"),
    }


def compact_anomaly(anomaly: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anomaly_id": anomaly.get("anomaly_id"),
        "event_id": anomaly.get("event_id"),
        "event_time": anomaly.get("event_time"),
        "vibration_mm_s": anomaly.get("vibration_mm_s"),
        "pressure_bar": anomaly.get("pressure_bar"),
        "anomaly_score": anomaly.get("anomaly_score"),
        "anomaly_flag": anomaly.get("anomaly_flag"),
        "score_semantics": anomaly.get("score_semantics"),
    }


def compact_alert(alert: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alert_id": alert.get("alert_id"),
        "machine_code": alert.get("machine_code"),
        "severity": alert.get("severity"),
        "alert_type": alert.get("alert_type"),
        "title": alert.get("title"),
        "status": alert.get("status"),
        "source_kind": alert.get("source_kind"),
        "source_event_id": alert.get("source_event_id"),
        "source_observed_at": alert.get("source_observed_at"),
    }


def compact_explanation(explanation: Mapping[str, Any]) -> dict[str, Any]:
    contributions = explanation.get("feature_contributions")
    compact_contributions = contributions if isinstance(contributions, list) else []
    return {
        "prediction_explanation_id": explanation.get("prediction_explanation_id"),
        "model_prediction_id": explanation.get("model_prediction_id"),
        "event_id": explanation.get("event_id"),
        "machine_code": explanation.get("machine_code"),
        "event_time": explanation.get("event_time"),
        "failure_probability": explanation.get("failure_probability"),
        "failure_prediction": explanation.get("failure_prediction"),
        "decision_semantics": explanation.get("decision_semantics"),
        "frozen_threshold": explanation.get("frozen_threshold"),
        "model_name": explanation.get("model_name"),
        "model_version": explanation.get("model_version"),
        "output_semantics": explanation.get("output_semantics"),
        "attribution_semantics": explanation.get("attribution_semantics"),
        "base_value": explanation.get("base_value"),
        "model_output_value": explanation.get("model_output_value"),
        "contribution_sum": explanation.get("contribution_sum"),
        "additivity_error": explanation.get("additivity_error"),
        "feature_contributions": compact_contributions[:8],
    }

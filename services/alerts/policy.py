"""Pure deterministic alert policy over already-persisted operational state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MODEL_FAILURE_RISK_ALERT_TYPE = "model_failure_risk"
TELEMETRY_ANOMALY_ALERT_TYPE = "telemetry_anomaly"
ALERT_SEVERITY = "warning"
ALERT_STATUS = "open"
DRIFT_ALERTS_SUPPORTED = False

AlertSourceKind = Literal["model_prediction", "anomaly"]


@dataclass(frozen=True)
class AlertCandidate:
    """A deterministic alert candidate derived from one persisted source row."""

    machine_id: int
    machine_code: str
    alert_type: str
    severity: str
    title: str
    description: str
    model_prediction_id: int | None = None
    anomaly_id: int | None = None
    status: str = ALERT_STATUS

    @property
    def source_kind(self) -> AlertSourceKind:
        if self.model_prediction_id is not None:
            return "model_prediction"
        return "anomaly"

    def identity_key(self) -> tuple[str, int]:
        if self.model_prediction_id is not None:
            return (self.alert_type, self.model_prediction_id)
        if self.anomaly_id is not None:
            return (self.alert_type, self.anomaly_id)
        raise ValueError("Alert candidate must reference a source row.")

    def immutable_values(self) -> tuple[Any, ...]:
        return (
            self.machine_id,
            self.model_prediction_id,
            self.anomaly_id,
            self.severity,
            self.alert_type,
            self.title,
            self.description,
        )


@dataclass(frozen=True)
class DriftAlertDecision:
    """Documented drift alert support decision for the current schema."""

    supported: bool
    eligible_alerts: int
    reason: str


def as_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else value in {"t", "true", "1", 1}


def derive_model_failure_risk_alert(row: Mapping[str, Any]) -> AlertCandidate | None:
    """Derive an alert only from a positive persisted AI4I model decision."""
    if not as_bool(row.get("failure_prediction")):
        return None
    machine_code = str(row["machine_code"])
    return AlertCandidate(
        machine_id=int(row["machine_id"]),
        machine_code=machine_code,
        model_prediction_id=int(row["model_prediction_id"]),
        anomaly_id=None,
        alert_type=MODEL_FAILURE_RISK_ALERT_TYPE,
        severity=ALERT_SEVERITY,
        title=f"Model-estimated failure risk for {machine_code}",
        description=(
            "The frozen AI4I classifier produced a positive persisted model decision for this "
            "telemetry event. This alert is a model-estimated risk signal for review, not direct "
            "evidence of a breakdown."
        ),
    )


def derive_telemetry_anomaly_alert(row: Mapping[str, Any]) -> AlertCandidate | None:
    """Derive an alert only from a persisted flagged telemetry anomaly row."""
    if not as_bool(row.get("anomaly_flag")):
        return None
    machine_code = str(row["machine_code"])
    return AlertCandidate(
        machine_id=int(row["machine_id"]),
        machine_code=machine_code,
        model_prediction_id=None,
        anomaly_id=int(row["anomaly_id"]),
        alert_type=TELEMETRY_ANOMALY_ALERT_TYPE,
        severity=ALERT_SEVERITY,
        title=f"Telemetry anomaly flagged for {machine_code}",
        description=(
            "The operational anomaly detector flagged a persisted vibration/pressure telemetry "
            "event as statistically unusual. The anomaly score is not a probability and this "
            "alert does not label the machine as failed."
        ),
    )


def derive_alert_candidates(
    prediction_rows: Sequence[Mapping[str, Any]],
    anomaly_rows: Sequence[Mapping[str, Any]],
) -> list[AlertCandidate]:
    """Return deterministic alert candidates without executing models or combining scores."""
    candidates: list[AlertCandidate] = []
    for row in prediction_rows:
        candidate = derive_model_failure_risk_alert(row)
        if candidate is not None:
            candidates.append(candidate)
    for row in anomaly_rows:
        candidate = derive_telemetry_anomaly_alert(row)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def drift_alert_decision(*, alerts_machine_id_required: bool) -> DriftAlertDecision:
    """Return the current drift alert materialization decision."""
    if alerts_machine_id_required:
        return DriftAlertDecision(
            supported=DRIFT_ALERTS_SUPPORTED,
            eligible_alerts=0,
            reason=(
                "The current alerts table requires machine_id, while drift monitoring is "
                "population-level state. Drift is exposed through /api/v1/drift/latest instead."
            ),
        )
    return DriftAlertDecision(
        supported=True,
        eligible_alerts=0,
        reason="Fleet-level drift alert materialization is not enabled in this local phase.",
    )


def alert_matches_candidate(existing: Mapping[str, Any], candidate: AlertCandidate) -> bool:
    """Compare immutable alert provenance fields while ignoring operational lifecycle fields."""
    existing_values = (
        int(existing["machine_id"]),
        existing.get("model_prediction_id"),
        existing.get("anomaly_id"),
        existing["severity"],
        existing["alert_type"],
        existing["title"],
        existing.get("description"),
    )
    normalized_existing = tuple(
        int(value) if index in {1, 2} and value is not None else value
        for index, value in enumerate(existing_values)
    )
    return normalized_existing == candidate.immutable_values()


def build_static_summary() -> dict[str, Any]:
    return {
        "eligible_source_types": {
            MODEL_FAILURE_RISK_ALERT_TYPE: (
                "Persisted AI4I prediction with failure_prediction=true."
            ),
            TELEMETRY_ANOMALY_ALERT_TYPE: "Persisted telemetry anomaly with anomaly_flag=true.",
        },
        "stable_identity_strategy": {
            MODEL_FAILURE_RISK_ALERT_TYPE: ["alert_type", "model_prediction_id"],
            TELEMETRY_ANOMALY_ALERT_TYPE: ["alert_type", "anomaly_id"],
        },
        "severity_mapping": {
            MODEL_FAILURE_RISK_ALERT_TYPE: ALERT_SEVERITY,
            TELEMETRY_ANOMALY_ALERT_TYPE: ALERT_SEVERITY,
        },
        "idempotency_policy": "Repeated execution reuses existing identical source-derived alerts.",
        "automatic_resolution_policy": "Alerts are not automatically resolved in this phase.",
        "drift_fleet_alert_decision": drift_alert_decision(alerts_machine_id_required=True).reason,
        "alert_semantics": "An alert is an operational interpretation, not a confirmed failure.",
        "score_policy": "AI4I and anomaly scores are never combined by this alert policy.",
    }

from __future__ import annotations

from services.alerts import policy


def prediction_row(*, positive: bool = True) -> dict[str, object]:
    return {
        "machine_id": 1,
        "machine_code": "MCH-0001",
        "model_prediction_id": 10,
        "failure_prediction": positive,
    }


def anomaly_row(*, flagged: bool = True) -> dict[str, object]:
    return {
        "machine_id": 1,
        "machine_code": "MCH-0001",
        "anomaly_id": 20,
        "anomaly_flag": flagged,
    }


def test_positive_ai4i_decision_creates_model_risk_alert() -> None:
    candidate = policy.derive_model_failure_risk_alert(prediction_row(positive=True))

    assert candidate is not None
    assert candidate.alert_type == policy.MODEL_FAILURE_RISK_ALERT_TYPE
    assert candidate.model_prediction_id == 10
    assert "model-estimated risk" in candidate.description


def test_negative_ai4i_decision_creates_no_model_risk_alert() -> None:
    assert policy.derive_model_failure_risk_alert(prediction_row(positive=False)) is None


def test_flagged_anomaly_creates_telemetry_anomaly_alert() -> None:
    candidate = policy.derive_telemetry_anomaly_alert(anomaly_row(flagged=True))

    assert candidate is not None
    assert candidate.alert_type == policy.TELEMETRY_ANOMALY_ALERT_TYPE
    assert candidate.anomaly_id == 20
    assert "not a probability" in candidate.description


def test_non_flagged_anomaly_creates_no_alert() -> None:
    assert policy.derive_telemetry_anomaly_alert(anomaly_row(flagged=False)) is None


def test_same_source_identity_is_stable() -> None:
    left = policy.derive_telemetry_anomaly_alert(anomaly_row(flagged=True))
    right = policy.derive_telemetry_anomaly_alert(anomaly_row(flagged=True))

    assert left is not None
    assert right is not None
    assert left.identity_key() == right.identity_key()


def test_severity_policy_uses_existing_warning_convention() -> None:
    candidates = policy.derive_alert_candidates(
        [prediction_row(positive=True)],
        [anomaly_row(flagged=True)],
    )

    assert {candidate.severity for candidate in candidates} == {"warning"}


def test_policy_does_not_create_combined_score() -> None:
    candidate = policy.derive_telemetry_anomaly_alert(anomaly_row(flagged=True))

    assert candidate is not None
    assert "combined" not in candidate.__dict__
    assert "score" not in candidate.__dict__


def test_alert_wording_does_not_claim_observed_or_confirmed_failure() -> None:
    candidates = policy.derive_alert_candidates(
        [prediction_row(positive=True)],
        [anomaly_row(flagged=True)],
    )
    text = " ".join(candidate.title + " " + candidate.description for candidate in candidates)
    lowered = text.lower()

    assert "machine failure detected" not in lowered
    assert "confirmed failure" not in lowered
    assert "detected failure" not in lowered


def test_drift_alert_decision_matches_machine_scoped_schema() -> None:
    decision = policy.drift_alert_decision(alerts_machine_id_required=True)

    assert decision.supported is False
    assert decision.eligible_alerts == 0
    assert "/api/v1/drift/latest" in decision.reason

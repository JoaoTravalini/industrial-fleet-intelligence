from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from apps.api.config import ApiSettings
from apps.api.db import DatabaseUnavailableError
from apps.api.dependencies import get_repository
from apps.api.main import create_app
from apps.api.repositories.platform import (
    AlertNotFoundError,
    MachineNotFoundError,
    PredictionExplanationNotFoundError,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = "092620c6-e580-579b-b806-484e4ee1d86f"


class FakeRepository:
    def health_check(self) -> bool:
        return True

    def list_machines(self, *, limit: int, offset: int, status: str | None) -> dict[str, Any]:
        item = {
            "machine_code": "MCH-0001",
            "machine_type": "excavator",
            "model_family": "EX-Series",
            "commissioned_on": date(2023, 1, 1),
            "operational_status": status or "active",
            "latest_prediction": {
                "event_time": NOW,
                "failure_probability": 0.1,
                "failure_prediction": False,
                "frozen_threshold": 0.14,
                "model_name": "ai4i-failure-risk-random-forest",
                "model_version": "1.0.0",
                "final_config_hash": "a" * 64,
            },
        }
        return {"items": [item], "limit": limit, "offset": offset, "count": 1, "total": 1}

    def get_machine(self, machine_code: str) -> dict[str, Any]:
        if machine_code != "MCH-0001":
            raise MachineNotFoundError(machine_code)
        return {
            **self.list_machines(limit=1, offset=0, status=None)["items"][0],
            "prediction_history_count": 2,
            "anomaly_audit_count": 3,
        }

    def list_machine_predictions(
        self,
        machine_code: str,
        *,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        if machine_code != "MCH-0001":
            raise MachineNotFoundError(machine_code)
        item = {
            "model_prediction_id": 10,
            "event_id": EVENT_ID,
            "event_time": NOW,
            "failure_probability": 0.1,
            "failure_prediction": False,
            "decision_semantics": "model_decision_not_observed_failure",
            "frozen_threshold": 0.14,
            "model_name": "ai4i-failure-risk-random-forest",
            "model_version": "1.0.0",
            "final_config_hash": "a" * 64,
            "adapter_version": "1.0.0",
            "model_input_sha256": "b" * 64,
            "lineage": lineage(),
        }
        return {
            "machine_code": machine_code,
            "items": [item],
            "limit": limit,
            "offset": offset,
            "count": 1,
            "total": 1,
        }

    def list_machine_anomalies(
        self,
        machine_code: str,
        *,
        limit: int,
        offset: int,
        flagged_only: bool,
    ) -> dict[str, Any]:
        if machine_code != "MCH-0001":
            raise MachineNotFoundError(machine_code)
        item = {
            "anomaly_id": 20,
            "event_id": EVENT_ID,
            "event_time": NOW,
            "vibration_mm_s": 2.1,
            "pressure_bar": 150.0,
            "anomaly_score": 0.42,
            "anomaly_flag": True,
            "score_semantics": "anomaly_score_not_probability",
            "model_name": "telemetry-isolation-forest",
            "model_version": "1.0.0",
            "model_config_hash": "c" * 64,
            "baseline_event_id_sha256": "d" * 64,
            "baseline_feature_data_sha256": "e" * 64,
            "lineage": lineage(),
        }
        return {
            "machine_code": machine_code,
            "flagged_only": flagged_only,
            "items": [item],
            "limit": limit,
            "offset": offset,
            "count": 1,
            "total": 1,
        }

    def get_prediction_explanation(self, machine_code: str, event_id: str) -> dict[str, Any]:
        if machine_code != "MCH-0001":
            raise MachineNotFoundError(machine_code)
        if event_id == "00000000-0000-4000-8000-000000000404":
            raise PredictionExplanationNotFoundError(event_id)
        if event_id != EVENT_ID:
            raise PredictionExplanationNotFoundError(event_id)
        return {
            "prediction_explanation_id": 30,
            "model_prediction_id": 10,
            "event_id": EVENT_ID,
            "machine_code": "MCH-0001",
            "event_time": NOW,
            "failure_probability": 0.1,
            "failure_prediction": False,
            "decision_semantics": "model_decision_not_observed_failure",
            "frozen_threshold": 0.14,
            "model_name": "ai4i-failure-risk-random-forest",
            "model_version": "1.0.0",
            "final_config_hash": "a" * 64,
            "model_input_sha256": "b" * 64,
            "explainer_name": "shap.TreeExplainer",
            "explainer_version": "0.52.0",
            "explanation_config_hash": "f" * 64,
            "output_semantics": "positive_class_failure_risk_model_output",
            "attribution_semantics": "shap_model_attribution_not_causality",
            "positive_contribution_semantics": (
                "positive_shap_pushes_model_output_toward_higher_failure_risk"
            ),
            "negative_contribution_semantics": (
                "negative_shap_pushes_model_output_toward_lower_failure_risk"
            ),
            "base_value": 0.12,
            "model_output_value": 0.1,
            "contribution_sum": -0.02,
            "additivity_error": 0.0,
            "feature_contributions": [
                {"feature_name": "Type", "feature_value": "L", "shap_value": 0.003},
                {
                    "feature_name": "Air temperature [K]",
                    "feature_value": 300.1,
                    "shap_value": -0.004,
                },
                {
                    "feature_name": "Process temperature [K]",
                    "feature_value": 309.2,
                    "shap_value": -0.002,
                },
                {
                    "feature_name": "Rotational speed [rpm]",
                    "feature_value": 1450.0,
                    "shap_value": 0.006,
                },
                {"feature_name": "Torque [Nm]", "feature_value": 42.0, "shap_value": -0.018},
                {
                    "feature_name": "Tool wear [min]",
                    "feature_value": 20.0,
                    "shap_value": -0.005,
                },
            ],
            "lineage": lineage(),
        }

    def fleet_overview(self) -> dict[str, Any]:
        return {
            "machine_count": 100,
            "machines_with_prediction_projection": 10,
            "prediction_history_count": 106,
            "positive_prediction_count": 0,
            "negative_prediction_count": 106,
            "mean_failure_probability": 0.01,
            "max_failure_probability": 0.1,
            "anomaly_audit_count": 106,
            "flagged_anomaly_count": 70,
            "non_flagged_anomaly_count": 36,
            "latest_ai4i_drift_status": "drift",
            "latest_anomaly_drift_status": "stable",
            "open_alert_count": 70,
        }

    def latest_drift(self) -> dict[str, Any]:
        metric = {
            "feature_name": "Type",
            "feature_type": "categorical",
            "psi": 0.2,
            "status": "watch",
            "reference_count": 100,
            "current_count": 100,
            "reference_mean": None,
            "current_mean": None,
            "reference_std": None,
            "current_std": None,
            "reference_min": None,
            "reference_max": None,
            "current_min": None,
            "current_max": None,
            "standardized_mean_shift": None,
            "outside_reference_range_count": None,
            "outside_reference_range_rate": None,
            "reference_proportions": {"L": 0.5},
            "current_proportions": {"L": 0.4},
            "bin_edges": None,
            "diagnostics": {"categories": ["L"]},
        }
        return {
            "drift_snapshot_id": 1,
            "monitor_version": "1.0.0",
            "reference_profile_sha256": "f" * 64,
            "ai4i_reference_identity": {"source": "ai4i"},
            "anomaly_reference_identity": {"source": "anomaly"},
            "ai4i_current_data_hash": "1" * 64,
            "anomaly_current_data_hash": "2" * 64,
            "ai4i_overall_status": "drift",
            "anomaly_overall_status": "stable",
            "ai4i_current_count": 106,
            "anomaly_current_count": 106,
            "created_at": NOW,
            "features_by_scope": {
                "ai4i_model_input": [metric],
                "operational_anomaly_inputs": [],
            },
        }

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
        item = alert_item()
        return {"items": [item], "limit": limit, "offset": offset, "count": 1, "total": 1}

    def get_alert(self, alert_id: int) -> dict[str, Any]:
        if alert_id != 1:
            raise AlertNotFoundError(str(alert_id))
        return alert_item()


class UnavailableRepository(FakeRepository):
    def health_check(self) -> bool:
        raise DatabaseUnavailableError("down")


class MissingExplanationRepository(FakeRepository):
    def get_prediction_explanation(self, machine_code: str, event_id: str) -> dict[str, Any]:
        if machine_code != "MCH-0001":
            raise MachineNotFoundError(machine_code)
        raise PredictionExplanationNotFoundError(event_id)


def lineage() -> dict[str, Any]:
    return {
        "source_kafka_topic": "industrial.telemetry.v1",
        "source_kafka_partition": 0,
        "source_kafka_offset": 1,
        "source_kafka_timestamp": NOW,
        "source_kafka_key": "MCH-0001",
        "payload_sha256": "9" * 64,
    }


def alert_item() -> dict[str, Any]:
    return {
        "alert_id": 1,
        "machine_code": "MCH-0001",
        "severity": "warning",
        "alert_type": "telemetry_anomaly",
        "title": "Telemetry anomaly flagged for MCH-0001",
        "description": "Persisted anomaly detector output for review.",
        "status": "open",
        "source_kind": "anomaly",
        "model_prediction_id": None,
        "anomaly_id": 20,
        "source_event_id": EVENT_ID,
        "source_observed_at": NOW,
        "created_at": NOW,
    }


def make_client(repository: Any) -> TestClient:
    settings = ApiSettings(
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_db="industrial_fleet_dev",
        postgres_user="industrial_fleet_dev",
        postgres_password="placeholder",
        api_host="127.0.0.1",
        api_port=8000,
        cors_origins=("http://localhost:5173",),
    )
    app = create_app(settings)
    app.dependency_overrides[get_repository] = lambda: repository
    return TestClient(app)


def test_health_response() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}


def test_fleet_response_schema() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get("/api/v1/fleet/overview")
    payload = response.json()
    assert response.status_code == 200
    assert payload["machine_count"] == 100
    assert payload["latest_ai4i_drift_status"] == "drift"


def test_machine_list_and_detail() -> None:
    with make_client(FakeRepository()) as client:
        list_response = client.get("/api/v1/machines", params={"status": "active"})
        detail_response = client.get("/api/v1/machines/MCH-0001")
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["machine_code"] == "MCH-0001"
    assert detail_response.status_code == 200
    assert detail_response.json()["prediction_history_count"] == 2


def test_unknown_machine_returns_404() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get("/api/v1/machines/MCH-9999")
    assert response.status_code == 404


def test_prediction_anomaly_drift_and_alert_schemas() -> None:
    with make_client(FakeRepository()) as client:
        predictions = client.get("/api/v1/machines/MCH-0001/predictions")
        anomalies = client.get("/api/v1/machines/MCH-0001/anomalies", params={"flagged_only": True})
        drift = client.get("/api/v1/drift/latest")
        alerts = client.get("/api/v1/alerts")
        alert_detail = client.get("/api/v1/alerts/1")
    assert predictions.status_code == 200
    decision_semantics = predictions.json()["items"][0]["decision_semantics"]
    assert decision_semantics == "model_decision_not_observed_failure"
    assert anomalies.status_code == 200
    assert anomalies.json()["items"][0]["score_semantics"] == "anomaly_score_not_probability"
    assert drift.status_code == 200
    assert "ai4i_model_input" in drift.json()["features_by_scope"]
    assert alerts.status_code == 200
    assert alert_detail.status_code == 200
    assert alert_detail.json()["source_kind"] == "anomaly"


def test_prediction_explanation_endpoint_success_schema() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get(f"/api/v1/machines/MCH-0001/predictions/{EVENT_ID}/explanation")
    payload = response.json()
    feature_names = [item["feature_name"] for item in payload["feature_contributions"]]

    assert response.status_code == 200
    assert payload["event_id"] == EVENT_ID
    assert payload["decision_semantics"] == "model_decision_not_observed_failure"
    assert payload["output_semantics"] == "positive_class_failure_risk_model_output"
    assert payload["attribution_semantics"] == "shap_model_attribution_not_causality"
    assert feature_names == [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]


def test_prediction_explanation_unknown_prediction_returns_404() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get(
            "/api/v1/machines/MCH-0001/predictions/00000000-0000-4000-8000-000000000404/explanation"
        )
    assert response.status_code == 404


def test_prediction_explanation_missing_materialization_returns_404() -> None:
    with make_client(MissingExplanationRepository()) as client:
        response = client.get(f"/api/v1/machines/MCH-0001/predictions/{EVENT_ID}/explanation")
    assert response.status_code == 404


def test_unknown_alert_returns_404() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get("/api/v1/alerts/999")
    assert response.status_code == 404


def test_pagination_boundaries() -> None:
    with make_client(FakeRepository()) as client:
        assert client.get("/api/v1/machines", params={"limit": 0}).status_code == 422
        assert client.get("/api/v1/machines", params={"limit": 201}).status_code == 422
        assert client.get("/api/v1/machines", params={"offset": -1}).status_code == 422


def test_cors_configuration_policy() -> None:
    with make_client(FakeRepository()) as client:
        response = client.options(
            "/api/v1/machines",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_database_unavailable_health_response() -> None:
    with make_client(UnavailableRepository()) as client:
        response = client.get("/health")
    assert response.status_code == 503


def test_openapi_generation() -> None:
    with make_client(FakeRepository()) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Industrial Fleet Intelligence API"


def test_source_guards() -> None:
    files = list((ROOT / "apps" / "api").rglob("*.py"))
    files += list((ROOT / "services" / "alerts").rglob("*.py"))
    files.append(ROOT / "scripts" / "materialize_operational_alerts.py")
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    forbidden = (
        ".predict(",
        "predict_proba(",
        ".fit(",
        "joblib.load",
        "shap.",
        "SparkSession",
        "KafkaConsumer",
        "confluent_kafka",
        "read_parquet",
        "latest_drift_report",
        "telemetry_predictions.jsonl",
        "telemetry_anomalies.jsonl",
        "INSERT INTO model_predictions",
        "UPDATE machine_health",
        "INSERT INTO anomalies",
        "INSERT INTO drift_feature_metrics",
    )
    assert not any(pattern in source for pattern in forbidden)

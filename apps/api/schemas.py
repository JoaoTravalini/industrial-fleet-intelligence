"""Explicit response schemas for the read-only operational API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["connected"]


class ErrorResponse(BaseModel):
    detail: str


class LatestPredictionProjection(BaseModel):
    event_time: datetime | None = None
    failure_probability: float | None = None
    failure_prediction: bool | None = Field(
        default=None,
        description="Model decision from the frozen AI4I classifier, not an observed failure.",
    )
    frozen_threshold: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    final_config_hash: str | None = None


class MachineSummary(BaseModel):
    machine_code: str
    machine_type: str
    model_family: str
    commissioned_on: date | None = None
    operational_status: str
    latest_prediction: LatestPredictionProjection | None = None


class MachineListResponse(BaseModel):
    items: list[MachineSummary]
    limit: int
    offset: int
    count: int
    total: int


class MachineDetailResponse(MachineSummary):
    prediction_history_count: int
    anomaly_audit_count: int


class SourceLineage(BaseModel):
    source_kafka_topic: str | None = None
    source_kafka_partition: int | None = None
    source_kafka_offset: int | None = None
    source_kafka_timestamp: datetime | None = None
    source_kafka_key: str | None = None
    payload_sha256: str | None = None


class PredictionResponse(BaseModel):
    model_prediction_id: int
    event_id: str | None = None
    event_time: datetime | None = None
    failure_probability: float | None = None
    failure_prediction: bool | None = Field(
        default=None,
        description="Model decision from the frozen AI4I classifier, not an observed failure.",
    )
    decision_semantics: Literal["model_decision_not_observed_failure"]
    frozen_threshold: float | None = None
    model_name: str
    model_version: str
    final_config_hash: str | None = None
    adapter_version: str | None = None
    model_input_sha256: str | None = None
    lineage: SourceLineage


class PredictionListResponse(BaseModel):
    machine_code: str
    items: list[PredictionResponse]
    limit: int
    offset: int
    count: int
    total: int


class AnomalyResponse(BaseModel):
    anomaly_id: int
    event_id: str | None = None
    event_time: datetime | None = None
    vibration_mm_s: float | None = None
    pressure_bar: float | None = None
    anomaly_score: float
    anomaly_flag: bool | None = None
    score_semantics: Literal["anomaly_score_not_probability"]
    model_name: str | None = None
    model_version: str | None = None
    model_config_hash: str | None = None
    baseline_event_id_sha256: str | None = None
    baseline_feature_data_sha256: str | None = None
    lineage: SourceLineage


class AnomalyListResponse(BaseModel):
    machine_code: str
    flagged_only: bool
    items: list[AnomalyResponse]
    limit: int
    offset: int
    count: int
    total: int


class FleetOverviewResponse(BaseModel):
    machine_count: int
    machines_with_prediction_projection: int
    prediction_history_count: int
    positive_prediction_count: int
    negative_prediction_count: int
    mean_failure_probability: float | None = None
    max_failure_probability: float | None = None
    anomaly_audit_count: int
    flagged_anomaly_count: int
    non_flagged_anomaly_count: int
    latest_ai4i_drift_status: str | None = None
    latest_anomaly_drift_status: str | None = None
    open_alert_count: int


class DriftFeatureMetric(BaseModel):
    feature_name: str
    feature_type: str
    psi: float
    status: str
    reference_count: int
    current_count: int
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None
    reference_min: float | None = None
    reference_max: float | None = None
    current_min: float | None = None
    current_max: float | None = None
    standardized_mean_shift: float | None = None
    outside_reference_range_count: int | None = None
    outside_reference_range_rate: float | None = None
    reference_proportions: Any
    current_proportions: Any
    bin_edges: Any | None = None
    diagnostics: dict[str, Any]


class DriftLatestResponse(BaseModel):
    drift_snapshot_id: int | None = None
    monitor_version: str | None = None
    reference_profile_sha256: str | None = None
    ai4i_reference_identity: dict[str, Any] | None = None
    anomaly_reference_identity: dict[str, Any] | None = None
    ai4i_current_data_hash: str | None = None
    anomaly_current_data_hash: str | None = None
    ai4i_overall_status: str | None = None
    anomaly_overall_status: str | None = None
    ai4i_current_count: int | None = None
    anomaly_current_count: int | None = None
    created_at: datetime | None = None
    features_by_scope: dict[str, list[DriftFeatureMetric]]


class AlertResponse(BaseModel):
    alert_id: int
    machine_code: str
    severity: str
    alert_type: str
    title: str
    description: str | None = None
    status: str
    source_kind: Literal["model_prediction", "anomaly", "unknown"]
    model_prediction_id: int | None = None
    anomaly_id: int | None = None
    source_event_id: str | None = None
    source_observed_at: datetime | None = None
    created_at: datetime


class AlertListResponse(BaseModel):
    items: list[AlertResponse]
    limit: int
    offset: int
    count: int
    total: int

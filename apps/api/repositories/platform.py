"""PostgreSQL data-access layer for materialized operational platform state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg

from apps.api.config import ApiSettings
from apps.api.db import DatabaseUnavailableError, open_connection

AI4I_PREDICTION_TYPE = "ai4i_failure_risk"
TELEMETRY_ANOMALY_TYPE = "telemetry_isolation_forest_score"
DRIFT_SCOPES = ("ai4i_model_input", "operational_anomaly_inputs")
EXPLANATION_OUTPUT_SEMANTICS = "positive_class_failure_risk_model_output"
EXPLANATION_ATTRIBUTION_SEMANTICS = "shap_model_attribution_not_causality"
POSITIVE_CONTRIBUTION_SEMANTICS = "positive_shap_pushes_model_output_toward_higher_failure_risk"
NEGATIVE_CONTRIBUTION_SEMANTICS = "negative_shap_pushes_model_output_toward_lower_failure_risk"


class MachineNotFoundError(LookupError):
    """Raised when a machine code does not exist."""


class AlertNotFoundError(LookupError):
    """Raised when an alert id does not exist."""


class PredictionExplanationNotFoundError(LookupError):
    """Raised when a prediction explanation is unavailable."""


def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_value(item) for item in value)
    return value


def normalize_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {str(key): normalize_value(value) for key, value in row.items()}


def normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_row(row) or {} for row in rows]


def latest_prediction_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("latest_model_name") is None:
        return None
    return {
        "event_time": row.get("latest_prediction_at"),
        "failure_probability": row.get("latest_failure_probability"),
        "failure_prediction": row.get("latest_failure_prediction"),
        "frozen_threshold": row.get("latest_frozen_threshold"),
        "model_name": row.get("latest_model_name"),
        "model_version": row.get("latest_model_version"),
        "final_config_hash": row.get("latest_final_config_hash"),
    }


def machine_summary_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "machine_code": row["machine_code"],
        "machine_type": row["machine_type"],
        "model_family": row["model_family"],
        "commissioned_on": row.get("commissioned_on"),
        "operational_status": row["operational_status"],
        "latest_prediction": latest_prediction_from_row(row),
    }


def source_lineage_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_kafka_topic": row.get("source_kafka_topic"),
        "source_kafka_partition": row.get("source_kafka_partition"),
        "source_kafka_offset": row.get("source_kafka_offset"),
        "source_kafka_timestamp": row.get("source_kafka_timestamp"),
        "source_kafka_key": row.get("source_kafka_key"),
        "payload_sha256": row.get("payload_sha256"),
    }


class PlatformRepository:
    """Small read-oriented DAL over PostgreSQL operational state."""

    def __init__(self, settings: ApiSettings) -> None:
        self._settings = settings

    def _fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        try:
            with open_connection(self._settings) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return normalize_row(cursor.fetchone())
        except (psycopg.Error, OSError) as exc:
            raise DatabaseUnavailableError("PostgreSQL query failed.") from exc

    def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        try:
            with open_connection(self._settings) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return normalize_rows(cursor.fetchall())
        except (psycopg.Error, OSError) as exc:
            raise DatabaseUnavailableError("PostgreSQL query failed.") from exc

    def health_check(self) -> bool:
        row = self._fetch_one("SELECT 1 AS ok;")
        return row is not None and row.get("ok") == 1

    def machine_exists(self, machine_code: str) -> int:
        row = self._fetch_one(
            """
            SELECT machine_id
            FROM machines
            WHERE machine_identifier = %s;
            """,
            (machine_code,),
        )
        if row is None:
            raise MachineNotFoundError(machine_code)
        return int(row["machine_id"])

    def list_machines(self, *, limit: int, offset: int, status: str | None) -> dict[str, Any]:
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("m.operational_status = %s")
            params.append(status)
        where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
        total_row = self._fetch_one(
            f"""
            SELECT count(*) AS total
            FROM machines m
            {where_sql};
            """,
            params,
        )
        rows = self._fetch_all(
            f"""
            SELECT
                m.machine_identifier AS machine_code,
                m.machine_type,
                m.model_family,
                m.commissioned_on,
                m.operational_status,
                mh.latest_prediction_at,
                mh.latest_failure_probability,
                mh.latest_failure_prediction,
                mh.latest_frozen_threshold,
                mh.latest_model_name,
                mh.latest_model_version,
                mh.latest_final_config_hash
            FROM machines m
            LEFT JOIN machine_health mh
              ON mh.machine_id = m.machine_id
            {where_sql}
            ORDER BY m.machine_identifier
            LIMIT %s OFFSET %s;
            """,
            (*params, limit, offset),
        )
        items = [machine_summary_from_row(row) for row in rows]
        return {
            "items": items,
            "limit": limit,
            "offset": offset,
            "count": len(items),
            "total": int(total_row["total"] if total_row else 0),
        }

    def get_machine(self, machine_code: str) -> dict[str, Any]:
        row = self._fetch_one(
            """
            SELECT
                m.machine_id,
                m.machine_identifier AS machine_code,
                m.machine_type,
                m.model_family,
                m.commissioned_on,
                m.operational_status,
                mh.latest_prediction_at,
                mh.latest_failure_probability,
                mh.latest_failure_prediction,
                mh.latest_frozen_threshold,
                mh.latest_model_name,
                mh.latest_model_version,
                mh.latest_final_config_hash,
                (
                    SELECT count(*)
                    FROM model_predictions p
                    WHERE p.machine_id = m.machine_id
                ) AS prediction_history_count,
                (
                    SELECT count(*)
                    FROM anomalies a
                    WHERE a.machine_id = m.machine_id
                ) AS anomaly_audit_count
            FROM machines m
            LEFT JOIN machine_health mh
              ON mh.machine_id = m.machine_id
            WHERE m.machine_identifier = %s;
            """,
            (machine_code,),
        )
        if row is None:
            raise MachineNotFoundError(machine_code)
        detail = machine_summary_from_row(row)
        detail["prediction_history_count"] = int(row["prediction_history_count"])
        detail["anomaly_audit_count"] = int(row["anomaly_audit_count"])
        return detail

    def list_machine_predictions(
        self,
        machine_code: str,
        *,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        machine_id = self.machine_exists(machine_code)
        total_row = self._fetch_one(
            """
            SELECT count(*) AS total
            FROM model_predictions
            WHERE machine_id = %s
              AND prediction_type = %s;
            """,
            (machine_id, AI4I_PREDICTION_TYPE),
        )
        rows = self._fetch_all(
            """
            SELECT
                model_prediction_id,
                event_id,
                event_time,
                failure_probability,
                failure_prediction,
                frozen_threshold,
                model_name,
                model_version,
                final_config_hash,
                adapter_version,
                model_input_sha256,
                source_kafka_topic,
                source_kafka_partition,
                source_kafka_offset,
                source_kafka_timestamp,
                source_kafka_key,
                payload_sha256
            FROM model_predictions
            WHERE machine_id = %s
              AND prediction_type = %s
            ORDER BY
                event_time DESC NULLS LAST,
                source_kafka_timestamp DESC NULLS LAST,
                source_kafka_topic DESC NULLS LAST,
                source_kafka_partition DESC NULLS LAST,
                source_kafka_offset DESC NULLS LAST,
                event_id DESC NULLS LAST
            LIMIT %s OFFSET %s;
            """,
            (machine_id, AI4I_PREDICTION_TYPE, limit, offset),
        )
        items = []
        for row in rows:
            items.append(
                {
                    "model_prediction_id": row["model_prediction_id"],
                    "event_id": row.get("event_id"),
                    "event_time": row.get("event_time"),
                    "failure_probability": row.get("failure_probability"),
                    "failure_prediction": row.get("failure_prediction"),
                    "decision_semantics": "model_decision_not_observed_failure",
                    "frozen_threshold": row.get("frozen_threshold"),
                    "model_name": row["model_name"],
                    "model_version": row["model_version"],
                    "final_config_hash": row.get("final_config_hash"),
                    "adapter_version": row.get("adapter_version"),
                    "model_input_sha256": row.get("model_input_sha256"),
                    "lineage": source_lineage_from_row(row),
                }
            )
        return {
            "machine_code": machine_code,
            "items": items,
            "limit": limit,
            "offset": offset,
            "count": len(items),
            "total": int(total_row["total"] if total_row else 0),
        }

    def get_prediction_explanation(self, machine_code: str, event_id: str) -> dict[str, Any]:
        machine_id = self.machine_exists(machine_code)
        try:
            event_uuid = str(UUID(event_id))
        except ValueError as exc:
            raise PredictionExplanationNotFoundError(event_id) from exc
        row = self._fetch_one(
            """
            SELECT
                e.prediction_explanation_id,
                e.model_prediction_id,
                e.event_id::text AS event_id,
                e.event_time,
                e.model_input_sha256,
                e.explainer_name,
                e.explainer_version,
                e.explanation_config_hash,
                e.output_semantics,
                e.attribution_semantics,
                e.base_value,
                e.model_output_value,
                e.contribution_sum,
                e.additivity_error,
                e.feature_contributions,
                m.machine_identifier AS machine_code,
                p.failure_probability,
                p.failure_prediction,
                p.frozen_threshold,
                p.model_name,
                p.model_version,
                p.final_config_hash,
                p.source_kafka_topic,
                p.source_kafka_partition,
                p.source_kafka_offset,
                p.source_kafka_timestamp,
                p.source_kafka_key,
                p.payload_sha256
            FROM prediction_explanations e
            JOIN model_predictions p
              ON p.model_prediction_id = e.model_prediction_id
            JOIN machines m
              ON m.machine_id = e.machine_id
            WHERE p.machine_id = %s
              AND p.prediction_type = %s
              AND p.event_id = %s::uuid
            ORDER BY e.created_at DESC, e.prediction_explanation_id DESC
            LIMIT 1;
            """,
            (machine_id, AI4I_PREDICTION_TYPE, event_uuid),
        )
        if row is None:
            raise PredictionExplanationNotFoundError(event_id)
        return {
            "prediction_explanation_id": row["prediction_explanation_id"],
            "model_prediction_id": row["model_prediction_id"],
            "event_id": row["event_id"],
            "machine_code": row["machine_code"],
            "event_time": row["event_time"],
            "failure_probability": row["failure_probability"],
            "failure_prediction": row["failure_prediction"],
            "decision_semantics": "model_decision_not_observed_failure",
            "frozen_threshold": row["frozen_threshold"],
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "final_config_hash": row["final_config_hash"],
            "model_input_sha256": row["model_input_sha256"],
            "explainer_name": row["explainer_name"],
            "explainer_version": row["explainer_version"],
            "explanation_config_hash": row["explanation_config_hash"],
            "output_semantics": row.get("output_semantics") or EXPLANATION_OUTPUT_SEMANTICS,
            "attribution_semantics": (
                row.get("attribution_semantics") or EXPLANATION_ATTRIBUTION_SEMANTICS
            ),
            "positive_contribution_semantics": POSITIVE_CONTRIBUTION_SEMANTICS,
            "negative_contribution_semantics": NEGATIVE_CONTRIBUTION_SEMANTICS,
            "base_value": row["base_value"],
            "model_output_value": row["model_output_value"],
            "contribution_sum": row["contribution_sum"],
            "additivity_error": row["additivity_error"],
            "feature_contributions": row["feature_contributions"],
            "lineage": source_lineage_from_row(row),
        }

    def list_machine_anomalies(
        self,
        machine_code: str,
        *,
        limit: int,
        offset: int,
        flagged_only: bool,
    ) -> dict[str, Any]:
        machine_id = self.machine_exists(machine_code)
        flag_condition = "AND anomaly_flag IS TRUE" if flagged_only else ""
        total_row = self._fetch_one(
            f"""
            SELECT count(*) AS total
            FROM anomalies
            WHERE machine_id = %s
              AND anomaly_type = %s
              {flag_condition};
            """,
            (machine_id, TELEMETRY_ANOMALY_TYPE),
        )
        rows = self._fetch_all(
            f"""
            SELECT
                anomaly_id,
                event_id,
                event_time,
                vibration_mm_s,
                pressure_bar,
                anomaly_score,
                anomaly_flag,
                model_name,
                model_version,
                model_config_hash,
                baseline_event_id_sha256,
                baseline_feature_data_sha256,
                source_kafka_topic,
                source_kafka_partition,
                source_kafka_offset,
                source_kafka_timestamp,
                source_kafka_key,
                payload_sha256
            FROM anomalies
            WHERE machine_id = %s
              AND anomaly_type = %s
              {flag_condition}
            ORDER BY
                event_time DESC NULLS LAST,
                source_kafka_timestamp DESC NULLS LAST,
                source_kafka_topic DESC NULLS LAST,
                source_kafka_partition DESC NULLS LAST,
                source_kafka_offset DESC NULLS LAST,
                event_id DESC NULLS LAST
            LIMIT %s OFFSET %s;
            """,
            (machine_id, TELEMETRY_ANOMALY_TYPE, limit, offset),
        )
        items = []
        for row in rows:
            items.append(
                {
                    "anomaly_id": row["anomaly_id"],
                    "event_id": row.get("event_id"),
                    "event_time": row.get("event_time"),
                    "vibration_mm_s": row.get("vibration_mm_s"),
                    "pressure_bar": row.get("pressure_bar"),
                    "anomaly_score": row["anomaly_score"],
                    "anomaly_flag": row.get("anomaly_flag"),
                    "score_semantics": "anomaly_score_not_probability",
                    "model_name": row.get("model_name"),
                    "model_version": row.get("model_version"),
                    "model_config_hash": row.get("model_config_hash"),
                    "baseline_event_id_sha256": row.get("baseline_event_id_sha256"),
                    "baseline_feature_data_sha256": row.get("baseline_feature_data_sha256"),
                    "lineage": source_lineage_from_row(row),
                }
            )
        return {
            "machine_code": machine_code,
            "flagged_only": flagged_only,
            "items": items,
            "limit": limit,
            "offset": offset,
            "count": len(items),
            "total": int(total_row["total"] if total_row else 0),
        }

    def fleet_overview(self) -> dict[str, Any]:
        row = self._fetch_one(
            """
            SELECT
                (SELECT count(*) FROM machines) AS machine_count,
                (
                    SELECT count(*)
                    FROM machine_health
                    WHERE latest_model_prediction_id IS NOT NULL
                ) AS machines_with_prediction_projection,
                (
                    SELECT count(*)
                    FROM model_predictions
                    WHERE prediction_type = %s
                ) AS prediction_history_count,
                (
                    SELECT count(*)
                    FROM model_predictions
                    WHERE prediction_type = %s
                      AND failure_prediction IS TRUE
                ) AS positive_prediction_count,
                (
                    SELECT count(*)
                    FROM model_predictions
                    WHERE prediction_type = %s
                      AND failure_prediction IS FALSE
                ) AS negative_prediction_count,
                (
                    SELECT avg(failure_probability)
                    FROM model_predictions
                    WHERE prediction_type = %s
                ) AS mean_failure_probability,
                (
                    SELECT max(failure_probability)
                    FROM model_predictions
                    WHERE prediction_type = %s
                ) AS max_failure_probability,
                (
                    SELECT count(*)
                    FROM anomalies
                    WHERE anomaly_type = %s
                ) AS anomaly_audit_count,
                (
                    SELECT count(*)
                    FROM anomalies
                    WHERE anomaly_type = %s
                      AND anomaly_flag IS TRUE
                ) AS flagged_anomaly_count,
                (
                    SELECT count(*)
                    FROM anomalies
                    WHERE anomaly_type = %s
                      AND anomaly_flag IS FALSE
                ) AS non_flagged_anomaly_count,
                (
                    SELECT ai4i_overall_status
                    FROM drift_snapshots
                    ORDER BY drift_snapshot_id DESC
                    LIMIT 1
                ) AS latest_ai4i_drift_status,
                (
                    SELECT anomaly_overall_status
                    FROM drift_snapshots
                    ORDER BY drift_snapshot_id DESC
                    LIMIT 1
                ) AS latest_anomaly_drift_status,
                (
                    SELECT count(*)
                    FROM alerts
                    WHERE status = 'open'
                ) AS open_alert_count;
            """,
            (
                AI4I_PREDICTION_TYPE,
                AI4I_PREDICTION_TYPE,
                AI4I_PREDICTION_TYPE,
                AI4I_PREDICTION_TYPE,
                AI4I_PREDICTION_TYPE,
                TELEMETRY_ANOMALY_TYPE,
                TELEMETRY_ANOMALY_TYPE,
                TELEMETRY_ANOMALY_TYPE,
            ),
        )
        return row or {}

    def latest_drift(self) -> dict[str, Any]:
        snapshot = self._fetch_one(
            """
            SELECT
                drift_snapshot_id,
                monitor_version,
                reference_profile_sha256,
                ai4i_reference_identity,
                anomaly_reference_identity,
                ai4i_current_data_hash,
                anomaly_current_data_hash,
                ai4i_overall_status,
                anomaly_overall_status,
                ai4i_current_count,
                anomaly_current_count,
                created_at
            FROM drift_snapshots
            ORDER BY drift_snapshot_id DESC
            LIMIT 1;
            """
        )
        features_by_scope: dict[str, list[dict[str, Any]]] = {scope: [] for scope in DRIFT_SCOPES}
        if snapshot is None:
            return {
                "drift_snapshot_id": None,
                "monitor_version": None,
                "reference_profile_sha256": None,
                "ai4i_reference_identity": None,
                "anomaly_reference_identity": None,
                "ai4i_current_data_hash": None,
                "anomaly_current_data_hash": None,
                "ai4i_overall_status": None,
                "anomaly_overall_status": None,
                "ai4i_current_count": None,
                "anomaly_current_count": None,
                "created_at": None,
                "features_by_scope": features_by_scope,
            }
        metrics = self._fetch_all(
            """
            SELECT
                monitor_scope,
                feature_name,
                feature_type,
                psi,
                status,
                reference_count,
                current_count,
                reference_mean,
                current_mean,
                reference_std,
                current_std,
                reference_min,
                reference_max,
                current_min,
                current_max,
                standardized_mean_shift,
                outside_reference_range_count,
                outside_reference_range_rate,
                reference_proportions,
                current_proportions,
                bin_edges,
                diagnostics
            FROM drift_feature_metrics
            WHERE drift_snapshot_id = %s
            ORDER BY monitor_scope, psi DESC, feature_name;
            """,
            (snapshot["drift_snapshot_id"],),
        )
        for metric in metrics:
            scope = str(metric.pop("monitor_scope"))
            features_by_scope.setdefault(scope, []).append(metric)
        snapshot["features_by_scope"] = features_by_scope
        return snapshot

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
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("al.status = %s")
            params.append(status)
        if severity is not None:
            conditions.append("al.severity = %s")
            params.append(severity)
        if alert_type is not None:
            conditions.append("al.alert_type = %s")
            params.append(alert_type)
        if machine_code is not None:
            conditions.append("m.machine_identifier = %s")
            params.append(machine_code)
        where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
        total_row = self._fetch_one(
            f"""
            SELECT count(*) AS total
            FROM alerts al
            JOIN machines m
              ON m.machine_id = al.machine_id
            {where_sql};
            """,
            params,
        )
        rows = self._fetch_all(
            f"""
            SELECT
                al.alert_id,
                m.machine_identifier AS machine_code,
                al.severity,
                al.alert_type,
                al.title,
                al.description,
                al.status,
                al.model_prediction_id,
                al.anomaly_id,
                CASE
                    WHEN al.model_prediction_id IS NOT NULL THEN 'model_prediction'
                    WHEN al.anomaly_id IS NOT NULL THEN 'anomaly'
                    ELSE 'unknown'
                END AS source_kind,
                COALESCE(mp.event_id::text, an.event_id::text) AS source_event_id,
                COALESCE(mp.event_time, an.event_time, al.created_at) AS source_observed_at,
                al.created_at
            FROM alerts al
            JOIN machines m
              ON m.machine_id = al.machine_id
            LEFT JOIN model_predictions mp
              ON mp.model_prediction_id = al.model_prediction_id
            LEFT JOIN anomalies an
              ON an.anomaly_id = al.anomaly_id
            {where_sql}
            ORDER BY
                COALESCE(mp.event_time, an.event_time, al.created_at) DESC,
                al.created_at DESC,
                al.alert_id DESC
            LIMIT %s OFFSET %s;
            """,
            (*params, limit, offset),
        )
        return {
            "items": rows,
            "limit": limit,
            "offset": offset,
            "count": len(rows),
            "total": int(total_row["total"] if total_row else 0),
        }

    def get_alert(self, alert_id: int) -> dict[str, Any]:
        row = self._fetch_one(
            """
            SELECT
                al.alert_id,
                m.machine_identifier AS machine_code,
                al.severity,
                al.alert_type,
                al.title,
                al.description,
                al.status,
                al.model_prediction_id,
                al.anomaly_id,
                CASE
                    WHEN al.model_prediction_id IS NOT NULL THEN 'model_prediction'
                    WHEN al.anomaly_id IS NOT NULL THEN 'anomaly'
                    ELSE 'unknown'
                END AS source_kind,
                COALESCE(mp.event_id::text, an.event_id::text) AS source_event_id,
                COALESCE(mp.event_time, an.event_time, al.created_at) AS source_observed_at,
                al.created_at
            FROM alerts al
            JOIN machines m
              ON m.machine_id = al.machine_id
            LEFT JOIN model_predictions mp
              ON mp.model_prediction_id = al.model_prediction_id
            LEFT JOIN anomalies an
              ON an.anomaly_id = al.anomaly_id
            WHERE al.alert_id = %s;
            """,
            (alert_id,),
        )
        if row is None:
            raise AlertNotFoundError(str(alert_id))
        return row

    def protected_state_counts(self) -> dict[str, int]:
        row = self._fetch_one(
            """
            SELECT
                (SELECT count(*) FROM model_predictions) AS model_predictions,
                (SELECT count(*) FROM prediction_explanations) AS prediction_explanations,
                (SELECT count(*) FROM anomalies) AS anomalies,
                (SELECT count(*) FROM drift_snapshots) AS drift_snapshots,
                (SELECT count(*) FROM drift_feature_metrics) AS drift_feature_metrics;
            """
        )
        return {key: int(value) for key, value in (row or {}).items()}

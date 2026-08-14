"""PostgreSQL persistence helpers for deterministic drift reports."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.monitoring import drift

SUMMARY_RELATIVE_PATH = Path("reports") / "database" / "drift_monitoring_persistence_summary.json"


class DriftPersistenceError(ValueError):
    """Raised when drift persistence validation fails."""


@dataclass(frozen=True)
class DriftSnapshotIdentity:
    """Stable business identity for one logical drift snapshot."""

    monitor_version: str
    reference_profile_sha256: str
    ai4i_current_data_hash: str
    anomaly_current_data_hash: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.monitor_version,
            self.reference_profile_sha256,
            self.ai4i_current_data_hash,
            self.anomaly_current_data_hash,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "ai4i_current_data_hash": self.ai4i_current_data_hash,
            "anomaly_current_data_hash": self.anomaly_current_data_hash,
            "monitor_version": self.monitor_version,
            "reference_profile_sha256": self.reference_profile_sha256,
        }


@dataclass(frozen=True)
class ConflictDetail:
    """Immutable mismatch for an existing drift identity."""

    identity: DriftSnapshotIdentity
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "identity": self.identity.to_dict(),
        }


@dataclass(frozen=True)
class DriftReuseSummary:
    """Pure idempotency summary before persistence runs."""

    new_snapshots: int
    existing_identical_snapshots_reused: int
    new_feature_metrics: int
    existing_identical_feature_metrics_reused: int
    conflicts: tuple[ConflictDetail, ...]


@dataclass(frozen=True)
class PersistenceSummary:
    """Summary of one drift report persistence run."""

    input_feature_metrics: int
    new_snapshots_inserted: int
    existing_identical_snapshots_reused: int
    new_feature_metrics_inserted: int
    existing_identical_feature_metrics_reused: int
    conflicts: int

    def to_dict(self) -> dict[str, int]:
        return {
            "conflicts": self.conflicts,
            "existing_identical_feature_metrics_reused": (
                self.existing_identical_feature_metrics_reused
            ),
            "existing_identical_snapshots_reused": self.existing_identical_snapshots_reused,
            "input_feature_metrics": self.input_feature_metrics,
            "new_feature_metrics_inserted": self.new_feature_metrics_inserted,
            "new_snapshots_inserted": self.new_snapshots_inserted,
        }


@dataclass(frozen=True)
class DriftStateSummary:
    """Read-only database state for the current drift monitor/reference."""

    snapshot_count: int
    latest_logical_snapshot_identity: dict[str, Any] | None
    ai4i_overall_status: str | None
    anomaly_overall_status: str | None
    feature_metric_count: int
    features_by_status: dict[str, int]
    highest_ai4i_psi_feature: dict[str, Any] | None
    highest_anomaly_psi_feature: dict[str, Any] | None
    duplicate_snapshot_identity_count: int
    duplicate_feature_metric_identity_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai4i_overall_status": self.ai4i_overall_status,
            "anomaly_overall_status": self.anomaly_overall_status,
            "duplicate_feature_metric_identity_count": self.duplicate_feature_metric_identity_count,
            "duplicate_snapshot_identity_count": self.duplicate_snapshot_identity_count,
            "feature_metric_count": self.feature_metric_count,
            "features_by_status": self.features_by_status,
            "highest_ai4i_psi_feature": self.highest_ai4i_psi_feature,
            "highest_anomaly_psi_feature": self.highest_anomaly_psi_feature,
            "latest_logical_snapshot_identity": self.latest_logical_snapshot_identity,
            "snapshot_count": self.snapshot_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def load_report(path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root_path = root or project_root()
    report_path = path or drift.drift_report_path(root_path)
    config = drift.load_config(root_path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftPersistenceError(f"Drift report does not exist: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise DriftPersistenceError(f"Drift report is invalid JSON: {report_path}") from exc
    if not isinstance(payload, dict):
        raise DriftPersistenceError("Drift report must be a JSON object.")
    try:
        drift.validate_drift_report(payload, config)
    except drift.DriftMonitoringError as exc:
        raise DriftPersistenceError(str(exc)) from exc
    return payload


def snapshot_identity(report: Mapping[str, Any]) -> DriftSnapshotIdentity:
    return DriftSnapshotIdentity(
        monitor_version=str(report["monitor_version"]),
        reference_profile_sha256=str(report["reference_profile_sha256"]),
        ai4i_current_data_hash=str(report[drift.AI4I_SCOPE]["current_data_hash"]),
        anomaly_current_data_hash=str(report[drift.ANOMALY_SCOPE]["current_data_hash"]),
    )


def snapshot_values(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ai4i_current_count": int(report[drift.AI4I_SCOPE]["current_record_count"]),
        "ai4i_current_data_hash": report[drift.AI4I_SCOPE]["current_data_hash"],
        "ai4i_overall_status": report[drift.AI4I_SCOPE]["overall_status"],
        "ai4i_reference_identity": report[drift.AI4I_SCOPE]["reference_identity"],
        "anomaly_current_count": int(report[drift.ANOMALY_SCOPE]["current_record_count"]),
        "anomaly_current_data_hash": report[drift.ANOMALY_SCOPE]["current_data_hash"],
        "anomaly_overall_status": report[drift.ANOMALY_SCOPE]["overall_status"],
        "anomaly_reference_identity": report[drift.ANOMALY_SCOPE]["reference_identity"],
        "monitor_version": report["monitor_version"],
        "reference_profile_sha256": report["reference_profile_sha256"],
    }


def metric_values(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for monitor_scope in (drift.AI4I_SCOPE, drift.ANOMALY_SCOPE):
        for metric in report[monitor_scope]["features"]:
            row = {
                "bin_edges": metric.get("bin_edges"),
                "current_count": int(metric["current_count"]),
                "current_max": metric.get("current_max"),
                "current_mean": metric.get("current_mean"),
                "current_min": metric.get("current_min"),
                "current_proportions": current_proportions(metric),
                "current_std": metric.get("current_std"),
                "diagnostics": diagnostic_payload(metric),
                "feature_name": metric["feature_name"],
                "feature_type": metric["feature_type"],
                "monitor_scope": monitor_scope,
                "outside_reference_range_count": metric.get("outside_reference_range_count"),
                "outside_reference_range_rate": metric.get("outside_reference_range_rate"),
                "psi": float(metric["psi"]),
                "reference_count": int(metric["reference_count"]),
                "reference_max": metric.get("reference_max"),
                "reference_mean": metric.get("reference_mean"),
                "reference_min": metric.get("reference_min"),
                "reference_proportions": reference_proportions(metric),
                "reference_std": metric.get("reference_std"),
                "standardized_mean_shift": metric.get("standardized_mean_shift"),
                "status": metric["status"],
            }
            rows.append(row)
    return rows


def reference_proportions(metric: Mapping[str, Any]) -> Any:
    if metric["feature_type"] == "numeric":
        return metric["reference_bin_proportions"]
    return metric["reference_proportions"]


def current_proportions(metric: Mapping[str, Any]) -> Any:
    if metric["feature_type"] == "numeric":
        return metric["current_bin_proportions"]
    return metric["current_proportions"]


def diagnostic_payload(metric: Mapping[str, Any]) -> dict[str, Any]:
    if metric["feature_type"] == "categorical":
        return {
            "categories": metric["categories"],
            "unexpected_category_count": metric["unexpected_category_count"],
        }
    return {
        "outside_reference_range_count": metric["outside_reference_range_count"],
        "outside_reference_range_rate": metric["outside_reference_range_rate"],
        "standardized_mean_shift": metric["standardized_mean_shift"],
    }


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def numeric_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DriftPersistenceError("Numeric persistence value must be finite.")
    return repr(numeric)


def int_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    return str(int(value))


def jsonb_sql(value: Any) -> str:
    return sql_literal(drift.canonical_json(value)) + "::jsonb"


def snapshot_values_sql(values: Mapping[str, Any]) -> str:
    return (
        "("
        + ", ".join(
            [
                sql_literal(str(values["monitor_version"])),
                sql_literal(str(values["reference_profile_sha256"])),
                jsonb_sql(values["ai4i_reference_identity"]),
                jsonb_sql(values["anomaly_reference_identity"]),
                sql_literal(str(values["ai4i_current_data_hash"])),
                sql_literal(str(values["anomaly_current_data_hash"])),
                sql_literal(str(values["ai4i_overall_status"])),
                sql_literal(str(values["anomaly_overall_status"])),
                str(int(values["ai4i_current_count"])),
                str(int(values["anomaly_current_count"])),
            ]
        )
        + ")"
    )


def metric_values_sql(identity: DriftSnapshotIdentity, row: Mapping[str, Any]) -> str:
    return (
        "("
        + ", ".join(
            [
                sql_literal(identity.monitor_version),
                sql_literal(identity.reference_profile_sha256),
                sql_literal(identity.ai4i_current_data_hash),
                sql_literal(identity.anomaly_current_data_hash),
                sql_literal(str(row["monitor_scope"])),
                sql_literal(str(row["feature_name"])),
                sql_literal(str(row["feature_type"])),
                numeric_sql(row["psi"]),
                sql_literal(str(row["status"])),
                str(int(row["reference_count"])),
                str(int(row["current_count"])),
                numeric_sql(row["reference_mean"]),
                numeric_sql(row["current_mean"]),
                numeric_sql(row["reference_std"]),
                numeric_sql(row["current_std"]),
                numeric_sql(row["reference_min"]),
                numeric_sql(row["reference_max"]),
                numeric_sql(row["current_min"]),
                numeric_sql(row["current_max"]),
                numeric_sql(row["standardized_mean_shift"]),
                int_sql(row["outside_reference_range_count"]),
                numeric_sql(row["outside_reference_range_rate"]),
                jsonb_sql(row["reference_proportions"]),
                jsonb_sql(row["current_proportions"]),
                "NULL" if row["bin_edges"] is None else jsonb_sql(row["bin_edges"]),
                jsonb_sql(row["diagnostics"]),
            ]
        )
        + ")"
    )


def build_persistence_transaction(report: Mapping[str, Any]) -> str:
    identity = snapshot_identity(report)
    snapshot = snapshot_values(report)
    metrics = metric_values(report)
    if not metrics:
        raise DriftPersistenceError("At least one drift feature metric is required.")
    metric_rows_sql = ",\n".join(metric_values_sql(identity, row) for row in metrics)
    return f"""
BEGIN;

CREATE TEMP TABLE staging_drift_snapshot (
    monitor_version TEXT NOT NULL,
    reference_profile_sha256 TEXT NOT NULL,
    ai4i_reference_identity JSONB NOT NULL,
    anomaly_reference_identity JSONB NOT NULL,
    ai4i_current_data_hash TEXT NOT NULL,
    anomaly_current_data_hash TEXT NOT NULL,
    ai4i_overall_status TEXT NOT NULL,
    anomaly_overall_status TEXT NOT NULL,
    ai4i_current_count INTEGER NOT NULL,
    anomaly_current_count INTEGER NOT NULL
) ON COMMIT DROP;

INSERT INTO staging_drift_snapshot (
    monitor_version,
    reference_profile_sha256,
    ai4i_reference_identity,
    anomaly_reference_identity,
    ai4i_current_data_hash,
    anomaly_current_data_hash,
    ai4i_overall_status,
    anomaly_overall_status,
    ai4i_current_count,
    anomaly_current_count
)
VALUES
{snapshot_values_sql(snapshot)};

CREATE TEMP TABLE staging_drift_feature_metrics (
    monitor_version TEXT NOT NULL,
    reference_profile_sha256 TEXT NOT NULL,
    ai4i_current_data_hash TEXT NOT NULL,
    anomaly_current_data_hash TEXT NOT NULL,
    monitor_scope TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    psi NUMERIC NOT NULL,
    status TEXT NOT NULL,
    reference_count INTEGER NOT NULL,
    current_count INTEGER NOT NULL,
    reference_mean NUMERIC,
    current_mean NUMERIC,
    reference_std NUMERIC,
    current_std NUMERIC,
    reference_min NUMERIC,
    reference_max NUMERIC,
    current_min NUMERIC,
    current_max NUMERIC,
    standardized_mean_shift NUMERIC,
    outside_reference_range_count INTEGER,
    outside_reference_range_rate NUMERIC,
    reference_proportions JSONB NOT NULL,
    current_proportions JSONB NOT NULL,
    bin_edges JSONB,
    diagnostics JSONB NOT NULL
) ON COMMIT DROP;

INSERT INTO staging_drift_feature_metrics (
    monitor_version,
    reference_profile_sha256,
    ai4i_current_data_hash,
    anomaly_current_data_hash,
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
)
VALUES
{metric_rows_sql};

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM staging_drift_snapshot s
        JOIN drift_snapshots d
          ON d.monitor_version = s.monitor_version
         AND d.reference_profile_sha256 = s.reference_profile_sha256
         AND d.ai4i_current_data_hash = s.ai4i_current_data_hash
         AND d.anomaly_current_data_hash = s.anomaly_current_data_hash
        WHERE d.ai4i_reference_identity <> s.ai4i_reference_identity
           OR d.anomaly_reference_identity <> s.anomaly_reference_identity
           OR d.ai4i_overall_status <> s.ai4i_overall_status
           OR d.anomaly_overall_status <> s.anomaly_overall_status
           OR d.ai4i_current_count <> s.ai4i_current_count
           OR d.anomaly_current_count <> s.anomaly_current_count
    ) THEN
        RAISE EXCEPTION 'Conflicting drift snapshot identity already exists.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM staging_drift_feature_metrics s
        JOIN drift_snapshots d
          ON d.monitor_version = s.monitor_version
         AND d.reference_profile_sha256 = s.reference_profile_sha256
         AND d.ai4i_current_data_hash = s.ai4i_current_data_hash
         AND d.anomaly_current_data_hash = s.anomaly_current_data_hash
        JOIN drift_feature_metrics m
          ON m.drift_snapshot_id = d.drift_snapshot_id
         AND m.monitor_scope = s.monitor_scope
         AND m.feature_name = s.feature_name
        WHERE m.feature_type <> s.feature_type
           OR m.psi <> s.psi
           OR m.status <> s.status
           OR m.reference_count <> s.reference_count
           OR m.current_count <> s.current_count
           OR m.reference_mean IS DISTINCT FROM s.reference_mean
           OR m.current_mean IS DISTINCT FROM s.current_mean
           OR m.reference_std IS DISTINCT FROM s.reference_std
           OR m.current_std IS DISTINCT FROM s.current_std
           OR m.reference_min IS DISTINCT FROM s.reference_min
           OR m.reference_max IS DISTINCT FROM s.reference_max
           OR m.current_min IS DISTINCT FROM s.current_min
           OR m.current_max IS DISTINCT FROM s.current_max
           OR m.standardized_mean_shift IS DISTINCT FROM s.standardized_mean_shift
           OR m.outside_reference_range_count IS DISTINCT FROM s.outside_reference_range_count
           OR m.outside_reference_range_rate IS DISTINCT FROM s.outside_reference_range_rate
           OR m.reference_proportions <> s.reference_proportions
           OR m.current_proportions <> s.current_proportions
           OR m.bin_edges IS DISTINCT FROM s.bin_edges
           OR m.diagnostics <> s.diagnostics
    ) THEN
        RAISE EXCEPTION 'Conflicting drift feature metric identity already exists.';
    END IF;
END $$;

INSERT INTO drift_snapshots (
    monitor_version,
    reference_profile_sha256,
    ai4i_reference_identity,
    anomaly_reference_identity,
    ai4i_current_data_hash,
    anomaly_current_data_hash,
    ai4i_overall_status,
    anomaly_overall_status,
    ai4i_current_count,
    anomaly_current_count
)
SELECT
    monitor_version,
    reference_profile_sha256,
    ai4i_reference_identity,
    anomaly_reference_identity,
    ai4i_current_data_hash,
    anomaly_current_data_hash,
    ai4i_overall_status,
    anomaly_overall_status,
    ai4i_current_count,
    anomaly_current_count
FROM staging_drift_snapshot
ON CONFLICT (
    monitor_version,
    reference_profile_sha256,
    ai4i_current_data_hash,
    anomaly_current_data_hash
)
DO NOTHING;

INSERT INTO drift_feature_metrics (
    drift_snapshot_id,
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
)
SELECT
    d.drift_snapshot_id,
    s.monitor_scope,
    s.feature_name,
    s.feature_type,
    s.psi,
    s.status,
    s.reference_count,
    s.current_count,
    s.reference_mean,
    s.current_mean,
    s.reference_std,
    s.current_std,
    s.reference_min,
    s.reference_max,
    s.current_min,
    s.current_max,
    s.standardized_mean_shift,
    s.outside_reference_range_count,
    s.outside_reference_range_rate,
    s.reference_proportions,
    s.current_proportions,
    s.bin_edges,
    s.diagnostics
FROM staging_drift_feature_metrics s
JOIN drift_snapshots d
  ON d.monitor_version = s.monitor_version
 AND d.reference_profile_sha256 = s.reference_profile_sha256
 AND d.ai4i_current_data_hash = s.ai4i_current_data_hash
 AND d.anomaly_current_data_hash = s.anomaly_current_data_hash
ON CONFLICT (drift_snapshot_id, monitor_scope, feature_name)
DO NOTHING;

COMMIT;
"""


def build_existing_report_query(report: Mapping[str, Any]) -> str:
    identity = snapshot_identity(report)
    return f"""
    SELECT COALESCE(jsonb_agg(snapshot_payload ORDER BY drift_snapshot_id), '[]'::jsonb)
    FROM (
        SELECT
            d.drift_snapshot_id,
            jsonb_build_object(
                'drift_snapshot_id', d.drift_snapshot_id,
                'monitor_version', d.monitor_version,
                'reference_profile_sha256', d.reference_profile_sha256,
                'ai4i_reference_identity', d.ai4i_reference_identity,
                'anomaly_reference_identity', d.anomaly_reference_identity,
                'ai4i_current_data_hash', d.ai4i_current_data_hash,
                'anomaly_current_data_hash', d.anomaly_current_data_hash,
                'ai4i_overall_status', d.ai4i_overall_status,
                'anomaly_overall_status', d.anomaly_overall_status,
                'ai4i_current_count', d.ai4i_current_count,
                'anomaly_current_count', d.anomaly_current_count,
                'feature_metrics', (
                    SELECT COALESCE(
                        jsonb_agg(
                            jsonb_build_object(
                                'monitor_scope', m.monitor_scope,
                                'feature_name', m.feature_name,
                                'feature_type', m.feature_type,
                                'psi', m.psi,
                                'status', m.status,
                                'reference_count', m.reference_count,
                                'current_count', m.current_count,
                                'reference_mean', m.reference_mean,
                                'current_mean', m.current_mean,
                                'reference_std', m.reference_std,
                                'current_std', m.current_std,
                                'reference_min', m.reference_min,
                                'reference_max', m.reference_max,
                                'current_min', m.current_min,
                                'current_max', m.current_max,
                                'standardized_mean_shift', m.standardized_mean_shift,
                                'outside_reference_range_count',
                                m.outside_reference_range_count,
                                'outside_reference_range_rate',
                                m.outside_reference_range_rate,
                                'reference_proportions', m.reference_proportions,
                                'current_proportions', m.current_proportions,
                                'bin_edges', m.bin_edges,
                                'diagnostics', m.diagnostics
                            )
                            ORDER BY m.monitor_scope, m.feature_name
                        ),
                        '[]'::jsonb
                    )
                    FROM drift_feature_metrics m
                    WHERE m.drift_snapshot_id = d.drift_snapshot_id
                )
            ) AS snapshot_payload
        FROM drift_snapshots d
        WHERE d.monitor_version = {sql_literal(identity.monitor_version)}
          AND d.reference_profile_sha256 = {sql_literal(identity.reference_profile_sha256)}
          AND d.ai4i_current_data_hash = {sql_literal(identity.ai4i_current_data_hash)}
          AND d.anomaly_current_data_hash = {sql_literal(identity.anomaly_current_data_hash)}
    ) snapshots;
    """


def parse_json_query_output(output: str) -> Any:
    text = output.replace("\x00", "").strip()
    if not text:
        return None
    return json.loads(text)


def parse_count_output(output: str) -> int:
    lines = [line.strip() for line in output.replace("\x00", "").splitlines() if line.strip()]
    if not lines:
        raise DriftPersistenceError("Expected a count query result, but output was empty.")
    return int(lines[0])


def normalize_existing_number(value: Any) -> Any:
    if value is None:
        return None
    return float(value)


def normalize_existing_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in (
        "current_max",
        "current_mean",
        "current_min",
        "current_std",
        "outside_reference_range_rate",
        "psi",
        "reference_max",
        "reference_mean",
        "reference_min",
        "reference_std",
        "standardized_mean_shift",
    ):
        normalized[field] = normalize_existing_number(row.get(field))
    return normalized


def values_match(expected: Any, existing: Any) -> bool:
    if expected is None or existing is None:
        return expected is None and existing is None
    if isinstance(expected, float) or isinstance(existing, float):
        return math.isclose(float(expected), float(existing), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(expected, dict) and isinstance(existing, dict):
        return drift.canonical_json(expected) == drift.canonical_json(existing)
    if isinstance(expected, list) and isinstance(existing, list):
        return len(expected) == len(existing) and all(
            values_match(left, right) for left, right in zip(expected, existing, strict=True)
        )
    return expected == existing


def compare_mapping(expected: Mapping[str, Any], existing: Mapping[str, Any]) -> tuple[str, ...]:
    fields: list[str] = []
    for key, expected_value in expected.items():
        if not values_match(expected_value, existing.get(key)):
            fields.append(key)
    return tuple(fields)


def summarize_report_reuse(
    report: Mapping[str, Any],
    existing_snapshots: Sequence[Mapping[str, Any]],
) -> DriftReuseSummary:
    identity = snapshot_identity(report)
    metrics = metric_values(report)
    if not existing_snapshots:
        return DriftReuseSummary(
            new_snapshots=1,
            existing_identical_snapshots_reused=0,
            new_feature_metrics=len(metrics),
            existing_identical_feature_metrics_reused=0,
            conflicts=(),
        )
    if len(existing_snapshots) > 1:
        return DriftReuseSummary(
            new_snapshots=0,
            existing_identical_snapshots_reused=0,
            new_feature_metrics=0,
            existing_identical_feature_metrics_reused=0,
            conflicts=(ConflictDetail(identity, ("duplicate_drift_snapshot_identity",)),),
        )

    existing = existing_snapshots[0]
    snapshot_conflicts = compare_mapping(
        snapshot_values(report),
        existing,
    )
    expected_by_identity = {(row["monitor_scope"], row["feature_name"]): row for row in metrics}
    existing_metrics = [normalize_existing_metric(row) for row in existing["feature_metrics"]]
    existing_by_identity = {
        (row["monitor_scope"], row["feature_name"]): row for row in existing_metrics
    }
    metric_conflict_fields: list[str] = []
    reused_metrics = 0
    for metric_identity, expected_metric in expected_by_identity.items():
        existing_metric = existing_by_identity.get(metric_identity)
        if existing_metric is None:
            continue
        fields = compare_mapping(expected_metric, existing_metric)
        if fields:
            metric_conflict_fields.extend(
                f"{metric_identity[0]}.{metric_identity[1]}.{field}" for field in fields
            )
        else:
            reused_metrics += 1
    missing_existing = len(expected_by_identity) - reused_metrics
    conflicts = []
    if snapshot_conflicts:
        conflicts.append(ConflictDetail(identity, snapshot_conflicts))
    if metric_conflict_fields:
        conflicts.append(ConflictDetail(identity, tuple(sorted(metric_conflict_fields))))
    return DriftReuseSummary(
        new_snapshots=0 if not snapshot_conflicts else 0,
        existing_identical_snapshots_reused=0 if snapshot_conflicts else 1,
        new_feature_metrics=missing_existing if not metric_conflict_fields else 0,
        existing_identical_feature_metrics_reused=reused_metrics,
        conflicts=tuple(conflicts),
    )


def build_static_summary() -> dict[str, Any]:
    return {
        "conflict_policy": "Identical snapshot identities with different immutable values fail.",
        "feature_metric_identity": ["drift_snapshot_id", "monitor_scope", "feature_name"],
        "persistence_tables": ["drift_snapshots", "drift_feature_metrics"],
        "stable_snapshot_identity": [
            "monitor_version",
            "reference_profile_sha256",
            "ai4i_current_data_hash",
            "anomaly_current_data_hash",
        ],
        "transactional": True,
    }


def write_static_summary(root: Path | None = None) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_static_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def feature_status_counts(metrics: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(metric["status"]) for metric in metrics)
    return {status: int(counts.get(status, 0)) for status in ("stable", "watch", "drift")}


def state_summary_from_payload(payload: Mapping[str, Any]) -> DriftStateSummary:
    latest = payload.get("latest_logical_snapshot_identity")
    highest_ai4i = payload.get("highest_ai4i_psi_feature")
    highest_anomaly = payload.get("highest_anomaly_psi_feature")
    return DriftStateSummary(
        snapshot_count=int(payload["snapshot_count"]),
        latest_logical_snapshot_identity=latest if isinstance(latest, dict) else None,
        ai4i_overall_status=payload.get("ai4i_overall_status"),
        anomaly_overall_status=payload.get("anomaly_overall_status"),
        feature_metric_count=int(payload["feature_metric_count"]),
        features_by_status={
            "drift": int(payload["features_by_status"].get("drift", 0)),
            "stable": int(payload["features_by_status"].get("stable", 0)),
            "watch": int(payload["features_by_status"].get("watch", 0)),
        },
        highest_ai4i_psi_feature=highest_ai4i if isinstance(highest_ai4i, dict) else None,
        highest_anomaly_psi_feature=highest_anomaly if isinstance(highest_anomaly, dict) else None,
        duplicate_snapshot_identity_count=int(payload["duplicate_snapshot_identity_count"]),
        duplicate_feature_metric_identity_count=int(
            payload["duplicate_feature_metric_identity_count"]
        ),
    )


def build_state_summary_query(monitor_version: str, reference_profile_sha256: str) -> str:
    return f"""
    WITH relevant_snapshots AS (
        SELECT *
        FROM drift_snapshots
        WHERE monitor_version = {sql_literal(monitor_version)}
          AND reference_profile_sha256 = {sql_literal(reference_profile_sha256)}
    ),
    latest_snapshot AS (
        SELECT *
        FROM relevant_snapshots
        ORDER BY drift_snapshot_id DESC
        LIMIT 1
    ),
    relevant_metrics AS (
        SELECT m.*
        FROM drift_feature_metrics m
        JOIN relevant_snapshots d
          ON d.drift_snapshot_id = m.drift_snapshot_id
    ),
    duplicate_snapshots AS (
        SELECT count(*) AS duplicate_count
        FROM (
            SELECT
                monitor_version,
                reference_profile_sha256,
                ai4i_current_data_hash,
                anomaly_current_data_hash
            FROM drift_snapshots
            GROUP BY
                monitor_version,
                reference_profile_sha256,
                ai4i_current_data_hash,
                anomaly_current_data_hash
            HAVING count(*) > 1
        ) duplicates
    ),
    duplicate_metrics AS (
        SELECT count(*) AS duplicate_count
        FROM (
            SELECT drift_snapshot_id, monitor_scope, feature_name
            FROM drift_feature_metrics
            GROUP BY drift_snapshot_id, monitor_scope, feature_name
            HAVING count(*) > 1
        ) duplicates
    )
    SELECT jsonb_build_object(
        'snapshot_count', (SELECT count(*) FROM relevant_snapshots),
        'latest_logical_snapshot_identity', (
            SELECT CASE
                WHEN count(*) = 0 THEN NULL
                ELSE jsonb_build_object(
                    'monitor_version', monitor_version,
                    'reference_profile_sha256', reference_profile_sha256,
                    'ai4i_current_data_hash', ai4i_current_data_hash,
                    'anomaly_current_data_hash', anomaly_current_data_hash
                )
            END
            FROM latest_snapshot
            GROUP BY
                monitor_version,
                reference_profile_sha256,
                ai4i_current_data_hash,
                anomaly_current_data_hash
        ),
        'ai4i_overall_status', (SELECT ai4i_overall_status FROM latest_snapshot),
        'anomaly_overall_status', (SELECT anomaly_overall_status FROM latest_snapshot),
        'feature_metric_count', (SELECT count(*) FROM relevant_metrics),
        'features_by_status', jsonb_build_object(
            'stable', (SELECT count(*) FROM relevant_metrics WHERE status = 'stable'),
            'watch', (SELECT count(*) FROM relevant_metrics WHERE status = 'watch'),
            'drift', (SELECT count(*) FROM relevant_metrics WHERE status = 'drift')
        ),
        'highest_ai4i_psi_feature', (
            SELECT jsonb_build_object(
                'feature_name', feature_name,
                'psi', psi,
                'status', status
            )
            FROM relevant_metrics
            WHERE monitor_scope = 'ai4i_model_input'
            ORDER BY psi DESC, feature_name
            LIMIT 1
        ),
        'highest_anomaly_psi_feature', (
            SELECT jsonb_build_object(
                'feature_name', feature_name,
                'psi', psi,
                'status', status
            )
            FROM relevant_metrics
            WHERE monitor_scope = 'operational_anomaly_inputs'
            ORDER BY psi DESC, feature_name
            LIMIT 1
        ),
        'duplicate_snapshot_identity_count', (
            SELECT duplicate_count FROM duplicate_snapshots
        ),
        'duplicate_feature_metric_identity_count', (
            SELECT duplicate_count FROM duplicate_metrics
        )
    );
    """

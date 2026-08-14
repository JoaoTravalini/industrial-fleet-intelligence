"""PostgreSQL persistence helpers for telemetry anomaly detector outputs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ml.anomaly import telemetry_detector

ANOMALY_TYPE = "telemetry_isolation_forest_score"
SUMMARY_RELATIVE_PATH = Path("reports") / "database" / "telemetry_anomaly_persistence_summary.json"
REQUIRED_ANOMALY_FIELDS = telemetry_detector.ANOMALY_OUTPUT_FIELDS


class TelemetryAnomalyPersistenceError(ValueError):
    """Raised when telemetry anomaly persistence validation fails."""


@dataclass(frozen=True)
class AnomalyIdentity:
    """Stable business identity for one telemetry anomaly detector output."""

    event_id: str
    model_name: str
    model_version: str
    model_config_hash: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.event_id, self.model_name, self.model_version, self.model_config_hash)


@dataclass(frozen=True)
class AnomalyRecord:
    """Validated telemetry anomaly output record."""

    event_id: str
    machine_code: str
    event_time: str
    vibration_mm_s: float
    pressure_bar: float
    anomaly_score: float
    anomaly_flag: bool
    model_name: str
    model_version: str
    model_config_hash: str
    baseline_event_id_sha256: str
    baseline_feature_data_sha256: str
    source_kafka_topic: str
    source_kafka_partition: int
    source_kafka_offset: int
    source_kafka_timestamp: str
    source_kafka_key: str
    payload_sha256: str

    @property
    def identity(self) -> AnomalyIdentity:
        return AnomalyIdentity(
            event_id=self.event_id,
            model_name=self.model_name,
            model_version=self.model_version,
            model_config_hash=self.model_config_hash,
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.event_time,
            self.machine_code,
            self.event_id,
            self.source_kafka_timestamp,
            self.source_kafka_topic,
            self.source_kafka_partition,
            self.source_kafka_offset,
            self.source_kafka_key,
            self.payload_sha256,
        )

    def to_output_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "machine_code": self.machine_code,
            "event_time": self.event_time,
            "vibration_mm_s": self.vibration_mm_s,
            "pressure_bar": self.pressure_bar,
            "anomaly_score": self.anomaly_score,
            "anomaly_flag": self.anomaly_flag,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_config_hash": self.model_config_hash,
            "baseline_event_id_sha256": self.baseline_event_id_sha256,
            "baseline_feature_data_sha256": self.baseline_feature_data_sha256,
            "source_kafka_topic": self.source_kafka_topic,
            "source_kafka_partition": self.source_kafka_partition,
            "source_kafka_offset": self.source_kafka_offset,
            "source_kafka_timestamp": self.source_kafka_timestamp,
            "source_kafka_key": self.source_kafka_key,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class ExistingAnomalyRow:
    """Existing database anomaly row relevant to idempotency checks."""

    anomaly_id: int
    machine_id: int
    record: AnomalyRecord


@dataclass(frozen=True)
class ConflictDetail:
    """Material mismatch for an already persisted anomaly identity."""

    identity: AnomalyIdentity
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.identity.event_id,
            "fields": list(self.fields),
            "model_config_hash": self.identity.model_config_hash,
            "model_name": self.identity.model_name,
            "model_version": self.identity.model_version,
        }


@dataclass(frozen=True)
class AnomalyReuseSummary:
    """Pure idempotency summary before persistence runs."""

    new_records: int
    existing_identical_records: int
    conflicts: tuple[ConflictDetail, ...]


@dataclass(frozen=True)
class PersistenceSummary:
    """Summary of one telemetry anomaly persistence run."""

    input_anomaly_records: int
    new_anomaly_rows_inserted: int
    existing_identical_anomalies_reused: int
    conflicting_anomalies: int
    distinct_machines_in_batch: int

    def to_dict(self) -> dict[str, int]:
        return {
            "conflicting_anomalies": self.conflicting_anomalies,
            "distinct_machines_in_batch": self.distinct_machines_in_batch,
            "existing_identical_anomalies_reused": self.existing_identical_anomalies_reused,
            "input_anomaly_records": self.input_anomaly_records,
            "new_anomaly_rows_inserted": self.new_anomaly_rows_inserted,
        }


@dataclass(frozen=True)
class AnomalyStateSummary:
    """Read-only current-model anomaly detector state summary."""

    anomaly_row_count: int
    distinct_anomaly_event_count: int
    distinct_machine_count_with_anomalies: int
    anomaly_flag_count: int
    non_anomaly_count: int
    min_anomaly_score: float | None
    max_anomaly_score: float | None
    mean_anomaly_score: float | None
    duplicate_anomaly_identity_count: int
    machine_reference_mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_flag_count": self.anomaly_flag_count,
            "anomaly_row_count": self.anomaly_row_count,
            "distinct_anomaly_event_count": self.distinct_anomaly_event_count,
            "distinct_machine_count_with_anomalies": self.distinct_machine_count_with_anomalies,
            "duplicate_anomaly_identity_count": self.duplicate_anomaly_identity_count,
            "machine_reference_mismatch_count": self.machine_reference_mismatch_count,
            "max_anomaly_score": self.max_anomaly_score,
            "mean_anomaly_score": self.mean_anomaly_score,
            "min_anomaly_score": self.min_anomaly_score,
            "non_anomaly_count": self.non_anomaly_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def anomaly_output_path(root: Path | None = None) -> Path:
    return telemetry_detector.anomaly_output_path(root or project_root())


def validate_anomaly_record(record: Mapping[str, Any]) -> AnomalyRecord:
    try:
        validated = telemetry_detector.validate_anomaly_output_record(record)
    except telemetry_detector.TelemetryAnomalyError as exc:
        raise TelemetryAnomalyPersistenceError(str(exc)) from exc
    if validated["model_name"] != telemetry_detector.MODEL_NAME:
        raise TelemetryAnomalyPersistenceError("Anomaly model_name is not current detector.")
    if validated["model_version"] != telemetry_detector.MODEL_VERSION:
        raise TelemetryAnomalyPersistenceError("Anomaly model_version is not current detector.")
    return AnomalyRecord(
        event_id=str(validated["event_id"]),
        machine_code=str(validated["machine_code"]),
        event_time=str(validated["event_time"]),
        vibration_mm_s=float(validated["vibration_mm_s"]),
        pressure_bar=float(validated["pressure_bar"]),
        anomaly_score=float(validated["anomaly_score"]),
        anomaly_flag=bool(validated["anomaly_flag"]),
        model_name=str(validated["model_name"]),
        model_version=str(validated["model_version"]),
        model_config_hash=str(validated["model_config_hash"]),
        baseline_event_id_sha256=str(validated["baseline_event_id_sha256"]),
        baseline_feature_data_sha256=str(validated["baseline_feature_data_sha256"]),
        source_kafka_topic=str(validated["source_kafka_topic"]),
        source_kafka_partition=int(validated["source_kafka_partition"]),
        source_kafka_offset=int(validated["source_kafka_offset"]),
        source_kafka_timestamp=str(validated["source_kafka_timestamp"]),
        source_kafka_key=str(validated["source_kafka_key"]),
        payload_sha256=str(validated["payload_sha256"]),
    )


def validate_anomaly_records(records: Sequence[Mapping[str, Any]]) -> list[AnomalyRecord]:
    if not records:
        raise TelemetryAnomalyPersistenceError("Anomaly file must contain at least one record.")
    validated = [validate_anomaly_record(record) for record in records]
    identities = [record.identity.as_tuple() for record in validated]
    duplicate_identities = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicate_identities:
        raise TelemetryAnomalyPersistenceError(
            "Duplicate anomaly stable identity in input: " + "|".join(duplicate_identities[0])
        )
    event_ids = [record.event_id for record in validated]
    duplicate_events = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicate_events:
        raise TelemetryAnomalyPersistenceError(
            "Duplicate event_id in anomaly file: " + duplicate_events[0]
        )
    model_identities = {
        (
            record.model_name,
            record.model_version,
            record.model_config_hash,
            record.baseline_event_id_sha256,
            record.baseline_feature_data_sha256,
        )
        for record in validated
    }
    if len(model_identities) != 1:
        raise TelemetryAnomalyPersistenceError(
            "Anomaly file must contain one internally consistent model identity."
        )
    return sorted(validated, key=lambda record: record.sort_key())


def current_config_hash(root: Path | None = None) -> str:
    active_config = telemetry_detector.load_config(root or project_root())
    return telemetry_detector.model_config_hash(active_config)


def validate_current_model_identity(
    records: Sequence[AnomalyRecord],
    *,
    root: Path | None = None,
) -> None:
    if not records:
        raise TelemetryAnomalyPersistenceError("At least one anomaly record is required.")
    expected_hash = current_config_hash(root)
    actual_identities = {
        (record.model_name, record.model_version, record.model_config_hash) for record in records
    }
    expected_identity = (
        telemetry_detector.MODEL_NAME,
        telemetry_detector.MODEL_VERSION,
        expected_hash,
    )
    if actual_identities != {expected_identity}:
        raise TelemetryAnomalyPersistenceError(
            "Anomaly records do not match the current model identity/config hash."
        )


def load_anomaly_records(path: Path, *, root: Path | None = None) -> list[AnomalyRecord]:
    if not path.exists():
        raise TelemetryAnomalyPersistenceError(
            f"Anomaly file is missing: {path}. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\score_telemetry_anomalies.py first."
        )
    raw_records = telemetry_detector.read_anomalies_jsonl(path)
    records = validate_anomaly_records(raw_records)
    validate_current_model_identity(records, root=root)
    return records


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def bool_sql_literal(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def score_sql_literal(value: float) -> str:
    return format(value, ".5f")


def numeric_sql_literal(value: float) -> str:
    return format(value, ".6f")


def anomaly_description(record: AnomalyRecord) -> str:
    if record.anomaly_flag:
        return "Isolation Forest flagged this Silver telemetry record as statistically unusual."
    return "Isolation Forest scored this Silver telemetry record inside the reference boundary."


def anomaly_values_sql(record: AnomalyRecord, machine_id: int) -> str:
    return (
        f"({machine_id}, "
        f"{sql_literal(record.event_time)}::timestamptz, "
        f"{score_sql_literal(record.anomaly_score)}, "
        f"{sql_literal(ANOMALY_TYPE)}, "
        f"{sql_literal(anomaly_description(record))}, "
        f"{sql_literal(record.event_id)}::uuid, "
        f"{sql_literal(record.event_time)}::timestamptz, "
        f"{bool_sql_literal(record.anomaly_flag)}, "
        f"{sql_literal(record.model_name)}, "
        f"{sql_literal(record.model_version)}, "
        f"{sql_literal(record.model_config_hash)}, "
        f"{sql_literal(record.baseline_event_id_sha256)}, "
        f"{sql_literal(record.baseline_feature_data_sha256)}, "
        f"{numeric_sql_literal(record.vibration_mm_s)}, "
        f"{numeric_sql_literal(record.pressure_bar)}, "
        f"{sql_literal(record.source_kafka_topic)}, "
        f"{record.source_kafka_partition}, "
        f"{record.source_kafka_offset}, "
        f"{sql_literal(record.source_kafka_timestamp)}::timestamptz, "
        f"{sql_literal(record.source_kafka_key)}, "
        f"{sql_literal(record.payload_sha256)})"
    )


def identity_values_sql(records: Sequence[AnomalyRecord]) -> str:
    return ",\n".join(
        (
            f"({sql_literal(record.event_id)}::uuid, "
            f"{sql_literal(record.model_name)}, "
            f"{sql_literal(record.model_version)}, "
            f"{sql_literal(record.model_config_hash)})"
        )
        for record in records
    )


def machine_code_values_sql(machine_codes: Sequence[str]) -> str:
    return ",\n".join(f"({sql_literal(machine_code)})" for machine_code in machine_codes)


def model_identity(records: Sequence[AnomalyRecord]) -> tuple[str, str, str]:
    identities = {
        (record.model_name, record.model_version, record.model_config_hash) for record in records
    }
    if len(identities) != 1:
        raise TelemetryAnomalyPersistenceError("Anomaly records must contain one model identity.")
    return next(iter(identities))


def build_machine_lookup_query(machine_codes: Sequence[str]) -> str:
    if not machine_codes:
        raise TelemetryAnomalyPersistenceError("At least one machine_code is required.")
    values = machine_code_values_sql(sorted(set(machine_codes)))
    return f"""
WITH wanted(machine_identifier) AS (
    VALUES
{values}
)
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'machine_code', m.machine_identifier,
            'machine_id', m.machine_id
        )
        ORDER BY m.machine_identifier
    )::text,
    '[]'
)
FROM wanted w
JOIN machines m
  ON m.machine_identifier = w.machine_identifier;
"""


def build_existing_anomalies_query(records: Sequence[AnomalyRecord]) -> str:
    if not records:
        raise TelemetryAnomalyPersistenceError("At least one anomaly record is required.")
    values = identity_values_sql(records)
    return f"""
WITH wanted(event_id, model_name, model_version, model_config_hash) AS (
    VALUES
{values}
)
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'anomaly_flag', a.anomaly_flag,
            'anomaly_id', a.anomaly_id,
            'anomaly_score', a.anomaly_score,
            'baseline_event_id_sha256', a.baseline_event_id_sha256,
            'baseline_feature_data_sha256', a.baseline_feature_data_sha256,
            'event_id', a.event_id::text,
            'event_time', to_char(a.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'machine_code', m.machine_identifier,
            'machine_id', a.machine_id,
            'model_config_hash', a.model_config_hash,
            'model_name', a.model_name,
            'model_version', a.model_version,
            'payload_sha256', a.payload_sha256,
            'pressure_bar', a.pressure_bar,
            'source_kafka_key', a.source_kafka_key,
            'source_kafka_offset', a.source_kafka_offset,
            'source_kafka_partition', a.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(a.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', a.source_kafka_topic,
            'vibration_mm_s', a.vibration_mm_s
        )
        ORDER BY a.anomaly_id
    )::text,
    '[]'
)
FROM wanted w
JOIN anomalies a
  ON a.event_id = w.event_id
 AND a.model_name = w.model_name
 AND a.model_version = w.model_version
 AND a.model_config_hash = w.model_config_hash
 AND a.anomaly_type = {sql_literal(ANOMALY_TYPE)}
JOIN machines m
  ON m.machine_id = a.machine_id;
"""


def build_current_anomalies_query(
    model_name: str,
    model_version: str,
    model_config_hash: str,
) -> str:
    return f"""
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'anomaly_flag', a.anomaly_flag,
            'anomaly_id', a.anomaly_id,
            'anomaly_score', a.anomaly_score,
            'baseline_event_id_sha256', a.baseline_event_id_sha256,
            'baseline_feature_data_sha256', a.baseline_feature_data_sha256,
            'event_id', a.event_id::text,
            'event_time', to_char(a.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'machine_code', m.machine_identifier,
            'machine_id', a.machine_id,
            'model_config_hash', a.model_config_hash,
            'model_name', a.model_name,
            'model_version', a.model_version,
            'payload_sha256', a.payload_sha256,
            'pressure_bar', a.pressure_bar,
            'source_kafka_key', a.source_kafka_key,
            'source_kafka_offset', a.source_kafka_offset,
            'source_kafka_partition', a.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(a.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', a.source_kafka_topic,
            'vibration_mm_s', a.vibration_mm_s
        )
        ORDER BY
            a.event_time,
            a.source_kafka_timestamp,
            a.source_kafka_topic,
            a.source_kafka_partition,
            a.source_kafka_offset,
            a.event_id
    )::text,
    '[]'
)
FROM anomalies a
JOIN machines m
  ON m.machine_id = a.machine_id
WHERE a.anomaly_type = {sql_literal(ANOMALY_TYPE)}
  AND a.model_name = {sql_literal(model_name)}
  AND a.model_version = {sql_literal(model_version)}
  AND a.model_config_hash = {sql_literal(model_config_hash)};
"""


def build_duplicate_identity_count_query(
    model_name: str,
    model_version: str,
    model_config_hash: str,
) -> str:
    return f"""
WITH grouped AS (
    SELECT event_id, model_name, model_version, model_config_hash, count(*) AS row_count
    FROM anomalies
    WHERE anomaly_type = {sql_literal(ANOMALY_TYPE)}
      AND model_name = {sql_literal(model_name)}
      AND model_version = {sql_literal(model_version)}
      AND model_config_hash = {sql_literal(model_config_hash)}
    GROUP BY event_id, model_name, model_version, model_config_hash
)
SELECT count(*)
FROM grouped
WHERE row_count > 1;
"""


def build_machine_reference_mismatch_count_query(
    model_name: str,
    model_version: str,
    model_config_hash: str,
) -> str:
    return f"""
SELECT count(*)
FROM anomalies a
LEFT JOIN machines m
  ON m.machine_id = a.machine_id
WHERE a.anomaly_type = {sql_literal(ANOMALY_TYPE)}
  AND a.model_name = {sql_literal(model_name)}
  AND a.model_version = {sql_literal(model_version)}
  AND a.model_config_hash = {sql_literal(model_config_hash)}
  AND m.machine_id IS NULL;
"""


def build_persistence_transaction(
    records: Sequence[AnomalyRecord],
    machine_ids_by_code: Mapping[str, int],
) -> str:
    if not records:
        raise TelemetryAnomalyPersistenceError("At least one anomaly record is required.")
    missing_machines = sorted(
        {record.machine_code for record in records} - set(machine_ids_by_code)
    )
    if missing_machines:
        raise TelemetryAnomalyPersistenceError(
            "Anomaly machine_code values are missing in PostgreSQL: " + ", ".join(missing_machines)
        )
    values = ",\n".join(
        anomaly_values_sql(record, machine_ids_by_code[record.machine_code]) for record in records
    )
    return f"""
BEGIN;

CREATE TEMP TABLE staging_telemetry_anomalies (
    machine_id BIGINT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    anomaly_score NUMERIC(6, 5) NOT NULL,
    anomaly_type TEXT NOT NULL,
    description TEXT NOT NULL,
    event_id UUID NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    anomaly_flag BOOLEAN NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_config_hash TEXT NOT NULL,
    baseline_event_id_sha256 TEXT NOT NULL,
    baseline_feature_data_sha256 TEXT NOT NULL,
    vibration_mm_s NUMERIC(10, 6) NOT NULL,
    pressure_bar NUMERIC(10, 6) NOT NULL,
    source_kafka_topic TEXT NOT NULL,
    source_kafka_partition INTEGER NOT NULL,
    source_kafka_offset BIGINT NOT NULL,
    source_kafka_timestamp TIMESTAMPTZ NOT NULL,
    source_kafka_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
) ON COMMIT DROP;

INSERT INTO staging_telemetry_anomalies (
    machine_id,
    detected_at,
    anomaly_score,
    anomaly_type,
    description,
    event_id,
    event_time,
    anomaly_flag,
    model_name,
    model_version,
    model_config_hash,
    baseline_event_id_sha256,
    baseline_feature_data_sha256,
    vibration_mm_s,
    pressure_bar,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
)
VALUES
{values};

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM staging_telemetry_anomalies s
        JOIN anomalies a
          ON a.event_id = s.event_id
         AND a.model_name = s.model_name
         AND a.model_version = s.model_version
         AND a.model_config_hash = s.model_config_hash
         AND a.anomaly_type = s.anomaly_type
        WHERE a.machine_id <> s.machine_id
           OR a.detected_at <> s.detected_at
           OR a.anomaly_score <> s.anomaly_score
           OR a.event_time <> s.event_time
           OR a.anomaly_flag <> s.anomaly_flag
           OR a.baseline_event_id_sha256 <> s.baseline_event_id_sha256
           OR a.baseline_feature_data_sha256 <> s.baseline_feature_data_sha256
           OR a.vibration_mm_s <> s.vibration_mm_s
           OR a.pressure_bar <> s.pressure_bar
           OR a.source_kafka_topic <> s.source_kafka_topic
           OR a.source_kafka_partition <> s.source_kafka_partition
           OR a.source_kafka_offset <> s.source_kafka_offset
           OR a.source_kafka_timestamp <> s.source_kafka_timestamp
           OR a.source_kafka_key <> s.source_kafka_key
           OR a.payload_sha256 <> s.payload_sha256
    ) THEN
        RAISE EXCEPTION 'Conflicting telemetry anomaly identity already exists.';
    END IF;
END $$;

INSERT INTO anomalies (
    machine_id,
    detected_at,
    anomaly_score,
    anomaly_type,
    description,
    event_id,
    event_time,
    anomaly_flag,
    model_name,
    model_version,
    model_config_hash,
    baseline_event_id_sha256,
    baseline_feature_data_sha256,
    vibration_mm_s,
    pressure_bar,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
)
SELECT
    machine_id,
    detected_at,
    anomaly_score,
    anomaly_type,
    description,
    event_id,
    event_time,
    anomaly_flag,
    model_name,
    model_version,
    model_config_hash,
    baseline_event_id_sha256,
    baseline_feature_data_sha256,
    vibration_mm_s,
    pressure_bar,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
FROM staging_telemetry_anomalies
ON CONFLICT (event_id, model_name, model_version, model_config_hash)
WHERE anomaly_type = 'telemetry_isolation_forest_score'
DO NOTHING;

COMMIT;
"""


def records_match(
    expected: AnomalyRecord,
    existing: AnomalyRecord,
    *,
    expected_machine_id: int | None = None,
    existing_machine_id: int | None = None,
) -> bool:
    if expected_machine_id is not None and existing_machine_id is not None:
        if expected_machine_id != existing_machine_id:
            return False
    return expected == existing


def conflicting_anomaly_fields(
    expected: AnomalyRecord,
    existing: AnomalyRecord,
    *,
    expected_machine_id: int,
    existing_machine_id: int,
) -> list[str]:
    fields: list[str] = []
    if expected_machine_id != existing_machine_id:
        fields.append("machine_id")
    for field in AnomalyRecord.__dataclass_fields__:
        if getattr(expected, field) != getattr(existing, field):
            fields.append(field)
    return fields


def summarize_anomaly_reuse(
    records: Sequence[AnomalyRecord],
    existing_by_identity: Mapping[tuple[str, str, str, str], ExistingAnomalyRow],
    machine_ids_by_code: Mapping[str, int],
) -> AnomalyReuseSummary:
    new_records = 0
    reused_records = 0
    conflicts: list[ConflictDetail] = []
    for record in records:
        identity = record.identity.as_tuple()
        existing = existing_by_identity.get(identity)
        if existing is None:
            new_records += 1
            continue
        expected_machine_id = machine_ids_by_code[record.machine_code]
        if records_match(
            record,
            existing.record,
            expected_machine_id=expected_machine_id,
            existing_machine_id=existing.machine_id,
        ):
            reused_records += 1
            continue
        conflicts.append(
            ConflictDetail(
                identity=record.identity,
                fields=tuple(
                    conflicting_anomaly_fields(
                        record,
                        existing.record,
                        expected_machine_id=expected_machine_id,
                        existing_machine_id=existing.machine_id,
                    )
                ),
            )
        )
    return AnomalyReuseSummary(new_records, reused_records, tuple(conflicts))


def db_row_to_anomaly_record(row: Mapping[str, Any]) -> AnomalyRecord:
    return validate_anomaly_record(
        {
            "anomaly_flag": row["anomaly_flag"],
            "anomaly_score": row["anomaly_score"],
            "baseline_event_id_sha256": row["baseline_event_id_sha256"],
            "baseline_feature_data_sha256": row["baseline_feature_data_sha256"],
            "event_id": row["event_id"],
            "event_time": row["event_time"],
            "machine_code": row["machine_code"],
            "model_config_hash": row["model_config_hash"],
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "payload_sha256": row["payload_sha256"],
            "pressure_bar": row["pressure_bar"],
            "source_kafka_key": row["source_kafka_key"],
            "source_kafka_offset": row["source_kafka_offset"],
            "source_kafka_partition": row["source_kafka_partition"],
            "source_kafka_timestamp": row["source_kafka_timestamp"],
            "source_kafka_topic": row["source_kafka_topic"],
            "vibration_mm_s": row["vibration_mm_s"],
        }
    )


def db_row_to_existing_anomaly(row: Mapping[str, Any]) -> ExistingAnomalyRow:
    return ExistingAnomalyRow(
        anomaly_id=int(row["anomaly_id"]),
        machine_id=int(row["machine_id"]),
        record=db_row_to_anomaly_record(row),
    )


def parse_json_query_output(output: str) -> Any:
    stripped = output.strip()
    if not stripped:
        return []
    return json.loads(stripped)


def parse_count_output(output: str) -> int:
    stripped = output.strip()
    if not stripped:
        raise TelemetryAnomalyPersistenceError("Expected one count value, found empty output.")
    return int(stripped.splitlines()[0].strip())


def anomaly_state_summary_from_rows(
    rows: Sequence[ExistingAnomalyRow],
    *,
    duplicate_anomaly_identity_count: int,
    machine_reference_mismatch_count: int,
) -> AnomalyStateSummary:
    scores = [row.record.anomaly_score for row in rows]
    flag_count = sum(1 for row in rows if row.record.anomaly_flag)
    return AnomalyStateSummary(
        anomaly_row_count=len(rows),
        distinct_anomaly_event_count=len({row.record.event_id for row in rows}),
        distinct_machine_count_with_anomalies=len({row.machine_id for row in rows}),
        anomaly_flag_count=flag_count,
        non_anomaly_count=len(rows) - flag_count,
        min_anomaly_score=round(min(scores), 5) if scores else None,
        max_anomaly_score=round(max(scores), 5) if scores else None,
        mean_anomaly_score=round(float(fmean(scores)), 5) if scores else None,
        duplicate_anomaly_identity_count=duplicate_anomaly_identity_count,
        machine_reference_mismatch_count=machine_reference_mismatch_count,
    )


def build_static_summary() -> dict[str, Any]:
    return {
        "source_anomaly_path": "data/anomalies/telemetry_anomalies.jsonl",
        "target_anomaly_table": "anomalies",
        "stable_anomaly_identity": [
            "event_id",
            "model_name",
            "model_version",
            "model_config_hash",
        ],
        "model_identity_dimensions": [
            "model_name",
            "model_version",
            "model_config_hash",
        ],
        "idempotency_policy": "Identical anomaly identity is reused without duplicate rows.",
        "conflict_policy": (
            "Existing identical stable identity with different immutable values fails."
        ),
        "persistence_scope": "All scored telemetry anomaly outputs are persisted for audit.",
        "machine_lookup_policy": "machine_code must resolve to an existing machines row.",
        "runtime_counts": "intentionally excluded from tracked summary",
        "alerts_policy": "No alerts are created by this persistence phase.",
        "machine_health_policy": "machine_health is not updated by anomaly persistence.",
        "ai4i_policy": "AI4I model_predictions are not read or modified by anomaly persistence.",
    }


def write_static_summary(root: Path | None = None) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_static_summary(), indent=2, sort_keys=False) + "\n")
    return path

"""PostgreSQL persistence helpers for AI4I telemetry prediction records."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import UUID

from ml.inference import ai4i_predictor, ai4i_telemetry

PREDICTION_TYPE = "ai4i_failure_risk"
SUMMARY_RELATIVE_PATH = Path("reports") / "database" / "ai4i_prediction_persistence_summary.json"
REQUIRED_PREDICTION_FIELDS = (
    "adapter_version",
    "event_id",
    "event_time",
    "failure_prediction",
    "failure_probability",
    "final_config_hash",
    "frozen_threshold",
    "machine_code",
    "model_input_sha256",
    "model_name",
    "model_version",
    "payload_sha256",
    "source_kafka_key",
    "source_kafka_offset",
    "source_kafka_partition",
    "source_kafka_timestamp",
    "source_kafka_topic",
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AI4IPredictionPersistenceError(ValueError):
    """Raised when AI4I prediction persistence validation fails."""


@dataclass(frozen=True)
class PredictionIdentity:
    """Stable business identity for one model prediction."""

    event_id: str
    model_name: str
    model_version: str
    final_config_hash: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.event_id, self.model_name, self.model_version, self.final_config_hash)


@dataclass(frozen=True)
class PredictionRecord:
    """Validated AI4I telemetry prediction record."""

    adapter_version: str
    event_id: str
    event_time: str
    failure_prediction: bool
    failure_probability: float
    final_config_hash: str
    frozen_threshold: float
    machine_code: str
    model_input_sha256: str
    model_name: str
    model_version: str
    payload_sha256: str
    source_kafka_key: str
    source_kafka_offset: int
    source_kafka_partition: int
    source_kafka_timestamp: str
    source_kafka_topic: str

    @property
    def identity(self) -> PredictionIdentity:
        return PredictionIdentity(
            event_id=self.event_id,
            model_name=self.model_name,
            model_version=self.model_version,
            final_config_hash=self.final_config_hash,
        )

    def latest_order_key(self) -> tuple[Any, ...]:
        return (
            self.event_time,
            self.source_kafka_timestamp,
            self.source_kafka_topic,
            self.source_kafka_partition,
            self.source_kafka_offset,
            self.event_id,
        )


@dataclass(frozen=True)
class ExistingPredictionRow:
    """Existing database prediction row relevant to idempotency checks."""

    model_prediction_id: int
    machine_id: int
    record: PredictionRecord


@dataclass(frozen=True)
class MachineHealthProjection:
    """Latest prediction projection prepared for machine_health."""

    machine_id: int
    model_prediction_id: int
    record: PredictionRecord


@dataclass(frozen=True)
class ConflictDetail:
    """Material mismatch for an already persisted prediction identity."""

    identity: PredictionIdentity
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.identity.event_id,
            "fields": list(self.fields),
            "final_config_hash": self.identity.final_config_hash,
            "model_name": self.identity.model_name,
            "model_version": self.identity.model_version,
        }


@dataclass(frozen=True)
class PredictionReuseSummary:
    """Pure idempotency summary before persistence runs."""

    new_records: int
    existing_identical_records: int
    conflicts: tuple[ConflictDetail, ...]


@dataclass(frozen=True)
class HealthProjectionChangeSummary:
    """Projected machine_health row change summary."""

    inserted: int
    updated: int
    unchanged: int


@dataclass(frozen=True)
class PersistenceSummary:
    """Summary of one AI4I prediction persistence run."""

    input_prediction_records: int
    new_prediction_rows_inserted: int
    existing_identical_predictions_reused: int
    conflicting_predictions: int
    distinct_machines_in_batch: int
    machine_health_rows_inserted: int
    machine_health_rows_updated: int
    machine_health_rows_unchanged: int

    def to_dict(self) -> dict[str, int]:
        return {
            "conflicting_predictions": self.conflicting_predictions,
            "distinct_machines_in_batch": self.distinct_machines_in_batch,
            "existing_identical_predictions_reused": (self.existing_identical_predictions_reused),
            "input_prediction_records": self.input_prediction_records,
            "machine_health_rows_inserted": self.machine_health_rows_inserted,
            "machine_health_rows_unchanged": self.machine_health_rows_unchanged,
            "machine_health_rows_updated": self.machine_health_rows_updated,
            "new_prediction_rows_inserted": self.new_prediction_rows_inserted,
        }


@dataclass(frozen=True)
class PredictionStateSummary:
    """Read-only current-model prediction state summary."""

    prediction_row_count: int
    distinct_prediction_event_count: int
    distinct_machine_count_with_predictions: int
    positive_prediction_count: int
    negative_prediction_count: int
    min_failure_probability: float | None
    max_failure_probability: float | None
    mean_failure_probability: float | None
    machine_health_projection_count: int
    machine_health_prediction_mismatch_count: int
    machine_health_latest_event_mismatch_count: int
    duplicate_prediction_business_identity_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct_machine_count_with_predictions": (
                self.distinct_machine_count_with_predictions
            ),
            "distinct_prediction_event_count": self.distinct_prediction_event_count,
            "duplicate_prediction_business_identity_count": (
                self.duplicate_prediction_business_identity_count
            ),
            "machine_health_latest_event_mismatch_count": (
                self.machine_health_latest_event_mismatch_count
            ),
            "machine_health_prediction_mismatch_count": (
                self.machine_health_prediction_mismatch_count
            ),
            "machine_health_projection_count": self.machine_health_projection_count,
            "max_failure_probability": self.max_failure_probability,
            "mean_failure_probability": self.mean_failure_probability,
            "min_failure_probability": self.min_failure_probability,
            "negative_prediction_count": self.negative_prediction_count,
            "positive_prediction_count": self.positive_prediction_count,
            "prediction_row_count": self.prediction_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def prediction_output_path(root: Path | None = None) -> Path:
    return ai4i_telemetry.prediction_output_path(root or project_root())


def as_non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AI4IPredictionPersistenceError(f"{field_name} must be non-empty text.")
    return value


def as_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise AI4IPredictionPersistenceError(f"{field_name} must be numeric.")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AI4IPredictionPersistenceError(f"{field_name} must be numeric.") from exc
    if not Decimal("0") <= decimal_value <= Decimal("1"):
        raise AI4IPredictionPersistenceError(f"{field_name} must be in [0, 1].")
    return float(decimal_value)


def as_int_at_least(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise AI4IPredictionPersistenceError(f"{field_name} must be an integer.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise AI4IPredictionPersistenceError(f"{field_name} must be an integer.") from exc
    if int_value < minimum:
        raise AI4IPredictionPersistenceError(f"{field_name} must be >= {minimum}.")
    return int_value


def validate_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise AI4IPredictionPersistenceError(f"{field_name} must be a UUID string.") from exc


def validate_sha256(value: str, field_name: str) -> str:
    if HASH_PATTERN.fullmatch(value) is None:
        raise AI4IPredictionPersistenceError(f"{field_name} must be a lowercase SHA-256 hex.")
    return value


def normalize_timestamp_text(value: str, field_name: str) -> str:
    raw_value = as_non_empty_text(value, field_name)
    parseable = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise AI4IPredictionPersistenceError(
            f"{field_name} must be an ISO-like timestamp."
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def validate_prediction_record(record: Mapping[str, Any]) -> PredictionRecord:
    actual = set(record)
    expected = set(REQUIRED_PREDICTION_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AI4IPredictionPersistenceError(
            "Prediction record is missing field(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4IPredictionPersistenceError(
            "Prediction record has unexpected field(s): " + ", ".join(extra)
        )

    event_id = validate_uuid(as_non_empty_text(record["event_id"], "event_id"), "event_id")
    model_name = as_non_empty_text(record["model_name"], "model_name")
    model_version = as_non_empty_text(record["model_version"], "model_version")
    final_config_hash = validate_sha256(
        as_non_empty_text(record["final_config_hash"], "final_config_hash"),
        "final_config_hash",
    )
    failure_probability = as_probability(
        record["failure_probability"],
        "failure_probability",
    )
    frozen_threshold = as_probability(record["frozen_threshold"], "frozen_threshold")
    failure_prediction = record["failure_prediction"]
    if isinstance(failure_prediction, bool):
        prediction_bool = failure_prediction
    elif failure_prediction in {0, 1}:
        prediction_bool = bool(failure_prediction)
    else:
        raise AI4IPredictionPersistenceError("failure_prediction must be boolean-like.")
    if prediction_bool != (failure_probability >= frozen_threshold):
        raise AI4IPredictionPersistenceError(
            "failure_prediction is inconsistent with the frozen threshold."
        )

    if model_name != ai4i_predictor.MODEL_NAME:
        raise AI4IPredictionPersistenceError("Prediction model_name is not current AI4I model.")
    if model_version != ai4i_predictor.MODEL_VERSION:
        raise AI4IPredictionPersistenceError("Prediction model_version is not current AI4I model.")

    return PredictionRecord(
        adapter_version=as_non_empty_text(record["adapter_version"], "adapter_version"),
        event_id=event_id,
        event_time=normalize_timestamp_text(record["event_time"], "event_time"),
        failure_prediction=prediction_bool,
        failure_probability=failure_probability,
        final_config_hash=final_config_hash,
        frozen_threshold=frozen_threshold,
        machine_code=as_non_empty_text(record["machine_code"], "machine_code"),
        model_input_sha256=validate_sha256(
            as_non_empty_text(record["model_input_sha256"], "model_input_sha256"),
            "model_input_sha256",
        ),
        model_name=model_name,
        model_version=model_version,
        payload_sha256=validate_sha256(
            as_non_empty_text(record["payload_sha256"], "payload_sha256"),
            "payload_sha256",
        ),
        source_kafka_key=as_non_empty_text(record["source_kafka_key"], "source_kafka_key"),
        source_kafka_offset=as_int_at_least(
            record["source_kafka_offset"],
            "source_kafka_offset",
            0,
        ),
        source_kafka_partition=as_int_at_least(
            record["source_kafka_partition"],
            "source_kafka_partition",
            0,
        ),
        source_kafka_timestamp=normalize_timestamp_text(
            record["source_kafka_timestamp"],
            "source_kafka_timestamp",
        ),
        source_kafka_topic=as_non_empty_text(
            record["source_kafka_topic"],
            "source_kafka_topic",
        ),
    )


def validate_prediction_records(records: Sequence[Mapping[str, Any]]) -> list[PredictionRecord]:
    if not records:
        raise AI4IPredictionPersistenceError("Prediction file must contain at least one record.")
    validated = [validate_prediction_record(record) for record in records]
    identities = [record.identity.as_tuple() for record in validated]
    duplicate_identities = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicate_identities:
        first = duplicate_identities[0]
        raise AI4IPredictionPersistenceError(
            "Duplicate prediction business identity in input: " + "|".join(first)
        )
    event_ids = [record.event_id for record in validated]
    duplicate_events = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicate_events:
        raise AI4IPredictionPersistenceError(
            "Duplicate event_id in input prediction file: " + duplicate_events[0]
        )
    model_identities = {
        (record.model_name, record.model_version, record.final_config_hash) for record in validated
    }
    if len(model_identities) != 1:
        raise AI4IPredictionPersistenceError(
            "Prediction file must contain one internally consistent model identity."
        )
    return sorted(validated, key=latest_sort_key_ascending)


def load_prediction_records(path: Path) -> list[PredictionRecord]:
    if not path.exists():
        raise AI4IPredictionPersistenceError(
            f"Prediction file is missing: {path}. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\check_ai4i_telemetry_inference.py first."
        )
    raw_records = ai4i_telemetry.read_predictions_jsonl(path)
    return validate_prediction_records(raw_records)


def latest_sort_key_ascending(record: PredictionRecord) -> tuple[Any, ...]:
    return record.latest_order_key()


def newer_than(left: PredictionRecord, right: PredictionRecord) -> bool:
    return left.latest_order_key() > right.latest_order_key()


def latest_prediction_by_machine(
    records: Iterable[PredictionRecord],
) -> dict[str, PredictionRecord]:
    latest: dict[str, PredictionRecord] = {}
    for record in records:
        current = latest.get(record.machine_code)
        if current is None or newer_than(record, current):
            latest[record.machine_code] = record
    return latest


def latest_prediction_rows_by_machine(
    rows: Iterable[ExistingPredictionRow],
) -> dict[int, ExistingPredictionRow]:
    latest: dict[int, ExistingPredictionRow] = {}
    for row in rows:
        current = latest.get(row.machine_id)
        if current is None or newer_than(row.record, current.record):
            latest[row.machine_id] = row
    return latest


def prediction_records_match(
    expected: PredictionRecord,
    existing: PredictionRecord,
    *,
    expected_machine_id: int | None = None,
    existing_machine_id: int | None = None,
) -> bool:
    if expected_machine_id is not None and existing_machine_id is not None:
        if expected_machine_id != existing_machine_id:
            return False
    return expected == existing


def conflicting_prediction_fields(
    expected: PredictionRecord,
    existing: PredictionRecord,
    *,
    expected_machine_id: int,
    existing_machine_id: int,
) -> list[str]:
    fields: list[str] = []
    if expected_machine_id != existing_machine_id:
        fields.append("machine_id")
    for field in PredictionRecord.__dataclass_fields__:
        if getattr(expected, field) != getattr(existing, field):
            fields.append(field)
    return fields


def summarize_prediction_reuse(
    records: Sequence[PredictionRecord],
    existing_by_identity: Mapping[tuple[str, str, str, str], ExistingPredictionRow],
    machine_ids_by_code: Mapping[str, int],
) -> PredictionReuseSummary:
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
        if prediction_records_match(
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
                    conflicting_prediction_fields(
                        record,
                        existing.record,
                        expected_machine_id=expected_machine_id,
                        existing_machine_id=existing.machine_id,
                    )
                ),
            )
        )
    return PredictionReuseSummary(new_records, reused_records, tuple(conflicts))


def prepare_latest_projections(
    records: Sequence[PredictionRecord],
    machine_ids_by_code: Mapping[str, int],
    prediction_ids_by_identity: Mapping[tuple[str, str, str, str], int],
) -> list[MachineHealthProjection]:
    latest = latest_prediction_by_machine(records)
    projections: list[MachineHealthProjection] = []
    for machine_code, record in sorted(latest.items()):
        projections.append(
            MachineHealthProjection(
                machine_id=machine_ids_by_code[machine_code],
                model_prediction_id=prediction_ids_by_identity[record.identity.as_tuple()],
                record=record,
            )
        )
    return projections


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def nullable_sql_literal(value: str | None) -> str:
    return "NULL" if value is None else sql_literal(value)


def bool_sql_literal(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def numeric_sql_literal(value: float) -> str:
    return format(value, ".6f")


def prediction_values_sql(record: PredictionRecord, machine_id: int) -> str:
    return (
        f"({machine_id}, "
        f"{sql_literal(record.event_id)}::uuid, "
        f"{sql_literal(record.event_time)}::timestamptz, "
        f"{sql_literal(record.model_name)}, "
        f"{sql_literal(record.model_version)}, "
        f"{sql_literal(PREDICTION_TYPE)}, "
        f"{numeric_sql_literal(record.failure_probability)}, "
        f"{bool_sql_literal(record.failure_prediction)}, "
        f"{numeric_sql_literal(record.frozen_threshold)}, "
        f"{sql_literal(record.final_config_hash)}, "
        f"{sql_literal(record.adapter_version)}, "
        f"{sql_literal(record.model_input_sha256)}, "
        f"{sql_literal(record.source_kafka_topic)}, "
        f"{record.source_kafka_partition}, "
        f"{record.source_kafka_offset}, "
        f"{sql_literal(record.source_kafka_timestamp)}::timestamptz, "
        f"{sql_literal(record.source_kafka_key)}, "
        f"{sql_literal(record.payload_sha256)})"
    )


def projection_values_sql(projection: MachineHealthProjection) -> str:
    record = projection.record
    return (
        f"({projection.machine_id}, "
        f"{projection.model_prediction_id}, "
        f"{sql_literal(record.event_id)}::uuid, "
        f"{sql_literal(record.event_time)}::timestamptz, "
        f"{numeric_sql_literal(record.failure_probability)}, "
        f"{bool_sql_literal(record.failure_prediction)}, "
        f"{numeric_sql_literal(record.frozen_threshold)}, "
        f"{sql_literal(record.model_name)}, "
        f"{sql_literal(record.model_version)}, "
        f"{sql_literal(record.final_config_hash)}, "
        f"{sql_literal(record.model_input_sha256)}, "
        f"{sql_literal(record.source_kafka_topic)}, "
        f"{record.source_kafka_partition}, "
        f"{record.source_kafka_offset}, "
        f"{sql_literal(record.source_kafka_timestamp)}::timestamptz, "
        f"{sql_literal(record.source_kafka_key)}, "
        f"{sql_literal(record.payload_sha256)})"
    )


def identity_values_sql(records: Sequence[PredictionRecord]) -> str:
    return ",\n".join(
        (
            f"({sql_literal(record.event_id)}::uuid, "
            f"{sql_literal(record.model_name)}, "
            f"{sql_literal(record.model_version)}, "
            f"{sql_literal(record.final_config_hash)})"
        )
        for record in records
    )


def machine_code_values_sql(machine_codes: Sequence[str]) -> str:
    return ",\n".join(f"({sql_literal(machine_code)})" for machine_code in machine_codes)


def model_identity(records: Sequence[PredictionRecord]) -> tuple[str, str, str]:
    identities = {
        (record.model_name, record.model_version, record.final_config_hash) for record in records
    }
    if len(identities) != 1:
        raise AI4IPredictionPersistenceError("Prediction records must contain one model identity.")
    return next(iter(identities))


def build_machine_lookup_query(machine_codes: Sequence[str]) -> str:
    if not machine_codes:
        raise AI4IPredictionPersistenceError("At least one machine_code is required.")
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


def build_existing_predictions_query(records: Sequence[PredictionRecord]) -> str:
    if not records:
        raise AI4IPredictionPersistenceError("At least one prediction record is required.")
    values = identity_values_sql(records)
    return f"""
WITH wanted(event_id, model_name, model_version, final_config_hash) AS (
    VALUES
{values}
)
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'adapter_version', p.adapter_version,
            'event_id', p.event_id::text,
            'event_time', to_char(p.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'failure_prediction', p.failure_prediction,
            'failure_probability', p.failure_probability,
            'final_config_hash', p.final_config_hash,
            'frozen_threshold', p.frozen_threshold,
            'machine_code', m.machine_identifier,
            'machine_id', p.machine_id,
            'model_input_sha256', p.model_input_sha256,
            'model_name', p.model_name,
            'model_prediction_id', p.model_prediction_id,
            'model_version', p.model_version,
            'payload_sha256', p.payload_sha256,
            'source_kafka_key', p.source_kafka_key,
            'source_kafka_offset', p.source_kafka_offset,
            'source_kafka_partition', p.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(p.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', p.source_kafka_topic
        )
        ORDER BY p.model_prediction_id
    )::text,
    '[]'
)
FROM wanted w
JOIN model_predictions p
  ON p.event_id = w.event_id
 AND p.model_name = w.model_name
 AND p.model_version = w.model_version
 AND p.final_config_hash = w.final_config_hash
 AND p.prediction_type = {sql_literal(PREDICTION_TYPE)}
JOIN machines m
  ON m.machine_id = p.machine_id;
"""


def build_current_model_predictions_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
) -> str:
    return f"""
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'adapter_version', p.adapter_version,
            'event_id', p.event_id::text,
            'event_time', to_char(p.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'failure_prediction', p.failure_prediction,
            'failure_probability', p.failure_probability,
            'final_config_hash', p.final_config_hash,
            'frozen_threshold', p.frozen_threshold,
            'machine_code', m.machine_identifier,
            'machine_id', p.machine_id,
            'model_input_sha256', p.model_input_sha256,
            'model_name', p.model_name,
            'model_prediction_id', p.model_prediction_id,
            'model_version', p.model_version,
            'payload_sha256', p.payload_sha256,
            'source_kafka_key', p.source_kafka_key,
            'source_kafka_offset', p.source_kafka_offset,
            'source_kafka_partition', p.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(p.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', p.source_kafka_topic
        )
        ORDER BY
            p.event_time,
            p.source_kafka_timestamp,
            p.source_kafka_topic,
            p.source_kafka_partition,
            p.source_kafka_offset,
            p.event_id
    )::text,
    '[]'
)
FROM model_predictions p
JOIN machines m
  ON m.machine_id = p.machine_id
WHERE p.prediction_type = {sql_literal(PREDICTION_TYPE)}
  AND p.model_name = {sql_literal(model_name)}
  AND p.model_version = {sql_literal(model_version)}
  AND p.final_config_hash = {sql_literal(final_config_hash)};
"""


def build_machine_health_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
    machine_ids: Sequence[int] | None = None,
) -> str:
    machine_filter = ""
    if machine_ids is not None:
        if not machine_ids:
            raise AI4IPredictionPersistenceError("At least one machine_id is required.")
        machine_filter = "AND mh.machine_id IN (" + ", ".join(str(id_) for id_ in machine_ids) + ")"
    return f"""
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'latest_failure_prediction', mh.latest_failure_prediction,
            'latest_failure_probability', mh.latest_failure_probability,
            'latest_final_config_hash', mh.latest_final_config_hash,
            'latest_frozen_threshold', mh.latest_frozen_threshold,
            'latest_model_input_sha256', mh.latest_model_input_sha256,
            'latest_model_name', mh.latest_model_name,
            'latest_model_prediction_id', mh.latest_model_prediction_id,
            'latest_model_version', mh.latest_model_version,
            'latest_payload_sha256', mh.latest_payload_sha256,
            'latest_prediction_at',
                to_char(mh.latest_prediction_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'latest_prediction_event_id', mh.latest_prediction_event_id::text,
            'latest_source_kafka_key', mh.latest_source_kafka_key,
            'latest_source_kafka_offset', mh.latest_source_kafka_offset,
            'latest_source_kafka_partition', mh.latest_source_kafka_partition,
            'latest_source_kafka_timestamp',
                to_char(
                    mh.latest_source_kafka_timestamp AT TIME ZONE 'UTC',
                    'YYYY-MM-DD HH24:MI:SS.MS'
                ),
            'latest_source_kafka_topic', mh.latest_source_kafka_topic,
            'machine_code', m.machine_identifier,
            'machine_id', mh.machine_id
        )
        ORDER BY m.machine_identifier
    )::text,
    '[]'
)
FROM machine_health mh
JOIN machines m
  ON m.machine_id = mh.machine_id
WHERE mh.latest_model_name = {sql_literal(model_name)}
  AND mh.latest_model_version = {sql_literal(model_version)}
  AND mh.latest_final_config_hash = {sql_literal(final_config_hash)}
  {machine_filter};
"""


def build_duplicate_identity_count_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
) -> str:
    return f"""
WITH grouped AS (
    SELECT event_id, model_name, model_version, final_config_hash, count(*) AS row_count
    FROM model_predictions
    WHERE prediction_type = {sql_literal(PREDICTION_TYPE)}
      AND model_name = {sql_literal(model_name)}
      AND model_version = {sql_literal(model_version)}
      AND final_config_hash = {sql_literal(final_config_hash)}
    GROUP BY event_id, model_name, model_version, final_config_hash
)
SELECT count(*)
FROM grouped
WHERE row_count > 1;
"""


def build_persistence_transaction(
    records: Sequence[PredictionRecord],
    machine_ids_by_code: Mapping[str, int],
) -> str:
    if not records:
        raise AI4IPredictionPersistenceError("At least one prediction record is required.")
    missing_machines = sorted(
        {record.machine_code for record in records} - set(machine_ids_by_code)
    )
    if missing_machines:
        raise AI4IPredictionPersistenceError(
            "Prediction machine_code values are missing in PostgreSQL: "
            + ", ".join(missing_machines)
        )

    values = ",\n".join(
        prediction_values_sql(record, machine_ids_by_code[record.machine_code])
        for record in records
    )
    return f"""
BEGIN;

CREATE TEMP TABLE staging_ai4i_predictions (
    machine_id BIGINT NOT NULL,
    event_id UUID NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    failure_probability NUMERIC(10, 6) NOT NULL,
    failure_prediction BOOLEAN NOT NULL,
    frozen_threshold NUMERIC(10, 6) NOT NULL,
    final_config_hash TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    model_input_sha256 TEXT NOT NULL,
    source_kafka_topic TEXT NOT NULL,
    source_kafka_partition INTEGER NOT NULL,
    source_kafka_offset BIGINT NOT NULL,
    source_kafka_timestamp TIMESTAMPTZ NOT NULL,
    source_kafka_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
) ON COMMIT DROP;

INSERT INTO staging_ai4i_predictions (
    machine_id,
    event_id,
    event_time,
    model_name,
    model_version,
    prediction_type,
    failure_probability,
    failure_prediction,
    frozen_threshold,
    final_config_hash,
    adapter_version,
    model_input_sha256,
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
        FROM staging_ai4i_predictions s
        JOIN model_predictions p
          ON p.event_id = s.event_id
         AND p.model_name = s.model_name
         AND p.model_version = s.model_version
         AND p.final_config_hash = s.final_config_hash
         AND p.prediction_type = s.prediction_type
        WHERE p.machine_id <> s.machine_id
           OR p.event_time <> s.event_time
           OR p.failure_probability <> s.failure_probability
           OR p.failure_prediction <> s.failure_prediction
           OR p.frozen_threshold <> s.frozen_threshold
           OR p.adapter_version <> s.adapter_version
           OR p.model_input_sha256 <> s.model_input_sha256
           OR p.source_kafka_topic <> s.source_kafka_topic
           OR p.source_kafka_partition <> s.source_kafka_partition
           OR p.source_kafka_offset <> s.source_kafka_offset
           OR p.source_kafka_timestamp <> s.source_kafka_timestamp
           OR p.source_kafka_key <> s.source_kafka_key
           OR p.payload_sha256 <> s.payload_sha256
    ) THEN
        RAISE EXCEPTION 'Conflicting AI4I prediction identity already exists.';
    END IF;
END $$;

INSERT INTO model_predictions (
    machine_id,
    prediction_at,
    model_name,
    model_version,
    prediction_type,
    predicted_value,
    confidence,
    event_id,
    event_time,
    failure_probability,
    failure_prediction,
    frozen_threshold,
    final_config_hash,
    adapter_version,
    model_input_sha256,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
)
SELECT
    machine_id,
    event_time,
    model_name,
    model_version,
    prediction_type,
    failure_probability,
    failure_probability,
    event_id,
    event_time,
    failure_probability,
    failure_prediction,
    frozen_threshold,
    final_config_hash,
    adapter_version,
    model_input_sha256,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
FROM staging_ai4i_predictions
ON CONFLICT (event_id, model_name, model_version, final_config_hash)
WHERE prediction_type = 'ai4i_failure_risk'
DO NOTHING;

WITH current_model_predictions AS (
    SELECT DISTINCT p.*
    FROM model_predictions p
    JOIN staging_ai4i_predictions s
      ON s.machine_id = p.machine_id
     AND s.model_name = p.model_name
     AND s.model_version = p.model_version
     AND s.final_config_hash = p.final_config_hash
     AND s.prediction_type = p.prediction_type
),
ranked AS (
    SELECT
        current_model_predictions.*,
        ROW_NUMBER() OVER (
            PARTITION BY machine_id, model_name, model_version, final_config_hash
            ORDER BY
                event_time DESC,
                source_kafka_timestamp DESC,
                source_kafka_topic DESC,
                source_kafka_partition DESC,
                source_kafka_offset DESC,
                event_id DESC
        ) AS latest_rank
    FROM current_model_predictions
),
latest AS (
    SELECT *
    FROM ranked
    WHERE latest_rank = 1
)
INSERT INTO machine_health (
    machine_id,
    health_score,
    failure_risk,
    anomaly_score,
    health_classification,
    last_telemetry_at,
    latest_model_prediction_id,
    latest_prediction_event_id,
    latest_prediction_at,
    latest_failure_probability,
    latest_failure_prediction,
    latest_frozen_threshold,
    latest_model_name,
    latest_model_version,
    latest_final_config_hash,
    latest_model_input_sha256,
    latest_source_kafka_topic,
    latest_source_kafka_partition,
    latest_source_kafka_offset,
    latest_source_kafka_timestamp,
    latest_source_kafka_key,
    latest_payload_sha256
)
SELECT
    machine_id,
    NULL,
    failure_probability,
    NULL,
    'unknown',
    event_time,
    model_prediction_id,
    event_id,
    event_time,
    failure_probability,
    failure_prediction,
    frozen_threshold,
    model_name,
    model_version,
    final_config_hash,
    model_input_sha256,
    source_kafka_topic,
    source_kafka_partition,
    source_kafka_offset,
    source_kafka_timestamp,
    source_kafka_key,
    payload_sha256
FROM latest
ON CONFLICT (machine_id) DO UPDATE
SET
    failure_risk = EXCLUDED.failure_risk,
    last_telemetry_at = EXCLUDED.last_telemetry_at,
    latest_model_prediction_id = EXCLUDED.latest_model_prediction_id,
    latest_prediction_event_id = EXCLUDED.latest_prediction_event_id,
    latest_prediction_at = EXCLUDED.latest_prediction_at,
    latest_failure_probability = EXCLUDED.latest_failure_probability,
    latest_failure_prediction = EXCLUDED.latest_failure_prediction,
    latest_frozen_threshold = EXCLUDED.latest_frozen_threshold,
    latest_model_name = EXCLUDED.latest_model_name,
    latest_model_version = EXCLUDED.latest_model_version,
    latest_final_config_hash = EXCLUDED.latest_final_config_hash,
    latest_model_input_sha256 = EXCLUDED.latest_model_input_sha256,
    latest_source_kafka_topic = EXCLUDED.latest_source_kafka_topic,
    latest_source_kafka_partition = EXCLUDED.latest_source_kafka_partition,
    latest_source_kafka_offset = EXCLUDED.latest_source_kafka_offset,
    latest_source_kafka_timestamp = EXCLUDED.latest_source_kafka_timestamp,
    latest_source_kafka_key = EXCLUDED.latest_source_kafka_key,
    latest_payload_sha256 = EXCLUDED.latest_payload_sha256
WHERE machine_health.latest_prediction_at IS NULL
   OR (
    EXCLUDED.latest_prediction_at,
    EXCLUDED.latest_source_kafka_timestamp,
    EXCLUDED.latest_source_kafka_topic,
    EXCLUDED.latest_source_kafka_partition,
    EXCLUDED.latest_source_kafka_offset,
    EXCLUDED.latest_prediction_event_id
   ) > (
    machine_health.latest_prediction_at,
    machine_health.latest_source_kafka_timestamp,
    machine_health.latest_source_kafka_topic,
    machine_health.latest_source_kafka_partition,
    machine_health.latest_source_kafka_offset,
    machine_health.latest_prediction_event_id
);

COMMIT;
"""


def db_row_to_prediction_record(row: Mapping[str, Any]) -> PredictionRecord:
    return validate_prediction_record(
        {
            "adapter_version": row["adapter_version"],
            "event_id": row["event_id"],
            "event_time": row["event_time"],
            "failure_prediction": row["failure_prediction"],
            "failure_probability": row["failure_probability"],
            "final_config_hash": row["final_config_hash"],
            "frozen_threshold": row["frozen_threshold"],
            "machine_code": row["machine_code"],
            "model_input_sha256": row["model_input_sha256"],
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "payload_sha256": row["payload_sha256"],
            "source_kafka_key": row["source_kafka_key"],
            "source_kafka_offset": row["source_kafka_offset"],
            "source_kafka_partition": row["source_kafka_partition"],
            "source_kafka_timestamp": row["source_kafka_timestamp"],
            "source_kafka_topic": row["source_kafka_topic"],
        }
    )


def db_row_to_existing_prediction(row: Mapping[str, Any]) -> ExistingPredictionRow:
    return ExistingPredictionRow(
        model_prediction_id=int(row["model_prediction_id"]),
        machine_id=int(row["machine_id"]),
        record=db_row_to_prediction_record(row),
    )


def parse_json_query_output(output: str) -> Any:
    stripped = output.strip()
    if not stripped:
        return []
    return json.loads(stripped)


def parse_count_output(output: str) -> int:
    stripped = output.strip()
    if not stripped:
        raise AI4IPredictionPersistenceError("Expected one count value, found empty output.")
    return int(stripped.splitlines()[0].strip())


def projection_signature(row: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    if row is None:
        return None
    return (
        row.get("latest_model_prediction_id"),
        row.get("latest_prediction_event_id"),
        row.get("latest_prediction_at"),
        row.get("latest_failure_probability"),
        row.get("latest_failure_prediction"),
        row.get("latest_frozen_threshold"),
        row.get("latest_model_name"),
        row.get("latest_model_version"),
        row.get("latest_final_config_hash"),
        row.get("latest_model_input_sha256"),
        row.get("latest_source_kafka_topic"),
        row.get("latest_source_kafka_partition"),
        row.get("latest_source_kafka_offset"),
        row.get("latest_source_kafka_timestamp"),
        row.get("latest_source_kafka_key"),
        row.get("latest_payload_sha256"),
    )


def summarize_health_projection_changes(
    before_by_machine_id: Mapping[int, Mapping[str, Any]],
    after_by_machine_id: Mapping[int, Mapping[str, Any]],
    machine_ids: Iterable[int],
) -> HealthProjectionChangeSummary:
    inserted = 0
    updated = 0
    unchanged = 0
    for machine_id in set(machine_ids):
        before = before_by_machine_id.get(machine_id)
        after = after_by_machine_id.get(machine_id)
        if before is None and after is not None:
            inserted += 1
        elif before is not None and projection_signature(before) != projection_signature(after):
            updated += 1
        else:
            unchanged += 1
    return HealthProjectionChangeSummary(inserted, updated, unchanged)


def expected_projection_matches_health_row(
    expected: ExistingPredictionRow,
    health_row: Mapping[str, Any] | None,
) -> bool:
    if health_row is None:
        return False
    record = expected.record
    expected_values = {
        "latest_failure_prediction": record.failure_prediction,
        "latest_failure_probability": record.failure_probability,
        "latest_final_config_hash": record.final_config_hash,
        "latest_frozen_threshold": record.frozen_threshold,
        "latest_model_input_sha256": record.model_input_sha256,
        "latest_model_name": record.model_name,
        "latest_model_prediction_id": expected.model_prediction_id,
        "latest_model_version": record.model_version,
        "latest_payload_sha256": record.payload_sha256,
        "latest_prediction_at": record.event_time,
        "latest_prediction_event_id": record.event_id,
        "latest_source_kafka_key": record.source_kafka_key,
        "latest_source_kafka_offset": record.source_kafka_offset,
        "latest_source_kafka_partition": record.source_kafka_partition,
        "latest_source_kafka_timestamp": record.source_kafka_timestamp,
        "latest_source_kafka_topic": record.source_kafka_topic,
    }
    return all(health_row.get(key) == value for key, value in expected_values.items())


def count_projection_mismatches(
    expected_latest_by_machine_id: Mapping[int, ExistingPredictionRow],
    health_rows_by_machine_id: Mapping[int, Mapping[str, Any]],
) -> tuple[int, int]:
    prediction_mismatches = 0
    latest_event_mismatches = 0
    for machine_id, expected in expected_latest_by_machine_id.items():
        health_row = health_rows_by_machine_id.get(machine_id)
        if health_row is None:
            prediction_mismatches += 1
            latest_event_mismatches += 1
            continue
        if health_row.get("latest_prediction_event_id") != expected.record.event_id:
            latest_event_mismatches += 1
        if not expected_projection_matches_health_row(expected, health_row):
            prediction_mismatches += 1
    return prediction_mismatches, latest_event_mismatches


def prediction_state_summary_from_rows(
    rows: Sequence[ExistingPredictionRow],
    *,
    machine_health_projection_count: int,
    machine_health_prediction_mismatch_count: int,
    machine_health_latest_event_mismatch_count: int,
    duplicate_prediction_business_identity_count: int,
) -> PredictionStateSummary:
    probabilities = [row.record.failure_probability for row in rows]
    positive = sum(1 for row in rows if row.record.failure_prediction)
    return PredictionStateSummary(
        prediction_row_count=len(rows),
        distinct_prediction_event_count=len({row.record.event_id for row in rows}),
        distinct_machine_count_with_predictions=len({row.machine_id for row in rows}),
        positive_prediction_count=positive,
        negative_prediction_count=len(rows) - positive,
        min_failure_probability=round(min(probabilities), 6) if probabilities else None,
        max_failure_probability=round(max(probabilities), 6) if probabilities else None,
        mean_failure_probability=round(float(fmean(probabilities)), 6) if probabilities else None,
        machine_health_projection_count=machine_health_projection_count,
        machine_health_prediction_mismatch_count=machine_health_prediction_mismatch_count,
        machine_health_latest_event_mismatch_count=machine_health_latest_event_mismatch_count,
        duplicate_prediction_business_identity_count=duplicate_prediction_business_identity_count,
    )


def build_static_summary() -> dict[str, Any]:
    return {
        "source_prediction_path": "data/predictions/ai4i/telemetry_predictions.jsonl",
        "target_prediction_table": "model_predictions",
        "target_latest_projection_table": "machine_health",
        "stable_prediction_business_identity": [
            "event_id",
            "model_name",
            "model_version",
            "final_config_hash",
        ],
        "model_identity_dimensions": [
            "model_name",
            "model_version",
            "final_config_hash",
        ],
        "idempotency_policy": "Identical prediction identity is reused without duplicate rows.",
        "conflict_policy": (
            "Existing identical business identity with different immutable values fails."
        ),
        "latest_projection_ordering": [
            "event_time DESC",
            "source_kafka_timestamp DESC",
            "source_kafka_topic DESC",
            "source_kafka_partition DESC",
            "source_kafka_offset DESC",
            "event_id DESC",
        ],
        "transaction_policy": "Prediction history inserts and machine_health projection updates "
        "share one transaction.",
        "machine_lookup_policy": "machine_code must resolve to an existing machines row.",
        "runtime_counts": "intentionally excluded from tracked summary",
        "alerts_policy": "No alerts are created by this persistence phase.",
        "anomaly_policy": "No anomaly records are created by this persistence phase.",
        "model_execution_policy": "Persistence consumes existing prediction JSONL only.",
    }


def write_static_summary(root: Path | None = None) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_static_summary(), indent=2, sort_keys=False) + "\n")
    return path


def prediction_state_summary_from_records(
    records: Sequence[PredictionRecord],
    *,
    machine_health_projection_count: int,
    machine_health_prediction_mismatch_count: int,
    machine_health_latest_event_mismatch_count: int,
    duplicate_prediction_business_identity_count: int,
) -> PredictionStateSummary:
    probabilities = [record.failure_probability for record in records]
    positive = sum(1 for record in records if record.failure_prediction)
    return PredictionStateSummary(
        prediction_row_count=len(records),
        distinct_prediction_event_count=len({record.event_id for record in records}),
        distinct_machine_count_with_predictions=len({record.machine_code for record in records}),
        positive_prediction_count=positive,
        negative_prediction_count=len(records) - positive,
        min_failure_probability=round(min(probabilities), 6) if probabilities else None,
        max_failure_probability=round(max(probabilities), 6) if probabilities else None,
        mean_failure_probability=round(float(fmean(probabilities)), 6) if probabilities else None,
        machine_health_projection_count=machine_health_projection_count,
        machine_health_prediction_mismatch_count=machine_health_prediction_mismatch_count,
        machine_health_latest_event_mismatch_count=machine_health_latest_event_mismatch_count,
        duplicate_prediction_business_identity_count=duplicate_prediction_business_identity_count,
    )

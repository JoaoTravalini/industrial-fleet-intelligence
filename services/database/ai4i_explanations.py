"""PostgreSQL persistence helpers for operational AI4I explanation records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ml.explainability import ai4i_telemetry_shap
from services.database import ai4i_predictions

PREDICTION_TYPE = ai4i_predictions.PREDICTION_TYPE
SUMMARY_RELATIVE_PATH = Path("reports") / "database" / "ai4i_explanation_persistence_summary.json"


class AI4IExplanationPersistenceError(ValueError):
    """Raised when AI4I explanation persistence validation fails."""


@dataclass(frozen=True)
class PredictionLookupRow:
    """Persisted prediction row required before an explanation can be stored."""

    model_prediction_id: int
    machine_id: int
    machine_code: str
    event_id: str
    event_time: str
    failure_probability: float
    failure_prediction: bool
    frozen_threshold: float
    model_name: str
    model_version: str
    final_config_hash: str
    model_input_sha256: str

    @property
    def prediction_identity(self) -> tuple[str, str, str, str]:
        return (self.event_id, self.model_name, self.model_version, self.final_config_hash)


@dataclass(frozen=True)
class ExistingExplanationRow:
    """Existing persisted explanation row used for idempotency checks."""

    model_prediction_id: int
    machine_id: int
    record: ai4i_telemetry_shap.ExplanationRecord

    @property
    def db_stable_identity(self) -> tuple[int, str, str, str]:
        return (
            self.model_prediction_id,
            self.record.explainer_name,
            self.record.explainer_version,
            self.record.explanation_config_hash,
        )


@dataclass(frozen=True)
class ConflictDetail:
    """Material mismatch for an already persisted explanation identity."""

    identity: tuple[int, str, str, str]
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"fields": list(self.fields), "identity": list(self.identity)}


@dataclass(frozen=True)
class ExplanationReuseSummary:
    """Pure idempotency summary before persistence runs."""

    new_records: int
    existing_identical_records: int
    conflicts: tuple[ConflictDetail, ...]


@dataclass(frozen=True)
class PersistenceSummary:
    """Summary of one AI4I explanation persistence run."""

    input_explanations: int
    explanation_rows_inserted: int
    existing_identical_explanations_reused: int
    conflicting_explanations: int
    distinct_machines: int

    def to_dict(self) -> dict[str, int]:
        return {
            "conflicting_explanations": self.conflicting_explanations,
            "distinct_machines": self.distinct_machines,
            "existing_identical_explanations_reused": (self.existing_identical_explanations_reused),
            "explanation_rows_inserted": self.explanation_rows_inserted,
            "input_explanations": self.input_explanations,
        }


@dataclass(frozen=True)
class ExplanationStateSummary:
    """Read-only persisted explanation state summary."""

    explanation_row_count: int
    distinct_prediction_count: int
    distinct_machine_count: int
    duplicate_explanation_identity_count: int
    model_input_hash_mismatch_count: int
    max_additivity_error: float | None
    mean_additivity_error: float | None
    mean_absolute_contribution_by_feature: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct_machine_count": self.distinct_machine_count,
            "distinct_prediction_count": self.distinct_prediction_count,
            "duplicate_explanation_identity_count": self.duplicate_explanation_identity_count,
            "explanation_row_count": self.explanation_row_count,
            "max_additivity_error": self.max_additivity_error,
            "mean_absolute_contribution_by_feature": self.mean_absolute_contribution_by_feature,
            "mean_additivity_error": self.mean_additivity_error,
            "model_input_hash_mismatch_count": self.model_input_hash_mismatch_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def explanation_output_path(root: Path | None = None) -> Path:
    return ai4i_telemetry_shap.explanation_output_path(root or project_root())


def load_explanation_records(path: Path) -> list[ai4i_telemetry_shap.ExplanationRecord]:
    return ai4i_telemetry_shap.load_explanation_records(path)


def parse_json_query_output(output: str) -> Any:
    text = output.replace("\x00", "").strip()
    if not text:
        return []
    return json.loads(text)


def parse_count_output(output: str) -> int:
    lines = [line.strip() for line in output.replace("\x00", "").splitlines() if line.strip()]
    if not lines:
        raise AI4IExplanationPersistenceError("Expected one count value, found empty output.")
    return int(lines[0])


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def bool_sql_literal(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def numeric_sql(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AI4IExplanationPersistenceError("Numeric persistence value must be finite.")
    return repr(numeric)


def jsonb_sql(value: Any) -> str:
    return sql_literal(ai4i_telemetry_shap.canonical_json(value)) + "::jsonb"


def record_prediction_identity_values_sql(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
) -> str:
    return ",\n".join(
        (
            f"({sql_literal(record.event_id)}::uuid, "
            f"{sql_literal(record.model_name)}, "
            f"{sql_literal(record.model_version)}, "
            f"{sql_literal(record.final_config_hash)})"
        )
        for record in records
    )


def build_prediction_lookup_query(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
) -> str:
    if not records:
        raise AI4IExplanationPersistenceError("At least one explanation record is required.")
    values = record_prediction_identity_values_sql(records)
    return f"""
WITH wanted(event_id, model_name, model_version, final_config_hash) AS (
    VALUES
{values}
)
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
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
            'model_version', p.model_version
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


def db_row_to_prediction_lookup(row: Mapping[str, Any]) -> PredictionLookupRow:
    return PredictionLookupRow(
        model_prediction_id=int(row["model_prediction_id"]),
        machine_id=int(row["machine_id"]),
        machine_code=str(row["machine_code"]),
        event_id=ai4i_telemetry_shap.validate_uuid(str(row["event_id"]), "event_id"),
        event_time=ai4i_telemetry_shap.normalize_timestamp_text(row["event_time"], "event_time"),
        failure_probability=round(float(row["failure_probability"]), 6),
        failure_prediction=ai4i_telemetry_shap.as_bool(
            row["failure_prediction"],
            "failure_prediction",
        ),
        frozen_threshold=float(row["frozen_threshold"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        final_config_hash=ai4i_telemetry_shap.validate_sha256(
            str(row["final_config_hash"]),
            "final_config_hash",
        ),
        model_input_sha256=ai4i_telemetry_shap.validate_sha256(
            str(row["model_input_sha256"]),
            "model_input_sha256",
        ),
    )


def prediction_matches_explanation(
    prediction: PredictionLookupRow,
    explanation: ai4i_telemetry_shap.ExplanationRecord,
) -> bool:
    return (
        prediction.machine_code == explanation.machine_code
        and prediction.event_id == explanation.event_id
        and prediction.event_time == explanation.event_time
        and prediction.failure_probability == explanation.failure_probability
        and prediction.failure_prediction == explanation.failure_prediction
        and math.isclose(
            prediction.frozen_threshold,
            explanation.frozen_threshold,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and prediction.model_name == explanation.model_name
        and prediction.model_version == explanation.model_version
        and prediction.final_config_hash == explanation.final_config_hash
        and prediction.model_input_sha256 == explanation.model_input_sha256
    )


def validate_prediction_lookup(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    prediction_rows: Sequence[PredictionLookupRow],
) -> dict[tuple[str, str, str, str], PredictionLookupRow]:
    by_identity = {row.prediction_identity: row for row in prediction_rows}
    missing = [
        record.prediction_identity
        for record in records
        if record.prediction_identity not in by_identity
    ]
    if missing:
        raise AI4IExplanationPersistenceError(
            "Persisted model_predictions are missing for explanation event_id: " + missing[0][0]
        )
    for record in records:
        prediction = by_identity[record.prediction_identity]
        if not prediction_matches_explanation(prediction, record):
            raise AI4IExplanationPersistenceError(
                "Persisted prediction does not match explanation values for event_id "
                + record.event_id
            )
    return by_identity


def db_stable_identity(
    record: ai4i_telemetry_shap.ExplanationRecord,
    prediction: PredictionLookupRow,
) -> tuple[int, str, str, str]:
    return (
        prediction.model_prediction_id,
        record.explainer_name,
        record.explainer_version,
        record.explanation_config_hash,
    )


def stable_identity_values_sql(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    prediction_lookup: Mapping[tuple[str, str, str, str], PredictionLookupRow],
) -> str:
    return ",\n".join(
        (
            f"({prediction_lookup[record.prediction_identity].model_prediction_id}, "
            f"{sql_literal(record.explainer_name)}, "
            f"{sql_literal(record.explainer_version)}, "
            f"{sql_literal(record.explanation_config_hash)})"
        )
        for record in records
    )


def build_existing_explanations_query(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    prediction_lookup: Mapping[tuple[str, str, str, str], PredictionLookupRow],
) -> str:
    if not records:
        raise AI4IExplanationPersistenceError("At least one explanation record is required.")
    values = stable_identity_values_sql(records, prediction_lookup)
    return f"""
WITH wanted(model_prediction_id, explainer_name, explainer_version, explanation_config_hash) AS (
    VALUES
{values}
)
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'additivity_error', e.additivity_error,
            'attribution_semantics', e.attribution_semantics,
            'base_value', e.base_value,
            'contribution_sum', e.contribution_sum,
            'event_id', e.event_id::text,
            'event_time', to_char(e.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'explainer_name', e.explainer_name,
            'explainer_version', e.explainer_version,
            'explanation_config_hash', e.explanation_config_hash,
            'failure_prediction', p.failure_prediction,
            'failure_probability', p.failure_probability,
            'feature_contributions', e.feature_contributions,
            'final_config_hash', p.final_config_hash,
            'frozen_threshold', p.frozen_threshold,
            'machine_code', m.machine_identifier,
            'machine_id', e.machine_id,
            'model_input_sha256', e.model_input_sha256,
            'model_name', p.model_name,
            'model_output_value', e.model_output_value,
            'model_prediction_id', e.model_prediction_id,
            'model_version', p.model_version,
            'negative_contribution_semantics',
                'negative_shap_pushes_model_output_toward_lower_failure_risk',
            'output_semantics', e.output_semantics,
            'payload_sha256', p.payload_sha256,
            'positive_contribution_semantics',
                'positive_shap_pushes_model_output_toward_higher_failure_risk',
            'source_kafka_key', p.source_kafka_key,
            'source_kafka_offset', p.source_kafka_offset,
            'source_kafka_partition', p.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(p.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', p.source_kafka_topic
        )
        ORDER BY e.model_prediction_id
    )::text,
    '[]'
)
FROM wanted w
JOIN prediction_explanations e
  ON e.model_prediction_id = w.model_prediction_id
 AND e.explainer_name = w.explainer_name
 AND e.explainer_version = w.explainer_version
 AND e.explanation_config_hash = w.explanation_config_hash
JOIN model_predictions p
  ON p.model_prediction_id = e.model_prediction_id
JOIN machines m
  ON m.machine_id = e.machine_id;
"""


def db_row_to_explanation_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the runtime explanation fields from an enriched database row."""
    return {field: row[field] for field in ai4i_telemetry_shap.REQUIRED_EXPLANATION_FIELDS}


def db_row_to_existing_explanation(row: Mapping[str, Any]) -> ExistingExplanationRow:
    record = ai4i_telemetry_shap.validate_explanation_record(db_row_to_explanation_payload(row))
    return ExistingExplanationRow(
        model_prediction_id=int(row["model_prediction_id"]),
        machine_id=int(row["machine_id"]),
        record=record,
    )


def values_match(expected: Any, existing: Any) -> bool:
    if isinstance(expected, float) or isinstance(existing, float):
        return math.isclose(float(expected), float(existing), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(expected, list) and isinstance(existing, list):
        return ai4i_telemetry_shap.canonical_json(expected) == ai4i_telemetry_shap.canonical_json(
            existing
        )
    return expected == existing


def conflicting_fields(
    expected: ai4i_telemetry_shap.ExplanationRecord,
    existing: ai4i_telemetry_shap.ExplanationRecord,
    *,
    expected_machine_id: int,
    existing_machine_id: int,
) -> tuple[str, ...]:
    fields: list[str] = []
    if expected_machine_id != existing_machine_id:
        fields.append("machine_id")
    expected_dict = expected.to_dict()
    existing_dict = existing.to_dict()
    for field, expected_value in expected_dict.items():
        if not values_match(expected_value, existing_dict.get(field)):
            fields.append(field)
    return tuple(fields)


def summarize_explanation_reuse(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    existing_by_identity: Mapping[tuple[int, str, str, str], ExistingExplanationRow],
    prediction_lookup: Mapping[tuple[str, str, str, str], PredictionLookupRow],
) -> ExplanationReuseSummary:
    new_records = 0
    reused_records = 0
    conflicts: list[ConflictDetail] = []
    for record in records:
        prediction = prediction_lookup[record.prediction_identity]
        identity = db_stable_identity(record, prediction)
        existing = existing_by_identity.get(identity)
        if existing is None:
            new_records += 1
            continue
        fields = conflicting_fields(
            record,
            existing.record,
            expected_machine_id=prediction.machine_id,
            existing_machine_id=existing.machine_id,
        )
        if fields:
            conflicts.append(ConflictDetail(identity, fields))
        else:
            reused_records += 1
    return ExplanationReuseSummary(new_records, reused_records, tuple(conflicts))


def explanation_values_sql(
    record: ai4i_telemetry_shap.ExplanationRecord,
    prediction: PredictionLookupRow,
) -> str:
    return (
        f"({prediction.model_prediction_id}, "
        f"{prediction.machine_id}, "
        f"{sql_literal(record.event_id)}::uuid, "
        f"{sql_literal(record.event_time)}::timestamptz, "
        f"{sql_literal(record.model_input_sha256)}, "
        f"{sql_literal(record.explainer_name)}, "
        f"{sql_literal(record.explainer_version)}, "
        f"{sql_literal(record.explanation_config_hash)}, "
        f"{sql_literal(record.output_semantics)}, "
        f"{sql_literal(record.attribution_semantics)}, "
        f"{numeric_sql(record.base_value)}, "
        f"{numeric_sql(record.model_output_value)}, "
        f"{numeric_sql(record.contribution_sum)}, "
        f"{numeric_sql(record.additivity_error)}, "
        f"{jsonb_sql([item.to_dict() for item in record.feature_contributions])})"
    )


def build_persistence_transaction(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    prediction_lookup: Mapping[tuple[str, str, str, str], PredictionLookupRow],
) -> str:
    if not records:
        raise AI4IExplanationPersistenceError("At least one explanation record is required.")
    missing = [
        record.event_id for record in records if record.prediction_identity not in prediction_lookup
    ]
    if missing:
        raise AI4IExplanationPersistenceError(
            "Missing persisted prediction lookup for event_id: " + missing[0]
        )
    values = ",\n".join(
        explanation_values_sql(record, prediction_lookup[record.prediction_identity])
        for record in records
    )
    return f"""
BEGIN;

CREATE TEMP TABLE staging_ai4i_prediction_explanations (
    model_prediction_id BIGINT NOT NULL,
    machine_id BIGINT NOT NULL,
    event_id UUID NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    model_input_sha256 TEXT NOT NULL,
    explainer_name TEXT NOT NULL,
    explainer_version TEXT NOT NULL,
    explanation_config_hash TEXT NOT NULL,
    output_semantics TEXT NOT NULL,
    attribution_semantics TEXT NOT NULL,
    base_value NUMERIC NOT NULL,
    model_output_value NUMERIC NOT NULL,
    contribution_sum NUMERIC NOT NULL,
    additivity_error NUMERIC NOT NULL,
    feature_contributions JSONB NOT NULL
) ON COMMIT DROP;

INSERT INTO staging_ai4i_prediction_explanations (
    model_prediction_id,
    machine_id,
    event_id,
    event_time,
    model_input_sha256,
    explainer_name,
    explainer_version,
    explanation_config_hash,
    output_semantics,
    attribution_semantics,
    base_value,
    model_output_value,
    contribution_sum,
    additivity_error,
    feature_contributions
)
VALUES
{values};

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM staging_ai4i_prediction_explanations s
        JOIN model_predictions p
          ON p.model_prediction_id = s.model_prediction_id
        WHERE p.prediction_type <> {sql_literal(PREDICTION_TYPE)}
           OR p.machine_id <> s.machine_id
           OR p.event_id <> s.event_id
           OR p.event_time <> s.event_time
           OR p.model_input_sha256 <> s.model_input_sha256
    ) THEN
        RAISE EXCEPTION 'AI4I explanation does not match persisted prediction identity.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM staging_ai4i_prediction_explanations s
        JOIN prediction_explanations e
          ON e.model_prediction_id = s.model_prediction_id
         AND e.explainer_name = s.explainer_name
         AND e.explainer_version = s.explainer_version
         AND e.explanation_config_hash = s.explanation_config_hash
        WHERE e.machine_id <> s.machine_id
           OR e.event_id <> s.event_id
           OR e.event_time <> s.event_time
           OR e.model_input_sha256 <> s.model_input_sha256
           OR e.output_semantics <> s.output_semantics
           OR e.attribution_semantics <> s.attribution_semantics
           OR e.base_value <> s.base_value
           OR e.model_output_value <> s.model_output_value
           OR e.contribution_sum <> s.contribution_sum
           OR e.additivity_error <> s.additivity_error
           OR e.feature_contributions <> s.feature_contributions
    ) THEN
        RAISE EXCEPTION 'Conflicting AI4I explanation identity already exists.';
    END IF;
END $$;

INSERT INTO prediction_explanations (
    model_prediction_id,
    machine_id,
    event_id,
    event_time,
    model_input_sha256,
    explainer_name,
    explainer_version,
    explanation_config_hash,
    output_semantics,
    attribution_semantics,
    base_value,
    model_output_value,
    contribution_sum,
    additivity_error,
    feature_contributions
)
SELECT
    model_prediction_id,
    machine_id,
    event_id,
    event_time,
    model_input_sha256,
    explainer_name,
    explainer_version,
    explanation_config_hash,
    output_semantics,
    attribution_semantics,
    base_value,
    model_output_value,
    contribution_sum,
    additivity_error,
    feature_contributions
FROM staging_ai4i_prediction_explanations
ON CONFLICT (
    model_prediction_id,
    explainer_name,
    explainer_version,
    explanation_config_hash
)
DO NOTHING;

COMMIT;
"""


def build_current_explanations_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
    explanation_config_hash: str | None = None,
) -> str:
    explanation_filter = ""
    if explanation_config_hash is not None:
        explanation_filter = "AND e.explanation_config_hash = " + sql_literal(
            explanation_config_hash
        )
    return f"""
SELECT COALESCE(
    jsonb_agg(
        jsonb_build_object(
            'additivity_error', e.additivity_error,
            'attribution_semantics', e.attribution_semantics,
            'base_value', e.base_value,
            'contribution_sum', e.contribution_sum,
            'event_id', e.event_id::text,
            'event_time', to_char(e.event_time AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'explainer_name', e.explainer_name,
            'explainer_version', e.explainer_version,
            'explanation_config_hash', e.explanation_config_hash,
            'failure_prediction', p.failure_prediction,
            'failure_probability', p.failure_probability,
            'feature_contributions', e.feature_contributions,
            'final_config_hash', p.final_config_hash,
            'frozen_threshold', p.frozen_threshold,
            'machine_code', m.machine_identifier,
            'machine_id', e.machine_id,
            'model_input_sha256', e.model_input_sha256,
            'model_name', p.model_name,
            'model_output_value', e.model_output_value,
            'model_prediction_id', e.model_prediction_id,
            'model_version', p.model_version,
            'negative_contribution_semantics',
                'negative_shap_pushes_model_output_toward_lower_failure_risk',
            'output_semantics', e.output_semantics,
            'payload_sha256', p.payload_sha256,
            'positive_contribution_semantics',
                'positive_shap_pushes_model_output_toward_higher_failure_risk',
            'source_kafka_key', p.source_kafka_key,
            'source_kafka_offset', p.source_kafka_offset,
            'source_kafka_partition', p.source_kafka_partition,
            'source_kafka_timestamp',
                to_char(p.source_kafka_timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS'),
            'source_kafka_topic', p.source_kafka_topic
        )
        ORDER BY e.event_time, m.machine_identifier, e.event_id
    )::text,
    '[]'
)
FROM prediction_explanations e
JOIN model_predictions p
  ON p.model_prediction_id = e.model_prediction_id
JOIN machines m
  ON m.machine_id = e.machine_id
WHERE p.prediction_type = {sql_literal(PREDICTION_TYPE)}
  AND p.model_name = {sql_literal(model_name)}
  AND p.model_version = {sql_literal(model_version)}
  AND p.final_config_hash = {sql_literal(final_config_hash)}
  {explanation_filter};
"""


def build_duplicate_identity_count_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
    explanation_config_hash: str | None = None,
) -> str:
    explanation_filter = ""
    if explanation_config_hash is not None:
        explanation_filter = "AND e.explanation_config_hash = " + sql_literal(
            explanation_config_hash
        )
    return f"""
WITH grouped AS (
    SELECT
        e.model_prediction_id,
        e.explainer_name,
        e.explainer_version,
        e.explanation_config_hash,
        count(*) AS row_count
    FROM prediction_explanations e
    JOIN model_predictions p
      ON p.model_prediction_id = e.model_prediction_id
    WHERE p.prediction_type = {sql_literal(PREDICTION_TYPE)}
      AND p.model_name = {sql_literal(model_name)}
      AND p.model_version = {sql_literal(model_version)}
      AND p.final_config_hash = {sql_literal(final_config_hash)}
      {explanation_filter}
    GROUP BY
        e.model_prediction_id,
        e.explainer_name,
        e.explainer_version,
        e.explanation_config_hash
)
SELECT count(*)
FROM grouped
WHERE row_count > 1;
"""


def build_model_input_hash_mismatch_count_query(
    model_name: str,
    model_version: str,
    final_config_hash: str,
    explanation_config_hash: str | None = None,
) -> str:
    explanation_filter = ""
    if explanation_config_hash is not None:
        explanation_filter = "AND e.explanation_config_hash = " + sql_literal(
            explanation_config_hash
        )
    return f"""
SELECT count(*)
FROM prediction_explanations e
JOIN model_predictions p
  ON p.model_prediction_id = e.model_prediction_id
WHERE p.prediction_type = {sql_literal(PREDICTION_TYPE)}
  AND p.model_name = {sql_literal(model_name)}
  AND p.model_version = {sql_literal(model_version)}
  AND p.final_config_hash = {sql_literal(final_config_hash)}
  AND e.model_input_sha256 <> p.model_input_sha256
  {explanation_filter};
"""


def explanation_state_summary_from_rows(
    rows: Sequence[ExistingExplanationRow],
    *,
    duplicate_explanation_identity_count: int,
    model_input_hash_mismatch_count: int,
) -> ExplanationStateSummary:
    additivity_errors = [row.record.additivity_error for row in rows]
    return ExplanationStateSummary(
        explanation_row_count=len(rows),
        distinct_prediction_count=len({row.model_prediction_id for row in rows}),
        distinct_machine_count=len({row.machine_id for row in rows}),
        duplicate_explanation_identity_count=duplicate_explanation_identity_count,
        model_input_hash_mismatch_count=model_input_hash_mismatch_count,
        max_additivity_error=(
            ai4i_telemetry_shap.rounded_float(max(additivity_errors)) if additivity_errors else None
        ),
        mean_additivity_error=(
            ai4i_telemetry_shap.rounded_float(fmean(additivity_errors))
            if additivity_errors
            else None
        ),
        mean_absolute_contribution_by_feature=(
            ai4i_telemetry_shap.mean_absolute_contribution_by_feature([row.record for row in rows])
            if rows
            else {feature: 0.0 for feature in ai4i_telemetry_shap.EXPECTED_SEMANTIC_FEATURES}
        ),
    )


def build_static_summary() -> dict[str, Any]:
    return {
        "conflict_policy": "Identical stable identities with different immutable values fail.",
        "runtime_explanation_path": "data/explanations/ai4i/telemetry_explanations.jsonl",
        "stable_explanation_identity": [
            "model_prediction_id",
            "explainer_name",
            "explainer_version",
            "explanation_config_hash",
        ],
        "target_table": "prediction_explanations",
        "transactional": True,
    }


def write_static_summary(root: Path | None = None) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_static_summary(), indent=2, sort_keys=True) + "\n")
    return path

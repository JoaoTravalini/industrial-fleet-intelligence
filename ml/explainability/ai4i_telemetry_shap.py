"""Operational SHAP materialization for persisted AI4I telemetry predictions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import shap

from ml.explainability import ai4i_shap
from ml.inference import ai4i_predictor, ai4i_telemetry

EXPLANATION_OUTPUT_RELATIVE_PATH = (
    Path("data") / "explanations" / "ai4i" / "telemetry_explanations.jsonl"
)
STATIC_SUMMARY_RELATIVE_PATH = (
    Path("reports") / "explainability" / "operational_ai4i_explainability_summary.json"
)
EXPLAINER_NAME = "shap.TreeExplainer"
EXPECTED_SEMANTIC_FEATURES = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
OUTPUT_SEMANTICS = "positive_class_failure_risk_model_output"
ATTRIBUTION_SEMANTICS = "shap_model_attribution_not_causality"
POSITIVE_CONTRIBUTION_SEMANTICS = "positive_shap_pushes_model_output_toward_higher_failure_risk"
NEGATIVE_CONTRIBUTION_SEMANTICS = "negative_shap_pushes_model_output_toward_lower_failure_risk"
PREDICTION_OUTPUT_TOLERANCE = 1e-6
HASH_PATTERN_LENGTH = 64
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
REQUIRED_EXPLANATION_FIELDS = (
    "additivity_error",
    "attribution_semantics",
    "base_value",
    "contribution_sum",
    "event_id",
    "event_time",
    "explainer_name",
    "explainer_version",
    "explanation_config_hash",
    "failure_prediction",
    "failure_probability",
    "feature_contributions",
    "final_config_hash",
    "frozen_threshold",
    "machine_code",
    "model_input_sha256",
    "model_name",
    "model_output_value",
    "model_version",
    "negative_contribution_semantics",
    "output_semantics",
    "payload_sha256",
    "positive_contribution_semantics",
    "source_kafka_key",
    "source_kafka_offset",
    "source_kafka_partition",
    "source_kafka_timestamp",
    "source_kafka_topic",
)


class AI4ITelemetryExplainabilityError(ValueError):
    """Raised when operational AI4I explanation validation fails."""


@dataclass(frozen=True)
class OperationalPredictionRecord:
    """Validated persisted telemetry prediction source record."""

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
    def model_identity(self) -> tuple[str, str, str]:
        return (self.model_name, self.model_version, self.final_config_hash)

    @property
    def prediction_identity(self) -> tuple[str, str, str, str]:
        return (self.event_id, self.model_name, self.model_version, self.final_config_hash)

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


@dataclass(frozen=True)
class FeatureContribution:
    """One semantic AI4I feature attribution for one prediction."""

    feature_name: str
    feature_value: str | float
    shap_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
        }


@dataclass(frozen=True)
class ExplanationRecord:
    """Deterministic operational explanation record."""

    event_id: str
    machine_code: str
    event_time: str
    model_name: str
    model_version: str
    final_config_hash: str
    model_input_sha256: str
    failure_probability: float
    failure_prediction: bool
    frozen_threshold: float
    explainer_name: str
    explainer_version: str
    explanation_config_hash: str
    base_value: float
    model_output_value: float
    contribution_sum: float
    additivity_error: float
    feature_contributions: tuple[FeatureContribution, ...]
    output_semantics: str
    attribution_semantics: str
    positive_contribution_semantics: str
    negative_contribution_semantics: str
    source_kafka_topic: str
    source_kafka_partition: int
    source_kafka_offset: int
    source_kafka_timestamp: str
    source_kafka_key: str
    payload_sha256: str

    @property
    def prediction_identity(self) -> tuple[str, str, str, str]:
        return (self.event_id, self.model_name, self.model_version, self.final_config_hash)

    @property
    def stable_identity(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.event_id,
            self.model_name,
            self.model_version,
            self.final_config_hash,
            self.explainer_name,
            self.explanation_config_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "additivity_error": self.additivity_error,
            "attribution_semantics": self.attribution_semantics,
            "base_value": self.base_value,
            "contribution_sum": self.contribution_sum,
            "event_id": self.event_id,
            "event_time": self.event_time,
            "explainer_name": self.explainer_name,
            "explainer_version": self.explainer_version,
            "explanation_config_hash": self.explanation_config_hash,
            "failure_prediction": self.failure_prediction,
            "failure_probability": self.failure_probability,
            "feature_contributions": [item.to_dict() for item in self.feature_contributions],
            "final_config_hash": self.final_config_hash,
            "frozen_threshold": self.frozen_threshold,
            "machine_code": self.machine_code,
            "model_input_sha256": self.model_input_sha256,
            "model_name": self.model_name,
            "model_output_value": self.model_output_value,
            "model_version": self.model_version,
            "negative_contribution_semantics": self.negative_contribution_semantics,
            "output_semantics": self.output_semantics,
            "payload_sha256": self.payload_sha256,
            "positive_contribution_semantics": self.positive_contribution_semantics,
            "source_kafka_key": self.source_kafka_key,
            "source_kafka_offset": self.source_kafka_offset,
            "source_kafka_partition": self.source_kafka_partition,
            "source_kafka_timestamp": self.source_kafka_timestamp,
            "source_kafka_topic": self.source_kafka_topic,
        }


@dataclass(frozen=True)
class OperationalExplainabilitySummary:
    """Runtime summary for one operational explanation materialization run."""

    prediction_record_count: int
    explanation_record_count: int
    distinct_event_count: int
    max_additivity_error: float
    mean_absolute_contribution_by_feature: dict[str, float]
    output_path: Path
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct_event_count": self.distinct_event_count,
            "explanation_record_count": self.explanation_record_count,
            "max_additivity_error": self.max_additivity_error,
            "mean_absolute_contribution_by_feature": self.mean_absolute_contribution_by_feature,
            "output_path": self.output_path.as_posix(),
            "output_sha256": self.output_sha256,
            "prediction_record_count": self.prediction_record_count,
        }


@dataclass(frozen=True)
class OperationalExplainabilityResult:
    """Generated operational explanation records and summary."""

    prediction_records: list[OperationalPredictionRecord]
    adapter_records: list[dict[str, Any]]
    explanation_records: list[ExplanationRecord]
    summary: OperationalExplainabilitySummary


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def explanation_output_path(root: Path | None = None) -> Path:
    return (root or project_root()) / EXPLANATION_OUTPUT_RELATIVE_PATH


def static_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / STATIC_SUMMARY_RELATIVE_PATH


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be non-empty text.")
    return value


def as_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be numeric.")
    try:
        numeric_value = float(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(numeric_value):
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be finite.")
    return numeric_value


def as_probability(value: Any, field_name: str) -> float:
    numeric_value = as_finite_float(value, field_name)
    if not 0 <= numeric_value <= 1:
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be in [0, 1].")
    return numeric_value


def as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    raise AI4ITelemetryExplainabilityError(f"{field_name} must be boolean-like.")


def as_int_at_least(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be an integer.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be an integer.") from exc
    if int_value < minimum:
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be >= {minimum}.")
    return int_value


def validate_uuid(value: str, field_name: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be a UUID string.") from exc


def validate_sha256(value: str, field_name: str) -> str:
    if len(value) != HASH_PATTERN_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AI4ITelemetryExplainabilityError(f"{field_name} must be a lowercase SHA-256 hex.")
    return value


def normalize_timestamp_text(value: Any, field_name: str) -> str:
    raw_value = as_non_empty_text(value, field_name)
    parseable = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise AI4ITelemetryExplainabilityError(
            f"{field_name} must be an ISO-like timestamp."
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def rounded_float(value: Any, digits: int = 12) -> float:
    return ai4i_shap.rounded_float(as_finite_float(value, "numeric value"), digits=digits)


def validate_feature_contract(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(feature) for feature in feature_names)
    if names != EXPECTED_SEMANTIC_FEATURES:
        raise AI4ITelemetryExplainabilityError(
            "Operational explanations must expose exactly the six semantic AI4I features."
        )
    return names


def semantic_features(final_config: Mapping[str, Any]) -> tuple[str, ...]:
    return validate_feature_contract(ai4i_shap.expected_grouped_features(final_config))


def validate_prediction_record(record: Mapping[str, Any]) -> OperationalPredictionRecord:
    actual = set(record)
    expected = set(REQUIRED_PREDICTION_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AI4ITelemetryExplainabilityError(
            "Prediction record is missing field(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4ITelemetryExplainabilityError(
            "Prediction record has unexpected field(s): " + ", ".join(extra)
        )

    failure_probability = as_probability(record["failure_probability"], "failure_probability")
    frozen_threshold = as_probability(record["frozen_threshold"], "frozen_threshold")
    failure_prediction = as_bool(record["failure_prediction"], "failure_prediction")
    if failure_prediction != (failure_probability >= frozen_threshold):
        raise AI4ITelemetryExplainabilityError(
            "failure_prediction is inconsistent with the frozen threshold."
        )

    model_name = as_non_empty_text(record["model_name"], "model_name")
    model_version = as_non_empty_text(record["model_version"], "model_version")
    if model_name != ai4i_predictor.MODEL_NAME:
        raise AI4ITelemetryExplainabilityError("Prediction model_name is not current AI4I model.")
    if model_version != ai4i_predictor.MODEL_VERSION:
        raise AI4ITelemetryExplainabilityError(
            "Prediction model_version is not current AI4I model."
        )

    return OperationalPredictionRecord(
        adapter_version=as_non_empty_text(record["adapter_version"], "adapter_version"),
        event_id=validate_uuid(as_non_empty_text(record["event_id"], "event_id"), "event_id"),
        event_time=normalize_timestamp_text(record["event_time"], "event_time"),
        failure_prediction=failure_prediction,
        failure_probability=round(failure_probability, 6),
        final_config_hash=validate_sha256(
            as_non_empty_text(record["final_config_hash"], "final_config_hash"),
            "final_config_hash",
        ),
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


def validate_prediction_records(
    records: Sequence[Mapping[str, Any]],
) -> list[OperationalPredictionRecord]:
    if not records:
        raise AI4ITelemetryExplainabilityError("Prediction file must contain at least one record.")
    validated = [validate_prediction_record(record) for record in records]
    identities = [record.prediction_identity for record in validated]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise AI4ITelemetryExplainabilityError(
            "Duplicate prediction business identity in input: " + "|".join(duplicates[0])
        )
    event_ids = [record.event_id for record in validated]
    duplicate_events = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicate_events:
        raise AI4ITelemetryExplainabilityError(
            "Duplicate event_id in input prediction file: " + duplicate_events[0]
        )
    model_identities = {record.model_identity for record in validated}
    if len(model_identities) != 1:
        raise AI4ITelemetryExplainabilityError(
            "Prediction file must contain one internally consistent model identity."
        )
    return sorted(validated, key=lambda item: item.sort_key())


def load_prediction_records(path: Path) -> list[OperationalPredictionRecord]:
    if not path.exists():
        raise AI4ITelemetryExplainabilityError(
            f"Telemetry prediction file is missing: {path}. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\predict_silver_telemetry.py first."
        )
    raw_records = ai4i_telemetry.read_predictions_jsonl(path)
    return validate_prediction_records(raw_records)


def adapter_records_by_event_id(
    adapter_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    event_ids = [str(record["event_id"]) for record in adapter_records]
    duplicates = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicates:
        raise AI4ITelemetryExplainabilityError(
            "Duplicate adapter event_id in input: " + duplicates[0]
        )
    return {str(record["event_id"]): record for record in adapter_records}


def validate_event_identity_alignment(
    prediction_records: Sequence[OperationalPredictionRecord],
    adapter_records: Sequence[Mapping[str, Any]],
) -> None:
    prediction_ids = {record.event_id for record in prediction_records}
    adapter_ids = {str(record["event_id"]) for record in adapter_records}
    missing_adapter = sorted(prediction_ids - adapter_ids)
    missing_prediction = sorted(adapter_ids - prediction_ids)
    if missing_adapter or missing_prediction:
        details = []
        if missing_adapter:
            details.append(
                f"missing adapter records for {len(missing_adapter)} prediction event(s)"
            )
        if missing_prediction:
            details.append(f"adapter records without predictions: {len(missing_prediction)}")
        raise AI4ITelemetryExplainabilityError(
            "Event identity alignment failed: " + "; ".join(details)
        )


def validate_prediction_model_identity(
    prediction_records: Sequence[OperationalPredictionRecord],
    predictor: ai4i_predictor.AI4IPredictor,
) -> None:
    for record in prediction_records:
        if record.model_name != predictor.model_name:
            raise AI4ITelemetryExplainabilityError("Prediction model name differs from predictor.")
        if record.model_version != predictor.model_version:
            raise AI4ITelemetryExplainabilityError(
                "Prediction model version differs from predictor."
            )
        if record.final_config_hash != predictor.final_config_hash:
            raise AI4ITelemetryExplainabilityError("Prediction config hash differs from predictor.")
        if not math.isclose(
            record.frozen_threshold,
            float(predictor.decision_threshold),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AI4ITelemetryExplainabilityError("Prediction threshold differs from predictor.")


def validate_prediction_input_alignment(
    prediction: OperationalPredictionRecord,
    adapter_record: Mapping[str, Any],
    final_config: Mapping[str, Any],
) -> dict[str, Any]:
    if prediction.event_id != str(adapter_record["event_id"]):
        raise AI4ITelemetryExplainabilityError("Adapter event_id does not match prediction.")
    if prediction.machine_code != adapter_record["machine_code"]:
        raise AI4ITelemetryExplainabilityError("Adapter machine_code does not match prediction.")
    if prediction.event_time != normalize_timestamp_text(
        adapter_record["event_time"], "event_time"
    ):
        raise AI4ITelemetryExplainabilityError("Adapter event_time does not match prediction.")
    if prediction.adapter_version != adapter_record["adapter_version"]:
        raise AI4ITelemetryExplainabilityError("Adapter version does not match prediction.")

    lineage = adapter_record["source_lineage"]
    lineage_checks = {
        "source_kafka_key": prediction.source_kafka_key,
        "source_kafka_offset": prediction.source_kafka_offset,
        "source_kafka_partition": prediction.source_kafka_partition,
        "source_kafka_topic": prediction.source_kafka_topic,
        "payload_sha256": prediction.payload_sha256,
    }
    for field, expected in lineage_checks.items():
        if lineage[field] != expected:
            raise AI4ITelemetryExplainabilityError(
                f"Adapter lineage {field} does not match prediction."
            )
    if prediction.source_kafka_timestamp != normalize_timestamp_text(
        lineage["source_kafka_timestamp"],
        "source_kafka_timestamp",
    ):
        raise AI4ITelemetryExplainabilityError(
            "Adapter lineage source_kafka_timestamp does not match prediction."
        )

    model_input = adapter_record["model_input"]
    actual_hash = ai4i_telemetry.model_input_sha256(model_input, final_config)
    if actual_hash != prediction.model_input_sha256:
        raise AI4ITelemetryExplainabilityError(
            "model_input_sha256 mismatch for event_id " + prediction.event_id
        )
    return ai4i_predictor.validate_inference_record(model_input, final_config)


def aligned_model_inputs(
    prediction_records: Sequence[OperationalPredictionRecord],
    adapter_records: Sequence[Mapping[str, Any]],
    final_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    validate_event_identity_alignment(prediction_records, adapter_records)
    adapter_by_event = adapter_records_by_event_id(adapter_records)
    model_inputs = [
        validate_prediction_input_alignment(
            prediction, adapter_by_event[prediction.event_id], final_config
        )
        for prediction in prediction_records
    ]
    frame = pd.DataFrame(model_inputs, columns=ai4i_predictor.required_input_fields(final_config))
    return model_inputs, frame


def explanation_config_payload(
    predictor: ai4i_predictor.AI4IPredictor,
    semantic_feature_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "additivity_tolerance": ai4i_shap.ADDITIVITY_TOLERANCE,
        "attribution_semantics": ATTRIBUTION_SEMANTICS,
        "explained_output": ai4i_shap.EXPLAINED_OUTPUT,
        "explainer_name": EXPLAINER_NAME,
        "explainer_version": shap.__version__,
        "final_config_hash": predictor.final_config_hash,
        "grouping_policy": "sum one-hot Type contributions back to the semantic Type feature",
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
        "output_semantics": OUTPUT_SEMANTICS,
        "positive_class": ai4i_shap.POSITIVE_CLASS,
        "semantic_features": list(semantic_feature_names),
    }


def explanation_config_hash(
    predictor: ai4i_predictor.AI4IPredictor,
    semantic_feature_names: Sequence[str],
) -> str:
    return hashlib.sha256(
        canonical_json(explanation_config_payload(predictor, semantic_feature_names)).encode(
            "utf-8"
        )
    ).hexdigest()


def normalized_feature_value(value: Any) -> str | float:
    if isinstance(value, str):
        return value
    return rounded_float(value, digits=6)


def build_feature_contributions(
    semantic_feature_names: Sequence[str],
    model_input: Mapping[str, Any],
    shap_values: Sequence[float],
) -> tuple[FeatureContribution, ...]:
    validate_feature_contract(semantic_feature_names)
    if len(shap_values) != len(EXPECTED_SEMANTIC_FEATURES):
        raise AI4ITelemetryExplainabilityError("SHAP contribution count is not six.")
    rows = []
    for feature_name, shap_value in zip(semantic_feature_names, shap_values, strict=True):
        rows.append(
            FeatureContribution(
                feature_name=feature_name,
                feature_value=normalized_feature_value(model_input[feature_name]),
                shap_value=rounded_float(shap_value),
            )
        )
    validate_feature_contributions(rows)
    return tuple(rows)


def validate_feature_contributions(
    contributions: Sequence[FeatureContribution | Mapping[str, Any]],
) -> tuple[FeatureContribution, ...]:
    rows: list[FeatureContribution] = []
    for contribution in contributions:
        if isinstance(contribution, FeatureContribution):
            row = contribution
        else:
            actual = set(contribution)
            expected = {"feature_name", "feature_value", "shap_value"}
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                raise AI4ITelemetryExplainabilityError("Feature contribution fields are invalid.")
            feature_name = as_non_empty_text(contribution["feature_name"], "feature_name")
            if feature_name == "Type":
                feature_value: str | float = as_non_empty_text(
                    contribution["feature_value"],
                    "feature_value",
                )
            else:
                feature_value = rounded_float(contribution["feature_value"], digits=6)
            row = FeatureContribution(
                feature_name=feature_name,
                feature_value=feature_value,
                shap_value=rounded_float(contribution["shap_value"]),
            )
        rows.append(row)
    if tuple(row.feature_name for row in rows) != EXPECTED_SEMANTIC_FEATURES:
        raise AI4ITelemetryExplainabilityError(
            "Feature contributions must contain the six semantic features exactly once in order."
        )
    return tuple(rows)


def build_explanation_records(
    prediction_records: Sequence[OperationalPredictionRecord],
    model_inputs: Sequence[Mapping[str, Any]],
    grouped_feature_names: Sequence[str],
    grouped_values: np.ndarray,
    base_value: float,
    model_outputs: Sequence[float],
    additivity_errors: Sequence[float],
    config_hash: str,
) -> list[ExplanationRecord]:
    validate_feature_contract(grouped_feature_names)
    records: list[ExplanationRecord] = []
    for index, prediction in enumerate(prediction_records):
        model_output = rounded_float(model_outputs[index])
        if not math.isclose(
            model_output,
            prediction.failure_probability,
            rel_tol=0.0,
            abs_tol=PREDICTION_OUTPUT_TOLERANCE,
        ):
            raise AI4ITelemetryExplainabilityError(
                "SHAP model output does not match stored failure_probability for event_id "
                + prediction.event_id
            )
        contributions = build_feature_contributions(
            grouped_feature_names,
            model_inputs[index],
            grouped_values[index],
        )
        contribution_sum = rounded_float(sum(item.shap_value for item in contributions))
        additivity_error = rounded_float(additivity_errors[index])
        if additivity_error > ai4i_shap.ADDITIVITY_TOLERANCE:
            raise AI4ITelemetryExplainabilityError(
                "SHAP additivity error exceeds tolerance for event_id " + prediction.event_id
            )
        records.append(
            ExplanationRecord(
                event_id=prediction.event_id,
                machine_code=prediction.machine_code,
                event_time=prediction.event_time,
                model_name=prediction.model_name,
                model_version=prediction.model_version,
                final_config_hash=prediction.final_config_hash,
                model_input_sha256=prediction.model_input_sha256,
                failure_probability=prediction.failure_probability,
                failure_prediction=prediction.failure_prediction,
                frozen_threshold=prediction.frozen_threshold,
                explainer_name=EXPLAINER_NAME,
                explainer_version=shap.__version__,
                explanation_config_hash=config_hash,
                base_value=rounded_float(base_value),
                model_output_value=model_output,
                contribution_sum=contribution_sum,
                additivity_error=additivity_error,
                feature_contributions=contributions,
                output_semantics=OUTPUT_SEMANTICS,
                attribution_semantics=ATTRIBUTION_SEMANTICS,
                positive_contribution_semantics=POSITIVE_CONTRIBUTION_SEMANTICS,
                negative_contribution_semantics=NEGATIVE_CONTRIBUTION_SEMANTICS,
                source_kafka_topic=prediction.source_kafka_topic,
                source_kafka_partition=prediction.source_kafka_partition,
                source_kafka_offset=prediction.source_kafka_offset,
                source_kafka_timestamp=prediction.source_kafka_timestamp,
                source_kafka_key=prediction.source_kafka_key,
                payload_sha256=prediction.payload_sha256,
            )
        )
    return validate_explanation_records([record.to_dict() for record in records])


def validate_explanation_record(record: Mapping[str, Any]) -> ExplanationRecord:
    actual = set(record)
    expected = set(REQUIRED_EXPLANATION_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AI4ITelemetryExplainabilityError(
            "Explanation record is missing field(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4ITelemetryExplainabilityError(
            "Explanation record has unexpected field(s): " + ", ".join(extra)
        )

    failure_probability = as_probability(record["failure_probability"], "failure_probability")
    frozen_threshold = as_probability(record["frozen_threshold"], "frozen_threshold")
    failure_prediction = as_bool(record["failure_prediction"], "failure_prediction")
    if failure_prediction != (failure_probability >= frozen_threshold):
        raise AI4ITelemetryExplainabilityError(
            "failure_prediction is inconsistent with the frozen threshold."
        )

    base_value = rounded_float(record["base_value"])
    model_output_value = rounded_float(record["model_output_value"])
    contribution_sum = rounded_float(record["contribution_sum"])
    additivity_error = rounded_float(record["additivity_error"])
    contributions = validate_feature_contributions(record["feature_contributions"])
    expected_sum = rounded_float(sum(item.shap_value for item in contributions))
    if not math.isclose(contribution_sum, expected_sum, rel_tol=0.0, abs_tol=1e-12):
        raise AI4ITelemetryExplainabilityError("contribution_sum does not match SHAP values.")
    reconstructed_error = abs((base_value + contribution_sum) - model_output_value)
    if reconstructed_error > ai4i_shap.ADDITIVITY_TOLERANCE:
        raise AI4ITelemetryExplainabilityError("Explanation additivity exceeds tolerance.")
    if additivity_error > ai4i_shap.ADDITIVITY_TOLERANCE:
        raise AI4ITelemetryExplainabilityError("Stored additivity_error exceeds tolerance.")
    if not math.isclose(
        model_output_value,
        failure_probability,
        rel_tol=0.0,
        abs_tol=PREDICTION_OUTPUT_TOLERANCE,
    ):
        raise AI4ITelemetryExplainabilityError(
            "model_output_value does not match stored failure_probability."
        )

    if record["output_semantics"] != OUTPUT_SEMANTICS:
        raise AI4ITelemetryExplainabilityError("Unexpected output_semantics.")
    if record["attribution_semantics"] != ATTRIBUTION_SEMANTICS:
        raise AI4ITelemetryExplainabilityError("Unexpected attribution_semantics.")
    if record["positive_contribution_semantics"] != POSITIVE_CONTRIBUTION_SEMANTICS:
        raise AI4ITelemetryExplainabilityError("Unexpected positive contribution semantics.")
    if record["negative_contribution_semantics"] != NEGATIVE_CONTRIBUTION_SEMANTICS:
        raise AI4ITelemetryExplainabilityError("Unexpected negative contribution semantics.")

    model_name = as_non_empty_text(record["model_name"], "model_name")
    model_version = as_non_empty_text(record["model_version"], "model_version")
    if model_name != ai4i_predictor.MODEL_NAME:
        raise AI4ITelemetryExplainabilityError("Explanation model_name is not current AI4I model.")
    if model_version != ai4i_predictor.MODEL_VERSION:
        raise AI4ITelemetryExplainabilityError(
            "Explanation model_version is not current AI4I model."
        )

    return ExplanationRecord(
        event_id=validate_uuid(as_non_empty_text(record["event_id"], "event_id"), "event_id"),
        machine_code=as_non_empty_text(record["machine_code"], "machine_code"),
        event_time=normalize_timestamp_text(record["event_time"], "event_time"),
        model_name=model_name,
        model_version=model_version,
        final_config_hash=validate_sha256(
            as_non_empty_text(record["final_config_hash"], "final_config_hash"),
            "final_config_hash",
        ),
        model_input_sha256=validate_sha256(
            as_non_empty_text(record["model_input_sha256"], "model_input_sha256"),
            "model_input_sha256",
        ),
        failure_probability=round(failure_probability, 6),
        failure_prediction=failure_prediction,
        frozen_threshold=frozen_threshold,
        explainer_name=as_non_empty_text(record["explainer_name"], "explainer_name"),
        explainer_version=as_non_empty_text(record["explainer_version"], "explainer_version"),
        explanation_config_hash=validate_sha256(
            as_non_empty_text(record["explanation_config_hash"], "explanation_config_hash"),
            "explanation_config_hash",
        ),
        base_value=base_value,
        model_output_value=model_output_value,
        contribution_sum=contribution_sum,
        additivity_error=additivity_error,
        feature_contributions=contributions,
        output_semantics=OUTPUT_SEMANTICS,
        attribution_semantics=ATTRIBUTION_SEMANTICS,
        positive_contribution_semantics=POSITIVE_CONTRIBUTION_SEMANTICS,
        negative_contribution_semantics=NEGATIVE_CONTRIBUTION_SEMANTICS,
        source_kafka_topic=as_non_empty_text(
            record["source_kafka_topic"],
            "source_kafka_topic",
        ),
        source_kafka_partition=as_int_at_least(
            record["source_kafka_partition"],
            "source_kafka_partition",
            0,
        ),
        source_kafka_offset=as_int_at_least(
            record["source_kafka_offset"],
            "source_kafka_offset",
            0,
        ),
        source_kafka_timestamp=normalize_timestamp_text(
            record["source_kafka_timestamp"],
            "source_kafka_timestamp",
        ),
        source_kafka_key=as_non_empty_text(record["source_kafka_key"], "source_kafka_key"),
        payload_sha256=validate_sha256(
            as_non_empty_text(record["payload_sha256"], "payload_sha256"),
            "payload_sha256",
        ),
    )


def validate_explanation_records(records: Sequence[Mapping[str, Any]]) -> list[ExplanationRecord]:
    if not records:
        raise AI4ITelemetryExplainabilityError("Explanation file must contain at least one record.")
    validated = [validate_explanation_record(record) for record in records]
    identities = [record.stable_identity for record in validated]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise AI4ITelemetryExplainabilityError(
            "Duplicate explanation stable identity in input: " + "|".join(duplicates[0])
        )
    event_ids = [record.event_id for record in validated]
    duplicate_events = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicate_events:
        raise AI4ITelemetryExplainabilityError(
            "Duplicate event_id in explanation file: " + duplicate_events[0]
        )
    return sorted(validated, key=lambda item: (item.event_time, item.machine_code, item.event_id))


def explanation_record_json(record: ExplanationRecord) -> str:
    return canonical_json(record.to_dict())


def write_explanations_jsonl(records: Sequence[ExplanationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in sorted(
            records, key=lambda item: (item.event_time, item.machine_code, item.event_id)
        ):
            file.write(explanation_record_json(record))
            file.write("\n")


def read_explanations_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AI4ITelemetryExplainabilityError(f"Explanation output does not exist: {path}")
    return ai4i_telemetry.read_json_lines(path)


def load_explanation_records(path: Path) -> list[ExplanationRecord]:
    return validate_explanation_records(read_explanations_jsonl(path))


def mean_absolute_contribution_by_feature(
    records: Sequence[ExplanationRecord],
) -> dict[str, float]:
    values_by_feature: dict[str, list[float]] = {
        feature: [] for feature in EXPECTED_SEMANTIC_FEATURES
    }
    for record in records:
        for contribution in record.feature_contributions:
            values_by_feature[contribution.feature_name].append(abs(contribution.shap_value))
    return {
        feature: rounded_float(fmean(values)) if values else 0.0
        for feature, values in values_by_feature.items()
    }


def explanation_summary(
    prediction_records: Sequence[OperationalPredictionRecord],
    explanation_records: Sequence[ExplanationRecord],
    output_path: Path,
) -> OperationalExplainabilitySummary:
    if len(prediction_records) != len(explanation_records):
        raise AI4ITelemetryExplainabilityError("Explanation count does not match prediction count.")
    return OperationalExplainabilitySummary(
        prediction_record_count=len(prediction_records),
        explanation_record_count=len(explanation_records),
        distinct_event_count=len({record.event_id for record in explanation_records}),
        max_additivity_error=rounded_float(
            max((record.additivity_error for record in explanation_records), default=0.0)
        ),
        mean_absolute_contribution_by_feature=mean_absolute_contribution_by_feature(
            explanation_records
        ),
        output_path=output_path,
        output_sha256=file_sha256(output_path),
    )


def run_operational_explainability(
    root: Path | None = None,
    output_path: Path | None = None,
) -> OperationalExplainabilityResult:
    root_path = root or project_root()
    predictor = ai4i_shap.load_trusted_predictor(root_path)
    validate_feature_contract(semantic_features(predictor.final_config))
    prediction_records = load_prediction_records(ai4i_telemetry.prediction_output_path(root_path))
    validate_prediction_model_identity(prediction_records, predictor)
    adapter_config = ai4i_telemetry.load_adapter_config(root_path)
    adapter_records = ai4i_telemetry.load_adapter_records(
        root=root_path,
        config=adapter_config,
        final_config=predictor.final_config,
    )
    model_inputs, model_input_frame = aligned_model_inputs(
        prediction_records,
        adapter_records,
        predictor.final_config,
    )
    components = ai4i_shap.extract_model_components(predictor.pipeline, predictor.final_config)
    transformed = ai4i_shap.transform_model_inputs(components.preprocessor, model_input_frame)
    shap_result = ai4i_shap.explain_positive_class(
        components.classifier,
        transformed,
        components.transformed_feature_names,
    )
    grouped_features, grouped_values = ai4i_shap.grouped_contribution_matrix(
        components.transformed_feature_names,
        shap_result.values,
        predictor.final_config,
    )
    grouped_features = list(validate_feature_contract(grouped_features))
    grouped_additivity_errors = ai4i_shap.additivity_errors(
        grouped_values,
        shap_result.base_value,
        shap_result.model_outputs,
    )
    config_hash = explanation_config_hash(predictor, grouped_features)
    explanation_records = build_explanation_records(
        prediction_records,
        model_inputs,
        grouped_features,
        grouped_values,
        shap_result.base_value,
        shap_result.model_outputs,
        grouped_additivity_errors,
        config_hash,
    )
    target_path = output_path or explanation_output_path(root_path)
    write_explanations_jsonl(explanation_records, target_path)
    summary = explanation_summary(prediction_records, explanation_records, target_path)
    write_static_summary(root_path, final_config_hash=predictor.final_config_hash)
    return OperationalExplainabilityResult(
        prediction_records=list(prediction_records),
        adapter_records=list(adapter_records),
        explanation_records=explanation_records,
        summary=summary,
    )


def build_static_summary(final_config_hash: str | None = None) -> dict[str, Any]:
    return {
        "api_endpoint": "/api/v1/machines/{machine_code}/predictions/{event_id}/explanation",
        "explanation_method": EXPLAINER_NAME,
        "explanation_version_source": "Installed SHAP library version",
        "explanations_materialized_offline": True,
        "final_config_hash": final_config_hash,
        "model_identity": {
            "model_name": ai4i_predictor.MODEL_NAME,
            "model_version": ai4i_predictor.MODEL_VERSION,
        },
        "model_retraining_occurs": False,
        "output_semantics": OUTPUT_SEMANTICS,
        "persistence_table": "prediction_explanations",
        "runtime_explanation_path": EXPLANATION_OUTPUT_RELATIVE_PATH.as_posix(),
        "semantic_feature_list": list(EXPECTED_SEMANTIC_FEATURES),
        "shap_interpretation": (
            "Model attribution for the positive-class output, not physical causality."
        ),
        "source_prediction_path": ai4i_telemetry.PREDICTION_OUTPUT_RELATIVE_PATH.as_posix(),
        "stable_explanation_identity": [
            "model_prediction_id",
            "explainer_name",
            "explainer_version",
            "explanation_config_hash",
        ],
    }


def write_static_summary(root: Path | None = None, *, final_config_hash: str | None = None) -> Path:
    root_path = root or project_root()
    if final_config_hash is None:
        try:
            final_config_hash = ai4i_predictor.current_final_config_hash(
                ai4i_predictor.load_final_config(root_path)
            )
        except (OSError, ValueError):
            final_config_hash = None
    path = static_summary_path(root_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_static_summary(final_config_hash), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path

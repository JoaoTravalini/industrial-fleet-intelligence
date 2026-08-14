"""Unsupervised anomaly detection for Silver operational telemetry sensors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any
from uuid import UUID

CONFIG_RELATIVE_PATH = Path("ml") / "config" / "telemetry_anomaly_model.json"
BASELINE_SUMMARY_RELATIVE_PATH = (
    Path("reports") / "anomaly" / "telemetry_anomaly_baseline_summary.json"
)
STATIC_SUMMARY_RELATIVE_PATH = (
    Path("reports") / "anomaly" / "telemetry_anomaly_detection_summary.json"
)
ARTIFACT_RELATIVE_PATH = Path("ml") / "artifacts" / "anomaly" / "telemetry_isolation_forest.joblib"
ARTIFACT_METADATA_RELATIVE_PATH = Path("ml") / "artifacts" / "anomaly" / "artifact_metadata.json"
FEATURE_EXPORT_RELATIVE_PATH = Path("data") / "model_input" / "anomaly" / "telemetry"
ANOMALY_OUTPUT_RELATIVE_PATH = Path("data") / "anomalies" / "telemetry_anomalies.jsonl"

MODEL_NAME = "telemetry-isolation-forest"
MODEL_VERSION = "1.0.0"
ALGORITHM = "IsolationForest"
SOURCE_LAYER = "silver"
SOURCE_PATH = "data/silver/telemetry"
FEATURES = ("vibration_mm_s", "pressure_bar")
LINEAGE_FIELDS = (
    "source_kafka_topic",
    "source_kafka_partition",
    "source_kafka_offset",
    "source_kafka_timestamp",
    "source_kafka_key",
    "payload_sha256",
)
FEATURE_RECORD_FIELDS = (
    "event_id",
    "machine_code",
    "event_time",
    *FEATURES,
    *LINEAGE_FIELDS,
)
ANOMALY_OUTPUT_FIELDS = (
    "event_id",
    "machine_code",
    "event_time",
    *FEATURES,
    "anomaly_score",
    "anomaly_flag",
    "model_name",
    "model_version",
    "model_config_hash",
    "baseline_event_id_sha256",
    "baseline_feature_data_sha256",
    *LINEAGE_FIELDS,
)
FORBIDDEN_ANOMALY_FIELDS = (
    "Machine failure",
    "actual_failure",
    "failure_prediction",
    "failure_probability",
    "failure_probability_threshold",
    "ground_truth",
    "product_quality_type",
    "shap",
    "shap_values",
    "anomaly_probability",
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TelemetryAnomalyError(ValueError):
    """Raised when telemetry anomaly configuration or records are invalid."""


@dataclass(frozen=True)
class TelemetryAnomalyConfig:
    """Static configuration for telemetry anomaly model version 1."""

    model_name: str
    model_version: str
    algorithm: str
    features: tuple[str, ...]
    n_estimators: int
    contamination: str
    random_state: int
    n_jobs: int
    source_layer: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "contamination": self.contamination,
            "features": list(self.features),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "n_estimators": self.n_estimators,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "source_layer": self.source_layer,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class FeatureRecord:
    """Validated anomaly feature record extracted from canonical Silver telemetry."""

    event_id: str
    machine_code: str
    event_time: str
    vibration_mm_s: float
    pressure_bar: float
    source_kafka_topic: str
    source_kafka_partition: int
    source_kafka_offset: int
    source_kafka_timestamp: str
    source_kafka_key: str
    payload_sha256: str

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

    def feature_values(self) -> tuple[float, float]:
        return (self.vibration_mm_s, self.pressure_bar)

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "machine_code": self.machine_code,
            "event_time": self.event_time,
            "vibration_mm_s": self.vibration_mm_s,
            "pressure_bar": self.pressure_bar,
            "source_kafka_topic": self.source_kafka_topic,
            "source_kafka_partition": self.source_kafka_partition,
            "source_kafka_offset": self.source_kafka_offset,
            "source_kafka_timestamp": self.source_kafka_timestamp,
            "source_kafka_key": self.source_kafka_key,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class TrustedAnomalyArtifact:
    """Trusted local anomaly model artifact plus verified metadata."""

    model: Any
    metadata: dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def model_config_hash(self) -> str:
        return str(self.metadata["model_config_hash"])

    @property
    def baseline_event_id_sha256(self) -> str:
        return str(self.metadata["baseline_event_id_sha256"])

    @property
    def baseline_feature_data_sha256(self) -> str:
        return str(self.metadata["baseline_feature_data_sha256"])

    @property
    def score_reference_min_decision(self) -> float:
        return float(self.metadata["score_reference_min_decision"])

    @property
    def score_reference_max_decision(self) -> float:
        return float(self.metadata["score_reference_max_decision"])


@dataclass(frozen=True)
class ScoredAnomalyRecord:
    """Runtime anomaly output record with internal decision value for validation."""

    record: dict[str, Any]
    decision_score: float


@dataclass(frozen=True)
class TelemetryAnomalyPackagingSummary:
    """Summary of one anomaly baseline packaging run."""

    baseline_event_count: int
    baseline_machine_count: int
    feature_statistics: dict[str, dict[str, float]]
    baseline_event_id_sha256: str
    baseline_feature_data_sha256: str
    artifact_sha256: str
    model_name: str
    model_version: str
    model_config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "baseline_event_count": self.baseline_event_count,
            "baseline_event_id_sha256": self.baseline_event_id_sha256,
            "baseline_feature_data_sha256": self.baseline_feature_data_sha256,
            "baseline_machine_count": self.baseline_machine_count,
            "feature_statistics": self.feature_statistics,
            "model_config_hash": self.model_config_hash,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class TelemetryAnomalyScoringSummary:
    """Summary of one runtime anomaly scoring run."""

    scored_event_count: int
    distinct_machine_count: int
    anomaly_flag_count: int
    non_anomaly_count: int
    min_anomaly_score: float
    max_anomaly_score: float
    mean_anomaly_score: float
    model_name: str
    model_version: str
    model_config_hash: str
    output_path: Path
    output_sha256: str
    representatives: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_flag_count": self.anomaly_flag_count,
            "distinct_machine_count": self.distinct_machine_count,
            "max_anomaly_score": self.max_anomaly_score,
            "mean_anomaly_score": self.mean_anomaly_score,
            "min_anomaly_score": self.min_anomaly_score,
            "model_config_hash": self.model_config_hash,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "non_anomaly_count": self.non_anomaly_count,
            "output_path": self.output_path.as_posix(),
            "output_sha256": self.output_sha256,
            "representatives": self.representatives,
            "scored_event_count": self.scored_event_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def artifact_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ARTIFACT_RELATIVE_PATH


def artifact_metadata_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ARTIFACT_METADATA_RELATIVE_PATH


def baseline_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / BASELINE_SUMMARY_RELATIVE_PATH


def static_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / STATIC_SUMMARY_RELATIVE_PATH


def feature_export_path(root: Path | None = None) -> Path:
    return (root or project_root()) / FEATURE_EXPORT_RELATIVE_PATH


def anomaly_output_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ANOMALY_OUTPUT_RELATIVE_PATH


def parse_config(raw_config: Mapping[str, Any]) -> TelemetryAnomalyConfig:
    expected_keys = {
        "algorithm",
        "contamination",
        "features",
        "model_name",
        "model_version",
        "n_estimators",
        "n_jobs",
        "random_state",
        "source_layer",
        "source_path",
    }
    actual_keys = set(raw_config)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise TelemetryAnomalyError("Anomaly config is missing key(s): " + ", ".join(missing))
    if extra:
        raise TelemetryAnomalyError("Anomaly config has unexpected key(s): " + ", ".join(extra))

    features = raw_config["features"]
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise TelemetryAnomalyError("features must be a list of strings.")
    config = TelemetryAnomalyConfig(
        model_name=as_non_empty_text(raw_config["model_name"], "model_name"),
        model_version=as_non_empty_text(raw_config["model_version"], "model_version"),
        algorithm=as_non_empty_text(raw_config["algorithm"], "algorithm"),
        features=tuple(features),
        n_estimators=as_int_at_least(raw_config["n_estimators"], "n_estimators", 1),
        contamination=as_non_empty_text(raw_config["contamination"], "contamination"),
        random_state=as_int(raw_config["random_state"], "random_state"),
        n_jobs=as_int(raw_config["n_jobs"], "n_jobs"),
        source_layer=as_non_empty_text(raw_config["source_layer"], "source_layer"),
        source_path=as_non_empty_text(raw_config["source_path"], "source_path"),
    )
    validate_config(config)
    return config


def validate_config(config: TelemetryAnomalyConfig) -> None:
    if config.model_name != MODEL_NAME:
        raise TelemetryAnomalyError("model_name must be telemetry-isolation-forest.")
    if config.model_version != MODEL_VERSION:
        raise TelemetryAnomalyError("model_version must be 1.0.0.")
    if config.algorithm != ALGORITHM:
        raise TelemetryAnomalyError("algorithm must be IsolationForest.")
    if config.features != FEATURES:
        raise TelemetryAnomalyError(
            "Anomaly model v1 must use exactly vibration_mm_s and pressure_bar."
        )
    if config.n_estimators != 300:
        raise TelemetryAnomalyError("n_estimators must be 300.")
    if config.contamination != "auto":
        raise TelemetryAnomalyError("contamination must be auto.")
    if config.random_state != 42:
        raise TelemetryAnomalyError("random_state must be 42.")
    if config.n_jobs != 1:
        raise TelemetryAnomalyError("n_jobs must be 1.")
    if config.source_layer != SOURCE_LAYER:
        raise TelemetryAnomalyError("source_layer must be silver.")
    if config.source_path != SOURCE_PATH:
        raise TelemetryAnomalyError("source_path must be data/silver/telemetry.")


def load_config(root: Path | None = None) -> TelemetryAnomalyConfig:
    path = config_path(root)
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TelemetryAnomalyError(f"Anomaly config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TelemetryAnomalyError(f"Anomaly config is invalid JSON: {path}") from exc
    if not isinstance(raw_config, dict):
        raise TelemetryAnomalyError("Anomaly config must be a JSON object.")
    return parse_config(raw_config)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def model_config_hash(config: TelemetryAnomalyConfig) -> str:
    return sha256_text(canonical_json(config.to_dict()))


def as_non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TelemetryAnomalyError(f"{field_name} must be non-empty text.")
    return value


def as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TelemetryAnomalyError(f"{field_name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryAnomalyError(f"{field_name} must be an integer.") from exc


def as_int_at_least(value: Any, field_name: str, minimum: int) -> int:
    int_value = as_int(value, field_name)
    if int_value < minimum:
        raise TelemetryAnomalyError(f"{field_name} must be >= {minimum}.")
    return int_value


def as_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TelemetryAnomalyError(f"{field_name} must be numeric.")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TelemetryAnomalyError(f"{field_name} must be numeric.") from exc
    float_value = float(decimal_value)
    if not math.isfinite(float_value):
        raise TelemetryAnomalyError(f"{field_name} must be finite.")
    return float_value


def as_probability_score(value: Any, field_name: str) -> float:
    score = as_finite_float(value, field_name)
    if not 0 <= score <= 1:
        raise TelemetryAnomalyError(f"{field_name} must be in [0, 1].")
    return score


def validate_sha256(value: Any, field_name: str) -> str:
    text = as_non_empty_text(value, field_name)
    if HASH_PATTERN.fullmatch(text) is None:
        raise TelemetryAnomalyError(f"{field_name} must be a lowercase SHA-256 hex string.")
    return text


def validate_uuid_text(value: Any, field_name: str) -> str:
    text = as_non_empty_text(value, field_name)
    try:
        return str(UUID(text))
    except ValueError as exc:
        raise TelemetryAnomalyError(f"{field_name} must be a UUID string.") from exc


def normalize_timestamp_text(value: Any, field_name: str) -> str:
    raw_value = as_non_empty_text(value, field_name)
    parseable = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise TelemetryAnomalyError(f"{field_name} must be an ISO-like timestamp.") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def validate_record_field_set(
    record: Mapping[str, Any],
    expected_fields: Sequence[str],
    record_name: str,
) -> None:
    actual = set(record)
    expected = set(expected_fields)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise TelemetryAnomalyError(f"{record_name} is missing field(s): " + ", ".join(missing))
    if extra:
        raise TelemetryAnomalyError(f"{record_name} has unexpected field(s): " + ", ".join(extra))


def validate_no_forbidden_fields(record: Mapping[str, Any], record_name: str) -> None:
    forbidden = sorted(set(record) & set(FORBIDDEN_ANOMALY_FIELDS))
    if forbidden:
        raise TelemetryAnomalyError(
            f"{record_name} contains forbidden field(s): " + ", ".join(forbidden)
        )


def validate_feature_record(record: Mapping[str, Any]) -> FeatureRecord:
    validate_record_field_set(record, FEATURE_RECORD_FIELDS, "Feature record")
    validate_no_forbidden_fields(record, "Feature record")
    return FeatureRecord(
        event_id=validate_uuid_text(record["event_id"], "event_id"),
        machine_code=as_non_empty_text(record["machine_code"], "machine_code"),
        event_time=normalize_timestamp_text(record["event_time"], "event_time"),
        vibration_mm_s=as_finite_float(record["vibration_mm_s"], "vibration_mm_s"),
        pressure_bar=as_finite_float(record["pressure_bar"], "pressure_bar"),
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
        payload_sha256=validate_sha256(record["payload_sha256"], "payload_sha256"),
    )


def prepare_feature_records(records: Sequence[Mapping[str, Any]]) -> list[FeatureRecord]:
    if not records:
        raise TelemetryAnomalyError("At least one feature record is required.")
    validated = [validate_feature_record(record) for record in records]
    event_ids = [record.event_id for record in validated]
    duplicates = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicates:
        raise TelemetryAnomalyError("Feature records contain duplicate event_id: " + duplicates[0])
    return sorted(validated, key=lambda record: record.sort_key())


def discover_spark_json_part_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        raise TelemetryAnomalyError(f"Feature export directory does not exist: {path}")
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.name.startswith("part-") and item.suffix == ".json"
    )
    if not files:
        raise TelemetryAnomalyError(
            f"Feature export directory contains no part-*.json files: {path}"
        )
    return files


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TelemetryAnomalyError(
                    f"{path.name}:{line_number} is not valid JSON."
                ) from exc
            if not isinstance(parsed, dict):
                raise TelemetryAnomalyError(f"{path.name}:{line_number} is not a JSON object.")
            records.append(parsed)
    return records


def load_feature_records_from_export(root: Path | None = None) -> list[FeatureRecord]:
    raw_records: list[dict[str, Any]] = []
    for path in discover_spark_json_part_files(feature_export_path(root)):
        raw_records.extend(read_json_lines(path))
    return prepare_feature_records(raw_records)


def feature_matrix(records: Sequence[FeatureRecord]) -> Any:
    import numpy as np

    if not records:
        raise TelemetryAnomalyError("At least one feature record is required.")
    return np.array([record.feature_values() for record in records], dtype=float)


def fit_isolation_forest_model(
    config: TelemetryAnomalyConfig,
    records: Sequence[FeatureRecord],
) -> Any:
    from sklearn.ensemble import IsolationForest

    matrix = feature_matrix(records)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    return model.fit(matrix)


def decision_values(model: Any, records: Sequence[FeatureRecord]) -> list[float]:
    matrix = feature_matrix(records)
    values = [float(value) for value in model.decision_function(matrix)]
    if not all(math.isfinite(value) for value in values):
        raise TelemetryAnomalyError("Model decision values must be finite.")
    return values


def model_predictions(model: Any, records: Sequence[FeatureRecord]) -> list[int]:
    matrix = feature_matrix(records)
    predictions = [int(value) for value in model.predict(matrix)]
    unexpected = sorted({value for value in predictions if value not in {-1, 1}})
    if unexpected:
        raise TelemetryAnomalyError("IsolationForest returned unexpected decision labels.")
    return predictions


def score_reference_bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise TelemetryAnomalyError("At least one reference decision value is required.")
    minimum = min(values)
    maximum = max(values)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise TelemetryAnomalyError("Reference decision bounds must be finite.")
    if minimum == maximum:
        raise TelemetryAnomalyError("Reference decision bounds cannot be identical.")
    return minimum, maximum


def anomaly_score_from_decision(
    decision_score: float,
    *,
    reference_min_decision: float,
    reference_max_decision: float,
) -> float:
    if reference_min_decision >= reference_max_decision:
        raise TelemetryAnomalyError("Reference min decision must be less than max decision.")
    raw_score = (reference_max_decision - decision_score) / (
        reference_max_decision - reference_min_decision
    )
    return round(min(max(raw_score, 0.0), 1.0), 5)


def feature_statistics(records: Sequence[FeatureRecord]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field in FEATURES:
        values = [float(getattr(record, field)) for record in records]
        stats[field] = {
            "max": round(max(values), 6),
            "mean": round(float(fmean(values)), 6),
            "min": round(min(values), 6),
            "std": round(float(pstdev(values)), 6),
        }
    return stats


def baseline_hashes(records: Sequence[FeatureRecord]) -> tuple[str, str]:
    ordered = sorted(records, key=lambda record: record.sort_key())
    event_id_payload = [record.event_id for record in ordered]
    feature_payload = [
        {
            "event_id": record.event_id,
            "pressure_bar": record.pressure_bar,
            "vibration_mm_s": record.vibration_mm_s,
        }
        for record in ordered
    ]
    return (
        sha256_text(canonical_json(event_id_payload)),
        sha256_text(canonical_json(feature_payload)),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_bundle(
    *,
    model: Any,
    config: TelemetryAnomalyConfig,
    config_hash: str,
    baseline_event_id_sha256: str,
    baseline_feature_data_sha256: str,
    score_reference_min_decision: float,
    score_reference_max_decision: float,
) -> dict[str, Any]:
    return {
        "algorithm": config.algorithm,
        "baseline_event_id_sha256": baseline_event_id_sha256,
        "baseline_feature_data_sha256": baseline_feature_data_sha256,
        "features": list(config.features),
        "model": model,
        "model_config_hash": config_hash,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "score_reference_max_decision": score_reference_max_decision,
        "score_reference_min_decision": score_reference_min_decision,
    }


def build_artifact_metadata(
    *,
    artifact_sha256: str,
    baseline_event_count: int,
    baseline_machine_count: int,
    baseline_event_id_sha256: str,
    baseline_feature_data_sha256: str,
    config: TelemetryAnomalyConfig,
    config_hash: str,
    score_reference_min_decision: float,
    score_reference_max_decision: float,
) -> dict[str, Any]:
    return {
        "algorithm": config.algorithm,
        "artifact_path": ARTIFACT_RELATIVE_PATH.as_posix(),
        "artifact_sha256": artifact_sha256,
        "baseline_event_count": baseline_event_count,
        "baseline_event_id_sha256": baseline_event_id_sha256,
        "baseline_feature_data_sha256": baseline_feature_data_sha256,
        "baseline_has_anomaly_labels": False,
        "baseline_machine_count": baseline_machine_count,
        "features": list(config.features),
        "model_config_hash": config_hash,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "score_reference_max_decision": round(score_reference_max_decision, 12),
        "score_reference_min_decision": round(score_reference_min_decision, 12),
        "score_semantics": (
            "anomaly_score is a bounded monotonic transform of IsolationForest "
            "decision_function; higher means more anomalous and it is not a probability."
        ),
        "source_layer": config.source_layer,
        "source_path": config.source_path,
        "trusted_artifact_policy": "Load only this project-generated local joblib artifact.",
    }


def validate_artifact_metadata(metadata: Mapping[str, Any], config: TelemetryAnomalyConfig) -> None:
    required = {
        "algorithm",
        "artifact_path",
        "artifact_sha256",
        "baseline_event_count",
        "baseline_event_id_sha256",
        "baseline_feature_data_sha256",
        "baseline_has_anomaly_labels",
        "baseline_machine_count",
        "features",
        "model_config_hash",
        "model_name",
        "model_version",
        "score_reference_max_decision",
        "score_reference_min_decision",
        "source_layer",
        "source_path",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise TelemetryAnomalyError("Artifact metadata is missing key(s): " + ", ".join(missing))
    if metadata["model_name"] != config.model_name:
        raise TelemetryAnomalyError("Artifact model_name does not match config.")
    if metadata["model_version"] != config.model_version:
        raise TelemetryAnomalyError("Artifact model_version does not match config.")
    if tuple(metadata["features"]) != config.features:
        raise TelemetryAnomalyError("Artifact feature contract does not match config.")
    if metadata["algorithm"] != config.algorithm:
        raise TelemetryAnomalyError("Artifact algorithm does not match config.")
    if metadata["model_config_hash"] != model_config_hash(config):
        raise TelemetryAnomalyError("Artifact config hash does not match current config.")
    validate_sha256(metadata["artifact_sha256"], "artifact_sha256")
    validate_sha256(metadata["baseline_event_id_sha256"], "baseline_event_id_sha256")
    validate_sha256(metadata["baseline_feature_data_sha256"], "baseline_feature_data_sha256")
    min_decision = as_finite_float(
        metadata["score_reference_min_decision"],
        "score_reference_min_decision",
    )
    max_decision = as_finite_float(
        metadata["score_reference_max_decision"],
        "score_reference_max_decision",
    )
    if min_decision >= max_decision:
        raise TelemetryAnomalyError(
            "score_reference_min_decision must be less than score_reference_max_decision."
        )


def write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def build_baseline_summary(
    *,
    config: TelemetryAnomalyConfig,
    records: Sequence[FeatureRecord],
    config_hash: str,
    baseline_event_id_sha256: str,
    baseline_feature_data_sha256: str,
) -> dict[str, Any]:
    return {
        "algorithm": config.algorithm,
        "baseline_event_count": len(records),
        "baseline_event_id_sha256": baseline_event_id_sha256,
        "baseline_feature_data_sha256": baseline_feature_data_sha256,
        "baseline_has_anomaly_labels": False,
        "baseline_machine_count": len({record.machine_code for record in records}),
        "baseline_policy": (
            "Synthetic operational reference baseline from the current canonical Silver snapshot; "
            "not known healthy ground truth."
        ),
        "configuration": config.to_dict(),
        "feature_statistics": feature_statistics(records),
        "model_config_hash": config_hash,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "source_layer": config.source_layer,
        "source_path": config.source_path,
    }


def write_baseline_summary(
    *,
    root: Path | None,
    config: TelemetryAnomalyConfig,
    records: Sequence[FeatureRecord],
    config_hash: str,
    baseline_event_id_sha256: str,
    baseline_feature_data_sha256: str,
) -> Path:
    return write_json_file(
        baseline_summary_path(root),
        build_baseline_summary(
            config=config,
            records=records,
            config_hash=config_hash,
            baseline_event_id_sha256=baseline_event_id_sha256,
            baseline_feature_data_sha256=baseline_feature_data_sha256,
        ),
    )


def package_anomaly_artifact(
    records: Sequence[FeatureRecord],
    *,
    root: Path | None = None,
    config: TelemetryAnomalyConfig | None = None,
) -> TelemetryAnomalyPackagingSummary:
    root_path = root or project_root()
    active_config = config or load_config(root_path)
    prepared_records = prepare_feature_records([record.to_feature_dict() for record in records])
    config_hash = model_config_hash(active_config)
    event_hash, feature_hash = baseline_hashes(prepared_records)
    model = fit_isolation_forest_model(active_config, prepared_records)
    reference_decisions = decision_values(model, prepared_records)
    min_decision, max_decision = score_reference_bounds(reference_decisions)

    bundle = build_artifact_bundle(
        model=model,
        config=active_config,
        config_hash=config_hash,
        baseline_event_id_sha256=event_hash,
        baseline_feature_data_sha256=feature_hash,
        score_reference_min_decision=min_decision,
        score_reference_max_decision=max_decision,
    )
    artifact_file = artifact_path(root_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, artifact_file)
    artifact_hash = file_sha256(artifact_file)

    metadata = build_artifact_metadata(
        artifact_sha256=artifact_hash,
        baseline_event_count=len(prepared_records),
        baseline_machine_count=len({record.machine_code for record in prepared_records}),
        baseline_event_id_sha256=event_hash,
        baseline_feature_data_sha256=feature_hash,
        config=active_config,
        config_hash=config_hash,
        score_reference_min_decision=min_decision,
        score_reference_max_decision=max_decision,
    )
    write_json_file(artifact_metadata_path(root_path), metadata)
    write_baseline_summary(
        root=root_path,
        config=active_config,
        records=prepared_records,
        config_hash=config_hash,
        baseline_event_id_sha256=event_hash,
        baseline_feature_data_sha256=feature_hash,
    )
    write_static_summary(root_path)

    return TelemetryAnomalyPackagingSummary(
        baseline_event_count=len(prepared_records),
        baseline_machine_count=len({record.machine_code for record in prepared_records}),
        feature_statistics=feature_statistics(prepared_records),
        baseline_event_id_sha256=event_hash,
        baseline_feature_data_sha256=feature_hash,
        artifact_sha256=artifact_hash,
        model_name=active_config.model_name,
        model_version=active_config.model_version,
        model_config_hash=config_hash,
    )


def load_artifact_metadata(root: Path | None = None) -> dict[str, Any]:
    path = artifact_metadata_path(root)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TelemetryAnomalyError(f"Artifact metadata is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TelemetryAnomalyError(f"Artifact metadata is invalid JSON: {path}") from exc
    if not isinstance(metadata, dict):
        raise TelemetryAnomalyError("Artifact metadata must be a JSON object.")
    return metadata


def load_trusted_artifact(root: Path | None = None) -> TrustedAnomalyArtifact:
    root_path = root or project_root()
    config = load_config(root_path)
    metadata = load_artifact_metadata(root_path)
    validate_artifact_metadata(metadata, config)
    artifact_file = artifact_path(root_path)
    if not artifact_file.exists():
        raise TelemetryAnomalyError(
            "Packaged telemetry anomaly artifact is missing. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\package_telemetry_anomaly_model.py first."
        )
    actual_hash = file_sha256(artifact_file)
    if actual_hash != metadata["artifact_sha256"]:
        raise TelemetryAnomalyError("Telemetry anomaly artifact SHA-256 does not match metadata.")

    import joblib

    bundle = joblib.load(artifact_file)
    if not isinstance(bundle, dict):
        raise TelemetryAnomalyError("Telemetry anomaly artifact bundle is invalid.")
    for key in (
        "algorithm",
        "baseline_event_id_sha256",
        "baseline_feature_data_sha256",
        "features",
        "model",
        "model_config_hash",
        "model_name",
        "model_version",
        "score_reference_max_decision",
        "score_reference_min_decision",
    ):
        if key not in bundle:
            raise TelemetryAnomalyError(f"Telemetry anomaly artifact is missing {key}.")
    if bundle["model_name"] != metadata["model_name"]:
        raise TelemetryAnomalyError("Artifact bundle model_name does not match metadata.")
    if bundle["model_version"] != metadata["model_version"]:
        raise TelemetryAnomalyError("Artifact bundle model_version does not match metadata.")
    if bundle["model_config_hash"] != metadata["model_config_hash"]:
        raise TelemetryAnomalyError("Artifact bundle config hash does not match metadata.")
    if tuple(bundle["features"]) != tuple(metadata["features"]):
        raise TelemetryAnomalyError("Artifact bundle features do not match metadata.")
    return TrustedAnomalyArtifact(model=bundle["model"], metadata=dict(metadata))


def build_scored_record(
    *,
    source: FeatureRecord,
    anomaly_score: float,
    anomaly_flag: bool,
    artifact: TrustedAnomalyArtifact,
) -> dict[str, Any]:
    return {
        "event_id": source.event_id,
        "machine_code": source.machine_code,
        "event_time": source.event_time,
        "vibration_mm_s": source.vibration_mm_s,
        "pressure_bar": source.pressure_bar,
        "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag,
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "model_config_hash": artifact.model_config_hash,
        "baseline_event_id_sha256": artifact.baseline_event_id_sha256,
        "baseline_feature_data_sha256": artifact.baseline_feature_data_sha256,
        "source_kafka_topic": source.source_kafka_topic,
        "source_kafka_partition": source.source_kafka_partition,
        "source_kafka_offset": source.source_kafka_offset,
        "source_kafka_timestamp": source.source_kafka_timestamp,
        "source_kafka_key": source.source_kafka_key,
        "payload_sha256": source.payload_sha256,
    }


def score_feature_records(
    records: Sequence[FeatureRecord],
    artifact: TrustedAnomalyArtifact,
) -> list[ScoredAnomalyRecord]:
    prepared_records = prepare_feature_records([record.to_feature_dict() for record in records])
    decisions = decision_values(artifact.model, prepared_records)
    predictions = model_predictions(artifact.model, prepared_records)
    scored_records: list[ScoredAnomalyRecord] = []
    for source, decision_score, prediction in zip(
        prepared_records,
        decisions,
        predictions,
        strict=True,
    ):
        anomaly_score = anomaly_score_from_decision(
            decision_score,
            reference_min_decision=artifact.score_reference_min_decision,
            reference_max_decision=artifact.score_reference_max_decision,
        )
        anomaly_flag = prediction == -1
        scored_record = build_scored_record(
            source=source,
            anomaly_score=anomaly_score,
            anomaly_flag=anomaly_flag,
            artifact=artifact,
        )
        validate_anomaly_output_record(scored_record)
        scored_records.append(ScoredAnomalyRecord(scored_record, decision_score))
    return sorted(scored_records, key=lambda item: anomaly_record_sort_key(item.record))


def anomaly_record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["event_time"],
        record["machine_code"],
        record["event_id"],
        record["source_kafka_timestamp"],
        record["source_kafka_topic"],
        int(record["source_kafka_partition"]),
        int(record["source_kafka_offset"]),
        record["source_kafka_key"],
        record["payload_sha256"],
    )


def validate_anomaly_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_record_field_set(record, ANOMALY_OUTPUT_FIELDS, "Anomaly output record")
    validate_no_forbidden_fields(record, "Anomaly output record")
    feature = validate_feature_record({field: record[field] for field in FEATURE_RECORD_FIELDS})
    anomaly_score = as_probability_score(record["anomaly_score"], "anomaly_score")
    anomaly_flag = record["anomaly_flag"]
    if not isinstance(anomaly_flag, bool):
        raise TelemetryAnomalyError("anomaly_flag must be boolean.")
    return {
        **feature.to_feature_dict(),
        "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag,
        "model_name": as_non_empty_text(record["model_name"], "model_name"),
        "model_version": as_non_empty_text(record["model_version"], "model_version"),
        "model_config_hash": validate_sha256(record["model_config_hash"], "model_config_hash"),
        "baseline_event_id_sha256": validate_sha256(
            record["baseline_event_id_sha256"],
            "baseline_event_id_sha256",
        ),
        "baseline_feature_data_sha256": validate_sha256(
            record["baseline_feature_data_sha256"],
            "baseline_feature_data_sha256",
        ),
    }


def validate_anomaly_output_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise TelemetryAnomalyError("Anomaly output must contain at least one record.")
    validated = [validate_anomaly_output_record(record) for record in records]
    identities = [
        (
            str(record["event_id"]),
            str(record["model_name"]),
            str(record["model_version"]),
            str(record["model_config_hash"]),
        )
        for record in validated
    ]
    duplicate_identities = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicate_identities:
        raise TelemetryAnomalyError(
            "Duplicate anomaly stable identity in output: " + "|".join(duplicate_identities[0])
        )
    event_ids = [str(record["event_id"]) for record in validated]
    duplicate_events = [event_id for event_id, count in Counter(event_ids).items() if count > 1]
    if duplicate_events:
        raise TelemetryAnomalyError("Duplicate anomaly event_id in output: " + duplicate_events[0])
    model_identities = {
        (
            str(record["model_name"]),
            str(record["model_version"]),
            str(record["model_config_hash"]),
            str(record["baseline_event_id_sha256"]),
            str(record["baseline_feature_data_sha256"]),
        )
        for record in validated
    }
    if len(model_identities) != 1:
        raise TelemetryAnomalyError("Anomaly output must use one model and baseline identity.")
    return sorted(validated, key=anomaly_record_sort_key)


def anomaly_record_json(record: Mapping[str, Any]) -> str:
    validate_anomaly_output_record(record)
    ordered = {field: record[field] for field in ANOMALY_OUTPUT_FIELDS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def write_anomalies_jsonl(records: Sequence[ScoredAnomalyRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for scored in sorted(records, key=lambda item: anomaly_record_sort_key(item.record)):
            file.write(anomaly_record_json(scored.record))
            file.write("\n")
    return path


def read_anomalies_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise TelemetryAnomalyError(f"Anomaly output does not exist: {path}")
    return validate_anomaly_output_records(read_json_lines(path))


def representative_anomaly_records(
    records: Sequence[ScoredAnomalyRecord],
) -> dict[str, dict[str, Any]]:
    if not records:
        raise TelemetryAnomalyError("No anomaly records are available.")
    lowest = min(
        records,
        key=lambda item: (
            float(item.record["anomaly_score"]),
            item.record["event_time"],
            item.record["machine_code"],
            item.record["event_id"],
        ),
    )
    highest = max(
        records,
        key=lambda item: (
            float(item.record["anomaly_score"]),
            item.record["event_time"],
            item.record["machine_code"],
            item.record["event_id"],
        ),
    )
    closest = min(
        records,
        key=lambda item: (
            abs(item.decision_score),
            item.record["event_time"],
            item.record["machine_code"],
            item.record["event_id"],
        ),
    )
    return {
        "closest_to_decision_boundary": dict(closest.record),
        "highest_anomaly_score": dict(highest.record),
        "lowest_anomaly_score": dict(lowest.record),
    }


def scoring_summary(
    *,
    records: Sequence[ScoredAnomalyRecord],
    output_path: Path,
    artifact: TrustedAnomalyArtifact,
) -> TelemetryAnomalyScoringSummary:
    if not records:
        raise TelemetryAnomalyError("At least one scored anomaly record is required.")
    scores = [float(item.record["anomaly_score"]) for item in records]
    flag_count = sum(1 for item in records if bool(item.record["anomaly_flag"]))
    return TelemetryAnomalyScoringSummary(
        scored_event_count=len(records),
        distinct_machine_count=len({str(item.record["machine_code"]) for item in records}),
        anomaly_flag_count=flag_count,
        non_anomaly_count=len(records) - flag_count,
        min_anomaly_score=round(min(scores), 5),
        max_anomaly_score=round(max(scores), 5),
        mean_anomaly_score=round(float(fmean(scores)), 5),
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        model_config_hash=artifact.model_config_hash,
        output_path=output_path,
        output_sha256=file_sha256(output_path),
        representatives=representative_anomaly_records(records),
    )


def run_scoring_pipeline(
    *,
    root: Path | None = None,
    output_path: Path | None = None,
) -> TelemetryAnomalyScoringSummary:
    root_path = root or project_root()
    artifact = load_trusted_artifact(root_path)
    records = load_feature_records_from_export(root_path)
    scored = score_feature_records(records, artifact)
    output_file = output_path or anomaly_output_path(root_path)
    write_anomalies_jsonl(scored, output_file)
    write_static_summary(root_path)
    return scoring_summary(records=scored, output_path=output_file, artifact=artifact)


def build_static_summary() -> dict[str, Any]:
    return {
        "algorithm": ALGORITHM,
        "artifact_path": ARTIFACT_RELATIVE_PATH.as_posix(),
        "baseline_policy": (
            "Frozen synthetic operational reference baseline built from canonical Silver "
            "telemetry; not known healthy data and not production calibration."
        ),
        "decision_semantics": (
            "anomaly_flag is directly consistent with IsolationForest.predict where -1 is flagged."
        ),
        "features": list(FEATURES),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "no_anomaly_labels": True,
        "no_supervised_evaluation_metrics": True,
        "persistence_table": "anomalies",
        "relationship_to_ai4i": (
            "AI4I failure prediction and telemetry anomaly detection are independent outputs."
        ),
        "runtime_output_path": ANOMALY_OUTPUT_RELATIVE_PATH.as_posix(),
        "score_semantics": (
            "anomaly_score is a bounded monotonic transform of IsolationForest decision_function; "
            "higher means more anomalous and it is not a calibrated probability."
        ),
        "source_layer": SOURCE_LAYER,
        "source_path": SOURCE_PATH,
        "stable_identity_policy": [
            "event_id",
            "model_name",
            "model_version",
            "model_config_hash",
        ],
        "future_alert_policy": "planned",
        "future_drift_monitoring": "planned",
        "future_streaming_detection": "planned",
    }


def write_static_summary(root: Path | None = None) -> Path:
    return write_json_file(static_summary_path(root), build_static_summary())

"""Host-side AI4I inference bridge for adapted canonical Silver telemetry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ml.inference import ai4i_predictor
from pipelines.batch import ai4i_feature_adapter

ADAPTER_CONFIG_RELATIVE_PATH = ai4i_feature_adapter.CONFIG_RELATIVE_PATH
PREDICTION_OUTPUT_RELATIVE_PATH = (
    Path("data") / "predictions" / "ai4i" / "telemetry_predictions.jsonl"
)
EXPECTED_TOP_LEVEL_FIELDS = (
    "adapter_version",
    "event_id",
    "machine_code",
    "event_time",
    "model_input",
    "source_lineage",
)
LINEAGE_FIELDS = ai4i_feature_adapter.LINEAGE_FIELDS
FORBIDDEN_PREDICTION_FIELDS = (
    "Machine failure",
    "actual_failure",
    "ground_truth",
    "shap_values",
    "shap",
    "anomaly_label",
    "anomaly_score",
)


class AI4ITelemetryInferenceError(ValueError):
    """Raised when telemetry adapter records or predictions fail validation."""


@dataclass(frozen=True)
class TelemetryPredictionSummary:
    """Summary of a telemetry batch inference run."""

    adapter_record_count: int
    prediction_record_count: int
    unique_event_id_count: int
    positive_prediction_count: int
    negative_prediction_count: int
    min_failure_probability: float
    max_failure_probability: float
    mean_failure_probability: float
    model_name: str
    model_version: str
    frozen_threshold: float
    final_config_hash: str
    output_path: Path
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_record_count": self.adapter_record_count,
            "final_config_hash": self.final_config_hash,
            "frozen_threshold": self.frozen_threshold,
            "max_failure_probability": self.max_failure_probability,
            "mean_failure_probability": self.mean_failure_probability,
            "min_failure_probability": self.min_failure_probability,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "negative_prediction_count": self.negative_prediction_count,
            "output_path": self.output_path.as_posix(),
            "output_sha256": self.output_sha256,
            "positive_prediction_count": self.positive_prediction_count,
            "prediction_record_count": self.prediction_record_count,
            "unique_event_id_count": self.unique_event_id_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prediction_output_path(root: Path | None = None) -> Path:
    return (root or project_root()) / PREDICTION_OUTPUT_RELATIVE_PATH


def load_adapter_config(
    root: Path | None = None,
) -> ai4i_feature_adapter.AI4IFeatureAdapterConfig:
    root_path = root or project_root()
    return ai4i_feature_adapter.load_adapter_config(root_path / ADAPTER_CONFIG_RELATIVE_PATH)


def adapter_output_path(
    config: ai4i_feature_adapter.AI4IFeatureAdapterConfig,
    root: Path | None = None,
) -> Path:
    return (root or project_root()) / Path(config.output)


def discover_adapter_part_files(adapter_path: Path) -> list[Path]:
    if not adapter_path.exists() or not adapter_path.is_dir():
        raise AI4ITelemetryInferenceError(
            f"Adapter output directory does not exist: {adapter_path}"
        )
    files = sorted(
        path
        for path in adapter_path.iterdir()
        if path.is_file() and path.name.startswith("part-") and path.suffix == ".json"
    )
    if not files:
        raise AI4ITelemetryInferenceError(
            f"Adapter output directory contains no Spark part-*.json files: {adapter_path}"
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
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AI4ITelemetryInferenceError(
                    f"{path.name}:{line_number} is not valid JSON."
                ) from exc
            if not isinstance(value, dict):
                raise AI4ITelemetryInferenceError(f"{path.name}:{line_number} is not an object.")
            records.append(value)
    return records


def exact_model_features(
    config: ai4i_feature_adapter.AI4IFeatureAdapterConfig,
) -> tuple[str, ...]:
    return tuple(config.model_input_features)


def validate_model_input(
    model_input: Mapping[str, Any],
    config: ai4i_feature_adapter.AI4IFeatureAdapterConfig,
    final_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(model_input, Mapping):
        raise AI4ITelemetryInferenceError("model_input must be a JSON object.")
    actual_fields = set(model_input)
    expected_fields = set(exact_model_features(config))
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    excluded = sorted(actual_fields & set(config.excluded_current_model_fields))
    if missing:
        raise AI4ITelemetryInferenceError(
            "model_input is missing feature(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4ITelemetryInferenceError(
            "model_input has unexpected feature(s): " + ", ".join(extra)
        )
    if excluded:
        raise AI4ITelemetryInferenceError(
            "Excluded telemetry field entered model_input: " + ", ".join(excluded)
        )
    try:
        validated = ai4i_predictor.validate_inference_record(model_input, final_config)
    except ValueError as exc:
        raise AI4ITelemetryInferenceError(str(exc)) from exc
    return {field: validated[field] for field in config.model_input_features}


def validate_source_lineage(source_lineage: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_lineage, Mapping):
        raise AI4ITelemetryInferenceError("source_lineage must be a JSON object.")
    actual = set(source_lineage)
    expected = set(LINEAGE_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AI4ITelemetryInferenceError(
            "source_lineage is missing field(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4ITelemetryInferenceError(
            "source_lineage has unexpected field(s): " + ", ".join(extra)
        )
    return {field: source_lineage[field] for field in LINEAGE_FIELDS}


def validate_adapter_record(
    record: Mapping[str, Any],
    config: ai4i_feature_adapter.AI4IFeatureAdapterConfig,
    final_config: Mapping[str, Any],
) -> dict[str, Any]:
    actual = set(record)
    expected = set(EXPECTED_TOP_LEVEL_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AI4ITelemetryInferenceError(
            "Adapter record is missing field(s): " + ", ".join(missing)
        )
    if extra:
        raise AI4ITelemetryInferenceError(
            "Adapter record has unexpected field(s): " + ", ".join(extra)
        )
    if record["adapter_version"] != config.adapter_version:
        raise AI4ITelemetryInferenceError("Adapter record version does not match config.")
    for field in ("event_id", "machine_code", "event_time"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise AI4ITelemetryInferenceError(f"Adapter record field {field} must be text.")
    model_input = validate_model_input(record["model_input"], config, final_config)
    source_lineage = validate_source_lineage(record["source_lineage"])
    return {
        "adapter_version": record["adapter_version"],
        "event_id": record["event_id"],
        "event_time": record["event_time"],
        "machine_code": record["machine_code"],
        "model_input": model_input,
        "source_lineage": source_lineage,
    }


def adapter_record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    lineage = record["source_lineage"]
    return (
        record["event_time"],
        record["machine_code"],
        record["event_id"],
        lineage.get("source_kafka_timestamp") or "",
        lineage.get("source_kafka_topic") or "",
        int(lineage.get("source_kafka_partition") or -1),
        int(lineage.get("source_kafka_offset") or -1),
        lineage.get("source_kafka_key") or "",
        lineage.get("payload_sha256") or "",
    )


def prediction_record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["event_time"],
        record["machine_code"],
        record["event_id"],
        record.get("source_kafka_timestamp") or "",
        record.get("source_kafka_topic") or "",
        int(record.get("source_kafka_partition") or -1),
        int(record.get("source_kafka_offset") or -1),
        record.get("source_kafka_key") or "",
        record.get("payload_sha256") or "",
    )


def validate_unique_event_ids(records: Sequence[Mapping[str, Any]]) -> None:
    event_ids = [str(record["event_id"]) for record in records]
    if len(event_ids) != len(set(event_ids)):
        raise AI4ITelemetryInferenceError("Adapter or prediction event_id values are not unique.")


def load_adapter_records(
    *,
    root: Path | None = None,
    config: ai4i_feature_adapter.AI4IFeatureAdapterConfig | None = None,
    final_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    root_path = root or project_root()
    config = config or load_adapter_config(root_path)
    final_config = final_config or ai4i_predictor.load_final_config(root_path)
    raw_records: list[dict[str, Any]] = []
    for path in discover_adapter_part_files(adapter_output_path(config, root_path)):
        raw_records.extend(read_json_lines(path))
    records = [validate_adapter_record(record, config, final_config) for record in raw_records]
    validate_unique_event_ids(records)
    return sorted(records, key=adapter_record_sort_key)


def canonical_model_input_json(
    model_input: Mapping[str, Any],
    final_config: Mapping[str, Any],
) -> str:
    validated = ai4i_predictor.validate_inference_record(model_input, final_config)
    ordered = {
        field: validated[field] for field in ai4i_predictor.required_input_fields(final_config)
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def model_input_sha256(model_input: Mapping[str, Any], final_config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_model_input_json(model_input, final_config).encode("utf-8")
    ).hexdigest()


def validate_predictor_identity(predictor: Any, final_config: Mapping[str, Any]) -> None:
    threshold = float(final_config["decision_threshold"])
    expected_hash = ai4i_predictor.current_final_config_hash(final_config)
    if predictor.model_name != ai4i_predictor.MODEL_NAME:
        raise AI4ITelemetryInferenceError("Predictor model name is unexpected.")
    if predictor.model_version != ai4i_predictor.MODEL_VERSION:
        raise AI4ITelemetryInferenceError("Predictor model version is unexpected.")
    if float(predictor.decision_threshold) != threshold:
        raise AI4ITelemetryInferenceError("Predictor threshold does not match frozen config.")
    if predictor.final_config_hash != expected_hash:
        raise AI4ITelemetryInferenceError("Predictor config hash does not match frozen config.")


def prediction_is_consistent(probability: float, prediction: int, threshold: float) -> bool:
    return int(float(probability) >= float(threshold)) == int(prediction)


def build_prediction_records(
    adapter_records: Sequence[Mapping[str, Any]],
    prediction_outputs: Sequence[Mapping[str, Any]],
    final_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(adapter_records) != len(prediction_outputs):
        raise AI4ITelemetryInferenceError("Prediction count does not match adapter record count.")
    records: list[dict[str, Any]] = []
    for adapter_record, prediction in zip(adapter_records, prediction_outputs, strict=True):
        probability = float(prediction["failure_probability"])
        threshold = float(prediction["decision_threshold"])
        failure_prediction = int(prediction["failure_prediction"])
        if not 0 <= probability <= 1:
            raise AI4ITelemetryInferenceError("Failure probability is outside [0, 1].")
        if not prediction_is_consistent(probability, failure_prediction, threshold):
            raise AI4ITelemetryInferenceError("Prediction does not follow the frozen threshold.")
        source_lineage = adapter_record["source_lineage"]
        record = {
            "adapter_version": adapter_record["adapter_version"],
            "event_id": adapter_record["event_id"],
            "failure_prediction": failure_prediction,
            "failure_probability": probability,
            "final_config_hash": prediction["final_config_hash"],
            "frozen_threshold": threshold,
            "machine_code": adapter_record["machine_code"],
            "model_input_sha256": model_input_sha256(adapter_record["model_input"], final_config),
            "model_name": prediction["model_name"],
            "model_version": prediction["model_version"],
            "event_time": adapter_record["event_time"],
        }
        for field in LINEAGE_FIELDS:
            record[field] = source_lineage[field]
        records.append(record)
    validate_unique_event_ids(records)
    return sorted(records, key=prediction_record_sort_key)


def prediction_record_json(record: Mapping[str, Any]) -> str:
    forbidden = [field for field in FORBIDDEN_PREDICTION_FIELDS if field in record]
    if forbidden:
        raise AI4ITelemetryInferenceError(
            "Forbidden prediction field(s): " + ", ".join(sorted(forbidden))
        )
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def write_predictions_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(prediction_record_json(record))
            file.write("\n")


def read_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AI4ITelemetryInferenceError(f"Prediction output does not exist: {path}")
    return read_json_lines(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_summary(
    *,
    adapter_record_count: int,
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> TelemetryPredictionSummary:
    if not records:
        raise AI4ITelemetryInferenceError("Prediction output must contain at least one record.")
    probabilities = [float(record["failure_probability"]) for record in records]
    positives = sum(int(record["failure_prediction"]) for record in records)
    first = records[0]
    return TelemetryPredictionSummary(
        adapter_record_count=adapter_record_count,
        prediction_record_count=len(records),
        unique_event_id_count=len({str(record["event_id"]) for record in records}),
        positive_prediction_count=positives,
        negative_prediction_count=len(records) - positives,
        min_failure_probability=round(min(probabilities), 6),
        max_failure_probability=round(max(probabilities), 6),
        mean_failure_probability=round(float(fmean(probabilities)), 6),
        model_name=str(first["model_name"]),
        model_version=str(first["model_version"]),
        frozen_threshold=float(first["frozen_threshold"]),
        final_config_hash=str(first["final_config_hash"]),
        output_path=output_path,
        output_sha256=file_sha256(output_path),
    )


def run_prediction_pipeline(
    *,
    root: Path | None = None,
    output_path: Path | None = None,
    predictor: Any | None = None,
) -> TelemetryPredictionSummary:
    root_path = root or project_root()
    config = load_adapter_config(root_path)
    final_config = ai4i_predictor.load_final_config(root_path)
    model_path = ai4i_predictor.artifact_path(root_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Packaged AI4I model artifact is missing. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\package_ai4i_final_model.py first."
        )
    predictor = predictor or ai4i_predictor.load_predictor(root_path, final_config=final_config)
    validate_predictor_identity(predictor, final_config)
    adapter_records = load_adapter_records(root=root_path, config=config, final_config=final_config)
    model_inputs = [record["model_input"] for record in adapter_records]
    prediction_outputs = predictor.predict_batch(model_inputs)
    prediction_records = build_prediction_records(adapter_records, prediction_outputs, final_config)
    output_file = output_path or prediction_output_path(root_path)
    write_predictions_jsonl(prediction_records, output_file)
    return prediction_summary(
        adapter_record_count=len(adapter_records),
        records=prediction_records,
        output_path=output_file,
    )


def representative_prediction_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not records:
        raise AI4ITelemetryInferenceError("No prediction records are available.")
    threshold = float(records[0]["frozen_threshold"])
    ordered = sorted(
        records,
        key=lambda record: (float(record["failure_probability"]), record["event_id"]),
    )
    closest = min(
        records,
        key=lambda record: (
            abs(float(record["failure_probability"]) - threshold),
            record["event_time"],
            record["machine_code"],
            record["event_id"],
        ),
    )
    return {
        "closest_to_threshold": dict(closest),
        "highest_probability": dict(ordered[-1]),
        "lowest_probability": dict(ordered[0]),
    }

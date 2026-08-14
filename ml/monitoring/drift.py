"""Deterministic input-distribution drift monitoring helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.anomaly import telemetry_detector
from ml.inference import ai4i_predictor, ai4i_telemetry
from pipelines.batch import ai4i_feature_adapter

CONFIG_RELATIVE_PATH = Path("ml") / "config" / "drift_monitoring.json"
REFERENCE_PROFILE_RELATIVE_PATH = Path("reports") / "drift" / "drift_reference_profiles.json"
STATIC_SUMMARY_RELATIVE_PATH = Path("reports") / "drift" / "data_drift_monitoring_summary.json"
DRIFT_REPORT_RELATIVE_PATH = Path("data") / "drift" / "latest_drift_report.json"

MONITOR_VERSION = "1.0.0"
AI4I_SCOPE = "ai4i_model_input"
ANOMALY_SCOPE = "operational_anomaly_inputs"
AI4I_REFERENCE_NAME = "ai4i_train_validation_development"
ANOMALY_REFERENCE_NAME = "telemetry_isolation_forest_v1_baseline"
AI4I_FEATURES = ai4i_feature_adapter.EXPECTED_MODEL_INPUT_FEATURES
AI4I_NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
AI4I_CATEGORICAL_FEATURES = ("Type",)
AI4I_TYPE_CATEGORIES = ("L", "M", "H")
ANOMALY_FEATURES = telemetry_detector.FEATURES
STATUS_ORDER = {"stable": 0, "watch": 1, "drift": 2}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_AI4I_SOURCE_PATH = "data/processed/ai4i/test.csv"
FORBIDDEN_DRIFT_FEATURES = {
    "Machine failure",
    "actual_failure",
    "failure_prediction",
    "failure_probability",
    "anomaly_score",
    "anomaly_flag",
    "shap",
    "shap_values",
}


class DriftMonitoringError(ValueError):
    """Raised when drift monitoring inputs or outputs are invalid."""


@dataclass(frozen=True)
class DriftConfig:
    """Static policy for deterministic drift monitoring."""

    monitor_version: str
    numeric_bin_count: int
    epsilon: float
    psi_watch_threshold: float
    psi_drift_threshold: float
    ai4i_model_input: dict[str, Any]
    operational_anomaly_inputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai4i_model_input": self.ai4i_model_input,
            "epsilon": self.epsilon,
            "monitor_version": self.monitor_version,
            "numeric_bin_count": self.numeric_bin_count,
            "operational_anomaly_inputs": self.operational_anomaly_inputs,
            "psi_drift_threshold": self.psi_drift_threshold,
            "psi_watch_threshold": self.psi_watch_threshold,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def reference_profile_path(root: Path | None = None) -> Path:
    return (root or project_root()) / REFERENCE_PROFILE_RELATIVE_PATH


def static_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / STATIC_SUMMARY_RELATIVE_PATH


def drift_report_path(root: Path | None = None) -> Path:
    return (root or project_root()) / DRIFT_REPORT_RELATIVE_PATH


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise DriftMonitoringError(f"{field_name} must be a lowercase SHA-256 hex string.")
    return value


def require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriftMonitoringError(f"{field_name} must be non-empty text.")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise DriftMonitoringError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise DriftMonitoringError(f"{field_name} must be positive.")
    return parsed


def require_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise DriftMonitoringError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise DriftMonitoringError(f"{field_name} must be finite and positive.")
    return parsed


def require_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise DriftMonitoringError(f"{field_name} must be a non-empty string list.")
    return tuple(value)


def parse_config(raw_config: Mapping[str, Any]) -> DriftConfig:
    expected = {
        "ai4i_model_input",
        "epsilon",
        "monitor_version",
        "numeric_bin_count",
        "operational_anomaly_inputs",
        "psi_drift_threshold",
        "psi_watch_threshold",
    }
    missing = sorted(expected - set(raw_config))
    extra = sorted(set(raw_config) - expected)
    if missing:
        raise DriftMonitoringError("Drift config is missing key(s): " + ", ".join(missing))
    if extra:
        raise DriftMonitoringError("Drift config has unexpected key(s): " + ", ".join(extra))
    config = DriftConfig(
        monitor_version=require_text(raw_config["monitor_version"], "monitor_version"),
        numeric_bin_count=require_positive_int(
            raw_config["numeric_bin_count"], "numeric_bin_count"
        ),
        epsilon=require_positive_float(raw_config["epsilon"], "epsilon"),
        psi_watch_threshold=require_positive_float(
            raw_config["psi_watch_threshold"],
            "psi_watch_threshold",
        ),
        psi_drift_threshold=require_positive_float(
            raw_config["psi_drift_threshold"],
            "psi_drift_threshold",
        ),
        ai4i_model_input=dict(raw_config["ai4i_model_input"]),
        operational_anomaly_inputs=dict(raw_config["operational_anomaly_inputs"]),
    )
    validate_config(config)
    return config


def validate_config(config: DriftConfig) -> None:
    if config.monitor_version != MONITOR_VERSION:
        raise DriftMonitoringError("monitor_version must be 1.0.0.")
    if config.numeric_bin_count != 10:
        raise DriftMonitoringError("numeric_bin_count must be 10.")
    if config.epsilon != 0.000001:
        raise DriftMonitoringError("epsilon must be 0.000001.")
    if config.psi_watch_threshold != 0.10:
        raise DriftMonitoringError("psi_watch_threshold must be 0.10.")
    if config.psi_drift_threshold != 0.25:
        raise DriftMonitoringError("psi_drift_threshold must be 0.25.")
    if config.psi_watch_threshold >= config.psi_drift_threshold:
        raise DriftMonitoringError("PSI watch threshold must be below drift threshold.")
    validate_ai4i_config(config.ai4i_model_input)
    validate_anomaly_config(config.operational_anomaly_inputs)


def validate_ai4i_config(section: Mapping[str, Any]) -> None:
    required = {
        "categories",
        "categorical_features",
        "features",
        "final_config_hash",
        "forbidden_source_paths",
        "model_name",
        "model_version",
        "numeric_features",
        "reference_name",
        "reference_policy",
        "source_paths",
        "training_data_policy",
    }
    missing = sorted(required - set(section))
    if missing:
        raise DriftMonitoringError("AI4I drift config is missing key(s): " + ", ".join(missing))
    if require_text(section["reference_name"], "ai4i.reference_name") != AI4I_REFERENCE_NAME:
        raise DriftMonitoringError("AI4I reference_name is unexpected.")
    if require_text(section["training_data_policy"], "ai4i.training_data_policy") != (
        "train + validation"
    ):
        raise DriftMonitoringError("AI4I training_data_policy must be train + validation.")
    if require_text(section["model_name"], "ai4i.model_name") != ai4i_predictor.MODEL_NAME:
        raise DriftMonitoringError("AI4I model_name is unexpected.")
    if require_text(section["model_version"], "ai4i.model_version") != ai4i_predictor.MODEL_VERSION:
        raise DriftMonitoringError("AI4I model_version is unexpected.")
    validate_sha256(section["final_config_hash"], "ai4i.final_config_hash")
    if require_string_list(section["features"], "ai4i.features") != AI4I_FEATURES:
        raise DriftMonitoringError("AI4I drift features must match the frozen model inputs.")
    if require_string_list(section["numeric_features"], "ai4i.numeric_features") != (
        AI4I_NUMERIC_FEATURES
    ):
        raise DriftMonitoringError("AI4I numeric features are unexpected.")
    if require_string_list(section["categorical_features"], "ai4i.categorical_features") != (
        AI4I_CATEGORICAL_FEATURES
    ):
        raise DriftMonitoringError("AI4I categorical features are unexpected.")
    source_paths = require_string_list(section["source_paths"], "ai4i.source_paths")
    if source_paths != (
        "data/processed/ai4i/train.csv",
        "data/processed/ai4i/validation.csv",
    ):
        raise DriftMonitoringError("AI4I source paths must be train.csv and validation.csv.")
    forbidden = require_string_list(section["forbidden_source_paths"], "ai4i.forbidden_paths")
    if FORBIDDEN_AI4I_SOURCE_PATH not in forbidden:
        raise DriftMonitoringError("AI4I forbidden source paths must include test.csv.")
    categories = section["categories"]
    if not isinstance(categories, Mapping):
        raise DriftMonitoringError("AI4I categories must be an object.")
    if tuple(categories.get("Type", ())) != AI4I_TYPE_CATEGORIES:
        raise DriftMonitoringError("AI4I Type categories must be L, M, H.")


def validate_anomaly_config(section: Mapping[str, Any]) -> None:
    required = {
        "baseline_event_id_sha256",
        "baseline_feature_data_sha256",
        "features",
        "model_config_hash",
        "model_name",
        "model_version",
        "numeric_features",
        "reference_name",
        "reference_policy",
        "source_layer",
        "source_path",
    }
    missing = sorted(required - set(section))
    if missing:
        raise DriftMonitoringError("Anomaly drift config is missing key(s): " + ", ".join(missing))
    if require_text(section["reference_name"], "anomaly.reference_name") != ANOMALY_REFERENCE_NAME:
        raise DriftMonitoringError("Anomaly reference_name is unexpected.")
    if require_text(section["model_name"], "anomaly.model_name") != telemetry_detector.MODEL_NAME:
        raise DriftMonitoringError("Anomaly model_name is unexpected.")
    if require_text(section["model_version"], "anomaly.model_version") != (
        telemetry_detector.MODEL_VERSION
    ):
        raise DriftMonitoringError("Anomaly model_version is unexpected.")
    validate_sha256(section["model_config_hash"], "anomaly.model_config_hash")
    validate_sha256(section["baseline_event_id_sha256"], "anomaly.baseline_event_id_sha256")
    validate_sha256(section["baseline_feature_data_sha256"], "anomaly.baseline_feature_data_sha256")
    if (
        require_text(section["source_layer"], "anomaly.source_layer")
        != telemetry_detector.SOURCE_LAYER
    ):
        raise DriftMonitoringError("Anomaly source_layer must be silver.")
    if (
        require_text(section["source_path"], "anomaly.source_path")
        != telemetry_detector.SOURCE_PATH
    ):
        raise DriftMonitoringError("Anomaly source_path is unexpected.")
    if require_string_list(section["features"], "anomaly.features") != ANOMALY_FEATURES:
        raise DriftMonitoringError(
            "Anomaly drift features must be vibration_mm_s and pressure_bar."
        )
    if require_string_list(section["numeric_features"], "anomaly.numeric_features") != (
        ANOMALY_FEATURES
    ):
        raise DriftMonitoringError("Anomaly numeric features are unexpected.")


def load_config(root: Path | None = None) -> DriftConfig:
    path = config_path(root)
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftMonitoringError(f"Drift config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DriftMonitoringError(f"Drift config is invalid JSON: {path}") from exc
    if not isinstance(raw_config, Mapping):
        raise DriftMonitoringError("Drift config must be a JSON object.")
    return parse_config(raw_config)


def status_for_psi(psi: float, config: DriftConfig) -> str:
    if not math.isfinite(psi) or psi < 0:
        raise DriftMonitoringError("PSI must be finite and non-negative.")
    if psi >= config.psi_drift_threshold:
        return "drift"
    if psi >= config.psi_watch_threshold:
        return "watch"
    return "stable"


def overall_status(statuses: Sequence[str]) -> str:
    if not statuses:
        raise DriftMonitoringError("At least one feature status is required.")
    unknown = sorted(set(statuses) - set(STATUS_ORDER))
    if unknown:
        raise DriftMonitoringError("Unknown monitoring status: " + ", ".join(unknown))
    return max(statuses, key=lambda value: STATUS_ORDER[value])


def finite_float_array(values: Sequence[Any], feature_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise DriftMonitoringError(f"{feature_name} requires at least one value.")
    if not np.all(np.isfinite(array)):
        raise DriftMonitoringError(f"{feature_name} contains non-finite values.")
    return array


def numeric_bin_edges(reference_values: Sequence[Any], bin_count: int) -> list[float]:
    values = finite_float_array(reference_values, "numeric reference")
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    raw_edges = np.quantile(values, quantiles, method="linear")
    unique_edges = np.unique(raw_edges)
    return [float(value) for value in unique_edges]


def comparison_edges(finite_edges: Sequence[float]) -> np.ndarray:
    if not finite_edges:
        raise DriftMonitoringError("Numeric reference profile has no bin edges.")
    values = finite_float_array(finite_edges, "numeric bin edges")
    unique_edges = np.unique(values)
    if len(unique_edges) == 1:
        return np.asarray([-np.inf, np.inf], dtype=float)
    return np.asarray([-np.inf, *unique_edges[1:-1], np.inf], dtype=float)


def proportions_from_edges(values: Sequence[Any], edges: Sequence[float]) -> list[float]:
    array = finite_float_array(values, "numeric values")
    counts, _ = np.histogram(array, bins=comparison_edges(edges))
    total = counts.sum()
    if total <= 0:
        raise DriftMonitoringError("Cannot calculate proportions for an empty array.")
    return [float(count / total) for count in counts]


def psi_from_proportions(
    reference_proportions: Sequence[Any],
    current_proportions: Sequence[Any],
    epsilon: float,
) -> float:
    reference = np.asarray(reference_proportions, dtype=float)
    current = np.asarray(current_proportions, dtype=float)
    if reference.shape != current.shape:
        raise DriftMonitoringError("PSI proportion arrays must have the same shape.")
    if reference.size == 0:
        raise DriftMonitoringError("PSI requires at least one bin.")
    if np.any(reference < 0) or np.any(current < 0):
        raise DriftMonitoringError("PSI proportions must be non-negative.")
    value = float(
        np.sum((current - reference) * np.log((current + epsilon) / (reference + epsilon)))
    )
    if not math.isfinite(value):
        raise DriftMonitoringError("PSI calculation produced a non-finite value.")
    return max(value, 0.0)


def numeric_statistics(values: Sequence[Any]) -> dict[str, float | int]:
    array = finite_float_array(values, "numeric values")
    return {
        "count": int(array.size),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "std": float(np.std(array, ddof=0)),
    }


def standardized_mean_shift(
    reference_mean: float, current_mean: float, reference_std: float
) -> float | None:
    if reference_std <= 0:
        return None
    return abs(current_mean - reference_mean) / reference_std


def outside_reference_range(
    values: Sequence[Any], reference_min: float, reference_max: float
) -> tuple[int, float]:
    array = finite_float_array(values, "current numeric values")
    outside = int(np.sum((array < reference_min) | (array > reference_max)))
    return outside, float(outside / array.size)


def categorical_proportions(
    values: Sequence[Any], categories: Sequence[str]
) -> tuple[dict[str, float], int]:
    if not values:
        raise DriftMonitoringError("Categorical PSI requires at least one current value.")
    category_set = set(categories)
    counts = Counter(str(value) for value in values)
    total = len(values)
    proportions = {category: float(counts.get(category, 0) / total) for category in categories}
    unexpected = sum(count for category, count in counts.items() if category not in category_set)
    return proportions, unexpected


def numeric_reference_profile(
    feature_name: str,
    reference_values: Sequence[Any],
    config: DriftConfig,
) -> dict[str, Any]:
    edges = numeric_bin_edges(reference_values, config.numeric_bin_count)
    proportions = proportions_from_edges(reference_values, edges)
    stats = numeric_statistics(reference_values)
    return {
        "bin_edges": edges,
        "feature_name": feature_name,
        "feature_type": "numeric",
        "reference_bin_proportions": proportions,
        "reference_count": stats["count"],
        "reference_max": stats["max"],
        "reference_mean": stats["mean"],
        "reference_min": stats["min"],
        "reference_std": stats["std"],
    }


def categorical_reference_profile(
    feature_name: str,
    reference_values: Sequence[Any],
    categories: Sequence[str],
) -> dict[str, Any]:
    proportions, unexpected = categorical_proportions(reference_values, categories)
    if unexpected:
        raise DriftMonitoringError(f"Reference feature {feature_name} has unexpected categories.")
    return {
        "categories": list(categories),
        "feature_name": feature_name,
        "feature_type": "categorical",
        "reference_count": len(reference_values),
        "reference_proportions": proportions,
    }


def numeric_feature_metric(
    reference_profile: Mapping[str, Any],
    current_values: Sequence[Any],
    config: DriftConfig,
) -> dict[str, Any]:
    feature_name = require_text(reference_profile["feature_name"], "feature_name")
    current_stats = numeric_statistics(current_values)
    current_proportions = proportions_from_edges(current_values, reference_profile["bin_edges"])
    psi = psi_from_proportions(
        reference_profile["reference_bin_proportions"],
        current_proportions,
        config.epsilon,
    )
    range_count, range_rate = outside_reference_range(
        current_values,
        float(reference_profile["reference_min"]),
        float(reference_profile["reference_max"]),
    )
    shift = standardized_mean_shift(
        float(reference_profile["reference_mean"]),
        float(current_stats["mean"]),
        float(reference_profile["reference_std"]),
    )
    return {
        "bin_edges": list(reference_profile["bin_edges"]),
        "current_bin_proportions": current_proportions,
        "current_count": current_stats["count"],
        "current_max": current_stats["max"],
        "current_mean": current_stats["mean"],
        "current_min": current_stats["min"],
        "current_std": current_stats["std"],
        "feature_name": feature_name,
        "feature_type": "numeric",
        "outside_reference_range_count": range_count,
        "outside_reference_range_rate": range_rate,
        "psi": psi,
        "reference_bin_proportions": list(reference_profile["reference_bin_proportions"]),
        "reference_count": int(reference_profile["reference_count"]),
        "reference_max": float(reference_profile["reference_max"]),
        "reference_mean": float(reference_profile["reference_mean"]),
        "reference_min": float(reference_profile["reference_min"]),
        "reference_std": float(reference_profile["reference_std"]),
        "standardized_mean_shift": shift,
        "status": status_for_psi(psi, config),
    }


def categorical_feature_metric(
    reference_profile: Mapping[str, Any],
    current_values: Sequence[Any],
    config: DriftConfig,
) -> dict[str, Any]:
    feature_name = require_text(reference_profile["feature_name"], "feature_name")
    categories = tuple(str(value) for value in reference_profile["categories"])
    current_proportions, unexpected = categorical_proportions(current_values, categories)
    psi = psi_from_proportions(
        [reference_profile["reference_proportions"][category] for category in categories],
        [current_proportions[category] for category in categories],
        config.epsilon,
    )
    return {
        "categories": list(categories),
        "current_count": len(current_values),
        "current_proportions": current_proportions,
        "feature_name": feature_name,
        "feature_type": "categorical",
        "psi": psi,
        "reference_count": int(reference_profile["reference_count"]),
        "reference_proportions": {
            category: float(reference_profile["reference_proportions"][category])
            for category in categories
        },
        "status": status_for_psi(psi, config),
        "unexpected_category_count": unexpected,
    }


def load_ai4i_development_reference(
    root: Path,
    config: DriftConfig,
) -> tuple[pd.DataFrame, list[str]]:
    section = config.ai4i_model_input
    source_paths = [root / Path(path) for path in section["source_paths"]]
    forbidden_paths = {Path(path).as_posix() for path in section["forbidden_source_paths"]}
    used_relative = [path.relative_to(root).as_posix() for path in source_paths]
    if FORBIDDEN_AI4I_SOURCE_PATH in used_relative:
        raise DriftMonitoringError("AI4I drift reference attempted to read test.csv.")
    if FORBIDDEN_AI4I_SOURCE_PATH not in forbidden_paths:
        raise DriftMonitoringError("AI4I drift config does not forbid test.csv.")
    frames = []
    for path in source_paths:
        if not path.exists():
            raise DriftMonitoringError(f"AI4I reference source does not exist: {path}")
        frames.append(pd.read_csv(path, usecols=list(AI4I_FEATURES)))
    frame = pd.concat(frames, axis=0, ignore_index=True)
    validate_ai4i_reference_frame(frame)
    return frame.loc[:, list(AI4I_FEATURES)], used_relative


def validate_ai4i_reference_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != AI4I_FEATURES:
        raise DriftMonitoringError("AI4I reference columns do not match the frozen contract.")
    unexpected = sorted(set(frame["Type"].astype(str)) - set(AI4I_TYPE_CATEGORIES))
    if unexpected:
        raise DriftMonitoringError(
            "AI4I reference Type has unexpected values: " + ", ".join(unexpected)
        )
    for feature_name in AI4I_NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature_name], errors="raise").to_numpy(dtype=float)
        finite_float_array(values, feature_name)


def ai4i_reference_identity(
    config: DriftConfig,
    used_source_paths: Sequence[str],
) -> dict[str, Any]:
    section = config.ai4i_model_input
    return {
        "final_config_hash": section["final_config_hash"],
        "forbidden_source_paths": list(section["forbidden_source_paths"]),
        "model_name": section["model_name"],
        "model_version": section["model_version"],
        "reference_name": section["reference_name"],
        "source_paths": list(used_source_paths),
        "training_data_policy": section["training_data_policy"],
    }


def anomaly_reference_identity(config: DriftConfig) -> dict[str, Any]:
    section = config.operational_anomaly_inputs
    return {
        "baseline_event_id_sha256": section["baseline_event_id_sha256"],
        "baseline_feature_data_sha256": section["baseline_feature_data_sha256"],
        "model_config_hash": section["model_config_hash"],
        "model_name": section["model_name"],
        "model_version": section["model_version"],
        "reference_name": section["reference_name"],
        "source_layer": section["source_layer"],
        "source_path": section["source_path"],
    }


def verify_anomaly_reference_identity(
    records: Sequence[telemetry_detector.FeatureRecord],
    config: DriftConfig,
    root: Path,
) -> None:
    if not records:
        raise DriftMonitoringError("Anomaly reference baseline contains no feature records.")
    event_hash, feature_hash = telemetry_detector.baseline_hashes(records)
    section = config.operational_anomaly_inputs
    if event_hash != section["baseline_event_id_sha256"]:
        raise DriftMonitoringError(
            "Available anomaly reference event hash does not match frozen baseline."
        )
    if feature_hash != section["baseline_feature_data_sha256"]:
        raise DriftMonitoringError(
            "Available anomaly reference feature hash does not match frozen baseline."
        )
    baseline_summary = load_json_file(telemetry_detector.baseline_summary_path(root))
    metadata = load_json_file(telemetry_detector.artifact_metadata_path(root))
    for source_name, payload in (
        ("baseline summary", baseline_summary),
        ("artifact metadata", metadata),
    ):
        if payload.get("baseline_event_id_sha256") != event_hash:
            raise DriftMonitoringError(f"Anomaly {source_name} event hash does not match baseline.")
        if payload.get("baseline_feature_data_sha256") != feature_hash:
            raise DriftMonitoringError(
                f"Anomaly {source_name} feature hash does not match baseline."
            )
        if payload.get("model_name") != section["model_name"]:
            raise DriftMonitoringError(f"Anomaly {source_name} model name does not match config.")
        if payload.get("model_version") != section["model_version"]:
            raise DriftMonitoringError(
                f"Anomaly {source_name} model version does not match config."
            )
        if payload.get("model_config_hash") != section["model_config_hash"]:
            raise DriftMonitoringError(f"Anomaly {source_name} config hash does not match config.")


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftMonitoringError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DriftMonitoringError(f"Required JSON file is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DriftMonitoringError(f"JSON file must contain an object: {path}")
    return payload


def build_ai4i_reference_section(
    frame: pd.DataFrame,
    config: DriftConfig,
    used_source_paths: Sequence[str],
) -> dict[str, Any]:
    feature_profiles: list[dict[str, Any]] = [
        categorical_reference_profile(
            "Type", frame["Type"].astype(str).tolist(), AI4I_TYPE_CATEGORIES
        )
    ]
    for feature_name in AI4I_NUMERIC_FEATURES:
        feature_profiles.append(
            numeric_reference_profile(feature_name, frame[feature_name].tolist(), config)
        )
    return {
        "features": feature_profiles,
        "reference_identity": ai4i_reference_identity(config, used_source_paths),
        "reference_row_count": int(len(frame)),
    }


def build_anomaly_reference_section(
    records: Sequence[telemetry_detector.FeatureRecord],
    config: DriftConfig,
    root: Path,
) -> dict[str, Any]:
    verify_anomaly_reference_identity(records, config, root)
    feature_profiles = [
        numeric_reference_profile(
            feature_name,
            [getattr(record, feature_name) for record in records],
            config,
        )
        for feature_name in ANOMALY_FEATURES
    ]
    return {
        "features": feature_profiles,
        "reference_identity": anomaly_reference_identity(config),
        "reference_row_count": len(records),
    }


def build_reference_profile(
    *,
    root: Path | None = None,
    config: DriftConfig | None = None,
    anomaly_reference_records: Sequence[telemetry_detector.FeatureRecord] | None = None,
) -> dict[str, Any]:
    root_path = root or project_root()
    config = config or load_config(root_path)
    ai4i_frame, used_source_paths = load_ai4i_development_reference(root_path, config)
    anomaly_records = (
        list(anomaly_reference_records)
        if anomaly_reference_records is not None
        else telemetry_detector.load_feature_records_from_export(root_path)
    )
    profile = {
        "ai4i_model_input": build_ai4i_reference_section(
            ai4i_frame,
            config,
            used_source_paths,
        ),
        "monitor_version": config.monitor_version,
        "operational_anomaly_inputs": build_anomaly_reference_section(
            anomaly_records,
            config,
            root_path,
        ),
    }
    validate_reference_profile(profile, config)
    return profile


def write_reference_profile(profile: Mapping[str, Any], root: Path | None = None) -> Path:
    path = reference_profile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pretty_json(profile)
    path.write_text(payload, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != payload:
        raise DriftMonitoringError("Reference profile write was not byte deterministic.")
    return path


def load_reference_profile(
    root: Path | None = None, config: DriftConfig | None = None
) -> dict[str, Any]:
    root_path = root or project_root()
    path = reference_profile_path(root_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftMonitoringError(
            "Frozen drift reference profile is missing. Run "
            ".\\.venv\\Scripts\\python.exe scripts\\build_drift_reference_profiles.py"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DriftMonitoringError(f"Drift reference profile is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DriftMonitoringError("Drift reference profile must be a JSON object.")
    validate_reference_profile(payload, config or load_config(root_path))
    return payload


def validate_reference_profile(profile: Mapping[str, Any], config: DriftConfig) -> None:
    required = {"ai4i_model_input", "monitor_version", "operational_anomaly_inputs"}
    missing = sorted(required - set(profile))
    if missing:
        raise DriftMonitoringError("Reference profile is missing key(s): " + ", ".join(missing))
    if profile["monitor_version"] != config.monitor_version:
        raise DriftMonitoringError("Reference profile monitor version differs from config.")
    validate_reference_scope(
        profile["ai4i_model_input"],
        AI4I_SCOPE,
        tuple(config.ai4i_model_input["features"]),
    )
    validate_reference_scope(
        profile["operational_anomaly_inputs"],
        ANOMALY_SCOPE,
        tuple(config.operational_anomaly_inputs["features"]),
    )
    if profile["ai4i_model_input"]["reference_identity"] != ai4i_reference_identity(
        config,
        config.ai4i_model_input["source_paths"],
    ):
        raise DriftMonitoringError("AI4I reference identity differs from config.")
    if profile["operational_anomaly_inputs"]["reference_identity"] != anomaly_reference_identity(
        config
    ):
        raise DriftMonitoringError("Anomaly reference identity differs from config.")


def validate_reference_scope(
    scope: Mapping[str, Any], scope_name: str, expected_features: Sequence[str]
) -> None:
    if set(scope) != {"features", "reference_identity", "reference_row_count"}:
        raise DriftMonitoringError(f"{scope_name} reference scope has unexpected keys.")
    features = scope["features"]
    if not isinstance(features, list) or not features:
        raise DriftMonitoringError(f"{scope_name} reference features must be a non-empty list.")
    actual_features = tuple(str(feature.get("feature_name")) for feature in features)
    if actual_features != tuple(expected_features):
        raise DriftMonitoringError(f"{scope_name} reference features do not match config order.")
    row_count = require_positive_int(scope["reference_row_count"], f"{scope_name}.reference_count")
    for feature in features:
        if int(feature["reference_count"]) != row_count:
            raise DriftMonitoringError(f"{scope_name} feature reference count mismatch.")
        if feature["feature_type"] == "numeric":
            if len(feature["bin_edges"]) < 1:
                raise DriftMonitoringError(f"{scope_name} numeric profile has no bin edges.")
            proportions = feature["reference_bin_proportions"]
            if not math.isclose(sum(float(value) for value in proportions), 1.0, abs_tol=1e-9):
                raise DriftMonitoringError(f"{scope_name} reference bin proportions must sum to 1.")
        elif feature["feature_type"] == "categorical":
            proportions = feature["reference_proportions"]
            if not math.isclose(
                sum(float(value) for value in proportions.values()), 1.0, abs_tol=1e-9
            ):
                raise DriftMonitoringError(f"{scope_name} category proportions must sum to 1.")
        else:
            raise DriftMonitoringError(f"{scope_name} has unsupported feature type.")


def ai4i_current_records_from_adapter(root: Path | None = None) -> list[dict[str, Any]]:
    root_path = root or project_root()
    config = ai4i_telemetry.load_adapter_config(root_path)
    final_config = ai4i_predictor.load_final_config(root_path)
    final_hash = ai4i_predictor.current_final_config_hash(final_config)
    if final_hash != load_config(root_path).ai4i_model_input["final_config_hash"]:
        raise DriftMonitoringError("AI4I final config hash differs from drift config.")
    records = ai4i_telemetry.load_adapter_records(
        root=root_path,
        config=config,
        final_config=final_config,
    )
    validate_ai4i_current_records(records)
    return records


def validate_ai4i_current_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise DriftMonitoringError("AI4I current adapter records are empty.")
    for record in records:
        if set(record.get("model_input", {})) != set(AI4I_FEATURES):
            raise DriftMonitoringError(
                "AI4I current record does not use the exact model feature set."
            )
        forbidden = set(record.get("model_input", {})) & FORBIDDEN_DRIFT_FEATURES
        if forbidden:
            raise DriftMonitoringError(
                "Forbidden AI4I current feature(s): " + ", ".join(sorted(forbidden))
            )


def ai4i_current_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        model_input = record["model_input"]
        rows.append({feature: model_input[feature] for feature in AI4I_FEATURES})
    frame = pd.DataFrame(rows, columns=list(AI4I_FEATURES))
    validate_ai4i_reference_frame(frame)
    return frame


def anomaly_current_records_from_export(
    root: Path | None = None,
) -> list[telemetry_detector.FeatureRecord]:
    records = telemetry_detector.load_feature_records_from_export(root or project_root())
    validate_anomaly_current_records(records)
    return records


def validate_anomaly_current_records(records: Sequence[telemetry_detector.FeatureRecord]) -> None:
    if not records:
        raise DriftMonitoringError("Anomaly current feature records are empty.")


def ai4i_current_data_hash(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "event_id": record["event_id"],
            "model_input": {feature: record["model_input"][feature] for feature in AI4I_FEATURES},
        }
        for record in sorted(records, key=ai4i_telemetry.adapter_record_sort_key)
    ]
    return sha256_text(canonical_json(payload))


def anomaly_current_data_hash(records: Sequence[telemetry_detector.FeatureRecord]) -> str:
    ordered = sorted(records, key=lambda record: record.sort_key())
    payload = [
        {
            "event_id": record.event_id,
            "pressure_bar": record.pressure_bar,
            "vibration_mm_s": record.vibration_mm_s,
        }
        for record in ordered
    ]
    return sha256_text(canonical_json(payload))


def compare_ai4i_current(
    reference_scope: Mapping[str, Any],
    current_records: Sequence[Mapping[str, Any]],
    config: DriftConfig,
) -> dict[str, Any]:
    frame = ai4i_current_frame(current_records)
    metrics = []
    profiles = {profile["feature_name"]: profile for profile in reference_scope["features"]}
    metrics.append(
        categorical_feature_metric(
            profiles["Type"],
            frame["Type"].astype(str).tolist(),
            config,
        )
    )
    for feature_name in AI4I_NUMERIC_FEATURES:
        metrics.append(
            numeric_feature_metric(profiles[feature_name], frame[feature_name].tolist(), config)
        )
    return {
        "current_data_hash": ai4i_current_data_hash(current_records),
        "current_record_count": int(len(frame)),
        "features": metrics,
        "overall_status": overall_status([metric["status"] for metric in metrics]),
        "reference_identity": reference_scope["reference_identity"],
    }


def compare_anomaly_current(
    reference_scope: Mapping[str, Any],
    current_records: Sequence[telemetry_detector.FeatureRecord],
    config: DriftConfig,
) -> dict[str, Any]:
    profiles = {profile["feature_name"]: profile for profile in reference_scope["features"]}
    metrics = [
        numeric_feature_metric(
            profiles[feature_name],
            [getattr(record, feature_name) for record in current_records],
            config,
        )
        for feature_name in ANOMALY_FEATURES
    ]
    return {
        "current_data_hash": anomaly_current_data_hash(current_records),
        "current_record_count": len(current_records),
        "features": metrics,
        "overall_status": overall_status([metric["status"] for metric in metrics]),
        "reference_identity": reference_scope["reference_identity"],
    }


def build_drift_report(
    *,
    reference_profile: Mapping[str, Any],
    reference_profile_sha256: str,
    ai4i_current_records: Sequence[Mapping[str, Any]],
    anomaly_current_records: Sequence[telemetry_detector.FeatureRecord],
    config: DriftConfig,
) -> dict[str, Any]:
    report = {
        "ai4i_model_input": compare_ai4i_current(
            reference_profile["ai4i_model_input"],
            ai4i_current_records,
            config,
        ),
        "monitor_version": config.monitor_version,
        "operational_anomaly_inputs": compare_anomaly_current(
            reference_profile["operational_anomaly_inputs"],
            anomaly_current_records,
            config,
        ),
        "reference_profile_sha256": reference_profile_sha256,
    }
    validate_drift_report(report, config)
    return report


def validate_drift_report(report: Mapping[str, Any], config: DriftConfig) -> None:
    required = {
        "ai4i_model_input",
        "monitor_version",
        "operational_anomaly_inputs",
        "reference_profile_sha256",
    }
    missing = sorted(required - set(report))
    if missing:
        raise DriftMonitoringError("Drift report is missing key(s): " + ", ".join(missing))
    if report["monitor_version"] != config.monitor_version:
        raise DriftMonitoringError("Drift report monitor version differs from config.")
    validate_sha256(report["reference_profile_sha256"], "reference_profile_sha256")
    validate_current_scope(report["ai4i_model_input"], AI4I_SCOPE, AI4I_FEATURES, config)
    validate_current_scope(
        report["operational_anomaly_inputs"], ANOMALY_SCOPE, ANOMALY_FEATURES, config
    )


def validate_current_scope(
    scope: Mapping[str, Any],
    scope_name: str,
    expected_features: Sequence[str],
    config: DriftConfig,
) -> None:
    expected_keys = {
        "current_data_hash",
        "current_record_count",
        "features",
        "overall_status",
        "reference_identity",
    }
    if set(scope) != expected_keys:
        raise DriftMonitoringError(f"{scope_name} current scope has unexpected keys.")
    validate_sha256(scope["current_data_hash"], f"{scope_name}.current_data_hash")
    require_positive_int(scope["current_record_count"], f"{scope_name}.current_record_count")
    features = scope["features"]
    if not isinstance(features, list) or not features:
        raise DriftMonitoringError(f"{scope_name} feature metrics must be a non-empty list.")
    if tuple(feature["feature_name"] for feature in features) != tuple(expected_features):
        raise DriftMonitoringError(f"{scope_name} feature metrics do not match expected features.")
    statuses = []
    for feature in features:
        psi = float(feature["psi"])
        if not math.isfinite(psi) or psi < 0:
            raise DriftMonitoringError(f"{scope_name} PSI must be finite and non-negative.")
        expected_status = status_for_psi(psi, config)
        if feature["status"] != expected_status:
            raise DriftMonitoringError(f"{scope_name} feature status does not match PSI.")
        statuses.append(str(feature["status"]))
        if feature["feature_type"] == "numeric":
            for field in (
                "current_count",
                "current_max",
                "current_mean",
                "current_min",
                "current_std",
                "outside_reference_range_count",
                "outside_reference_range_rate",
                "reference_count",
                "reference_max",
                "reference_mean",
                "reference_min",
                "reference_std",
            ):
                if field not in feature:
                    raise DriftMonitoringError(f"{scope_name} metric is missing {field}.")
        elif feature["feature_type"] == "categorical":
            if "unexpected_category_count" not in feature:
                raise DriftMonitoringError(
                    f"{scope_name} categorical metric lacks unexpected count."
                )
        else:
            raise DriftMonitoringError(f"{scope_name} has unsupported feature metric type.")
    if scope["overall_status"] != overall_status(statuses):
        raise DriftMonitoringError(f"{scope_name} overall status does not match feature statuses.")


def write_drift_report(report: Mapping[str, Any], root: Path | None = None) -> Path:
    path = drift_report_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pretty_json(report)
    path.write_text(payload, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != payload:
        raise DriftMonitoringError("Drift report write was not byte deterministic.")
    return path


def load_drift_report(
    root: Path | None = None, config: DriftConfig | None = None
) -> dict[str, Any]:
    root_path = root or project_root()
    path = drift_report_path(root_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriftMonitoringError(f"Drift report does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DriftMonitoringError(f"Drift report is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DriftMonitoringError("Drift report must be a JSON object.")
    validate_drift_report(payload, config or load_config(root_path))
    return payload


def deterministic_bytes(payload: Mapping[str, Any]) -> bytes:
    return pretty_json(payload).encode("utf-8")


def report_business_identity(report: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(report["monitor_version"]),
        str(report["reference_profile_sha256"]),
        str(report[AI4I_SCOPE]["current_data_hash"]),
        str(report[ANOMALY_SCOPE]["current_data_hash"]),
    )


def highest_psi_feature(scope: Mapping[str, Any]) -> dict[str, Any]:
    return max(
        scope["features"],
        key=lambda feature: (float(feature["psi"]), str(feature["feature_name"])),
    )


def build_static_summary(config: DriftConfig | None = None) -> dict[str, Any]:
    config = config or load_config(project_root())
    return {
        "drift_is_not_model_performance": True,
        "monitor_version": config.monitor_version,
        "no_alert_created_by_monitoring": True,
        "no_automatic_retraining": True,
        "numeric_bins": config.numeric_bin_count,
        "persistence_tables": ["drift_snapshots", "drift_feature_metrics"],
        "primary_metric": "Population Stability Index",
        "psi_epsilon": config.epsilon,
        "psi_monitoring_bands": {
            "drift": "PSI >= 0.25",
            "stable": "PSI < 0.10",
            "watch": "0.10 <= PSI < 0.25",
        },
        "runtime_report_path": DRIFT_REPORT_RELATIVE_PATH.as_posix(),
        "stable_snapshot_identity": [
            "monitor_version",
            "reference_profile_sha256",
            "ai4i_current_data_hash",
            "anomaly_current_data_hash",
        ],
        "supplemental_diagnostics": [
            "reference_count",
            "current_count",
            "reference_mean",
            "current_mean",
            "reference_std",
            "current_std",
            "reference_min",
            "reference_max",
            "current_min",
            "current_max",
            "standardized_mean_shift",
            "outside_reference_range_count",
            "outside_reference_range_rate",
            "categorical_proportions",
            "unexpected_category_count",
        ],
        "monitor_scopes": {
            AI4I_SCOPE: {
                "features": list(AI4I_FEATURES),
                "reference_identity": config.ai4i_model_input,
            },
            ANOMALY_SCOPE: {
                "features": list(ANOMALY_FEATURES),
                "reference_identity": config.operational_anomaly_inputs,
            },
        },
    }


def write_static_summary(root: Path | None = None) -> Path:
    root_path = root or project_root()
    config = load_config(root_path)
    path = static_summary_path(root_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(build_static_summary(config)), encoding="utf-8", newline="\n")
    return path

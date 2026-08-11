"""Local AI4I model packaging and inference utilities."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.evaluation import ai4i_final_evaluation
from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

MODEL_NAME = "ai4i-failure-risk-random-forest"
MODEL_VERSION = "1.0.0"
SERIALIZATION_FORMAT = "joblib"
ARTIFACT_RELATIVE_PATH = Path("ml") / "artifacts" / "ai4i" / "final_model.joblib"
ARTIFACT_METADATA_RELATIVE_PATH = Path("ml") / "artifacts" / "ai4i" / "artifact_metadata.json"
PACKAGING_SUMMARY_RELATIVE_PATH = Path("reports") / "ai4i" / "model_packaging_summary.json"
SAMPLE_INPUT_RELATIVE_PATH = Path("data") / "sample" / "ai4i_inference_examples.json"
ALLOWED_TYPE_VALUES = {"L", "M", "H"}
FORBIDDEN_INPUT_FIELDS = {
    "Machine failure",
    "source_udi",
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}


@dataclass(frozen=True)
class PackagedModelResult:
    """Information produced by local model packaging."""

    model_name: str
    model_version: str
    final_config_hash: str
    model_artifact_sha256: str
    artifact_path: Path
    metadata_path: Path
    packaging_summary_path: Path
    training_row_count: int
    training_positive_count: int
    joblib_version: str


@dataclass(frozen=True)
class AI4IPredictor:
    """Trusted local predictor wrapper for the packaged AI4I model."""

    pipeline: Pipeline
    final_config: dict[str, Any]
    final_config_hash: str
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION

    @property
    def decision_threshold(self) -> float:
        return float(self.final_config["decision_threshold"])

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        features = validate_inference_records(records, self.final_config)
        probabilities = self.pipeline.predict_proba(features)[:, ai4i_baseline.POSITIVE_CLASS]
        return prediction_outputs(
            probabilities,
            self.decision_threshold,
            self.model_name,
            self.model_version,
            self.final_config_hash,
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ARTIFACT_RELATIVE_PATH


def artifact_metadata_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ARTIFACT_METADATA_RELATIVE_PATH


def packaging_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / PACKAGING_SUMMARY_RELATIVE_PATH


def sample_input_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SAMPLE_INPUT_RELATIVE_PATH


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def required_input_fields(final_config: Mapping[str, Any]) -> list[str]:
    return list(final_config["predictive_features"])


def numerical_input_fields(final_config: Mapping[str, Any]) -> list[str]:
    return list(final_config["numerical_features"])


def validate_inference_record(
    record: Mapping[str, Any],
    final_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("Each inference record must be a JSON object.")

    expected_fields = required_input_fields(final_config)
    expected_set = set(expected_fields)
    actual_set = set(record.keys())
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    forbidden = sorted(actual_set & FORBIDDEN_INPUT_FIELDS)
    if missing:
        raise ValueError("Missing required inference field(s): " + ", ".join(missing))
    if forbidden:
        raise ValueError("Forbidden inference field(s): " + ", ".join(forbidden))
    if unexpected:
        raise ValueError("Unexpected inference field(s): " + ", ".join(unexpected))

    type_value = record["Type"]
    if not isinstance(type_value, str):
        raise ValueError("Inference field `Type` must be a string.")
    if type_value not in ALLOWED_TYPE_VALUES:
        raise ValueError("Inference field `Type` must be one of L, M, or H.")

    validated: dict[str, Any] = {"Type": type_value}
    for field in numerical_input_fields(final_config):
        value = record[field]
        if value is None:
            raise ValueError(f"Inference field `{field}` must not be null.")
        if isinstance(value, bool):
            raise ValueError(f"Inference field `{field}` must be numeric, not boolean.")
        if not isinstance(value, Real):
            raise ValueError(f"Inference field `{field}` must be numeric.")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"Inference field `{field}` must be finite.")
        validated[field] = numeric_value
    return {field: validated[field] for field in expected_fields}


def validate_inference_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    final_config: Mapping[str, Any],
) -> pd.DataFrame:
    if isinstance(payload, Mapping):
        records = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        records = list(payload)
    else:
        raise ValueError("Inference payload must be one JSON object or an array of objects.")

    if not records:
        raise ValueError("Inference payload must contain at least one record.")

    validated_records = [validate_inference_record(record, final_config) for record in records]
    return pd.DataFrame(validated_records, columns=required_input_fields(final_config))


def validate_inference_records(
    records: Sequence[Mapping[str, Any]],
    final_config: Mapping[str, Any],
) -> pd.DataFrame:
    return validate_inference_payload(records, final_config)


def probability_to_prediction(probability: float, threshold: float) -> int:
    return int(float(probability) >= float(threshold))


def prediction_outputs(
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    model_name: str,
    model_version: str,
    final_config_hash: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for probability in probabilities:
        probability_value = float(probability)
        outputs.append(
            {
                "failure_probability": round(probability_value, 6),
                "failure_prediction": probability_to_prediction(
                    probability_value,
                    threshold,
                ),
                "decision_threshold": float(threshold),
                "model_name": model_name,
                "model_version": model_version,
                "final_config_hash": final_config_hash,
            }
        )
    return outputs


def load_final_config(root: Path | None = None) -> dict[str, Any]:
    return ai4i_final_evaluation.load_final_model_config(
        ai4i_final_evaluation.final_config_path(root)
    )


def current_final_config_hash(final_config: Mapping[str, Any]) -> str:
    return ai4i_final_evaluation.final_config_hash(final_config)


def validate_final_config_provenance(root: Path, final_config: Mapping[str, Any]) -> str:
    config_hash = current_final_config_hash(final_config)
    metrics = ai4i_final_evaluation.load_json(ai4i_final_evaluation.final_test_metrics_path(root))
    decision = ai4i_final_evaluation.load_json(
        ai4i_final_evaluation.final_model_decision_path(root)
    )
    metrics_hash = metrics.get("final_model_configuration_hash")
    decision_hash = decision.get("final_decision", {}).get("configuration_hash")
    if metrics_hash != config_hash:
        raise ValueError("Frozen final config hash does not match final metrics provenance.")
    if decision_hash != config_hash:
        raise ValueError("Frozen final config hash does not match final decision provenance.")
    return config_hash


def load_and_validate_final_config(root: Path) -> tuple[dict[str, Any], str]:
    final_config = load_final_config(root)
    modeling_config = ai4i_modeling.load_modeling_config(ai4i_modeling.config_path(root))
    comparison_metrics = ai4i_final_evaluation.load_json(
        ai4i_final_evaluation.model_comparison_metrics_path(root)
    )
    tuning_metrics = ai4i_final_evaluation.load_json(
        ai4i_final_evaluation.random_forest_tuning_metrics_path(root)
    )
    ai4i_final_evaluation.validate_final_model_config(
        final_config,
        modeling_config,
        comparison_metrics,
        tuning_metrics,
    )
    config_hash = validate_final_config_provenance(root, final_config)
    return final_config, config_hash


def load_development_training_data(
    root: Path,
    modeling_config: ai4i_modeling.ModelingConfig,
) -> pd.DataFrame:
    split_summary = ai4i_final_evaluation.load_split_summary(root)
    train_df, validation_df = ai4i_baseline.load_training_and_validation_frames(root)
    ai4i_baseline.validate_training_inputs(
        train_df,
        validation_df,
        modeling_config,
        split_summary,
    )
    return ai4i_final_evaluation.combine_development_training_data(
        train_df,
        validation_df,
        modeling_config,
    )


def build_and_fit_packaged_pipeline(
    root: Path,
    final_config: Mapping[str, Any],
) -> tuple[Pipeline, pd.DataFrame]:
    modeling_config = ai4i_modeling.load_modeling_config(ai4i_modeling.config_path(root))
    development_df = load_development_training_data(root, modeling_config)
    pipeline = ai4i_final_evaluation.fit_final_pipeline(
        development_df,
        modeling_config,
        final_config,
    )
    return pipeline, development_df


def build_artifact_metadata(
    root: Path,
    final_config: Mapping[str, Any],
    config_hash: str,
    artifact_sha256: str,
    development_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "decision_threshold": float(final_config["decision_threshold"]),
        "final_config_hash": config_hash,
        "joblib_version": joblib.__version__,
        "model_artifact_sha256": artifact_sha256,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predictive_feature_list": required_input_fields(final_config),
        "python_version": platform.python_version(),
        "relative_local_artifact_path": relative_posix(artifact_path(root), root),
        "scikit_learn_version": sklearn.__version__,
        "serialization_format": SERIALIZATION_FORMAT,
        "training_positive_count": int(development_df[final_config["target"]].sum()),
        "training_row_count": int(len(development_df)),
    }


def build_packaging_summary(
    root: Path,
    final_config: Mapping[str, Any],
    config_hash: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    modeling_config = ai4i_modeling.load_modeling_config(ai4i_modeling.config_path(root))
    return {
        "decision_threshold": float(final_config["decision_threshold"]),
        "feature_policy": ai4i_final_evaluation.feature_policy_from_config(modeling_config),
        "final_config_hash": config_hash,
        "frozen_threshold": float(final_config["decision_threshold"]),
        "joblib_version": str(metadata["joblib_version"]),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "python_version": str(metadata["python_version"]),
        "relative_local_artifact_path": relative_posix(artifact_path(root), root),
        "scikit_learn_version": str(metadata["scikit_learn_version"]),
        "serialization_format": SERIALIZATION_FORMAT,
        "test_data_used_for_packaging": False,
        "training_positive_count": int(metadata["training_positive_count"]),
        "training_row_count": int(metadata["training_row_count"]),
    }


def package_model(root: Path | None = None) -> PackagedModelResult:
    root_path = root or project_root()
    final_config, config_hash = load_and_validate_final_config(root_path)
    pipeline, development_df = build_and_fit_packaged_pipeline(root_path, final_config)

    model_path = artifact_path(root_path)
    metadata_path = artifact_metadata_path(root_path)
    summary_path = packaging_summary_path(root_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    artifact_sha256 = file_sha256(model_path)

    metadata = build_artifact_metadata(
        root_path,
        final_config,
        config_hash,
        artifact_sha256,
        development_df,
    )
    write_json(metadata, metadata_path)
    write_json(
        build_packaging_summary(root_path, final_config, config_hash, metadata),
        summary_path,
    )

    return PackagedModelResult(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        final_config_hash=config_hash,
        model_artifact_sha256=artifact_sha256,
        artifact_path=model_path,
        metadata_path=metadata_path,
        packaging_summary_path=summary_path,
        training_row_count=int(len(development_df)),
        training_positive_count=int(development_df[final_config["target"]].sum()),
        joblib_version=joblib.__version__,
    )


def load_artifact_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact metadata was not found: {path}")
    metadata = load_json(path)
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact metadata must be a JSON object.")
    return metadata


def metadata_contains_test_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"test_metrics", "test_performance", "test_results"}:
                return True
            if key_text.startswith("test_") and key_text != "test_data_used_for_packaging":
                return True
            if metadata_contains_test_metrics(child):
                return True
    if isinstance(value, list):
        return any(metadata_contains_test_metrics(item) for item in value)
    return False


def validate_artifact_metadata(
    metadata: Mapping[str, Any],
    model_path: Path,
    final_config: Mapping[str, Any],
    config_hash: str,
) -> None:
    if metadata.get("model_name") != MODEL_NAME:
        raise ValueError("Model artifact metadata has an unexpected model name.")
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Model artifact metadata has an unexpected model version.")
    if metadata.get("final_config_hash") != config_hash:
        raise ValueError("Model artifact metadata config hash does not match the frozen config.")
    if metadata.get("decision_threshold") != float(final_config["decision_threshold"]):
        raise ValueError("Model artifact metadata threshold does not match the frozen config.")
    if metadata.get("serialization_format") != SERIALIZATION_FORMAT:
        raise ValueError("Model artifact metadata has an unexpected serialization format.")
    if metadata.get("predictive_feature_list") != required_input_fields(final_config):
        raise ValueError("Model artifact metadata predictive features do not match the config.")
    expected_sha = metadata.get("model_artifact_sha256")
    if not isinstance(expected_sha, str) or expected_sha != file_sha256(model_path):
        raise ValueError("Model artifact binary SHA-256 does not match metadata.")
    if metadata_contains_test_metrics(metadata):
        raise ValueError("Model artifact metadata must not contain test metrics.")


def validate_pipeline_structure(
    loaded_model: Any,
    final_config: Mapping[str, Any],
) -> Pipeline:
    if not isinstance(loaded_model, Pipeline):
        raise ValueError("Loaded model artifact must be a scikit-learn Pipeline.")
    if (
        "preprocessor" not in loaded_model.named_steps
        or "classifier" not in loaded_model.named_steps
    ):
        raise ValueError("Loaded pipeline must contain preprocessor and classifier steps.")

    preprocessor = loaded_model.named_steps["preprocessor"]
    classifier = loaded_model.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError("Loaded preprocessor must be a ColumnTransformer.")
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Loaded classifier must be a RandomForestClassifier.")

    params = final_config["hyperparameters"]
    for key, expected_value in params.items():
        if classifier.get_params()[key] != expected_value:
            raise ValueError(f"Random Forest hyperparameter `{key}` does not match config.")

    transformers = {name: transformer for name, transformer, _columns in preprocessor.transformers}
    if "categorical" not in transformers or "numerical" not in transformers:
        raise ValueError("Loaded preprocessor must contain categorical and numerical transformers.")
    categorical = transformers["categorical"]
    if not isinstance(categorical, Pipeline):
        raise ValueError("Categorical transformer must be a Pipeline.")
    encoder = categorical.named_steps.get("one_hot_encoder")
    if not isinstance(encoder, OneHotEncoder) or encoder.handle_unknown != "ignore":
        raise ValueError("Categorical transformer must use OneHotEncoder(handle_unknown='ignore').")
    if transformers["numerical"] != "passthrough":
        raise ValueError("Numerical transformer must use passthrough.")
    return loaded_model


def load_predictor(
    root: Path | None = None,
    *,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    final_config: dict[str, Any] | None = None,
) -> AI4IPredictor:
    root_path = root or project_root()
    final_config = final_config or load_final_config(root_path)
    config_hash = current_final_config_hash(final_config)
    model_artifact = model_path or artifact_path(root_path)
    metadata_file = metadata_path or artifact_metadata_path(root_path)
    if not model_artifact.exists():
        raise FileNotFoundError(
            "Packaged AI4I model artifact is missing. Run "
            "scripts/package_ai4i_final_model.py first."
        )
    metadata = load_artifact_metadata(metadata_file)
    validate_artifact_metadata(metadata, model_artifact, final_config, config_hash)
    loaded_model = joblib.load(model_artifact)
    pipeline = validate_pipeline_structure(loaded_model, final_config)
    return AI4IPredictor(
        pipeline=pipeline,
        final_config=final_config,
        final_config_hash=config_hash,
    )


def load_inference_payload(path: Path) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return payload
    raise ValueError("Inference input JSON must contain one object or an array of objects.")

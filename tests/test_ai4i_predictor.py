from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.evaluation import ai4i_final_evaluation
from ml.inference import ai4i_predictor
from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

NUMERIC_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
LEAKAGE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def modeling_config() -> ai4i_modeling.ModelingConfig:
    return ai4i_modeling.ModelingConfig(
        dataset_name="AI4I 2020 Predictive Maintenance Dataset",
        modeling_objective="Binary classification of Machine failure",
        target_column="Machine failure",
        categorical_features=("Type",),
        numerical_features=NUMERIC_COLUMNS,
        traceability_fields=("UDI",),
        derived_traceability_field="source_udi",
        excluded_identifiers=("Product ID",),
        excluded_leakage_sensitive_columns=LEAKAGE_COLUMNS,
        forbidden_feature_sources=("Machine failure", *LEAKAGE_COLUMNS),
        random_seed=42,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        stratify_on="Machine failure",
        future_preprocessing_design={},
    )


def final_config() -> dict[str, object]:
    return {
        "categorical_features": ["Type"],
        "decision_threshold": 0.14,
        "excluded_identifier_fields": ["Product ID"],
        "excluded_leakage_sensitive_fields": list(LEAKAGE_COLUMNS),
        "final_evaluation_data_policy": "test only",
        "hyperparameters": dict(ai4i_final_evaluation.FROZEN_HYPERPARAMETERS),
        "model_family": "RandomForestClassifier",
        "numerical_features": list(NUMERIC_COLUMNS),
        "predictive_features": ["Type", *NUMERIC_COLUMNS],
        "preprocessing_policy": ai4i_final_evaluation.preprocessing_policy(),
        "random_seed": 42,
        "target": "Machine failure",
        "traceability_field": "source_udi",
        "training_data_policy": "train + validation",
    }


def valid_record(record_type: str = "L") -> dict[str, object]:
    return {
        "Type": record_type,
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1450,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 92,
    }


def training_frame(row_count: int = 18) -> pd.DataFrame:
    rows = []
    for offset in range(row_count):
        rows.append(
            {
                "source_udi": offset + 1,
                "Type": ["L", "M", "H"][offset % 3],
                "Air temperature [K]": 296.0 + offset * 0.3,
                "Process temperature [K]": 306.0 + offset * 0.2,
                "Rotational speed [rpm]": 1200 + offset * 20,
                "Torque [Nm]": 32.0 + offset,
                "Tool wear [min]": offset * 5,
                "Machine failure": 1 if offset % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def fitted_pipeline() -> object:
    cfg = modeling_config()
    train = training_frame()
    features, target = ai4i_baseline.extract_features_and_target(train, cfg)
    pipeline = ai4i_final_evaluation.build_frozen_random_forest_pipeline(cfg, final_config())
    pipeline.fit(features, target)
    return pipeline


def write_temp_artifact(
    tmp_path: Path,
    *,
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    cfg = final_config()
    pipeline = fitted_pipeline()
    model_path = ai4i_predictor.artifact_path(tmp_path)
    metadata_path = ai4i_predictor.artifact_metadata_path(tmp_path)
    model_path.parent.mkdir(parents=True)
    joblib.dump(pipeline, model_path)
    config_hash = ai4i_predictor.current_final_config_hash(cfg)
    metadata = ai4i_predictor.build_artifact_metadata(
        tmp_path,
        cfg,
        config_hash,
        ai4i_predictor.file_sha256(model_path),
        training_frame(),
    )
    if metadata_overrides:
        metadata.update(metadata_overrides)
    ai4i_predictor.write_json(metadata, metadata_path)
    return model_path, metadata_path, cfg


def test_exact_required_input_feature_set():
    assert ai4i_predictor.required_input_fields(final_config()) == ["Type", *NUMERIC_COLUMNS]


def test_single_record_validation():
    frame = ai4i_predictor.validate_inference_payload(valid_record("L"), final_config())

    assert list(frame.columns) == ["Type", *NUMERIC_COLUMNS]
    assert len(frame) == 1
    assert frame.iloc[0]["Type"] == "L"


def test_batch_validation_maintains_input_order():
    frame = ai4i_predictor.validate_inference_payload(
        [valid_record("L"), valid_record("M"), valid_record("H")],
        final_config(),
    )

    assert frame["Type"].tolist() == ["L", "M", "H"]


def test_missing_field_rejection():
    record = valid_record()
    del record["Torque [Nm]"]

    with pytest.raises(ValueError, match="Missing required"):
        ai4i_predictor.validate_inference_payload(record, final_config())


def test_unknown_field_rejection():
    record = valid_record()
    record["unknown"] = 1

    with pytest.raises(ValueError, match="Unexpected"):
        ai4i_predictor.validate_inference_payload(record, final_config())


def test_invalid_type_rejection():
    record = valid_record("X")

    with pytest.raises(ValueError, match="one of L, M, or H"):
        ai4i_predictor.validate_inference_payload(record, final_config())


@pytest.mark.parametrize("bad_value", [np.nan, float("inf"), -float("inf")])
def test_nan_and_infinity_rejection(bad_value: float):
    record = valid_record()
    record["Torque [Nm]"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        ai4i_predictor.validate_inference_payload(record, final_config())


def test_boolean_numeric_rejection():
    record = valid_record()
    record["Torque [Nm]"] = True

    with pytest.raises(ValueError, match="not boolean"):
        ai4i_predictor.validate_inference_payload(record, final_config())


@pytest.mark.parametrize("field", ["TWF", "HDF", "PWF", "OSF", "RNF"])
def test_leakage_sensitive_field_rejection(field: str):
    record = valid_record()
    record[field] = 0

    with pytest.raises(ValueError, match="Forbidden"):
        ai4i_predictor.validate_inference_payload(record, final_config())


@pytest.mark.parametrize("field", ["source_udi", "UDI", "Product ID"])
def test_identifier_field_rejection(field: str):
    record = valid_record()
    record[field] = 123

    with pytest.raises(ValueError, match="Forbidden"):
        ai4i_predictor.validate_inference_payload(record, final_config())


def test_probability_to_prediction_uses_frozen_threshold():
    assert ai4i_predictor.probability_to_prediction(0.1399, 0.14) == 0
    assert ai4i_predictor.probability_to_prediction(0.14, 0.14) == 1


def test_prediction_output_identity_and_config_hash():
    config_hash = ai4i_predictor.current_final_config_hash(final_config())
    outputs = ai4i_predictor.prediction_outputs(
        [0.1, 0.2],
        0.14,
        ai4i_predictor.MODEL_NAME,
        ai4i_predictor.MODEL_VERSION,
        config_hash,
    )

    assert outputs[0]["failure_prediction"] == 0
    assert outputs[1]["failure_prediction"] == 1
    assert {item["model_name"] for item in outputs} == {ai4i_predictor.MODEL_NAME}
    assert {item["model_version"] for item in outputs} == {ai4i_predictor.MODEL_VERSION}
    assert {item["final_config_hash"] for item in outputs} == {config_hash}


def test_model_save_load_round_trip_with_temporary_artifact(tmp_path: Path):
    write_temp_artifact(tmp_path)
    predictor = ai4i_predictor.load_predictor(tmp_path, final_config=final_config())

    outputs = predictor.predict_batch([valid_record("L"), valid_record("M")])

    assert len(outputs) == 2
    assert all(0 <= item["failure_probability"] <= 1 for item in outputs)


def test_binary_hash_validation(tmp_path: Path):
    write_temp_artifact(
        tmp_path,
        metadata_overrides={"model_artifact_sha256": "0" * 64},
    )

    with pytest.raises(ValueError, match="SHA-256"):
        ai4i_predictor.load_predictor(tmp_path, final_config=final_config())


def test_metadata_mismatch_failure(tmp_path: Path):
    write_temp_artifact(tmp_path, metadata_overrides={"model_version": "9.9.9"})

    with pytest.raises(ValueError, match="model version"):
        ai4i_predictor.load_predictor(tmp_path, final_config=final_config())


def test_loaded_prediction_equals_in_memory_prediction(tmp_path: Path):
    model_path, _metadata_path, cfg = write_temp_artifact(tmp_path)
    in_memory_pipeline = joblib.load(model_path)
    config_hash = ai4i_predictor.current_final_config_hash(cfg)
    in_memory = ai4i_predictor.AI4IPredictor(
        pipeline=in_memory_pipeline,
        final_config=cfg,
        final_config_hash=config_hash,
    )
    loaded = ai4i_predictor.load_predictor(tmp_path, final_config=cfg)
    records = [valid_record("L"), valid_record("M"), valid_record("H")]

    assert loaded.predict_batch(records) == in_memory.predict_batch(records)


class ConflictingThresholdPipeline:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(features), 0.2)
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=int)


def test_inference_uses_predict_proba_threshold_not_estimator_predict():
    config_hash = ai4i_predictor.current_final_config_hash(final_config())
    predictor = ai4i_predictor.AI4IPredictor(
        pipeline=ConflictingThresholdPipeline(),
        final_config=final_config(),
        final_config_hash=config_hash,
    )

    output = predictor.predict_one(valid_record())

    assert output["failure_probability"] == 0.2
    assert output["failure_prediction"] == 1


def test_packaging_sources_do_not_reference_restricted_split_file():
    project_root = Path(__file__).resolve().parents[1]
    guarded_files = [
        project_root / "scripts" / "package_ai4i_final_model.py",
        project_root / "ml" / "inference" / "ai4i_predictor.py",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        assert "test.csv" not in source
        assert "TEST_RELATIVE_PATH" not in source


def test_load_inference_payload_accepts_object_and_array(tmp_path: Path):
    object_path = tmp_path / "single.json"
    array_path = tmp_path / "batch.json"
    object_path.write_text(json.dumps(valid_record()), encoding="utf-8")
    array_path.write_text(json.dumps([valid_record("L"), valid_record("H")]), encoding="utf-8")

    assert isinstance(ai4i_predictor.load_inference_payload(object_path), dict)
    assert len(ai4i_predictor.load_inference_payload(array_path)) == 2

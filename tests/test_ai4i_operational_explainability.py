from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.explainability import ai4i_shap, ai4i_telemetry_shap
from ml.inference import ai4i_predictor, ai4i_telemetry
from services.database import ai4i_explanations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_HASH = "a" * 64
EXPLANATION_HASH = "b" * 64
PAYLOAD_HASH = "c" * 64
EVENT_ID = "00000000-0000-4000-8000-000000000001"
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
PREDICTIVE_FEATURES = ["Type", *NUMERICAL_FEATURES]


def final_config() -> dict[str, Any]:
    return {
        "categorical_features": ["Type"],
        "decision_threshold": 0.14,
        "numerical_features": NUMERICAL_FEATURES,
        "predictive_features": PREDICTIVE_FEATURES,
    }


def model_input() -> dict[str, Any]:
    return {
        "Type": "L",
        "Air temperature [K]": 300.1,
        "Process temperature [K]": 309.2,
        "Rotational speed [rpm]": 1450.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 20.0,
    }


def input_hash() -> str:
    return ai4i_telemetry.model_input_sha256(model_input(), final_config())


def prediction_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "adapter_version": "1.0",
        "event_id": EVENT_ID,
        "event_time": "2026-01-01 12:00:00",
        "failure_prediction": 0,
        "failure_probability": 0.1,
        "final_config_hash": CONFIG_HASH,
        "frozen_threshold": 0.14,
        "machine_code": "MCH-0001",
        "model_input_sha256": input_hash(),
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_version": ai4i_predictor.MODEL_VERSION,
        "payload_sha256": PAYLOAD_HASH,
        "source_kafka_key": "MCH-0001",
        "source_kafka_offset": 10,
        "source_kafka_partition": 1,
        "source_kafka_timestamp": "2026-01-01 12:00:01.250",
        "source_kafka_topic": "industrial.telemetry.v1",
    }
    record.update(overrides)
    return record


def adapter_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "adapter_version": "1.0",
        "event_id": EVENT_ID,
        "event_time": "2026-01-01 12:00:00",
        "machine_code": "MCH-0001",
        "model_input": model_input(),
        "source_lineage": {
            "payload_sha256": PAYLOAD_HASH,
            "source_kafka_key": "MCH-0001",
            "source_kafka_offset": 10,
            "source_kafka_partition": 1,
            "source_kafka_timestamp": "2026-01-01 12:00:01.250",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
    }
    record.update(overrides)
    return record


def explanation_dict(**overrides: object) -> dict[str, object]:
    record = {
        "additivity_error": 0.0,
        "attribution_semantics": ai4i_telemetry_shap.ATTRIBUTION_SEMANTICS,
        "base_value": 0.12,
        "contribution_sum": -0.02,
        "event_id": EVENT_ID,
        "event_time": "2026-01-01 12:00:00.000",
        "explainer_name": ai4i_telemetry_shap.EXPLAINER_NAME,
        "explainer_version": "0.52.0",
        "explanation_config_hash": EXPLANATION_HASH,
        "failure_prediction": False,
        "failure_probability": 0.1,
        "feature_contributions": [
            {"feature_name": "Type", "feature_value": "L", "shap_value": 0.003},
            {
                "feature_name": "Air temperature [K]",
                "feature_value": 300.1,
                "shap_value": -0.004,
            },
            {
                "feature_name": "Process temperature [K]",
                "feature_value": 309.2,
                "shap_value": -0.002,
            },
            {
                "feature_name": "Rotational speed [rpm]",
                "feature_value": 1450.0,
                "shap_value": 0.006,
            },
            {"feature_name": "Torque [Nm]", "feature_value": 42.0, "shap_value": -0.018},
            {"feature_name": "Tool wear [min]", "feature_value": 20.0, "shap_value": -0.005},
        ],
        "final_config_hash": CONFIG_HASH,
        "frozen_threshold": 0.14,
        "machine_code": "MCH-0001",
        "model_input_sha256": input_hash(),
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_output_value": 0.1,
        "model_version": ai4i_predictor.MODEL_VERSION,
        "negative_contribution_semantics": ai4i_telemetry_shap.NEGATIVE_CONTRIBUTION_SEMANTICS,
        "output_semantics": ai4i_telemetry_shap.OUTPUT_SEMANTICS,
        "payload_sha256": PAYLOAD_HASH,
        "positive_contribution_semantics": ai4i_telemetry_shap.POSITIVE_CONTRIBUTION_SEMANTICS,
        "source_kafka_key": "MCH-0001",
        "source_kafka_offset": 10,
        "source_kafka_partition": 1,
        "source_kafka_timestamp": "2026-01-01 12:00:01.250",
        "source_kafka_topic": "industrial.telemetry.v1",
    }
    record.update(overrides)
    return record


def explanation_record(**overrides: object) -> ai4i_telemetry_shap.ExplanationRecord:
    return ai4i_telemetry_shap.validate_explanation_record(explanation_dict(**overrides))


def prediction_lookup() -> ai4i_explanations.PredictionLookupRow:
    return ai4i_explanations.PredictionLookupRow(
        model_prediction_id=10,
        machine_id=1,
        machine_code="MCH-0001",
        event_id=EVENT_ID,
        event_time="2026-01-01 12:00:00.000",
        failure_probability=0.1,
        failure_prediction=False,
        frozen_threshold=0.14,
        model_name=ai4i_predictor.MODEL_NAME,
        model_version=ai4i_predictor.MODEL_VERSION,
        final_config_hash=CONFIG_HASH,
        model_input_sha256=input_hash(),
    )


def test_runtime_prediction_validation_normalizes_identity() -> None:
    record = ai4i_telemetry_shap.validate_prediction_record(prediction_record())

    assert record.event_time == "2026-01-01 12:00:00.000"
    assert record.prediction_identity == (
        EVENT_ID,
        ai4i_predictor.MODEL_NAME,
        ai4i_predictor.MODEL_VERSION,
        CONFIG_HASH,
    )


def test_model_input_hash_alignment_uses_adapter_model_input() -> None:
    prediction = ai4i_telemetry_shap.validate_prediction_record(prediction_record())

    aligned = ai4i_telemetry_shap.validate_prediction_input_alignment(
        prediction,
        adapter_record(),
        final_config(),
    )

    assert aligned == model_input()


def test_model_input_hash_mismatch_is_rejected() -> None:
    prediction = ai4i_telemetry_shap.validate_prediction_record(
        prediction_record(model_input_sha256="d" * 64)
    )

    with pytest.raises(ai4i_telemetry_shap.AI4ITelemetryExplainabilityError, match="sha256"):
        ai4i_telemetry_shap.validate_prediction_input_alignment(
            prediction,
            adapter_record(),
            final_config(),
        )


def test_exact_six_feature_contract_is_enforced() -> None:
    assert ai4i_telemetry_shap.validate_feature_contract(PREDICTIVE_FEATURES) == tuple(
        PREDICTIVE_FEATURES
    )

    with pytest.raises(ai4i_telemetry_shap.AI4ITelemetryExplainabilityError):
        ai4i_telemetry_shap.validate_feature_contract(["Type", "Torque [Nm]"])


def test_build_explanation_record_from_grouped_values() -> None:
    prediction = ai4i_telemetry_shap.validate_prediction_record(prediction_record())
    records = ai4i_telemetry_shap.build_explanation_records(
        [prediction],
        [model_input()],
        PREDICTIVE_FEATURES,
        np.array([[0.003, -0.004, -0.002, 0.006, -0.018, -0.005]]),
        0.12,
        [0.1],
        [0.0],
        EXPLANATION_HASH,
    )

    assert len(records) == 1
    assert records[0].feature_contributions[0].feature_name == "Type"
    assert records[0].contribution_sum == -0.02


def test_shap_serialization_is_deterministic() -> None:
    record = explanation_record()

    first = ai4i_telemetry_shap.explanation_record_json(record)
    second = ai4i_telemetry_shap.explanation_record_json(record)

    assert first == second
    assert json.loads(first)["feature_contributions"][0]["feature_name"] == "Type"


def test_additivity_validation_rejects_inconsistent_records() -> None:
    with pytest.raises(ai4i_telemetry_shap.AI4ITelemetryExplainabilityError, match="additivity"):
        explanation_record(model_output_value=0.2)


def test_stable_explanation_identity_includes_explainer_config() -> None:
    record = explanation_record()

    assert record.stable_identity == (
        EVENT_ID,
        ai4i_predictor.MODEL_NAME,
        ai4i_predictor.MODEL_VERSION,
        CONFIG_HASH,
        ai4i_telemetry_shap.EXPLAINER_NAME,
        EXPLANATION_HASH,
    )


def test_explanation_reuse_and_conflict_detection_are_pure() -> None:
    record = explanation_record()
    lookup = prediction_lookup()
    existing = ai4i_explanations.ExistingExplanationRow(10, 1, record)

    reuse = ai4i_explanations.summarize_explanation_reuse(
        [record],
        {existing.db_stable_identity: existing},
        {record.prediction_identity: lookup},
    )

    assert reuse.new_records == 0
    assert reuse.existing_identical_records == 1
    assert reuse.conflicts == ()

    conflicting_existing = ai4i_explanations.ExistingExplanationRow(
        10,
        1,
        replace(record, base_value=0.11),
    )
    conflict = ai4i_explanations.summarize_explanation_reuse(
        [record],
        {conflicting_existing.db_stable_identity: conflicting_existing},
        {record.prediction_identity: lookup},
    )

    assert conflict.conflicts
    assert "base_value" in conflict.conflicts[0].fields


def test_database_row_conversion_strips_database_only_fields() -> None:
    row = explanation_dict(model_prediction_id=10, machine_id=1)

    existing = ai4i_explanations.db_row_to_existing_explanation(row)

    assert existing.model_prediction_id == 10
    assert existing.machine_id == 1
    assert existing.record.event_id == EVENT_ID


def test_persistence_sql_is_idempotent_and_scoped() -> None:
    record = explanation_record()
    sql = ai4i_explanations.build_persistence_transaction(
        [record],
        {record.prediction_identity: prediction_lookup()},
    )

    assert "INSERT INTO prediction_explanations" in sql
    assert "ON CONFLICT (" in sql
    assert "DO NOTHING" in sql
    assert "INSERT INTO model_predictions" not in sql
    assert "UPDATE machine_health" not in sql
    assert "INSERT INTO anomalies" not in sql
    assert "TRUNCATE" not in sql
    assert "DELETE FROM" not in sql


def test_static_summaries_are_runtime_free() -> None:
    operational = ai4i_telemetry_shap.build_static_summary(CONFIG_HASH)
    persistence = ai4i_explanations.build_static_summary()
    rendered = json.dumps({"operational": operational, "persistence": persistence})

    assert "data/explanations/ai4i/telemetry_explanations.jsonl" in rendered
    assert "generated_at" not in rendered
    assert "runtime_timestamp" not in rendered
    assert "password" not in rendered


def test_source_guards_prevent_wrong_operational_sources() -> None:
    guarded_files = [
        PROJECT_ROOT / "ml" / "explainability" / "ai4i_telemetry_shap.py",
        PROJECT_ROOT / "scripts" / "explain_ai4i_telemetry_predictions.py",
    ]
    forbidden_terms = [
        "test" + ".csv",
        "." + "fit(",
        "." + "fit_transform(",
        "data/gold",
        "Silver duplicates",
        "Silver quarantine",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"


def test_persistence_source_does_not_compute_explanations() -> None:
    guarded_files = [
        PROJECT_ROOT / "services" / "database" / "ai4i_explanations.py",
        PROJECT_ROOT / "scripts" / "persist_ai4i_explanations.py",
        PROJECT_ROOT / "scripts" / "inspect_ai4i_explanation_state.py",
    ]
    forbidden_terms = [
        "Tree" + "Explainer",
        "predict_" + "proba",
        "predict_" + "batch",
        "load_" + "trusted_predictor",
        "." + "fit(",
        "test" + ".csv",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"


def test_frontend_source_does_not_compute_shap_or_read_runtime_jsonl() -> None:
    runtime_source_paths = [
        path
        for path in (PROJECT_ROOT / "apps" / "web" / "src").rglob("*.ts*")
        if "test" not in path.relative_to(PROJECT_ROOT / "apps" / "web" / "src").parts
        and ".test." not in path.name
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_source_paths)
    forbidden_terms = [
        "Tree" + "Explainer",
        "predict_" + "proba",
        "final_model" + ".joblib",
        "telemetry_explanations" + ".jsonl",
        "data/explanations",
    ]

    for term in forbidden_terms:
        assert term not in source


def test_existing_shap_grouping_matches_operational_feature_contract() -> None:
    grouped_names, grouped_values = ai4i_shap.grouped_contribution_matrix(
        ["Type_H", "Type_L", "Air temperature [K]", "Torque [Nm]"],
        np.array([[0.2, -0.1, 0.05, -0.03]]),
        final_config(),
    )

    assert tuple(grouped_names) == ai4i_telemetry_shap.EXPECTED_SEMANTIC_FEATURES
    assert grouped_values.shape == (1, 6)

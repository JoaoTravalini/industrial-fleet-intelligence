from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.inference import ai4i_predictor
from services.database import ai4i_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASH_A = "a" * 64
HASH_B = "b" * 64


def prediction_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "adapter_version": "1.0",
        "event_id": "00000000-0000-4000-8000-000000000001",
        "event_time": "2026-01-01 00:00:00",
        "failure_prediction": 0,
        "failure_probability": 0.05,
        "final_config_hash": HASH_A,
        "frozen_threshold": 0.14,
        "machine_code": "MCH-0001",
        "model_input_sha256": HASH_B,
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_version": ai4i_predictor.MODEL_VERSION,
        "payload_sha256": "c" * 64,
        "source_kafka_key": "MCH-0001",
        "source_kafka_offset": 10,
        "source_kafka_partition": 1,
        "source_kafka_timestamp": "2026-01-01 00:00:01.250",
        "source_kafka_topic": "industrial.telemetry.v1",
    }
    record.update(overrides)
    return record


def validated(**overrides: object) -> ai4i_predictions.PredictionRecord:
    return ai4i_predictions.validate_prediction_record(prediction_record(**overrides))


def test_prediction_contract_validation_normalizes_identity_and_timestamps() -> None:
    record = validated()

    assert record.identity.as_tuple() == (
        "00000000-0000-4000-8000-000000000001",
        ai4i_predictor.MODEL_NAME,
        ai4i_predictor.MODEL_VERSION,
        HASH_A,
    )
    assert record.event_time == "2026-01-01 00:00:00.000"
    assert record.source_kafka_timestamp == "2026-01-01 00:00:01.250"


def test_prediction_contract_rejects_missing_and_extra_fields() -> None:
    missing = prediction_record()
    del missing["payload_sha256"]
    with pytest.raises(ai4i_predictions.AI4IPredictionPersistenceError, match="missing"):
        ai4i_predictions.validate_prediction_record(missing)

    extra = prediction_record(unexpected="value")
    with pytest.raises(ai4i_predictions.AI4IPredictionPersistenceError, match="unexpected"):
        ai4i_predictions.validate_prediction_record(extra)


@pytest.mark.parametrize("probability", [-0.01, 1.01, True])
def test_probability_bounds_are_validated(probability: object) -> None:
    with pytest.raises(ai4i_predictions.AI4IPredictionPersistenceError):
        validated(failure_probability=probability)


def test_threshold_consistency_is_validated() -> None:
    with pytest.raises(ai4i_predictions.AI4IPredictionPersistenceError, match="threshold"):
        validated(failure_probability=0.2, failure_prediction=0)


def test_duplicate_input_identity_is_rejected() -> None:
    first = prediction_record()
    second = prediction_record(machine_code="MCH-0002")

    with pytest.raises(ai4i_predictions.AI4IPredictionPersistenceError, match="Duplicate"):
        ai4i_predictions.validate_prediction_records([first, second])


def test_identical_existing_prediction_is_modeled_as_idempotent_reuse() -> None:
    record = validated()
    existing = ai4i_predictions.ExistingPredictionRow(
        model_prediction_id=100,
        machine_id=1,
        record=record,
    )

    summary = ai4i_predictions.summarize_prediction_reuse(
        [record],
        {record.identity.as_tuple(): existing},
        {"MCH-0001": 1},
    )

    assert summary.new_records == 0
    assert summary.existing_identical_records == 1
    assert summary.conflicts == ()


def test_conflicting_existing_prediction_reports_material_fields() -> None:
    expected = validated()
    existing_record = validated(failure_probability=0.2, failure_prediction=1)
    existing = ai4i_predictions.ExistingPredictionRow(
        model_prediction_id=100,
        machine_id=1,
        record=existing_record,
    )

    summary = ai4i_predictions.summarize_prediction_reuse(
        [expected],
        {expected.identity.as_tuple(): existing},
        {"MCH-0001": 1},
    )

    assert summary.new_records == 0
    assert summary.existing_identical_records == 0
    assert len(summary.conflicts) == 1
    assert "failure_probability" in summary.conflicts[0].fields
    assert "failure_prediction" in summary.conflicts[0].fields


def test_latest_selection_prefers_newer_low_probability_over_older_high_probability() -> None:
    older_high = validated(
        event_id="00000000-0000-4000-8000-000000000001",
        event_time="2026-01-01 00:00:00",
        failure_probability=0.8,
        failure_prediction=1,
        source_kafka_offset=1,
    )
    newer_low = validated(
        event_id="00000000-0000-4000-8000-000000000002",
        event_time="2026-01-01 00:05:00",
        failure_probability=0.05,
        failure_prediction=0,
        source_kafka_offset=2,
    )

    latest = ai4i_predictions.latest_prediction_by_machine([older_high, newer_low])

    assert latest["MCH-0001"] == newer_low
    assert latest["MCH-0001"].failure_probability == 0.05


def test_latest_selection_uses_lineage_tie_breakers() -> None:
    low_offset = validated(
        event_id="00000000-0000-4000-8000-000000000001",
        event_time="2026-01-01 00:00:00",
        source_kafka_timestamp="2026-01-01 00:00:01.000",
        source_kafka_partition=0,
        source_kafka_offset=10,
    )
    high_offset = validated(
        event_id="00000000-0000-4000-8000-000000000002",
        event_time="2026-01-01 00:00:00",
        source_kafka_timestamp="2026-01-01 00:00:01.000",
        source_kafka_partition=0,
        source_kafka_offset=11,
    )

    latest = ai4i_predictions.latest_prediction_by_machine([low_offset, high_offset])

    assert latest["MCH-0001"] == high_offset


def test_machine_health_projection_preparation_uses_latest_prediction_identity() -> None:
    older = validated(
        event_id="00000000-0000-4000-8000-000000000001",
        event_time="2026-01-01 00:00:00",
    )
    newer = validated(
        event_id="00000000-0000-4000-8000-000000000002",
        event_time="2026-01-01 00:01:00",
    )

    projections = ai4i_predictions.prepare_latest_projections(
        [older, newer],
        {"MCH-0001": 1},
        {
            older.identity.as_tuple(): 10,
            newer.identity.as_tuple(): 11,
        },
    )

    assert len(projections) == 1
    assert projections[0].model_prediction_id == 11
    assert projections[0].record == newer


def test_static_summary_is_deterministic_and_runtime_free() -> None:
    first = ai4i_predictions.build_static_summary()
    second = ai4i_predictions.build_static_summary()
    rendered = json.dumps(first, sort_keys=True)

    assert first == second
    assert "data/predictions/ai4i/telemetry_predictions.jsonl" in rendered
    assert "runtime_counts" in first
    assert "generated_at" not in rendered
    assert "runtime_timestamp" not in rendered
    assert "container" not in rendered
    assert "password" not in rendered


def test_persistence_sql_uses_idempotent_conflict_target_and_no_alerts_or_deletes() -> None:
    record = validated()
    sql = ai4i_predictions.build_persistence_transaction([record], {"MCH-0001": 1})

    assert "ON CONFLICT (event_id, model_name, model_version, final_config_hash)" in sql
    assert "WHERE prediction_type = 'ai4i_failure_risk'" in sql
    assert "INSERT INTO machine_health" in sql
    assert "INSERT INTO alerts" not in sql
    assert "INSERT INTO anomalies" not in sql
    assert "TRUNCATE" not in sql
    assert "DELETE FROM" not in sql


def test_source_guards_prevent_model_execution_and_new_db_clients() -> None:
    guarded_files = [
        PROJECT_ROOT / "services" / "database" / "ai4i_predictions.py",
        PROJECT_ROOT / "scripts" / "persist_ai4i_predictions.py",
        PROJECT_ROOT / "scripts" / "inspect_ai4i_prediction_state.py",
        PROJECT_ROOT / "scripts" / "check_ai4i_prediction_persistence.py",
    ]
    forbidden_terms = [
        "load_" + "predictor",
        "predict_" + "batch",
        "predict_" + "proba",
        "final_model" + ".joblib",
        "." + "fit(",
        "test" + ".csv",
        "Tree" + "Explainer",
        "Isolation" + "Forest",
        "psy" + "copg",
        "sql" + "alchemy",
        "async" + "pg",
        "ale" + "mbic",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"

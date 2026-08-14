from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.anomaly import telemetry_detector
from services.database import telemetry_anomalies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def feature_record(index: int = 1, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "event_id": f"00000000-0000-4000-8000-{index:012d}",
        "machine_code": f"MCH-{index:04d}",
        "event_time": f"2026-01-01 00:{index % 60:02d}:00",
        "vibration_mm_s": 2.0 + (index % 5) * 0.05,
        "pressure_bar": 6.0 + (index % 7) * 0.04,
        "source_kafka_topic": "industrial.telemetry.v1",
        "source_kafka_partition": index % 3,
        "source_kafka_offset": index,
        "source_kafka_timestamp": f"2026-01-01 00:{index % 60:02d}:01.000",
        "source_kafka_key": f"MCH-{index:04d}",
        "payload_sha256": HASH_A,
    }
    record.update(overrides)
    return record


def validated_feature(index: int = 1, **overrides: object) -> telemetry_detector.FeatureRecord:
    return telemetry_detector.validate_feature_record(feature_record(index, **overrides))


def config() -> telemetry_detector.TelemetryAnomalyConfig:
    return telemetry_detector.load_config(PROJECT_ROOT)


def artifact_for(
    records: list[telemetry_detector.FeatureRecord],
) -> telemetry_detector.TrustedAnomalyArtifact:
    active_config = config()
    model = telemetry_detector.fit_isolation_forest_model(active_config, records)
    decisions = telemetry_detector.decision_values(model, records)
    min_decision, max_decision = telemetry_detector.score_reference_bounds(decisions)
    return telemetry_detector.TrustedAnomalyArtifact(
        model=model,
        metadata={
            "algorithm": active_config.algorithm,
            "artifact_sha256": HASH_C,
            "baseline_event_count": len(records),
            "baseline_event_id_sha256": HASH_A,
            "baseline_feature_data_sha256": HASH_B,
            "baseline_machine_count": len({record.machine_code for record in records}),
            "features": list(active_config.features),
            "model_config_hash": telemetry_detector.model_config_hash(active_config),
            "model_name": active_config.model_name,
            "model_version": active_config.model_version,
            "score_reference_max_decision": max_decision,
            "score_reference_min_decision": min_decision,
        },
    )


def anomaly_output_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        **feature_record(),
        "anomaly_score": 0.25,
        "anomaly_flag": False,
        "model_name": telemetry_detector.MODEL_NAME,
        "model_version": telemetry_detector.MODEL_VERSION,
        "model_config_hash": telemetry_detector.model_config_hash(config()),
        "baseline_event_id_sha256": HASH_A,
        "baseline_feature_data_sha256": HASH_B,
    }
    record.update(overrides)
    return record


def validated_anomaly(**overrides: object) -> telemetry_anomalies.AnomalyRecord:
    return telemetry_anomalies.validate_anomaly_record(anomaly_output_record(**overrides))


def test_config_uses_exact_operational_sensor_contract() -> None:
    active_config = config()

    assert active_config.features == ("vibration_mm_s", "pressure_bar")
    assert active_config.algorithm == "IsolationForest"
    assert active_config.contamination == "auto"
    assert "failure_probability" not in active_config.features
    assert "product_quality_type" not in active_config.features


def test_feature_record_rejects_missing_extra_and_ai4i_fields() -> None:
    missing = feature_record()
    del missing["pressure_bar"]
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="missing"):
        telemetry_detector.validate_feature_record(missing)

    extra = feature_record(failure_probability=0.99)
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="unexpected"):
        telemetry_detector.validate_feature_record(extra)

    product_quality = feature_record(product_quality_type="H")
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="unexpected"):
        telemetry_detector.validate_feature_record(product_quality)


def test_feature_validation_requires_finite_values() -> None:
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="finite"):
        validated_feature(vibration_mm_s="NaN")


def test_baseline_preparation_and_hashing_are_deterministic() -> None:
    late = feature_record(2, event_time="2026-01-01 00:02:00")
    early = feature_record(1, event_time="2026-01-01 00:01:00")

    first = telemetry_detector.prepare_feature_records([late, early])
    second = telemetry_detector.prepare_feature_records([early, late])
    first_hashes = telemetry_detector.baseline_hashes(first)
    second_hashes = telemetry_detector.baseline_hashes(second)

    assert [record.event_id for record in first] == [early["event_id"], late["event_id"]]
    assert first == second
    assert first_hashes == second_hashes


def test_duplicate_feature_event_ids_are_rejected() -> None:
    duplicate = feature_record(1, machine_code="MCH-0002")

    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="duplicate"):
        telemetry_detector.prepare_feature_records([feature_record(1), duplicate])


def test_score_direction_ranks_extreme_point_above_central_point() -> None:
    baseline = [validated_feature(index) for index in range(1, 41)]
    artifact = artifact_for(baseline)
    central = validated_feature(
        101,
        event_id="00000000-0000-4000-8000-000000000101",
        vibration_mm_s=2.1,
        pressure_bar=6.1,
    )
    extreme = validated_feature(
        102,
        event_id="00000000-0000-4000-8000-000000000102",
        vibration_mm_s=15.0,
        pressure_bar=12.0,
    )

    scored = telemetry_detector.score_feature_records([central, extreme], artifact)
    by_event_id = {item.record["event_id"]: item for item in scored}

    assert (
        by_event_id[extreme.event_id].record["anomaly_score"]
        > by_event_id[central.event_id].record["anomaly_score"]
    )
    assert "anomaly_probability" not in by_event_id[extreme.event_id].record


def test_flag_semantics_match_isolation_forest_prediction() -> None:
    baseline = [validated_feature(index) for index in range(1, 41)]
    artifact = artifact_for(baseline)
    scored = telemetry_detector.score_feature_records(baseline[:5], artifact)
    expected_predictions = telemetry_detector.model_predictions(artifact.model, baseline[:5])

    assert [bool(item.record["anomaly_flag"]) for item in scored] == [
        prediction == -1 for prediction in expected_predictions
    ]


def test_anomaly_output_serialization_is_deterministic_and_guarded() -> None:
    record = anomaly_output_record()
    first = telemetry_detector.anomaly_record_json(record)
    second = telemetry_detector.anomaly_record_json(dict(reversed(record.items())))

    assert first == second
    assert "failure_probability" not in first

    forbidden = anomaly_output_record(shap_values=[1.0])
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="unexpected"):
        telemetry_detector.anomaly_record_json(forbidden)


def test_artifact_metadata_validation_checks_identity_hashes_and_bounds() -> None:
    active_config = config()
    metadata = telemetry_detector.build_artifact_metadata(
        artifact_sha256=HASH_C,
        baseline_event_count=10,
        baseline_machine_count=3,
        baseline_event_id_sha256=HASH_A,
        baseline_feature_data_sha256=HASH_B,
        config=active_config,
        config_hash=telemetry_detector.model_config_hash(active_config),
        score_reference_min_decision=-0.1,
        score_reference_max_decision=0.2,
    )

    telemetry_detector.validate_artifact_metadata(metadata, active_config)
    metadata["score_reference_min_decision"] = 1.0
    metadata["score_reference_max_decision"] = 0.2
    with pytest.raises(telemetry_detector.TelemetryAnomalyError, match="less than"):
        telemetry_detector.validate_artifact_metadata(metadata, active_config)


def test_static_summary_is_deterministic_and_runtime_free() -> None:
    first = telemetry_detector.build_static_summary()
    second = telemetry_detector.build_static_summary()
    rendered = json.dumps(first, sort_keys=True)

    assert first == second
    assert first["features"] == ["vibration_mm_s", "pressure_bar"]
    assert "runtime_counts" not in rendered
    assert "generated_at" not in rendered
    assert "timestamp" not in rendered
    assert "anomaly_probability" not in rendered


def test_anomaly_persistence_identity_reuse_and_conflict_detection() -> None:
    record = validated_anomaly()
    existing = telemetry_anomalies.ExistingAnomalyRow(
        anomaly_id=1,
        machine_id=1,
        record=record,
    )

    reused = telemetry_anomalies.summarize_anomaly_reuse(
        [record],
        {record.identity.as_tuple(): existing},
        {"MCH-0001": 1},
    )

    assert reused.new_records == 0
    assert reused.existing_identical_records == 1
    assert reused.conflicts == ()

    conflict_record = validated_anomaly(anomaly_score=0.75, anomaly_flag=True)
    conflict = telemetry_anomalies.summarize_anomaly_reuse(
        [record],
        {record.identity.as_tuple(): telemetry_anomalies.ExistingAnomalyRow(1, 1, conflict_record)},
        {"MCH-0001": 1},
    )

    assert len(conflict.conflicts) == 1
    assert "anomaly_score" in conflict.conflicts[0].fields
    assert "anomaly_flag" in conflict.conflicts[0].fields


def test_persistence_sql_is_idempotent_and_does_not_touch_ai4i_or_alerts() -> None:
    record = validated_anomaly()
    sql = telemetry_anomalies.build_persistence_transaction([record], {"MCH-0001": 1})

    assert "INSERT INTO anomalies" in sql
    assert "ON CONFLICT (event_id, model_name, model_version, model_config_hash)" in sql
    assert "telemetry_isolation_forest_score" in sql
    assert "INSERT INTO alerts" not in sql
    assert "INSERT INTO model_predictions" not in sql
    assert "INSERT INTO machine_health" not in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE" not in sql


def test_database_static_summary_preserves_boundaries() -> None:
    summary = telemetry_anomalies.build_static_summary()
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["target_anomaly_table"] == "anomalies"
    assert "machine_health is not updated" in rendered
    assert "AI4I model_predictions are not read or modified" in rendered
    assert "runtime_counts" in summary

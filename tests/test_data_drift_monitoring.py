from __future__ import annotations

import copy
import math
from pathlib import Path

from ml.anomaly.telemetry_detector import FeatureRecord
from ml.monitoring import drift
from services.database import drift_monitoring

VALID_HASH = "a" * 64
REFERENCE_HASH = "b" * 64
AI4I_FEATURE_VALUES = {
    "Type": "L",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 20.0,
}


def sample_config() -> drift.DriftConfig:
    return drift.parse_config(
        {
            "monitor_version": "1.0.0",
            "numeric_bin_count": 10,
            "epsilon": 0.000001,
            "psi_watch_threshold": 0.10,
            "psi_drift_threshold": 0.25,
            "ai4i_model_input": {
                "reference_name": "ai4i_train_validation_development",
                "reference_policy": "train + validation only",
                "training_data_policy": "train + validation",
                "model_name": "ai4i-failure-risk-random-forest",
                "model_version": "1.0.0",
                "final_config_hash": VALID_HASH,
                "source_paths": [
                    "data/processed/ai4i/train.csv",
                    "data/processed/ai4i/validation.csv",
                ],
                "forbidden_source_paths": ["data/processed/ai4i/test.csv"],
                "features": list(drift.AI4I_FEATURES),
                "categorical_features": ["Type"],
                "numeric_features": list(drift.AI4I_NUMERIC_FEATURES),
                "categories": {"Type": list(drift.AI4I_TYPE_CATEGORIES)},
            },
            "operational_anomaly_inputs": {
                "reference_name": "telemetry_isolation_forest_v1_baseline",
                "reference_policy": "frozen baseline",
                "model_name": "telemetry-isolation-forest",
                "model_version": "1.0.0",
                "model_config_hash": VALID_HASH,
                "baseline_event_id_sha256": VALID_HASH,
                "baseline_feature_data_sha256": VALID_HASH,
                "source_layer": "silver",
                "source_path": "data/silver/telemetry",
                "features": list(drift.ANOMALY_FEATURES),
                "numeric_features": list(drift.ANOMALY_FEATURES),
            },
        }
    )


def adapter_record(
    event_number: int, model_input: dict[str, object] | None = None
) -> dict[str, object]:
    payload_hash = f"{event_number:064x}"[-64:]
    return {
        "adapter_version": "1.0",
        "event_id": f"00000000-0000-0000-0000-{event_number:012d}",
        "machine_code": "MCH-0001",
        "event_time": f"2026-01-01 00:00:0{event_number}.000",
        "model_input": dict(model_input or AI4I_FEATURE_VALUES),
        "source_lineage": {
            "source_kafka_topic": "telemetry.raw",
            "source_kafka_partition": 0,
            "source_kafka_offset": event_number,
            "source_kafka_timestamp": f"2026-01-01 00:00:0{event_number}.000",
            "source_kafka_key": "MCH-0001",
            "payload_sha256": payload_hash,
        },
    }


def anomaly_record(event_number: int, vibration: float, pressure: float) -> FeatureRecord:
    payload_hash = f"{event_number + 100:064x}"[-64:]
    return FeatureRecord(
        event_id=f"00000000-0000-0000-0000-{event_number:012d}",
        machine_code="MCH-0001",
        event_time=f"2026-01-01 00:00:0{event_number}.000",
        vibration_mm_s=vibration,
        pressure_bar=pressure,
        source_kafka_topic="telemetry.raw",
        source_kafka_partition=0,
        source_kafka_offset=event_number,
        source_kafka_timestamp=f"2026-01-01 00:00:0{event_number}.000",
        source_kafka_key="MCH-0001",
        payload_sha256=payload_hash,
    )


def sample_reference_profile(config: drift.DriftConfig) -> dict[str, object]:
    ai4i_rows = [
        {**AI4I_FEATURE_VALUES, "Type": "L", "Torque [Nm]": 30.0},
        {**AI4I_FEATURE_VALUES, "Type": "M", "Torque [Nm]": 40.0},
        {**AI4I_FEATURE_VALUES, "Type": "H", "Torque [Nm]": 50.0},
    ]
    ai4i_features = [
        drift.categorical_reference_profile(
            "Type",
            [row["Type"] for row in ai4i_rows],
            drift.AI4I_TYPE_CATEGORIES,
        )
    ]
    for feature_name in drift.AI4I_NUMERIC_FEATURES:
        ai4i_features.append(
            drift.numeric_reference_profile(
                feature_name,
                [row[feature_name] for row in ai4i_rows],
                config,
            )
        )
    anomaly_records = [
        anomaly_record(1, 1.0, 4.0),
        anomaly_record(2, 2.0, 5.0),
        anomaly_record(3, 3.0, 6.0),
    ]
    anomaly_features = [
        drift.numeric_reference_profile(
            feature_name,
            [getattr(record, feature_name) for record in anomaly_records],
            config,
        )
        for feature_name in drift.ANOMALY_FEATURES
    ]
    profile = {
        "monitor_version": config.monitor_version,
        drift.AI4I_SCOPE: {
            "reference_identity": drift.ai4i_reference_identity(
                config,
                config.ai4i_model_input["source_paths"],
            ),
            "reference_row_count": len(ai4i_rows),
            "features": ai4i_features,
        },
        drift.ANOMALY_SCOPE: {
            "reference_identity": drift.anomaly_reference_identity(config),
            "reference_row_count": len(anomaly_records),
            "features": anomaly_features,
        },
    }
    drift.validate_reference_profile(profile, config)
    return profile


def sample_report() -> dict[str, object]:
    config = sample_config()
    profile = sample_reference_profile(config)
    ai4i_current = [
        adapter_record(1, {**AI4I_FEATURE_VALUES, "Type": "L", "Torque [Nm]": 30.0}),
        adapter_record(2, {**AI4I_FEATURE_VALUES, "Type": "M", "Torque [Nm]": 40.0}),
        adapter_record(3, {**AI4I_FEATURE_VALUES, "Type": "H", "Torque [Nm]": 50.0}),
    ]
    anomaly_current = [
        anomaly_record(1, 1.0, 4.0),
        anomaly_record(2, 2.0, 5.0),
        anomaly_record(3, 3.0, 6.0),
    ]
    return drift.build_drift_report(
        reference_profile=profile,
        reference_profile_sha256=REFERENCE_HASH,
        ai4i_current_records=ai4i_current,
        anomaly_current_records=anomaly_current,
        config=config,
    )


def test_config_validation_accepts_static_policy() -> None:
    config = sample_config()
    assert config.monitor_version == "1.0.0"
    assert config.numeric_bin_count == 10


def test_identical_distribution_has_zero_psi() -> None:
    config = sample_config()
    profile = drift.numeric_reference_profile("x", [1, 2, 3, 4, 5], config)
    metric = drift.numeric_feature_metric(profile, [1, 2, 3, 4, 5], config)
    assert metric["psi"] == 0.0
    assert metric["status"] == "stable"


def test_shifted_distribution_has_larger_psi() -> None:
    config = sample_config()
    profile = drift.numeric_reference_profile("x", [1, 2, 3, 4, 5], config)
    identical = drift.numeric_feature_metric(profile, [1, 2, 3, 4, 5], config)
    shifted = drift.numeric_feature_metric(profile, [100, 101, 102, 103, 104], config)
    assert shifted["psi"] > identical["psi"]


def test_zero_bin_safety() -> None:
    config = sample_config()
    psi = drift.psi_from_proportions([1.0, 0.0], [0.0, 1.0], config.epsilon)
    assert math.isfinite(psi)
    assert psi >= 0


def test_numeric_quantile_duplicate_edges_are_handled() -> None:
    edges = drift.numeric_bin_edges([1, 1, 1, 2, 2, 3], 10)
    assert edges == sorted(set(edges))
    assert len(edges) < 11


def test_categorical_type_psi_and_unexpected_category() -> None:
    config = sample_config()
    profile = drift.categorical_reference_profile("Type", ["L", "M", "H"], ["L", "M", "H"])
    metric = drift.categorical_feature_metric(profile, ["L", "X", "X"], config)
    assert metric["unexpected_category_count"] == 2
    assert metric["psi"] >= 0


def test_threshold_boundaries() -> None:
    config = sample_config()
    assert drift.status_for_psi(0.099999, config) == "stable"
    assert drift.status_for_psi(0.10, config) == "watch"
    assert drift.status_for_psi(0.25, config) == "drift"


def test_standardized_mean_shift() -> None:
    assert drift.standardized_mean_shift(10.0, 13.0, 2.0) == 1.5
    assert drift.standardized_mean_shift(10.0, 13.0, 0.0) is None


def test_outside_reference_range_rate() -> None:
    count, rate = drift.outside_reference_range([0, 1, 2, 3], 1, 2)
    assert count == 2
    assert rate == 0.5


def test_overall_status_aggregation() -> None:
    assert drift.overall_status(["stable", "watch"]) == "watch"
    assert drift.overall_status(["stable", "drift", "watch"]) == "drift"


def test_current_data_hashing_is_deterministic() -> None:
    records = [adapter_record(2), adapter_record(1)]
    assert drift.ai4i_current_data_hash(records) == drift.ai4i_current_data_hash(
        list(reversed(records))
    )


def test_report_serialization_is_deterministic() -> None:
    report = sample_report()
    assert drift.deterministic_bytes(report) == drift.deterministic_bytes(copy.deepcopy(report))


def test_stable_db_business_identity() -> None:
    report = sample_report()
    identity = drift_monitoring.snapshot_identity(report)
    assert identity.as_tuple() == (
        "1.0.0",
        REFERENCE_HASH,
        report[drift.AI4I_SCOPE]["current_data_hash"],
        report[drift.ANOMALY_SCOPE]["current_data_hash"],
    )


def test_idempotency_comparison_reuses_identical_report() -> None:
    report = sample_report()
    existing = drift_monitoring.snapshot_values(report)
    existing["drift_snapshot_id"] = 1
    existing["feature_metrics"] = drift_monitoring.metric_values(report)
    summary = drift_monitoring.summarize_report_reuse(report, [existing])
    assert summary.existing_identical_snapshots_reused == 1
    assert summary.existing_identical_feature_metrics_reused == len(
        drift_monitoring.metric_values(report)
    )
    assert not summary.conflicts


def test_conflict_detection_finds_metric_mismatch() -> None:
    report = sample_report()
    existing = drift_monitoring.snapshot_values(report)
    existing["drift_snapshot_id"] = 1
    metrics = drift_monitoring.metric_values(report)
    metrics[0] = {**metrics[0], "psi": metrics[0]["psi"] + 0.2}
    existing["feature_metrics"] = metrics
    summary = drift_monitoring.summarize_report_reuse(report, [existing])
    assert summary.conflicts


def test_drift_features_exclude_model_outputs() -> None:
    report = sample_report()
    feature_names = {
        metric["feature_name"]
        for scope_name in (drift.AI4I_SCOPE, drift.ANOMALY_SCOPE)
        for metric in report[scope_name]["features"]
    }
    assert "failure_prediction" not in feature_names
    assert "anomaly_score" not in feature_names


def test_ai4i_test_csv_is_forbidden_not_source() -> None:
    config = sample_config()
    assert "data/processed/ai4i/test.csv" not in config.ai4i_model_input["source_paths"]
    assert "data/processed/ai4i/test.csv" in config.ai4i_model_input["forbidden_source_paths"]


def test_source_guards_do_not_call_model_scoring_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "ml" / "monitoring" / "drift.py",
        root / "scripts" / "build_drift_reference_profiles.py",
        root / "scripts" / "monitor_data_drift.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for forbidden in ("load_predictor", "predict_batch", "predict_proba", ".fit("):
        assert forbidden not in text


def test_static_summary_is_deterministic() -> None:
    config = sample_config()
    first = drift.canonical_json(drift.build_static_summary(config))
    second = drift.canonical_json(drift.build_static_summary(config))
    assert first == second

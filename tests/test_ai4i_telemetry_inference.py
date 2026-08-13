from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml.inference import ai4i_predictor, ai4i_telemetry
from pipelines.batch import ai4i_feature_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CONFIG = {
    "decision_threshold": 0.14,
    "numerical_features": [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ],
    "predictive_features": list(ai4i_feature_adapter.EXPECTED_MODEL_INPUT_FEATURES),
    "target": "Machine failure",
}
FINAL_CONFIG_HASH = ai4i_predictor.current_final_config_hash(FINAL_CONFIG)


@dataclass(frozen=True, order=True)
class FakePartPath:
    name: str
    file: bool = True

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix

    def is_file(self) -> bool:
        return self.file


class FakeAdapterDir:
    def __init__(self, entries: list[FakePartPath]) -> None:
        self._entries = entries

    def exists(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return True

    def iterdir(self) -> list[FakePartPath]:
        return self._entries


def load_raw_config() -> dict[str, object]:
    path = PROJECT_ROOT / ai4i_feature_adapter.CONFIG_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> ai4i_feature_adapter.AI4IFeatureAdapterConfig:
    return ai4i_feature_adapter.parse_adapter_config(load_raw_config())


def sample_silver_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "air_temperature_k": 298.638,
        "event_id": "event-001",
        "event_time": "2026-02-01 00:00:00",
        "machine_code": "MCH-0001",
        "machine_type": "excavator",
        "payload_sha256": "a" * 64,
        "pressure_bar": 133.1,
        "process_temperature_k": 306.837,
        "product_quality_type": "M",
        "rotational_speed_rpm": 1409,
        "source_kafka_key": "MCH-0001",
        "source_kafka_offset": 10,
        "source_kafka_partition": 0,
        "source_kafka_timestamp": "2026-02-01 00:00:01",
        "source_kafka_topic": "industrial.telemetry.v1",
        "tool_wear_min": 60,
        "torque_nm": 49.499,
        "vibration_mm_s": 2.4,
    }
    event.update(overrides)
    return event


def model_input(**overrides: object) -> dict[str, object]:
    value = ai4i_feature_adapter.adapt_silver_event_to_model_input(sample_silver_event())
    value.update(overrides)
    return value


def adapter_record(**overrides: object) -> dict[str, object]:
    silver_event = sample_silver_event()
    record: dict[str, object] = {
        "adapter_version": "1.0",
        "event_id": silver_event["event_id"],
        "event_time": silver_event["event_time"],
        "machine_code": silver_event["machine_code"],
        "model_input": ai4i_feature_adapter.adapt_silver_event_to_model_input(silver_event),
        "source_lineage": {
            field: silver_event[field] for field in ai4i_feature_adapter.LINEAGE_FIELDS
        },
    }
    record.update(overrides)
    return record


def prediction_output(probability: float) -> dict[str, object]:
    return {
        "decision_threshold": 0.14,
        "failure_prediction": int(probability >= 0.14),
        "failure_probability": probability,
        "final_config_hash": FINAL_CONFIG_HASH,
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_version": ai4i_predictor.MODEL_VERSION,
    }


def test_adapter_config_validation_accepts_expected_values() -> None:
    config = load_config()

    assert config.adapter_version == "1.0"
    assert config.spark_version == "4.0.4"
    assert config.master == "local[2]"
    assert config.source == "data/silver/telemetry"
    assert config.output == "data/model_input/ai4i/telemetry"
    assert config.output_format == "json"
    assert config.timezone == "UTC"
    assert config.shuffle_partitions == 3


def test_adapter_config_rejects_unsafe_paths() -> None:
    raw_config = load_raw_config()
    raw_config["source"] = "../data/silver/telemetry"

    with pytest.raises(ai4i_feature_adapter.AI4IFeatureAdapterConfigError, match="source"):
        ai4i_feature_adapter.parse_adapter_config(raw_config)


def test_exact_six_feature_mapping_and_numeric_values() -> None:
    adapted = ai4i_feature_adapter.adapt_silver_event_to_model_input(sample_silver_event())

    assert tuple(adapted) == ai4i_feature_adapter.EXPECTED_MODEL_INPUT_FEATURES
    assert adapted == {
        "Type": "M",
        "Air temperature [K]": 298.638,
        "Process temperature [K]": 306.837,
        "Rotational speed [rpm]": 1409,
        "Torque [Nm]": 49.499,
        "Tool wear [min]": 60,
    }


def test_product_quality_type_maps_per_event_and_not_from_machine_identity() -> None:
    low = sample_silver_event(product_quality_type="L", machine_code="MCH-0001")
    high = sample_silver_event(product_quality_type="H", machine_code="MCH-0001")

    assert ai4i_feature_adapter.adapt_silver_event_to_model_input(low)["Type"] == "L"
    assert ai4i_feature_adapter.adapt_silver_event_to_model_input(high)["Type"] == "H"
    assert ai4i_feature_adapter.adapt_silver_event_to_model_input(high)["Type"] != "excavator"


def test_vibration_and_pressure_are_excluded_from_model_input() -> None:
    adapted = ai4i_feature_adapter.adapt_silver_event_to_model_input(sample_silver_event())

    assert "vibration_mm_s" not in adapted
    assert "pressure_bar" not in adapted
    assert set(ai4i_feature_adapter.EXPECTED_EXCLUDED_CURRENT_MODEL_FIELDS) == {
        "pressure_bar",
        "vibration_mm_s",
    }


def test_model_input_exact_field_set_is_validated() -> None:
    config = load_config()
    validated = ai4i_telemetry.validate_model_input(model_input(), config, FINAL_CONFIG)

    assert tuple(validated) == ai4i_feature_adapter.EXPECTED_MODEL_INPUT_FEATURES


def test_missing_model_feature_is_rejected() -> None:
    config = load_config()
    incomplete = model_input()
    del incomplete["Torque [Nm]"]

    with pytest.raises(ai4i_telemetry.AI4ITelemetryInferenceError, match="missing"):
        ai4i_telemetry.validate_model_input(incomplete, config, FINAL_CONFIG)


def test_extra_model_feature_is_rejected() -> None:
    config = load_config()
    extra = model_input(vibration_mm_s=2.4)

    with pytest.raises(ai4i_telemetry.AI4ITelemetryInferenceError, match="unexpected"):
        ai4i_telemetry.validate_model_input(extra, config, FINAL_CONFIG)


def test_deterministic_model_input_serialization_and_sha256() -> None:
    first = ai4i_telemetry.canonical_model_input_json(model_input(), FINAL_CONFIG)
    second = ai4i_telemetry.canonical_model_input_json(
        dict(reversed(model_input().items())),
        FINAL_CONFIG,
    )
    expected = (
        '{"Type":"M","Air temperature [K]":298.638,'
        '"Process temperature [K]":306.837,"Rotational speed [rpm]":1409.0,'
        '"Torque [Nm]":49.499,"Tool wear [min]":60.0}'
    )

    assert first == second == expected
    assert (
        ai4i_telemetry.model_input_sha256(model_input(), FINAL_CONFIG)
        == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    )


def test_adapter_file_discovery_ignores_spark_metadata() -> None:
    adapter_dir = FakeAdapterDir(
        [
            FakePartPath("part-00001.json"),
            FakePartPath("_SUCCESS"),
            FakePartPath("part-00000.json"),
            FakePartPath("part-00002.crc"),
        ]
    )

    discovered = ai4i_telemetry.discover_adapter_part_files(
        adapter_dir  # type: ignore[arg-type]
    )

    assert [path.name for path in discovered] == ["part-00000.json", "part-00001.json"]


def test_adapter_records_sort_stably_by_event_and_lineage_fields() -> None:
    config = load_config()
    late = adapter_record(
        event_id="event-late",
        event_time="2026-02-01 00:01:00",
        machine_code="MCH-0001",
    )
    early = adapter_record(
        event_id="event-early",
        event_time="2026-02-01 00:00:00",
        machine_code="MCH-0002",
    )
    records = [
        ai4i_telemetry.validate_adapter_record(late, config, FINAL_CONFIG),
        ai4i_telemetry.validate_adapter_record(early, config, FINAL_CONFIG),
    ]
    ordered = sorted(records, key=ai4i_telemetry.adapter_record_sort_key)

    assert [record["event_id"] for record in ordered] == ["event-early", "event-late"]


def test_malformed_adapter_record_is_rejected() -> None:
    config = load_config()
    malformed = adapter_record()
    del malformed["source_lineage"]

    with pytest.raises(ai4i_telemetry.AI4ITelemetryInferenceError, match="missing"):
        ai4i_telemetry.validate_adapter_record(malformed, config, FINAL_CONFIG)


@pytest.mark.parametrize(
    ("probability", "prediction"),
    [(0.139999, 0), (0.14, 1), (0.140001, 1)],
)
def test_threshold_behavior_below_exactly_and_above(
    probability: float,
    prediction: int,
) -> None:
    assert ai4i_telemetry.prediction_is_consistent(probability, prediction, 0.14)


def test_failure_probability_bounds_are_validated() -> None:
    with pytest.raises(ai4i_telemetry.AI4ITelemetryInferenceError, match="outside"):
        ai4i_telemetry.build_prediction_records(
            [adapter_record()],
            [prediction_output(1.01)],
            FINAL_CONFIG,
        )


def test_prediction_records_preserve_lineage_and_hash_model_input_only() -> None:
    source = adapter_record()
    records = ai4i_telemetry.build_prediction_records(
        [source],
        [prediction_output(0.2)],
        FINAL_CONFIG,
    )
    record = records[0]

    assert record["failure_prediction"] == 1
    assert record["frozen_threshold"] == 0.14
    assert record["model_input_sha256"] == ai4i_telemetry.model_input_sha256(
        source["model_input"],
        FINAL_CONFIG,
    )
    for field in ai4i_telemetry.LINEAGE_FIELDS:
        assert record[field] == source["source_lineage"][field]


def test_prediction_serialization_is_deterministic() -> None:
    first = ai4i_telemetry.build_prediction_records(
        [adapter_record(event_id="b"), adapter_record(event_id="a")],
        [prediction_output(0.3), prediction_output(0.1)],
        FINAL_CONFIG,
    )
    rendered = "\n".join(ai4i_telemetry.prediction_record_json(record) for record in first)
    second = "\n".join(ai4i_telemetry.prediction_record_json(record) for record in first)

    assert second == rendered


@pytest.mark.parametrize("forbidden_key", ["Machine failure", "shap_values", "anomaly_score"])
def test_prediction_output_excludes_labels_shap_and_anomaly(forbidden_key: str) -> None:
    record = ai4i_telemetry.build_prediction_records(
        [adapter_record()],
        [prediction_output(0.2)],
        FINAL_CONFIG,
    )[0]
    record[forbidden_key] = "not allowed"

    with pytest.raises(ai4i_telemetry.AI4ITelemetryInferenceError, match="Forbidden"):
        ai4i_telemetry.prediction_record_json(record)


def test_static_bridge_summary_is_deterministic_and_runtime_free() -> None:
    config = load_config()
    first = ai4i_feature_adapter.build_static_bridge_summary(
        config,
        model_name=ai4i_predictor.MODEL_NAME,
        model_version=ai4i_predictor.MODEL_VERSION,
        frozen_threshold=0.14,
        final_config_hash=FINAL_CONFIG_HASH,
    )
    second = ai4i_feature_adapter.build_static_bridge_summary(
        config,
        model_name=ai4i_predictor.MODEL_NAME,
        model_version=ai4i_predictor.MODEL_VERSION,
        frozen_threshold=0.14,
        final_config_hash=FINAL_CONFIG_HASH,
    )
    rendered = json.dumps(first, sort_keys=True)

    assert first == second
    assert "prediction_count" not in rendered
    assert "row_count" not in rendered
    assert "timestamp" not in rendered
    assert "absolute" not in rendered


def test_source_guards_prevent_disallowed_integration_work() -> None:
    guarded_files = [
        PROJECT_ROOT / "pipelines" / "batch" / "ai4i_feature_adapter.py",
        PROJECT_ROOT / "scripts" / "run_spark_ai4i_adapter.py",
        PROJECT_ROOT / "scripts" / "run_spark_ai4i_adapter_docker.py",
        PROJECT_ROOT / "ml" / "inference" / "ai4i_telemetry.py",
        PROJECT_ROOT / "scripts" / "predict_silver_telemetry.py",
    ]
    forbidden_terms = [
        "data/gold",
        "psycopg",
        "sqlalchemy",
        "postgresql://",
        "pg_isready",
        "confluent_kafka",
        '.format("kafka")',
        "readStream",
        ".fit(",
        "fit_transform",
        "test.csv",
        "TreeExplainer",
        "IsolationForest",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"

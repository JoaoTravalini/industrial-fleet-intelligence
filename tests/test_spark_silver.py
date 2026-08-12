from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pipelines.streaming import silver_transformation
from scripts import run_spark_silver_docker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_config() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / silver_transformation.CONFIG_RELATIVE_PATH).read_text())


def test_silver_config_validation_accepts_expected_values() -> None:
    config = silver_transformation.load_silver_config(
        PROJECT_ROOT / silver_transformation.CONFIG_RELATIVE_PATH
    )

    assert config.spark_version == "4.0.4"
    assert config.application_name == "industrial-fleet-silver-telemetry"
    assert config.master == "local[2]"
    assert config.telemetry_schema_version == "1.0"
    assert config.business_event_key == "event_id"
    assert (
        config.duplicate_policy == "retain_one_canonical_record_and_audit_additional_valid_records"
    )


def test_silver_config_paths_and_runtime_policies_are_exact() -> None:
    config = silver_transformation.parse_silver_config(load_raw_config())

    assert config.bronze_input_path == "data/bronze/telemetry"
    assert config.silver_output_path == "data/silver/telemetry"
    assert config.duplicate_output_path == "data/silver/duplicates"
    assert config.quarantine_output_path == "data/silver/quarantine"
    assert config.output_format == "parquet"
    assert config.spark_sql_shuffle_partitions == 3
    assert config.spark_timezone == "UTC"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spark_version", "latest", "spark_version"),
        ("application_name", "silver", "application_name"),
        ("master", "spark://master:7077", "master"),
        ("output_format", "delta", "output_format"),
        ("business_event_key", "payload_sha256", "business_event_key"),
        ("duplicate_policy", "drop_duplicates", "duplicate_policy"),
    ],
)
def test_silver_config_rejects_incompatible_values(
    field: str,
    value: object,
    message: str,
) -> None:
    raw_config = load_raw_config()
    raw_config[field] = value

    with pytest.raises(silver_transformation.SparkSilverConfigError, match=message):
        silver_transformation.parse_silver_config(raw_config)


def test_silver_config_rejects_unsafe_paths() -> None:
    raw_config = load_raw_config()
    raw_config["silver_output_path"] = "/workspace/data/silver/telemetry"
    with pytest.raises(silver_transformation.SparkSilverConfigError, match="relative path"):
        silver_transformation.parse_silver_config(raw_config)

    raw_config = load_raw_config()
    raw_config["quarantine_output_path"] = "../outside"
    with pytest.raises(silver_transformation.SparkSilverConfigError, match="relative path"):
        silver_transformation.parse_silver_config(raw_config)


def test_exact_telemetry_field_list_and_types() -> None:
    assert silver_transformation.TELEMETRY_FIELD_NAMES == (
        "schema_version",
        "event_id",
        "machine_code",
        "sequence_number",
        "event_time",
        "source",
        "product_quality_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "vibration_mm_s",
        "pressure_bar",
    )
    assert silver_transformation.telemetry_field_types()["event_time"] == "timestamp"
    assert silver_transformation.telemetry_field_types()["sequence_number"] == "bigint"
    assert "Type" not in silver_transformation.TELEMETRY_FIELD_NAMES


def test_rejection_reason_identifiers_are_stable() -> None:
    expected = {
        "malformed_json",
        "missing_required_field",
        "unexpected_field",
        "invalid_schema_version",
        "invalid_event_id",
        "invalid_machine_code",
        "invalid_sequence_number",
        "invalid_event_time",
        "invalid_source",
        "invalid_product_quality_type",
        "invalid_air_temperature_k",
        "invalid_process_temperature_k",
        "invalid_rotational_speed_rpm",
        "invalid_torque_nm",
        "invalid_tool_wear_min",
        "invalid_vibration_mm_s",
        "invalid_pressure_bar",
        "process_temperature_not_above_air_temperature",
        "kafka_key_machine_code_mismatch",
    }

    assert set(silver_transformation.REJECTION_REASON_IDENTIFIERS) == expected


def test_machine_code_contract_matches_required_range() -> None:
    pattern = silver_transformation.machine_code_contract()

    assert re.fullmatch(pattern, "MCH-0001")
    assert re.fullmatch(pattern, "MCH-0100")
    assert not re.fullmatch(pattern, "MCH-0000")
    assert not re.fullmatch(pattern, "MCH-0101")
    assert not re.fullmatch(pattern, "mch-0001")


def test_sensor_bound_policy_matches_contract() -> None:
    bounds = silver_transformation.sensor_bound_policy()

    assert bounds["air_temperature_k"] == (294.0, 306.0)
    assert bounds["process_temperature_k"] == (304.0, 315.0)
    assert bounds["rotational_speed_rpm"] == (1000, 3000)
    assert bounds["torque_nm"] == (0.0, 80.0)
    assert bounds["tool_wear_min"] == (0, 300)
    assert bounds["vibration_mm_s"] == (0.0, 15.0)
    assert bounds["pressure_bar"] == (1.0, 12.0)


def test_canonical_ordering_policy_is_deterministic() -> None:
    assert silver_transformation.canonical_ordering_fields() == (
        "source_kafka_timestamp",
        "source_kafka_topic",
        "source_kafka_partition",
        "source_kafka_offset",
        "payload_sha256",
    )


def test_accounting_invariant_helper() -> None:
    assert silver_transformation.accounting_invariants_hold(
        bronze_row_count=10,
        valid_pre_dedup_row_count=8,
        canonical_silver_row_count=6,
        duplicate_audit_row_count=2,
        quarantine_row_count=2,
    )
    assert not silver_transformation.accounting_invariants_hold(
        bronze_row_count=10,
        valid_pre_dedup_row_count=8,
        canonical_silver_row_count=7,
        duplicate_audit_row_count=2,
        quarantine_row_count=2,
    )


def test_same_event_id_with_different_kafka_coordinates_is_audited_not_quarantined() -> None:
    records = [
        {
            "event_id": "same-event",
            "payload_sha256": "aaa",
            "source_kafka_offset": 10,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": "2026-02-01 00:00:00",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
        {
            "event_id": "same-event",
            "payload_sha256": "aaa",
            "source_kafka_offset": 11,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": "2026-02-01 00:00:01",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
    ]

    outcome = silver_transformation.conceptual_event_id_deduplication(records)

    assert outcome["canonical_count"] == 1
    assert outcome["duplicate_count"] == 1
    assert outcome["quarantine_count"] == 0


def test_duplicates_are_not_equivalent_to_invalid_data() -> None:
    assert silver_transformation.EXPECTED_DUPLICATE_OUTPUT_PATH == "data/silver/duplicates"
    assert silver_transformation.EXPECTED_QUARANTINE_OUTPUT_PATH == "data/silver/quarantine"
    assert "duplicate" not in silver_transformation.REJECTION_REASON_IDENTIFIERS


def test_static_summary_is_deterministic_and_runtime_free() -> None:
    config = silver_transformation.parse_silver_config(load_raw_config())
    first = silver_transformation.build_static_summary(config)
    second = silver_transformation.build_static_summary(config)
    tracked = json.loads(
        (PROJECT_ROOT / silver_transformation.SUMMARY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert first == second == tracked
    rendered = json.dumps(first, sort_keys=True)
    assert "row_count" not in rendered
    assert "container_id" not in rendered
    assert "execution_timestamp" not in rendered


def test_docker_wrapper_command_has_no_connector_package() -> None:
    command = run_spark_silver_docker.build_spark_submit_command()

    assert command[:5] == ["docker", "compose", "exec", "-T", "spark"]
    assert command[5] == "/opt/spark/bin/spark-submit"
    assert "/workspace/scripts/run_spark_silver.py" in command
    assert "--packages" not in command
    assert all(".venv" not in part for part in command)
    assert all(";" not in part and "&&" not in part for part in command)


def test_silver_runner_does_not_use_streaming_connector() -> None:
    runner = (PROJECT_ROOT / "scripts" / "run_spark_silver.py").read_text(encoding="utf-8")
    wrapper = (PROJECT_ROOT / "scripts" / "run_spark_silver_docker.py").read_text(encoding="utf-8")

    assert "readStream" not in runner
    assert '.format("kafka")' not in runner
    assert "spark-sql-kafka" not in wrapper
    assert "--packages" not in wrapper


def test_source_guards_prevent_unrelated_outputs_and_services() -> None:
    guarded_files = [
        PROJECT_ROOT / "pipelines" / "streaming" / "silver_transformation.py",
        PROJECT_ROOT / "scripts" / "run_spark_silver.py",
        PROJECT_ROOT / "scripts" / "inspect_spark_silver.py",
        PROJECT_ROOT / "scripts" / "check_spark_silver.py",
    ]
    forbidden_terms = [
        "ai4i_predictor",
        "final_model.joblib",
        "psycopg",
        "sqlalchemy",
        "postgresql://",
        "pg_isready",
        "kafka.bootstrap.servers",
        "confluent_kafka",
        "data/gold",
        "failure_probability",
        "failure_prediction",
        "machine failure",
        "shap",
        "anomaly_label",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"

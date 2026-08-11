from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.streaming import bronze_ingestion
from scripts import check_spark_bronze, run_spark_bronze_docker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_config() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / bronze_ingestion.CONFIG_RELATIVE_PATH).read_text())


def test_spark_config_validation_accepts_expected_values() -> None:
    config = bronze_ingestion.load_spark_config(
        PROJECT_ROOT / bronze_ingestion.CONFIG_RELATIVE_PATH
    )

    assert config.spark_version == "4.0.4"
    assert config.spark_docker_image == "apache/spark:4.0.4-scala2.13-java17-python3-ubuntu"
    assert config.master == "local[2]"
    assert config.application_name == "industrial-fleet-bronze-ingestion"
    assert config.kafka_connector == "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4"
    assert config.kafka_bootstrap_servers == "kafka:29092"
    assert config.kafka_topic == "industrial.telemetry.v1"


def test_spark_config_paths_and_streaming_policies_are_exact() -> None:
    config = bronze_ingestion.parse_spark_config(load_raw_config())

    assert config.bronze_output_path == "data/bronze/telemetry"
    assert config.checkpoint_path == "data/checkpoints/spark/bronze_telemetry"
    assert config.starting_offsets == "earliest"
    assert config.fail_on_data_loss is True
    assert config.output_format == "parquet"
    assert config.output_mode == "append"
    assert config.telemetry_schema_version == "1.0"
    assert config.spark_sql_shuffle_partitions == 3
    assert config.spark_timezone == "UTC"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spark_version", "latest", "spark_version"),
        ("spark_docker_image", "apache/spark:latest", "spark_docker_image"),
        ("master", "spark://spark-master:7077", "master"),
        ("kafka_connector", "manual-jar", "kafka_connector"),
        ("starting_offsets", "latest", "starting_offsets"),
        ("fail_on_data_loss", False, "fail_on_data_loss"),
        ("output_format", "json", "output_format"),
    ],
)
def test_spark_config_rejects_incompatible_values(
    field: str,
    value: object,
    message: str,
) -> None:
    raw_config = load_raw_config()
    raw_config[field] = value

    with pytest.raises(bronze_ingestion.SparkBronzeConfigError, match=message):
        bronze_ingestion.parse_spark_config(raw_config)


def test_spark_config_rejects_absolute_and_parent_relative_paths() -> None:
    raw_config = load_raw_config()
    raw_config["bronze_output_path"] = "/workspace/data/bronze/telemetry"
    with pytest.raises(bronze_ingestion.SparkBronzeConfigError, match="relative path"):
        bronze_ingestion.parse_spark_config(raw_config)

    raw_config = load_raw_config()
    raw_config["checkpoint_path"] = "../outside"
    with pytest.raises(bronze_ingestion.SparkBronzeConfigError, match="relative path"):
        bronze_ingestion.parse_spark_config(raw_config)


def test_bronze_expected_field_names_preserve_raw_kafka_record_shape() -> None:
    assert bronze_ingestion.BRONZE_FIELD_NAMES == (
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        "kafka_key",
        "raw_value",
        "bronze_ingested_at",
        "payload_sha256",
    )
    assert "raw_value" in bronze_ingestion.BRONZE_FIELD_NAMES
    assert "air_temperature_k" not in bronze_ingestion.BRONZE_FIELD_NAMES
    assert "failure_prediction" not in bronze_ingestion.BRONZE_FIELD_NAMES


def test_container_paths_are_workspace_relative() -> None:
    assert (
        bronze_ingestion.container_path("data/bronze/telemetry")
        == "/workspace/data/bronze/telemetry"
    )

    with pytest.raises(bronze_ingestion.SparkBronzeConfigError):
        bronze_ingestion.container_path("C:/absolute/path")


def test_kafka_coordinate_uniqueness_helper_accepts_unique_offsets() -> None:
    records = [
        {"kafka_topic": "industrial.telemetry.v1", "kafka_partition": 0, "kafka_offset": 10},
        {"kafka_topic": "industrial.telemetry.v1", "kafka_partition": 0, "kafka_offset": 11},
        {"kafka_topic": "industrial.telemetry.v1", "kafka_partition": 1, "kafka_offset": 10},
    ]

    assert bronze_ingestion.duplicate_kafka_coordinate_count(records) == 0
    bronze_ingestion.validate_kafka_coordinate_uniqueness(records)


def test_kafka_coordinate_uniqueness_helper_rejects_duplicate_coordinate() -> None:
    records = [
        {"kafka_topic": "industrial.telemetry.v1", "kafka_partition": 0, "kafka_offset": 10},
        {"kafka_topic": "industrial.telemetry.v1", "kafka_partition": 0, "kafka_offset": 10},
    ]

    assert bronze_ingestion.duplicate_kafka_coordinate_count(records) == 1
    with pytest.raises(bronze_ingestion.SparkBronzeValidationError, match="duplicate Kafka"):
        bronze_ingestion.validate_kafka_coordinate_uniqueness(records)


def test_bronze_policy_preserves_same_event_id_at_different_kafka_offsets() -> None:
    raw_payload = json.dumps({"event_id": "same-business-event", "machine_code": "MCH-0001"})
    records = [
        {
            "event_id": "same-business-event",
            "kafka_topic": "industrial.telemetry.v1",
            "kafka_partition": 0,
            "kafka_offset": 100,
            "raw_value": raw_payload,
        },
        {
            "event_id": "same-business-event",
            "kafka_topic": "industrial.telemetry.v1",
            "kafka_partition": 0,
            "kafka_offset": 101,
            "raw_value": raw_payload,
        },
    ]

    assert bronze_ingestion.bronze_policy_allows_event_id_duplicates(records) is True
    assert bronze_ingestion.duplicate_kafka_coordinate_count(records) == 0


def test_integration_summary_is_deterministic_and_runtime_free() -> None:
    config = bronze_ingestion.parse_spark_config(load_raw_config())

    first = bronze_ingestion.build_integration_summary(config)
    second = bronze_ingestion.build_integration_summary(config)

    assert first == second
    rendered = json.dumps(first, sort_keys=True)
    assert "batch_id" not in rendered
    assert "container_id" not in rendered
    assert "execution_timestamp" not in rendered
    assert "runtime offset" not in rendered.lower()
    assert first["preserved_kafka_metadata_fields"] == [
        "topic",
        "partition",
        "offset",
        "timestamp",
        "key",
        "value",
    ]


def test_docker_wrapper_command_construction_uses_pinned_connector() -> None:
    config = bronze_ingestion.parse_spark_config(load_raw_config())

    command = run_spark_bronze_docker.build_spark_submit_command(config)

    assert command[:5] == ["docker", "compose", "exec", "-T", "spark"]
    assert command[5] == "/opt/spark/bin/spark-submit"
    assert "--conf" in command
    assert "spark.jars.ivy=/tmp/spark-ivy" in command
    assert "--packages" in command
    assert "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4" in command
    assert "/workspace/scripts/run_spark_bronze.py" in command
    assert all(".venv" not in part for part in command)
    assert all(";" not in part and "&&" not in part for part in command)


def test_check_spark_bronze_inspection_command_carries_expected_records() -> None:
    command = check_spark_bronze.build_inspection_command(
        [
            {
                "kafka_key": "MCH-0001",
                "kafka_offset": 1,
                "kafka_partition": 0,
                "kafka_topic": "industrial.telemetry.v1",
                "raw_value": "{}",
            }
        ],
        ["{}"],
    )

    assert command[:5] == ["docker", "compose", "exec", "-T", "spark"]
    assert "/workspace/scripts/inspect_spark_bronze.py" in command
    assert "--expected-records-json" in command
    assert "--expected-payloads-json" in command


def test_pyproject_does_not_add_host_pyspark_dependency() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    assert "pyspark" not in pyproject


def test_source_guards_prevent_postgresql_ml_silver_gold_and_deduplication() -> None:
    guarded_files = [
        PROJECT_ROOT / "pipelines" / "streaming" / "bronze_ingestion.py",
        PROJECT_ROOT / "scripts" / "run_spark_bronze.py",
        PROJECT_ROOT / "scripts" / "inspect_spark_bronze.py",
        PROJECT_ROOT / "scripts" / "check_spark_bronze.py",
    ]
    forbidden_terms = [
        "ai4i_predictor",
        "psycopg",
        "sqlalchemy",
        "postgresql://",
        "data/silver",
        "data/gold",
        "final_model.joblib",
        "failure_probability",
        "failure_prediction",
        "machine failure",
        "dropduplicates",
        "drop_duplicates",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.batch import gold_transformation
from scripts import run_spark_gold_docker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_config() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / gold_transformation.CONFIG_RELATIVE_PATH).read_text())


def test_gold_config_validation_accepts_expected_values() -> None:
    config = gold_transformation.load_gold_config(
        PROJECT_ROOT / gold_transformation.CONFIG_RELATIVE_PATH
    )

    assert config.spark_version == "4.0.4"
    assert config.application_name == "industrial-fleet-gold-analytics"
    assert config.master == "local[2]"
    assert config.silver_input_path == "data/silver/telemetry"
    assert config.output_format == "parquet"
    assert config.window_duration == "1 minute"
    assert config.timezone == "UTC"
    assert config.shuffle_partitions == 3


def test_gold_output_paths_are_exact() -> None:
    config = gold_transformation.parse_gold_config(load_raw_config())

    assert config.machine_summary_output_path == "data/gold/machine_summary"
    assert config.machine_windows_output_path == "data/gold/machine_windows"
    assert config.fleet_summary_output_path == "data/gold/fleet_summary"
    assert gold_transformation.gold_output_paths() == {
        "fleet_summary": "data/gold/fleet_summary",
        "machine_summary": "data/gold/machine_summary",
        "machine_windows": "data/gold/machine_windows",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spark_version", "latest", "spark_version"),
        ("application_name", "gold", "application_name"),
        ("master", "spark://master:7077", "master"),
        ("output_format", "delta", "output_format"),
        ("window_duration", "5 minutes", "window_duration"),
        ("timezone", "local", "timezone"),
        ("shuffle_partitions", 10, "shuffle_partitions"),
    ],
)
def test_gold_config_rejects_incompatible_values(
    field: str,
    value: object,
    message: str,
) -> None:
    raw_config = load_raw_config()
    raw_config[field] = value

    with pytest.raises(gold_transformation.SparkGoldConfigError, match=message):
        gold_transformation.parse_gold_config(raw_config)


def test_gold_config_rejects_unsafe_paths() -> None:
    raw_config = load_raw_config()
    raw_config["machine_summary_output_path"] = "/workspace/data/gold/machine_summary"
    with pytest.raises(gold_transformation.SparkGoldConfigError, match="relative path"):
        gold_transformation.parse_gold_config(raw_config)

    raw_config = load_raw_config()
    raw_config["fleet_summary_output_path"] = "../outside"
    with pytest.raises(gold_transformation.SparkGoldConfigError, match="relative path"):
        gold_transformation.parse_gold_config(raw_config)


def test_gold_grains_are_explicit() -> None:
    assert gold_transformation.MACHINE_SUMMARY_GRAIN == ("machine_code",)
    assert gold_transformation.MACHINE_WINDOWS_GRAIN == (
        "machine_code",
        "window_start",
        "window_end",
    )
    assert gold_transformation.FLEET_SUMMARY_GRAIN == ("fleet_scope",)


def test_deterministic_latest_ordering_policy() -> None:
    assert gold_transformation.latest_observation_order() == (
        ("event_time", "desc"),
        ("source_kafka_timestamp", "desc"),
        ("source_kafka_topic", "desc"),
        ("source_kafka_partition", "desc"),
        ("source_kafka_offset", "desc"),
        ("event_id", "desc"),
    )


def test_expected_aggregate_field_definitions() -> None:
    aggregate_fields = gold_transformation.aggregate_field_names()

    assert "avg_air_temperature_k" in aggregate_fields
    assert "min_air_temperature_k" in aggregate_fields
    assert "max_pressure_bar" in aggregate_fields
    assert "health_score" not in aggregate_fields
    assert "risk_score" not in aggregate_fields


def test_event_accounting_helper() -> None:
    assert gold_transformation.event_accounting_holds(
        silver_row_count=10,
        machine_summary_event_count_sum=10,
        machine_windows_event_count_sum=10,
        fleet_event_count=10,
    )
    assert not gold_transformation.event_accounting_holds(
        silver_row_count=10,
        machine_summary_event_count_sum=9,
        machine_windows_event_count_sum=10,
        fleet_event_count=10,
    )


def mixed_product_quality_records() -> list[dict[str, object]]:
    return [
        {
            "event_id": "a",
            "event_time": "2026-02-01T00:00:05Z",
            "machine_code": "MCH-0001",
            "product_quality_type": "L",
            "source_kafka_offset": 1,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": "2026-02-01T00:00:06Z",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
        {
            "event_id": "b",
            "event_time": "2026-02-01T00:00:55Z",
            "machine_code": "MCH-0001",
            "product_quality_type": "H",
            "source_kafka_offset": 2,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": "2026-02-01T00:00:56Z",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
        {
            "event_id": "c",
            "event_time": "2026-02-01T00:01:05Z",
            "machine_code": "MCH-0001",
            "product_quality_type": "M",
            "source_kafka_offset": 3,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": "2026-02-01T00:01:06Z",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
        {
            "event_id": "d",
            "event_time": "2026-02-01T00:01:00Z",
            "machine_code": "MCH-0002",
            "product_quality_type": "H",
            "source_kafka_offset": 4,
            "source_kafka_partition": 1,
            "source_kafka_timestamp": "2026-02-01T00:01:01Z",
            "source_kafka_topic": "industrial.telemetry.v1",
        },
    ]


def test_multiple_product_quality_type_values_for_one_machine_are_valid() -> None:
    summary = gold_transformation.conceptual_machine_summary(mixed_product_quality_records())
    mch_0001 = [row for row in summary if row["machine_code"] == "MCH-0001"]

    assert len(mch_0001) == 1
    assert mch_0001[0]["event_count"] == 3


def test_machine_summary_type_event_counts_preserve_distribution() -> None:
    summary = gold_transformation.conceptual_machine_summary(mixed_product_quality_records())
    mch_0001 = next(row for row in summary if row["machine_code"] == "MCH-0001")

    assert mch_0001["product_quality_type_h_event_count"] == 1
    assert mch_0001["product_quality_type_l_event_count"] == 1
    assert mch_0001["product_quality_type_m_event_count"] == 1
    assert gold_transformation.type_event_counts_reconcile(mch_0001)


def test_latest_product_quality_type_comes_from_deterministic_latest_row() -> None:
    summary = gold_transformation.conceptual_machine_summary(mixed_product_quality_records())
    mch_0001 = next(row for row in summary if row["machine_code"] == "MCH-0001")

    assert mch_0001["latest_product_quality_type"] == "M"


def test_window_grain_is_not_split_by_product_quality_type_variation() -> None:
    windows = gold_transformation.conceptual_machine_windows(mixed_product_quality_records())
    mch_0001_first_window = [
        row
        for row in windows
        if row["machine_code"] == "MCH-0001"
        and str(row["window_start"]).startswith("2026-02-01T00:00:00")
    ]

    assert len(mch_0001_first_window) == 1
    assert mch_0001_first_window[0]["event_count"] == 2
    assert mch_0001_first_window[0]["product_quality_type_h_event_count"] == 1
    assert mch_0001_first_window[0]["product_quality_type_l_event_count"] == 1
    assert mch_0001_first_window[0]["product_quality_type_m_event_count"] == 0
    assert gold_transformation.type_event_counts_reconcile(mch_0001_first_window[0])


def test_fleet_type_event_counts_reconcile_with_event_count() -> None:
    records = mixed_product_quality_records()
    counts = gold_transformation.product_quality_type_event_counts(records)
    fleet_row = {
        "event_count": len(records),
        "product_quality_type_h_event_count": counts["H"],
        "product_quality_type_l_event_count": counts["L"],
        "product_quality_type_m_event_count": counts["M"],
    }

    assert counts == {"H": 2, "L": 1, "M": 1}
    assert gold_transformation.type_event_counts_reconcile(fleet_row)


def test_newer_event_supplies_latest_observation() -> None:
    older = {
        "event_id": "a",
        "event_time": "2026-02-01T00:00:00Z",
        "source_kafka_offset": 1,
        "source_kafka_partition": 0,
        "source_kafka_timestamp": "2026-02-01T00:00:01Z",
        "source_kafka_topic": "industrial.telemetry.v1",
        "torque_nm": 40.0,
    }
    newer = {
        "event_id": "b",
        "event_time": "2026-02-01T00:01:00Z",
        "source_kafka_offset": 2,
        "source_kafka_partition": 0,
        "source_kafka_timestamp": "2026-02-01T00:01:01Z",
        "source_kafka_topic": "industrial.telemetry.v1",
        "torque_nm": 50.0,
    }

    assert gold_transformation.conceptual_latest_record([older, newer]) == newer


def test_tied_event_time_uses_kafka_lineage_for_latest_observation() -> None:
    lower_offset = {
        "event_id": "a",
        "event_time": "2026-02-01T00:00:00Z",
        "product_quality_type": "L",
        "source_kafka_offset": 10,
        "source_kafka_partition": 0,
        "source_kafka_timestamp": "2026-02-01T00:00:01Z",
        "source_kafka_topic": "industrial.telemetry.v1",
    }
    higher_offset = {
        "event_id": "b",
        "event_time": "2026-02-01T00:00:00Z",
        "product_quality_type": "H",
        "source_kafka_offset": 11,
        "source_kafka_partition": 0,
        "source_kafka_timestamp": "2026-02-01T00:00:01Z",
        "source_kafka_topic": "industrial.telemetry.v1",
    }

    latest = gold_transformation.conceptual_latest_record([lower_offset, higher_offset])

    assert latest == higher_offset
    assert latest["product_quality_type"] == "H"


def test_window_key_assigns_each_event_to_one_minute_window() -> None:
    first = gold_transformation.conceptual_window_key({"event_time": "2026-02-01T00:00:59Z"})
    second = gold_transformation.conceptual_window_key({"event_time": "2026-02-01T00:01:00Z"})

    assert first != second
    assert first[0].startswith("2026-02-01T00:00:00")
    assert second[0].startswith("2026-02-01T00:01:00")


def test_static_summary_is_deterministic_and_runtime_free() -> None:
    config = gold_transformation.parse_gold_config(load_raw_config())
    first = gold_transformation.build_static_summary(config)
    second = gold_transformation.build_static_summary(config)
    tracked = json.loads(
        (PROJECT_ROOT / gold_transformation.SUMMARY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert first == second == tracked
    rendered = json.dumps(first, sort_keys=True)
    assert "row_count" not in rendered
    assert "container_id" not in rendered
    assert "execution_timestamp" not in rendered


def test_docker_wrapper_command_has_no_connector_package() -> None:
    command = run_spark_gold_docker.build_spark_submit_command()

    assert command[:5] == ["docker", "compose", "exec", "-T", "spark"]
    assert command[5] == "/opt/spark/bin/spark-submit"
    assert "/workspace/scripts/run_spark_gold.py" in command
    assert "--packages" not in command
    assert all(".venv" not in part for part in command)
    assert all(";" not in part and "&&" not in part for part in command)


def test_gold_runner_does_not_use_streaming_connector_or_external_systems() -> None:
    runner = (PROJECT_ROOT / "scripts" / "run_spark_gold.py").read_text(encoding="utf-8")
    wrapper = (PROJECT_ROOT / "scripts" / "run_spark_gold_docker.py").read_text(encoding="utf-8")

    assert "readStream" not in runner
    assert '.format("kafka")' not in runner
    assert "spark-sql-kafka" not in wrapper
    assert "--packages" not in wrapper
    assert "jdbc" not in runner.lower()


def test_source_guards_prevent_unrelated_fields_and_services() -> None:
    guarded_files = [
        PROJECT_ROOT / "pipelines" / "batch" / "gold_transformation.py",
        PROJECT_ROOT / "scripts" / "run_spark_gold.py",
        PROJECT_ROOT / "scripts" / "inspect_spark_gold.py",
        PROJECT_ROOT / "scripts" / "check_spark_gold.py",
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
        "failure_" + "probability",
        "failure_" + "prediction",
        "machine_" + "failure",
        "health_" + "score",
        "risk_" + "score",
        "risk_" + "level",
        "anomaly_" + "score",
        "anomaly_" + "label",
        "maintenance_" + "required",
        "s" + "hap",
    ]

    for path in guarded_files:
        source = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source, f"{term} found in {path}"


def test_pyproject_does_not_add_host_pyspark_dependency() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    assert "pyspark" not in pyproject

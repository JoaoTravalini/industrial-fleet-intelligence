"""Spark transformation from Bronze telemetry records into Silver datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_RELATIVE_PATH = Path("pipelines") / "streaming" / "silver_config.json"
SUMMARY_RELATIVE_PATH = Path("reports") / "streaming" / "spark_silver_summary.json"
CONTAINER_WORKSPACE = PurePosixPath("/workspace")

EXPECTED_SPARK_VERSION = "4.0.4"
EXPECTED_APPLICATION_NAME = "industrial-fleet-silver-telemetry"
EXPECTED_MASTER = "local[2]"
EXPECTED_BRONZE_INPUT_PATH = "data/bronze/telemetry"
EXPECTED_SILVER_OUTPUT_PATH = "data/silver/telemetry"
EXPECTED_DUPLICATE_OUTPUT_PATH = "data/silver/duplicates"
EXPECTED_QUARANTINE_OUTPUT_PATH = "data/silver/quarantine"
EXPECTED_OUTPUT_FORMAT = "parquet"
EXPECTED_TELEMETRY_SCHEMA_VERSION = "1.0"
EXPECTED_BUSINESS_EVENT_KEY = "event_id"
EXPECTED_DUPLICATE_POLICY = "retain_one_canonical_record_and_audit_additional_valid_records"
EXPECTED_SHUFFLE_PARTITIONS = 3
EXPECTED_TIMEZONE = "UTC"

BRONZE_REQUIRED_FIELD_NAMES = (
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "kafka_key",
    "raw_value",
    "bronze_ingested_at",
    "payload_sha256",
)
TELEMETRY_FIELD_NAMES = (
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
CANONICAL_SILVER_FIELD_NAMES = (
    *TELEMETRY_FIELD_NAMES,
    "source_kafka_topic",
    "source_kafka_partition",
    "source_kafka_offset",
    "source_kafka_timestamp",
    "source_kafka_key",
    "bronze_ingested_at",
    "payload_sha256",
)
DUPLICATE_AUDIT_FIELD_NAMES = (
    *CANONICAL_SILVER_FIELD_NAMES,
    "duplicate_rank",
    "canonical_source_kafka_topic",
    "canonical_source_kafka_partition",
    "canonical_source_kafka_offset",
)
QUARANTINE_FIELD_NAMES = (
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "kafka_key",
    "raw_value",
    "payload_sha256",
    "bronze_ingested_at",
    "event_id",
    "machine_code",
    "rejection_reasons",
)
REJECTION_REASON_IDENTIFIERS = (
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
)
SENSOR_BOUND_POLICY = {
    "air_temperature_k": (294.0, 306.0),
    "process_temperature_k": (304.0, 315.0),
    "rotational_speed_rpm": (1000, 3000),
    "torque_nm": (0.0, 80.0),
    "tool_wear_min": (0, 300),
    "vibration_mm_s": (0.0, 15.0),
    "pressure_bar": (1.0, 12.0),
}
PRODUCT_QUALITY_TYPES = ("L", "M", "H")
MACHINE_CODE_PATTERN = r"^MCH-(?:000[1-9]|00[1-9][0-9]|0100)$"
UUID_PATTERN = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
UTC_EVENT_TIME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
JSON_INTEGER_TOKEN_PATTERN = r"^-?(?:0|[1-9][0-9]*)$"
JSON_NUMBER_TOKEN_PATTERN = r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
JSON_STRING_TOKEN_PATTERN = r'^"(?:[^"\\]|\\.)*"$'
CANONICAL_ORDERING_FIELDS = (
    "source_kafka_timestamp",
    "source_kafka_topic",
    "source_kafka_partition",
    "source_kafka_offset",
    "payload_sha256",
)


class SparkSilverConfigError(ValueError):
    """Raised when Silver configuration is missing or incompatible."""


class SparkSilverValidationError(RuntimeError):
    """Raised when Bronze or Silver validation fails."""


@dataclass(frozen=True)
class SparkSilverConfig:
    """Static local Spark Silver transformation configuration."""

    spark_version: str
    application_name: str
    master: str
    bronze_input_path: str
    silver_output_path: str
    duplicate_output_path: str
    quarantine_output_path: str
    output_format: str
    telemetry_schema_version: str
    business_event_key: str
    duplicate_policy: str
    spark_sql_shuffle_partitions: int
    spark_timezone: str


@dataclass(frozen=True)
class SilverTransformResult:
    """DataFrames produced by applying Silver validation and deduplication."""

    valid_pre_dedup_df: Any
    canonical_df: Any
    duplicate_df: Any
    quarantine_df: Any


@dataclass(frozen=True)
class SilverWriteCounts:
    """Logical row counts from a Silver snapshot rebuild."""

    bronze_row_count: int
    valid_pre_dedup_row_count: int
    canonical_silver_row_count: int
    duplicate_audit_row_count: int
    quarantine_row_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "bronze_row_count": self.bronze_row_count,
            "canonical_silver_row_count": self.canonical_silver_row_count,
            "duplicate_audit_row_count": self.duplicate_audit_row_count,
            "quarantine_row_count": self.quarantine_row_count,
            "valid_pre_dedup_row_count": self.valid_pre_dedup_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def load_silver_config(path: Path | None = None) -> SparkSilverConfig:
    config_file = path or config_path()
    try:
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SparkSilverConfigError(f"Silver config file not found: {config_file}") from exc
    except json.JSONDecodeError as exc:
        raise SparkSilverConfigError(
            f"Silver config file is not valid JSON: {config_file}"
        ) from exc
    if not isinstance(raw_config, dict):
        raise SparkSilverConfigError("Silver config must be a JSON object.")
    return parse_silver_config(raw_config)


def parse_silver_config(raw_config: Mapping[str, Any]) -> SparkSilverConfig:
    required_keys = {
        "application_name",
        "bronze_input_path",
        "business_event_key",
        "duplicate_output_path",
        "duplicate_policy",
        "master",
        "output_format",
        "quarantine_output_path",
        "silver_output_path",
        "spark_sql_shuffle_partitions",
        "spark_timezone",
        "spark_version",
        "telemetry_schema_version",
    }
    actual_keys = set(raw_config)
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if missing:
        raise SparkSilverConfigError("Missing Silver config key(s): " + ", ".join(missing))
    if unknown:
        raise SparkSilverConfigError("Unknown Silver config key(s): " + ", ".join(unknown))

    config = SparkSilverConfig(
        spark_version=require_text(raw_config, "spark_version"),
        application_name=require_text(raw_config, "application_name"),
        master=require_text(raw_config, "master"),
        bronze_input_path=require_text(raw_config, "bronze_input_path"),
        silver_output_path=require_text(raw_config, "silver_output_path"),
        duplicate_output_path=require_text(raw_config, "duplicate_output_path"),
        quarantine_output_path=require_text(raw_config, "quarantine_output_path"),
        output_format=require_text(raw_config, "output_format"),
        telemetry_schema_version=require_text(raw_config, "telemetry_schema_version"),
        business_event_key=require_text(raw_config, "business_event_key"),
        duplicate_policy=require_text(raw_config, "duplicate_policy"),
        spark_sql_shuffle_partitions=require_int(raw_config, "spark_sql_shuffle_partitions"),
        spark_timezone=require_text(raw_config, "spark_timezone"),
    )
    validate_silver_config(config)
    return config


def require_text(raw_config: Mapping[str, Any], key: str) -> str:
    value = raw_config[key]
    if not isinstance(value, str) or not value.strip():
        raise SparkSilverConfigError(f"Silver config key {key} must be a non-empty string.")
    return value


def require_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SparkSilverConfigError(f"Silver config key {key} must be an integer.")
    return value


def validate_relative_path(value: str, field_name: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise SparkSilverConfigError(f"{field_name} must be a safe relative path.")


def validate_silver_config(config: SparkSilverConfig) -> None:
    for field_name, value in (
        ("bronze_input_path", config.bronze_input_path),
        ("silver_output_path", config.silver_output_path),
        ("duplicate_output_path", config.duplicate_output_path),
        ("quarantine_output_path", config.quarantine_output_path),
    ):
        validate_relative_path(value, field_name)

    expected_values: tuple[tuple[str, str, str], ...] = (
        ("spark_version", config.spark_version, EXPECTED_SPARK_VERSION),
        ("application_name", config.application_name, EXPECTED_APPLICATION_NAME),
        ("master", config.master, EXPECTED_MASTER),
        ("bronze_input_path", config.bronze_input_path, EXPECTED_BRONZE_INPUT_PATH),
        ("silver_output_path", config.silver_output_path, EXPECTED_SILVER_OUTPUT_PATH),
        (
            "duplicate_output_path",
            config.duplicate_output_path,
            EXPECTED_DUPLICATE_OUTPUT_PATH,
        ),
        (
            "quarantine_output_path",
            config.quarantine_output_path,
            EXPECTED_QUARANTINE_OUTPUT_PATH,
        ),
        ("output_format", config.output_format, EXPECTED_OUTPUT_FORMAT),
        (
            "telemetry_schema_version",
            config.telemetry_schema_version,
            EXPECTED_TELEMETRY_SCHEMA_VERSION,
        ),
        ("business_event_key", config.business_event_key, EXPECTED_BUSINESS_EVENT_KEY),
        ("duplicate_policy", config.duplicate_policy, EXPECTED_DUPLICATE_POLICY),
        ("spark_timezone", config.spark_timezone, EXPECTED_TIMEZONE),
    )
    for field_name, actual, expected in expected_values:
        if actual != expected:
            raise SparkSilverConfigError(f"{field_name} must be {expected}.")
    if config.spark_sql_shuffle_partitions != EXPECTED_SHUFFLE_PARTITIONS:
        raise SparkSilverConfigError("spark_sql_shuffle_partitions must be 3.")


def container_path(relative_path: str) -> str:
    validate_relative_path(relative_path, "container path")
    return str(CONTAINER_WORKSPACE / PurePosixPath(relative_path.replace("\\", "/")))


def telemetry_field_types() -> dict[str, str]:
    return {
        "air_temperature_k": "double",
        "event_id": "string",
        "event_time": "timestamp",
        "machine_code": "string",
        "pressure_bar": "double",
        "process_temperature_k": "double",
        "product_quality_type": "string",
        "rotational_speed_rpm": "int",
        "schema_version": "string",
        "sequence_number": "bigint",
        "source": "string",
        "tool_wear_min": "int",
        "torque_nm": "double",
        "vibration_mm_s": "double",
    }


def sensor_bound_policy() -> dict[str, tuple[int | float, int | float]]:
    return dict(SENSOR_BOUND_POLICY)


def machine_code_contract() -> str:
    return MACHINE_CODE_PATTERN


def canonical_ordering_fields() -> tuple[str, ...]:
    return CANONICAL_ORDERING_FIELDS


def accounting_invariants_hold(
    *,
    bronze_row_count: int,
    valid_pre_dedup_row_count: int,
    canonical_silver_row_count: int,
    duplicate_audit_row_count: int,
    quarantine_row_count: int,
) -> bool:
    return (
        bronze_row_count == quarantine_row_count + valid_pre_dedup_row_count
        and valid_pre_dedup_row_count == canonical_silver_row_count + duplicate_audit_row_count
        and bronze_row_count
        == quarantine_row_count + canonical_silver_row_count + duplicate_audit_row_count
    )


def conceptual_event_id_deduplication(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | str | None]:
    if not records:
        return {
            "canonical_count": 0,
            "duplicate_count": 0,
            "quarantine_count": 0,
            "canonical_event_id": None,
        }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["event_id"]), []).append(record)

    canonical_count = 0
    duplicate_count = 0
    canonical_event_id: str | None = None
    for event_id, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda row: (
                str(row["source_kafka_timestamp"]),
                str(row["source_kafka_topic"]),
                int(row["source_kafka_partition"]),
                int(row["source_kafka_offset"]),
                str(row.get("payload_sha256") or ""),
            ),
        )
        if ordered:
            canonical_count += 1
            duplicate_count += len(ordered) - 1
            canonical_event_id = event_id if canonical_event_id is None else canonical_event_id
    return {
        "canonical_count": canonical_count,
        "duplicate_count": duplicate_count,
        "quarantine_count": 0,
        "canonical_event_id": canonical_event_id,
    }


def build_static_summary(config: SparkSilverConfig) -> dict[str, Any]:
    return {
        "bronze_input_path": config.bronze_input_path,
        "business_event_key": config.business_event_key,
        "canonical_ordering_strategy": list(CANONICAL_ORDERING_FIELDS),
        "canonical_silver_path": config.silver_output_path,
        "deduplication_strategy": (
            "Valid telemetry records are partitioned by event_id; the first deterministic "
            "record is retained as canonical and later valid records are audited."
        ),
        "duplicate_audit_path": config.duplicate_output_path,
        "duplicate_audit_policy": (
            "Valid non-canonical records with repeated event_id values are preserved in "
            "the duplicate audit dataset."
        ),
        "execution_model": "deterministic local Spark snapshot rebuild on local[2]",
        "output_format": config.output_format,
        "quarantine_path": config.quarantine_output_path,
        "quarantine_policy": (
            "Invalid telemetry records retain raw payload and Kafka lineage with stable "
            "machine-readable rejection reasons."
        ),
        "runtime_counts": "intentionally excluded from tracked summary",
        "spark_version": config.spark_version,
        "telemetry_schema_version": config.telemetry_schema_version,
        "validation_strategy": (
            "Spark parses raw Bronze JSON with an explicit schema, checks exact contract "
            "fields, validates typed sensor and identity rules, and separates invalid rows."
        ),
    }


def write_static_summary(root: Path | None, config: SparkSilverConfig) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_static_summary(config), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def create_spark_session(config: SparkSilverConfig) -> Any:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(config.application_name)
        .master(config.master)
        .config("spark.sql.shuffle.partitions", str(config.spark_sql_shuffle_partitions))
        .config("spark.sql.session.timeZone", config.spark_timezone)
        .getOrCreate()
    )


def telemetry_json_schema() -> Any:
    from pyspark.sql import types as spark_types

    return spark_types.StructType(
        [
            spark_types.StructField("schema_version", spark_types.StringType(), True),
            spark_types.StructField("event_id", spark_types.StringType(), True),
            spark_types.StructField("machine_code", spark_types.StringType(), True),
            spark_types.StructField("sequence_number", spark_types.LongType(), True),
            spark_types.StructField("event_time", spark_types.StringType(), True),
            spark_types.StructField("source", spark_types.StringType(), True),
            spark_types.StructField("product_quality_type", spark_types.StringType(), True),
            spark_types.StructField("air_temperature_k", spark_types.DoubleType(), True),
            spark_types.StructField("process_temperature_k", spark_types.DoubleType(), True),
            spark_types.StructField("rotational_speed_rpm", spark_types.IntegerType(), True),
            spark_types.StructField("torque_nm", spark_types.DoubleType(), True),
            spark_types.StructField("tool_wear_min", spark_types.IntegerType(), True),
            spark_types.StructField("vibration_mm_s", spark_types.DoubleType(), True),
            spark_types.StructField("pressure_bar", spark_types.DoubleType(), True),
        ]
    )


def validate_bronze_schema(bronze_df: Any) -> None:
    missing = sorted(set(BRONZE_REQUIRED_FIELD_NAMES) - set(bronze_df.columns))
    if missing:
        raise SparkSilverValidationError(
            "Bronze input is missing required column(s): " + ", ".join(missing)
        )


def read_bronze_snapshot(spark: Any, config: SparkSilverConfig) -> Any:
    bronze_path = Path(container_path(config.bronze_input_path))
    if not bronze_path.exists():
        raise SparkSilverValidationError(f"Bronze input does not exist: {config.bronze_input_path}")
    bronze_df = spark.read.parquet(str(bronze_path))
    validate_bronze_schema(bronze_df)
    return bronze_df


def _json_token(raw_column: str, field_name: str) -> Any:
    from pyspark.sql import functions as spark_fn

    pattern = rf'"{re.escape(field_name)}"\s*:\s*([^,}}\r\n]+)'
    return spark_fn.trim(spark_fn.regexp_extract(spark_fn.col(raw_column), pattern, 1))


def _is_true(condition: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    return spark_fn.coalesce(condition, spark_fn.lit(False))


def _is_json_string_token(token: Any) -> Any:
    return _is_true(token.rlike(JSON_STRING_TOKEN_PATTERN))


def _is_json_integer_token(token: Any) -> Any:
    return _is_true(token.rlike(JSON_INTEGER_TOKEN_PATTERN))


def _is_json_number_token(token: Any) -> Any:
    return _is_true(token.rlike(JSON_NUMBER_TOKEN_PATTERN))


def _valid_text_field(field_name: str, parsed_column: Any) -> Any:
    token = _json_token("raw_value", field_name)
    return _is_json_string_token(token) & parsed_column.isNotNull()


def _valid_exact_text_field(field_name: str, parsed_column: Any, expected_value: str) -> Any:
    return _valid_text_field(field_name, parsed_column) & (parsed_column == expected_value)


def _valid_integer_field(field_name: str, parsed_column: Any, lower: int, upper: int | None) -> Any:
    valid = _is_json_integer_token(_json_token("raw_value", field_name)) & parsed_column.isNotNull()
    valid = valid & (parsed_column >= lower)
    if upper is not None:
        valid = valid & (parsed_column <= upper)
    return _is_true(valid)


def _valid_number_field(
    field_name: str,
    parsed_column: Any,
    lower: float,
    upper: float,
) -> Any:
    from pyspark.sql import functions as spark_fn

    return _is_true(
        _is_json_number_token(_json_token("raw_value", field_name))
        & parsed_column.isNotNull()
        & ~spark_fn.isnan(parsed_column)
        & (parsed_column >= lower)
        & (parsed_column <= upper)
    )


def _reason_when(condition: Any, reason: str) -> Any:
    from pyspark.sql import functions as spark_fn

    return spark_fn.when(_is_true(condition), spark_fn.lit(reason))


def parse_and_validate_bronze(bronze_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn
    from pyspark.sql import types as spark_types

    validate_bronze_schema(bronze_df)
    expected_keys = spark_fn.array(*[spark_fn.lit(field) for field in TELEMETRY_FIELD_NAMES])
    json_map_schema = spark_types.MapType(spark_types.StringType(), spark_types.StringType())
    parsed_df = (
        bronze_df.withColumn("_json_map", spark_fn.from_json("raw_value", json_map_schema))
        .withColumn("_parsed", spark_fn.from_json("raw_value", telemetry_json_schema()))
        .withColumn("_json_keys", spark_fn.map_keys("_json_map"))
        .withColumn("_event_time", spark_fn.try_to_timestamp(spark_fn.col("_parsed.event_time")))
    )
    valid_json_object = spark_fn.col("_json_map").isNotNull()
    missing_fields = valid_json_object & (
        spark_fn.size(spark_fn.array_except(expected_keys, spark_fn.col("_json_keys"))) > 0
    )
    unexpected_fields = valid_json_object & (
        spark_fn.size(spark_fn.array_except(spark_fn.col("_json_keys"), expected_keys)) > 0
    )

    schema_version = spark_fn.col("_parsed.schema_version")
    event_id = spark_fn.col("_parsed.event_id")
    machine_code = spark_fn.col("_parsed.machine_code")
    sequence_number = spark_fn.col("_parsed.sequence_number")
    event_time_text = spark_fn.col("_parsed.event_time")
    event_time = spark_fn.col("_event_time")
    source = spark_fn.col("_parsed.source")
    product_quality_type = spark_fn.col("_parsed.product_quality_type")
    air_temperature_k = spark_fn.col("_parsed.air_temperature_k")
    process_temperature_k = spark_fn.col("_parsed.process_temperature_k")
    rotational_speed_rpm = spark_fn.col("_parsed.rotational_speed_rpm")
    torque_nm = spark_fn.col("_parsed.torque_nm")
    tool_wear_min = spark_fn.col("_parsed.tool_wear_min")
    vibration_mm_s = spark_fn.col("_parsed.vibration_mm_s")
    pressure_bar = spark_fn.col("_parsed.pressure_bar")

    valid_event_id = _valid_text_field("event_id", event_id) & _is_true(
        event_id.rlike(UUID_PATTERN)
    )
    valid_machine_code = _valid_text_field("machine_code", machine_code) & _is_true(
        machine_code.rlike(MACHINE_CODE_PATTERN)
    )
    valid_sequence_number = _valid_integer_field("sequence_number", sequence_number, 1, None)
    valid_event_time = (
        _valid_text_field("event_time", event_time_text)
        & _is_true(event_time_text.rlike(UTC_EVENT_TIME_PATTERN))
        & event_time.isNotNull()
    )
    valid_product_quality_type = _valid_text_field(
        "product_quality_type",
        product_quality_type,
    ) & product_quality_type.isin(*PRODUCT_QUALITY_TYPES)
    valid_air_temperature = _valid_number_field(
        "air_temperature_k",
        air_temperature_k,
        294.0,
        306.0,
    )
    valid_process_temperature = _valid_number_field(
        "process_temperature_k",
        process_temperature_k,
        304.0,
        315.0,
    )
    valid_rotational_speed = _valid_integer_field(
        "rotational_speed_rpm",
        rotational_speed_rpm,
        1000,
        3000,
    )
    valid_torque = _valid_number_field("torque_nm", torque_nm, 0.0, 80.0)
    valid_tool_wear = _valid_integer_field("tool_wear_min", tool_wear_min, 0, 300)
    valid_vibration = _valid_number_field("vibration_mm_s", vibration_mm_s, 0.0, 15.0)
    valid_pressure = _valid_number_field("pressure_bar", pressure_bar, 1.0, 12.0)

    process_not_above_air = (
        valid_air_temperature
        & valid_process_temperature
        & (process_temperature_k <= air_temperature_k)
    )
    kafka_key_mismatch = valid_machine_code & (
        spark_fn.col("kafka_key").isNull() | (spark_fn.col("kafka_key") != machine_code)
    )

    reasons = spark_fn.array(
        _reason_when(~valid_json_object, "malformed_json"),
        _reason_when(missing_fields, "missing_required_field"),
        _reason_when(unexpected_fields, "unexpected_field"),
        _reason_when(
            valid_json_object & ~_valid_exact_text_field("schema_version", schema_version, "1.0"),
            "invalid_schema_version",
        ),
        _reason_when(valid_json_object & ~valid_event_id, "invalid_event_id"),
        _reason_when(valid_json_object & ~valid_machine_code, "invalid_machine_code"),
        _reason_when(valid_json_object & ~valid_sequence_number, "invalid_sequence_number"),
        _reason_when(valid_json_object & ~valid_event_time, "invalid_event_time"),
        _reason_when(
            valid_json_object & ~_valid_exact_text_field("source", source, "synthetic_simulator"),
            "invalid_source",
        ),
        _reason_when(
            valid_json_object & ~valid_product_quality_type,
            "invalid_product_quality_type",
        ),
        _reason_when(valid_json_object & ~valid_air_temperature, "invalid_air_temperature_k"),
        _reason_when(
            valid_json_object & ~valid_process_temperature,
            "invalid_process_temperature_k",
        ),
        _reason_when(valid_json_object & ~valid_rotational_speed, "invalid_rotational_speed_rpm"),
        _reason_when(valid_json_object & ~valid_torque, "invalid_torque_nm"),
        _reason_when(valid_json_object & ~valid_tool_wear, "invalid_tool_wear_min"),
        _reason_when(valid_json_object & ~valid_vibration, "invalid_vibration_mm_s"),
        _reason_when(valid_json_object & ~valid_pressure, "invalid_pressure_bar"),
        _reason_when(
            valid_json_object & process_not_above_air,
            "process_temperature_not_above_air_temperature",
        ),
        _reason_when(
            valid_json_object & kafka_key_mismatch,
            "kafka_key_machine_code_mismatch",
        ),
    )

    return parsed_df.withColumn(
        "rejection_reasons",
        spark_fn.array_sort(
            spark_fn.array_distinct(spark_fn.filter(reasons, lambda reason: reason.isNotNull()))
        ),
    )


def build_valid_pre_dedup(validated_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    return validated_df.where(spark_fn.size("rejection_reasons") == 0).select(
        spark_fn.col("_parsed.schema_version").alias("schema_version"),
        spark_fn.col("_parsed.event_id").alias("event_id"),
        spark_fn.col("_parsed.machine_code").alias("machine_code"),
        spark_fn.col("_parsed.sequence_number").cast("long").alias("sequence_number"),
        spark_fn.col("_event_time").alias("event_time"),
        spark_fn.col("_parsed.source").alias("source"),
        spark_fn.col("_parsed.product_quality_type").alias("product_quality_type"),
        spark_fn.col("_parsed.air_temperature_k").cast("double").alias("air_temperature_k"),
        spark_fn.col("_parsed.process_temperature_k").cast("double").alias("process_temperature_k"),
        spark_fn.col("_parsed.rotational_speed_rpm").cast("int").alias("rotational_speed_rpm"),
        spark_fn.col("_parsed.torque_nm").cast("double").alias("torque_nm"),
        spark_fn.col("_parsed.tool_wear_min").cast("int").alias("tool_wear_min"),
        spark_fn.col("_parsed.vibration_mm_s").cast("double").alias("vibration_mm_s"),
        spark_fn.col("_parsed.pressure_bar").cast("double").alias("pressure_bar"),
        spark_fn.col("kafka_topic").alias("source_kafka_topic"),
        spark_fn.col("kafka_partition").alias("source_kafka_partition"),
        spark_fn.col("kafka_offset").alias("source_kafka_offset"),
        spark_fn.col("kafka_timestamp").alias("source_kafka_timestamp"),
        spark_fn.col("kafka_key").alias("source_kafka_key"),
        "bronze_ingested_at",
        "payload_sha256",
    )


def build_quarantine(validated_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    return validated_df.where(spark_fn.size("rejection_reasons") > 0).select(
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        "kafka_key",
        "raw_value",
        "payload_sha256",
        "bronze_ingested_at",
        spark_fn.col("_parsed.event_id").alias("event_id"),
        spark_fn.col("_parsed.machine_code").alias("machine_code"),
        "rejection_reasons",
    )


def build_canonical_and_duplicates(valid_pre_dedup_df: Any) -> tuple[Any, Any]:
    from pyspark.sql import Window
    from pyspark.sql import functions as spark_fn

    ordering = [
        spark_fn.col("source_kafka_timestamp").asc(),
        spark_fn.col("source_kafka_topic").asc(),
        spark_fn.col("source_kafka_partition").asc(),
        spark_fn.col("source_kafka_offset").asc(),
        spark_fn.col("payload_sha256").asc_nulls_last(),
    ]
    rank_window = Window.partitionBy("event_id").orderBy(*ordering)
    ranked = valid_pre_dedup_df.withColumn(
        "duplicate_rank",
        spark_fn.row_number().over(rank_window),
    )
    canonical = ranked.where(spark_fn.col("duplicate_rank") == 1).select(
        *CANONICAL_SILVER_FIELD_NAMES
    )
    canonical_lookup = canonical.select(
        "event_id",
        spark_fn.col("source_kafka_topic").alias("canonical_source_kafka_topic"),
        spark_fn.col("source_kafka_partition").alias("canonical_source_kafka_partition"),
        spark_fn.col("source_kafka_offset").alias("canonical_source_kafka_offset"),
    )
    duplicates = (
        ranked.where(spark_fn.col("duplicate_rank") > 1)
        .join(canonical_lookup, on="event_id", how="left")
        .select(*DUPLICATE_AUDIT_FIELD_NAMES)
    )
    return canonical, duplicates


def transform_bronze_to_silver(bronze_df: Any) -> SilverTransformResult:
    validated = parse_and_validate_bronze(bronze_df)
    valid_pre_dedup = build_valid_pre_dedup(validated)
    quarantine = build_quarantine(validated)
    canonical, duplicates = build_canonical_and_duplicates(valid_pre_dedup)
    return SilverTransformResult(
        valid_pre_dedup_df=valid_pre_dedup,
        canonical_df=canonical,
        duplicate_df=duplicates,
        quarantine_df=quarantine,
    )


def write_silver_outputs(result: SilverTransformResult, config: SparkSilverConfig) -> None:
    result.canonical_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.silver_output_path)
    )
    result.duplicate_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.duplicate_output_path)
    )
    result.quarantine_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.quarantine_output_path)
    )


def rebuild_silver_snapshot(spark: Any, config: SparkSilverConfig) -> SilverWriteCounts:
    bronze_df = read_bronze_snapshot(spark, config)
    result = transform_bronze_to_silver(bronze_df)
    counts = SilverWriteCounts(
        bronze_row_count=int(bronze_df.count()),
        valid_pre_dedup_row_count=int(result.valid_pre_dedup_df.count()),
        canonical_silver_row_count=int(result.canonical_df.count()),
        duplicate_audit_row_count=int(result.duplicate_df.count()),
        quarantine_row_count=int(result.quarantine_df.count()),
    )
    if not accounting_invariants_hold(**counts.to_dict()):
        raise SparkSilverValidationError("Silver accounting invariants failed before write.")
    write_silver_outputs(result, config)
    return counts


def _path_exists(relative_path: str) -> bool:
    return Path(container_path(relative_path)).exists()


def _read_parquet(spark: Any, relative_path: str) -> Any:
    return spark.read.parquet(container_path(relative_path))


def _reason_counts(quarantine_df: Any) -> dict[str, int]:
    from pyspark.sql import functions as spark_fn

    if quarantine_df.count() == 0:
        return {}
    return {
        str(row["reason"]): int(row["count"])
        for row in quarantine_df.select(spark_fn.explode("rejection_reasons").alias("reason"))
        .groupBy("reason")
        .count()
        .orderBy("reason")
        .collect()
    }


def _product_quality_counts(canonical_df: Any) -> dict[str, int]:
    if canonical_df.count() == 0:
        return {quality: 0 for quality in PRODUCT_QUALITY_TYPES}
    counts = {
        str(row["product_quality_type"]): int(row["count"])
        for row in canonical_df.groupBy("product_quality_type")
        .count()
        .orderBy("product_quality_type")
        .collect()
    }
    return {quality: int(counts.get(quality, 0)) for quality in PRODUCT_QUALITY_TYPES}


def _max_duplicate_rank(duplicate_df: Any) -> int | None:
    from pyspark.sql import functions as spark_fn

    if duplicate_df.count() == 0:
        return None
    value = duplicate_df.agg(spark_fn.max("duplicate_rank").alias("max_rank")).collect()[0][
        "max_rank"
    ]
    return None if value is None else int(value)


def _event_time_bound(canonical_df: Any, aggregate_name: str) -> str | None:
    from pyspark.sql import functions as spark_fn

    if canonical_df.count() == 0:
        return None
    aggregate = spark_fn.min if aggregate_name == "min" else spark_fn.max
    value = canonical_df.agg(aggregate("event_time").cast("string").alias("bound")).collect()[0][
        "bound"
    ]
    return None if value is None else str(value)


def _selection_digest(canonical_df: Any) -> str | None:
    from pyspark.sql import functions as spark_fn

    if canonical_df.count() == 0:
        return None
    signature = spark_fn.concat_ws(
        "|",
        spark_fn.col("event_id"),
        spark_fn.col("source_kafka_topic"),
        spark_fn.col("source_kafka_partition").cast("string"),
        spark_fn.col("source_kafka_offset").cast("string"),
        spark_fn.coalesce(spark_fn.col("payload_sha256"), spark_fn.lit("")),
    )
    value = (
        canonical_df.select(signature.alias("signature"))
        .agg(
            spark_fn.sha2(
                spark_fn.concat_ws("\n", spark_fn.array_sort(spark_fn.collect_list("signature"))),
                256,
            ).alias("digest")
        )
        .collect()[0]["digest"]
    )
    return None if value is None else str(value)


def _schema_types(df: Any) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}


def _coordinate_overlap_count(left_df: Any, right_df: Any, *, right_prefix: str) -> int:
    left = left_df.select(
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
    )
    right = right_df.select(
        f"{right_prefix}_kafka_topic",
        f"{right_prefix}_kafka_partition",
        f"{right_prefix}_kafka_offset",
    )
    joined = left.join(
        right,
        (left.kafka_topic == right[f"{right_prefix}_kafka_topic"])
        & (left.kafka_partition == right[f"{right_prefix}_kafka_partition"])
        & (left.kafka_offset == right[f"{right_prefix}_kafka_offset"]),
        "inner",
    )
    return int(joined.count())


def inspect_silver_outputs(spark: Any, config: SparkSilverConfig) -> dict[str, Any]:
    from pyspark.sql import functions as spark_fn

    for path_name, relative_path in (
        ("bronze_input_path", config.bronze_input_path),
        ("silver_output_path", config.silver_output_path),
        ("duplicate_output_path", config.duplicate_output_path),
        ("quarantine_output_path", config.quarantine_output_path),
    ):
        if not _path_exists(relative_path):
            raise SparkSilverValidationError(f"{path_name} does not exist: {relative_path}")

    bronze_df = _read_parquet(spark, config.bronze_input_path)
    canonical_df = _read_parquet(spark, config.silver_output_path)
    duplicate_df = _read_parquet(spark, config.duplicate_output_path)
    quarantine_df = _read_parquet(spark, config.quarantine_output_path)

    bronze_row_count = int(bronze_df.count())
    canonical_count = int(canonical_df.count())
    duplicate_count = int(duplicate_df.count())
    quarantine_count = int(quarantine_df.count())
    valid_pre_dedup_count = canonical_count + duplicate_count
    duplicate_event_id_count = int(
        canonical_df.groupBy("event_id").count().where(spark_fn.col("count") > 1).count()
    )
    silver_distinct_event_id_count = int(canonical_df.select("event_id").distinct().count())
    duplicate_rank_max = _max_duplicate_rank(duplicate_df)
    return {
        "accounting_invariants_hold": accounting_invariants_hold(
            bronze_row_count=bronze_row_count,
            valid_pre_dedup_row_count=valid_pre_dedup_count,
            canonical_silver_row_count=canonical_count,
            duplicate_audit_row_count=duplicate_count,
            quarantine_row_count=quarantine_count,
        ),
        "bronze_row_count": bronze_row_count,
        "canonical_selection_sha256": _selection_digest(canonical_df),
        "canonical_silver_row_count": canonical_count,
        "duplicate_audit_row_count": duplicate_count,
        "duplicate_quarantine_coordinate_overlap_count": _coordinate_overlap_count(
            quarantine_df,
            duplicate_df,
            right_prefix="source",
        )
        if duplicate_count and quarantine_count
        else 0,
        "duplicate_rank_max": duplicate_rank_max,
        "event_time_max": _event_time_bound(canonical_df, "max"),
        "event_time_min": _event_time_bound(canonical_df, "min"),
        "machine_count": int(canonical_df.select("machine_code").distinct().count()),
        "product_quality_type_counts": _product_quality_counts(canonical_df),
        "quarantine_canonical_coordinate_overlap_count": _coordinate_overlap_count(
            quarantine_df,
            canonical_df,
            right_prefix="source",
        )
        if canonical_count and quarantine_count
        else 0,
        "quarantine_row_count": quarantine_count,
        "rejection_reason_counts": _reason_counts(quarantine_df),
        "silver_distinct_event_id_count": silver_distinct_event_id_count,
        "silver_duplicate_event_id_count": duplicate_event_id_count,
        "silver_field_types": _schema_types(canonical_df),
        "silver_null_event_id_count": int(canonical_df.where("event_id is null").count()),
        "silver_null_machine_code_count": int(canonical_df.where("machine_code is null").count()),
        "valid_pre_dedup_row_count": valid_pre_dedup_count,
        "null_source_kafka_topic_count": int(
            canonical_df.where("source_kafka_topic is null").count()
        ),
        "null_source_kafka_partition_count": int(
            canonical_df.where("source_kafka_partition is null").count()
        ),
        "null_source_kafka_offset_count": int(
            canonical_df.where("source_kafka_offset is null").count()
        ),
    }


def _synthetic_payload(
    *,
    event_id: str,
    machine_code: str,
    sequence_number: int,
    event_time: str,
    torque_nm: float = 40.0,
) -> str:
    event = {
        "schema_version": "1.0",
        "event_id": event_id,
        "machine_code": machine_code,
        "sequence_number": sequence_number,
        "event_time": event_time,
        "source": "synthetic_simulator",
        "product_quality_type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": torque_nm,
        "tool_wear_min": 10,
        "vibration_mm_s": 2.0,
        "pressure_bar": 6.0,
    }
    return json.dumps(
        {field: event[field] for field in TELEMETRY_FIELD_NAMES},
        separators=(",", ":"),
    )


def _payload_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_synthetic_bronze_rules_rows() -> list[dict[str, Any]]:
    base_time = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)  # noqa: UP017
    valid_event_id = "11111111-1111-4111-8111-111111111111"
    duplicate_payload = _synthetic_payload(
        event_id=valid_event_id,
        machine_code="MCH-0001",
        sequence_number=1,
        event_time="2026-02-01T00:00:00Z",
    )
    key_mismatch_payload = _synthetic_payload(
        event_id="22222222-2222-4222-8222-222222222222",
        machine_code="MCH-0002",
        sequence_number=1,
        event_time="2026-02-01T00:00:05Z",
    )
    invalid_sensor_payload = _synthetic_payload(
        event_id="33333333-3333-4333-8333-333333333333",
        machine_code="MCH-0003",
        sequence_number=1,
        event_time="2026-02-01T00:00:10Z",
        torque_nm=99.0,
    )
    malformed_payload = '{"schema_version":"1.0"'
    payloads = [
        ("MCH-0001", duplicate_payload),
        ("MCH-0001", duplicate_payload),
        ("MCH-0001", malformed_payload),
        ("MCH-0001", key_mismatch_payload),
        ("MCH-0003", invalid_sensor_payload),
    ]
    return [
        {
            "bronze_ingested_at": base_time,
            "kafka_key": key,
            "kafka_offset": index,
            "kafka_partition": 0,
            "kafka_timestamp": base_time,
            "kafka_topic": "industrial.telemetry.v1",
            "payload_sha256": _payload_sha256(payload),
            "raw_value": payload,
        }
        for index, (key, payload) in enumerate(payloads, start=1)
    ]


def build_synthetic_bronze_rules_dataframe(spark: Any) -> Any:
    from pyspark.sql import types as spark_types

    schema = spark_types.StructType(
        [
            spark_types.StructField("kafka_topic", spark_types.StringType(), False),
            spark_types.StructField("kafka_partition", spark_types.IntegerType(), False),
            spark_types.StructField("kafka_offset", spark_types.LongType(), False),
            spark_types.StructField("kafka_timestamp", spark_types.TimestampType(), False),
            spark_types.StructField("kafka_key", spark_types.StringType(), True),
            spark_types.StructField("raw_value", spark_types.StringType(), True),
            spark_types.StructField("bronze_ingested_at", spark_types.TimestampType(), False),
            spark_types.StructField("payload_sha256", spark_types.StringType(), True),
        ]
    )
    rows = [
        {
            "kafka_topic": row["kafka_topic"],
            "kafka_partition": row["kafka_partition"],
            "kafka_offset": row["kafka_offset"],
            "kafka_timestamp": row["kafka_timestamp"],
            "kafka_key": row["kafka_key"],
            "raw_value": row["raw_value"],
            "bronze_ingested_at": row["bronze_ingested_at"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in build_synthetic_bronze_rules_rows()
    ]
    return spark.createDataFrame(rows, schema=schema)


def run_synthetic_rules_check(spark: Any) -> dict[str, Any]:
    from pyspark.sql import functions as spark_fn

    bronze_df = build_synthetic_bronze_rules_dataframe(spark)
    result = transform_bronze_to_silver(bronze_df)
    quarantine_by_offset = {
        int(row["kafka_offset"]): list(row["rejection_reasons"])
        for row in result.quarantine_df.select("kafka_offset", "rejection_reasons").collect()
    }
    duplicate_event_ids = {
        str(row["event_id"]) for row in result.duplicate_df.select("event_id").collect()
    }
    duplicate_quarantine_count = int(
        result.quarantine_df.where(
            spark_fn.col("event_id") == "11111111-1111-4111-8111-111111111111"
        ).count()
    )
    return {
        "canonical_valid_count": int(result.canonical_df.count()),
        "invalid_sensor_quarantined": "invalid_torque_nm" in quarantine_by_offset.get(5, []),
        "kafka_key_mismatch_quarantined": (
            "kafka_key_machine_code_mismatch" in quarantine_by_offset.get(4, [])
        ),
        "malformed_json_quarantined": "malformed_json" in quarantine_by_offset.get(3, []),
        "quarantine_count": int(result.quarantine_df.count()),
        "rejection_reason_counts": _reason_counts(result.quarantine_df),
        "valid_duplicate_audited": (
            "11111111-1111-4111-8111-111111111111" in duplicate_event_ids
            and duplicate_quarantine_count == 0
        ),
        "valid_duplicate_count": int(result.duplicate_df.count()),
        "valid_pre_dedup_count": int(result.valid_pre_dedup_df.count()),
    }

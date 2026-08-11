"""PySpark Structured Streaming ingestion from Kafka into the Bronze layer."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_RELATIVE_PATH = Path("pipelines") / "streaming" / "spark_config.json"
SUMMARY_RELATIVE_PATH = Path("reports") / "streaming" / "spark_bronze_summary.json"
CONTAINER_WORKSPACE = PurePosixPath("/workspace")
EXPECTED_SPARK_VERSION = "4.0.4"
EXPECTED_SPARK_IMAGE = "apache/spark:4.0.4-scala2.13-java17-python3-ubuntu"
EXPECTED_MASTER = "local[2]"
EXPECTED_APPLICATION_NAME = "industrial-fleet-bronze-ingestion"
EXPECTED_KAFKA_CONNECTOR = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4"
EXPECTED_KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
EXPECTED_KAFKA_TOPIC = "industrial.telemetry.v1"
EXPECTED_BRONZE_OUTPUT_PATH = "data/bronze/telemetry"
EXPECTED_CHECKPOINT_PATH = "data/checkpoints/spark/bronze_telemetry"
EXPECTED_STARTING_OFFSETS = "earliest"
EXPECTED_OUTPUT_FORMAT = "parquet"
EXPECTED_OUTPUT_MODE = "append"
EXPECTED_TELEMETRY_SCHEMA_VERSION = "1.0"
EXPECTED_SHUFFLE_PARTITIONS = 3
EXPECTED_TIMEZONE = "UTC"
BRONZE_FIELD_NAMES = (
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "kafka_key",
    "raw_value",
    "bronze_ingested_at",
    "payload_sha256",
)
PRESERVED_KAFKA_METADATA_FIELDS = ("topic", "partition", "offset", "timestamp", "key", "value")


class SparkBronzeConfigError(ValueError):
    """Raised when Spark Bronze configuration is missing or incompatible."""


class SparkBronzeValidationError(RuntimeError):
    """Raised when Bronze inspection finds invalid persisted records."""


@dataclass(frozen=True)
class SparkBronzeConfig:
    """Static local Spark Structured Streaming configuration."""

    spark_version: str
    spark_docker_image: str
    master: str
    application_name: str
    kafka_connector: str
    kafka_bootstrap_servers: str
    kafka_topic: str
    bronze_output_path: str
    checkpoint_path: str
    starting_offsets: str
    fail_on_data_loss: bool
    output_format: str
    output_mode: str
    telemetry_schema_version: str
    spark_sql_shuffle_partitions: int
    spark_timezone: str


@dataclass(frozen=True)
class BronzeRecordCoordinate:
    """Kafka coordinate identity for a Bronze row."""

    kafka_topic: str
    kafka_partition: int
    kafka_offset: int


@dataclass(frozen=True)
class BronzeInspectionSummary:
    """Read-only diagnostic summary of the Bronze Parquet dataset."""

    total_row_count: int
    distinct_kafka_topics: tuple[str, ...]
    kafka_partition_counts: dict[str, int]
    kafka_offset_ranges: dict[str, dict[str, int]]
    null_kafka_key_count: int
    null_raw_value_count: int
    duplicate_kafka_coordinate_count: int
    distinct_payload_sha256_count: int | None
    expected_payload_match_count: int | None = None
    expected_payload_total: int | None = None
    expected_payloads_present: bool | None = None
    expected_key_mismatch_count: int | None = None
    expected_metadata_missing_count: int | None = None
    expected_payload_duplicate_rows: int | None = None
    expected_coordinate_total: int | None = None
    expected_coordinate_match_count: int | None = None
    expected_coordinate_payload_mismatch_count: int | None = None
    expected_coordinate_key_mismatch_count: int | None = None
    expected_coordinate_metadata_missing_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "distinct_kafka_topics": list(self.distinct_kafka_topics),
            "distinct_payload_sha256_count": self.distinct_payload_sha256_count,
            "duplicate_kafka_coordinate_count": self.duplicate_kafka_coordinate_count,
            "expected_coordinate_key_mismatch_count": self.expected_coordinate_key_mismatch_count,
            "expected_coordinate_match_count": self.expected_coordinate_match_count,
            "expected_coordinate_metadata_missing_count": (
                self.expected_coordinate_metadata_missing_count
            ),
            "expected_coordinate_payload_mismatch_count": (
                self.expected_coordinate_payload_mismatch_count
            ),
            "expected_coordinate_total": self.expected_coordinate_total,
            "expected_key_mismatch_count": self.expected_key_mismatch_count,
            "expected_metadata_missing_count": self.expected_metadata_missing_count,
            "expected_payload_duplicate_rows": self.expected_payload_duplicate_rows,
            "expected_payload_match_count": self.expected_payload_match_count,
            "expected_payload_total": self.expected_payload_total,
            "expected_payloads_present": self.expected_payloads_present,
            "kafka_offset_ranges": self.kafka_offset_ranges,
            "kafka_partition_counts": self.kafka_partition_counts,
            "null_kafka_key_count": self.null_kafka_key_count,
            "null_raw_value_count": self.null_raw_value_count,
            "total_row_count": self.total_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def load_spark_config(path: Path | None = None) -> SparkBronzeConfig:
    config_file = path or config_path()
    try:
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SparkBronzeConfigError(f"Spark config file not found: {config_file}") from exc
    except json.JSONDecodeError as exc:
        raise SparkBronzeConfigError(f"Spark config file is not valid JSON: {config_file}") from exc
    if not isinstance(raw_config, dict):
        raise SparkBronzeConfigError("Spark config must be a JSON object.")
    return parse_spark_config(raw_config)


def parse_spark_config(raw_config: Mapping[str, Any]) -> SparkBronzeConfig:
    required_keys = {
        "application_name",
        "bronze_output_path",
        "checkpoint_path",
        "fail_on_data_loss",
        "kafka_bootstrap_servers",
        "kafka_connector",
        "kafka_topic",
        "master",
        "output_format",
        "output_mode",
        "spark_docker_image",
        "spark_sql_shuffle_partitions",
        "spark_timezone",
        "spark_version",
        "starting_offsets",
        "telemetry_schema_version",
    }
    actual_keys = set(raw_config)
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if missing:
        raise SparkBronzeConfigError("Missing Spark config key(s): " + ", ".join(missing))
    if unknown:
        raise SparkBronzeConfigError("Unknown Spark config key(s): " + ", ".join(unknown))

    config = SparkBronzeConfig(
        spark_version=require_text(raw_config, "spark_version"),
        spark_docker_image=require_text(raw_config, "spark_docker_image"),
        master=require_text(raw_config, "master"),
        application_name=require_text(raw_config, "application_name"),
        kafka_connector=require_text(raw_config, "kafka_connector"),
        kafka_bootstrap_servers=require_text(raw_config, "kafka_bootstrap_servers"),
        kafka_topic=require_text(raw_config, "kafka_topic"),
        bronze_output_path=require_text(raw_config, "bronze_output_path"),
        checkpoint_path=require_text(raw_config, "checkpoint_path"),
        starting_offsets=require_text(raw_config, "starting_offsets"),
        fail_on_data_loss=require_bool(raw_config, "fail_on_data_loss"),
        output_format=require_text(raw_config, "output_format"),
        output_mode=require_text(raw_config, "output_mode"),
        telemetry_schema_version=require_text(raw_config, "telemetry_schema_version"),
        spark_sql_shuffle_partitions=require_int(raw_config, "spark_sql_shuffle_partitions"),
        spark_timezone=require_text(raw_config, "spark_timezone"),
    )
    validate_spark_config(config)
    return config


def require_text(raw_config: Mapping[str, Any], key: str) -> str:
    value = raw_config[key]
    if not isinstance(value, str) or not value.strip():
        raise SparkBronzeConfigError(f"Spark config key {key} must be a non-empty string.")
    return value


def require_bool(raw_config: Mapping[str, Any], key: str) -> bool:
    value = raw_config[key]
    if not isinstance(value, bool):
        raise SparkBronzeConfigError(f"Spark config key {key} must be true or false.")
    return value


def require_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SparkBronzeConfigError(f"Spark config key {key} must be an integer.")
    return value


def validate_relative_path(value: str, field_name: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise SparkBronzeConfigError(f"{field_name} must be a safe relative path.")


def validate_spark_config(config: SparkBronzeConfig) -> None:
    validate_relative_path(config.bronze_output_path, "bronze_output_path")
    validate_relative_path(config.checkpoint_path, "checkpoint_path")
    expected_values: tuple[tuple[str, str, str], ...] = (
        ("spark_version", config.spark_version, EXPECTED_SPARK_VERSION),
        ("spark_docker_image", config.spark_docker_image, EXPECTED_SPARK_IMAGE),
        ("master", config.master, EXPECTED_MASTER),
        ("application_name", config.application_name, EXPECTED_APPLICATION_NAME),
        ("kafka_connector", config.kafka_connector, EXPECTED_KAFKA_CONNECTOR),
        (
            "kafka_bootstrap_servers",
            config.kafka_bootstrap_servers,
            EXPECTED_KAFKA_BOOTSTRAP_SERVERS,
        ),
        ("kafka_topic", config.kafka_topic, EXPECTED_KAFKA_TOPIC),
        ("bronze_output_path", config.bronze_output_path, EXPECTED_BRONZE_OUTPUT_PATH),
        ("checkpoint_path", config.checkpoint_path, EXPECTED_CHECKPOINT_PATH),
        ("starting_offsets", config.starting_offsets, EXPECTED_STARTING_OFFSETS),
        ("output_format", config.output_format, EXPECTED_OUTPUT_FORMAT),
        ("output_mode", config.output_mode, EXPECTED_OUTPUT_MODE),
        (
            "telemetry_schema_version",
            config.telemetry_schema_version,
            EXPECTED_TELEMETRY_SCHEMA_VERSION,
        ),
        ("spark_timezone", config.spark_timezone, EXPECTED_TIMEZONE),
    )
    for field_name, actual, expected in expected_values:
        if actual != expected:
            raise SparkBronzeConfigError(f"{field_name} must be {expected}.")
    if config.fail_on_data_loss is not True:
        raise SparkBronzeConfigError("fail_on_data_loss must be true.")
    if config.spark_sql_shuffle_partitions != EXPECTED_SHUFFLE_PARTITIONS:
        raise SparkBronzeConfigError("spark_sql_shuffle_partitions must be 3.")


def container_path(relative_path: str) -> str:
    validate_relative_path(relative_path, "container path")
    return str(CONTAINER_WORKSPACE / PurePosixPath(relative_path.replace("\\", "/")))


def build_integration_summary(config: SparkBronzeConfig) -> dict[str, Any]:
    return {
        "bronze_format": config.output_format,
        "bronze_relative_path": config.bronze_output_path,
        "checkpoint_relative_path": config.checkpoint_path,
        "duplicate_event_policy": (
            "Bronze preserves every Kafka record; duplicate business event_id values "
            "are allowed and are handled in a later curated layer."
        ),
        "execution_mode": "available-now structured streaming on local[2]",
        "fail_on_data_loss": config.fail_on_data_loss,
        "kafka_bootstrap_servers": config.kafka_bootstrap_servers,
        "kafka_connector": config.kafka_connector,
        "kafka_topic": config.kafka_topic,
        "output_mode": config.output_mode,
        "preserved_kafka_metadata_fields": list(PRESERVED_KAFKA_METADATA_FIELDS),
        "schema_version": config.telemetry_schema_version,
        "spark_docker_image": config.spark_docker_image,
        "spark_version": config.spark_version,
        "starting_offsets": config.starting_offsets,
    }


def write_integration_summary(root: Path | None, config: SparkBronzeConfig) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(build_integration_summary(config), indent=2, sort_keys=False) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def create_spark_session(config: SparkBronzeConfig) -> Any:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(config.application_name)
        .master(config.master)
        .config("spark.sql.shuffle.partitions", str(config.spark_sql_shuffle_partitions))
        .config("spark.sql.session.timeZone", config.spark_timezone)
        .getOrCreate()
    )


def build_kafka_stream(spark: Any, config: SparkBronzeConfig) -> Any:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.kafka_bootstrap_servers)
        .option("subscribe", config.kafka_topic)
        .option("startingOffsets", config.starting_offsets)
        .option("failOnDataLoss", str(config.fail_on_data_loss).lower())
        .load()
    )


def build_bronze_projection(kafka_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    raw_value = spark_fn.col("value").cast("string")
    return kafka_df.select(
        spark_fn.col("topic").alias("kafka_topic"),
        spark_fn.col("partition").cast("int").alias("kafka_partition"),
        spark_fn.col("offset").cast("long").alias("kafka_offset"),
        spark_fn.col("timestamp").alias("kafka_timestamp"),
        spark_fn.col("key").cast("string").alias("kafka_key"),
        raw_value.alias("raw_value"),
        spark_fn.current_timestamp().alias("bronze_ingested_at"),
        spark_fn.sha2(raw_value, 256).alias("payload_sha256"),
    )


def build_bronze_sink(
    bronze_df: Any,
    config: SparkBronzeConfig,
    *,
    available_now: bool = True,
) -> Any:
    writer = (
        bronze_df.writeStream.format(config.output_format)
        .outputMode(config.output_mode)
        .option("checkpointLocation", container_path(config.checkpoint_path))
    )
    if available_now:
        writer = writer.trigger(availableNow=True)
    return writer


def start_bronze_query(
    spark: Any,
    config: SparkBronzeConfig,
    *,
    available_now: bool = True,
) -> Any:
    kafka_df = build_kafka_stream(spark, config)
    bronze_df = build_bronze_projection(kafka_df)
    writer = build_bronze_sink(bronze_df, config, available_now=available_now)
    return writer.start(container_path(config.bronze_output_path))


def duplicate_kafka_coordinate_count(records: Iterable[Mapping[str, Any]]) -> int:
    coordinates: list[BronzeRecordCoordinate] = []
    for record in records:
        coordinates.append(
            BronzeRecordCoordinate(
                kafka_topic=str(record["kafka_topic"]),
                kafka_partition=int(record["kafka_partition"]),
                kafka_offset=int(record["kafka_offset"]),
            )
        )
    counts = Counter(coordinates)
    return sum(count - 1 for count in counts.values() if count > 1)


def validate_kafka_coordinate_uniqueness(records: Iterable[Mapping[str, Any]]) -> None:
    duplicates = duplicate_kafka_coordinate_count(records)
    if duplicates:
        raise SparkBronzeValidationError(
            f"Bronze contains {duplicates} duplicate Kafka coordinate row(s)."
        )


def bronze_policy_allows_event_id_duplicates(records: Sequence[Mapping[str, Any]]) -> bool:
    return duplicate_kafka_coordinate_count(records) == 0


def _empty_summary(
    expected_payload_total: int | None = None,
    expected_coordinate_total: int | None = None,
) -> BronzeInspectionSummary:
    return BronzeInspectionSummary(
        total_row_count=0,
        distinct_kafka_topics=(),
        kafka_partition_counts={},
        kafka_offset_ranges={},
        null_kafka_key_count=0,
        null_raw_value_count=0,
        duplicate_kafka_coordinate_count=0,
        distinct_payload_sha256_count=0,
        expected_payload_match_count=0 if expected_payload_total is not None else None,
        expected_payload_total=expected_payload_total,
        expected_payloads_present=False if expected_payload_total is not None else None,
        expected_key_mismatch_count=0 if expected_payload_total is not None else None,
        expected_metadata_missing_count=0 if expected_payload_total is not None else None,
        expected_payload_duplicate_rows=0 if expected_payload_total is not None else None,
        expected_coordinate_total=expected_coordinate_total,
        expected_coordinate_match_count=0 if expected_coordinate_total is not None else None,
        expected_coordinate_payload_mismatch_count=(
            0 if expected_coordinate_total is not None else None
        ),
        expected_coordinate_key_mismatch_count=0 if expected_coordinate_total is not None else None,
        expected_coordinate_metadata_missing_count=(
            0 if expected_coordinate_total is not None else None
        ),
    )


def parse_expected_payload_machine_code(payload: str) -> str:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SparkBronzeValidationError("Expected payload must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise SparkBronzeValidationError("Expected payload must be a JSON object.")
    machine_code = parsed.get("machine_code")
    if not isinstance(machine_code, str) or not machine_code:
        raise SparkBronzeValidationError("Expected payload must contain machine_code.")
    return machine_code


def _expected_coordinate(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["kafka_topic"]),
        int(record["kafka_partition"]),
        int(record["kafka_offset"]),
    )


def _inspect_expected_payloads(
    bronze_df: Any,
    expected_payloads: Sequence[str],
) -> tuple[int, bool, int, int, int]:
    from pyspark.sql import functions as spark_fn

    unique_payloads = tuple(dict.fromkeys(expected_payloads))
    expected_keys = {
        payload: parse_expected_payload_machine_code(payload) for payload in unique_payloads
    }
    matched_rows = (
        bronze_df.where(spark_fn.col("raw_value").isin(list(unique_payloads)))
        .select(
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            "kafka_key",
            "raw_value",
        )
        .collect()
    )
    matched_payloads = {str(row["raw_value"]) for row in matched_rows}
    payload_counts = Counter(str(row["raw_value"]) for row in matched_rows)
    return (
        len(matched_payloads),
        len(matched_payloads) == len(unique_payloads),
        sum(1 for row in matched_rows if row["kafka_key"] != expected_keys[str(row["raw_value"])]),
        sum(
            1
            for row in matched_rows
            if row["kafka_topic"] is None
            or row["kafka_partition"] is None
            or row["kafka_offset"] is None
            or row["kafka_timestamp"] is None
        ),
        sum(count - 1 for count in payload_counts.values() if count > 1),
    )


def _inspect_expected_records(
    bronze_df: Any,
    expected_records: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int]:
    from pyspark.sql import functions as spark_fn

    expected_by_coordinate = {_expected_coordinate(record): record for record in expected_records}
    condition = None
    for topic, partition, offset in expected_by_coordinate:
        next_condition = (
            (spark_fn.col("kafka_topic") == topic)
            & (spark_fn.col("kafka_partition") == partition)
            & (spark_fn.col("kafka_offset") == offset)
        )
        condition = next_condition if condition is None else condition | next_condition

    expected_rows = [] if condition is None else bronze_df.where(condition).collect()
    matched_coordinates = {_expected_coordinate(row.asDict()) for row in expected_rows}
    payload_mismatches = 0
    key_mismatches = 0
    metadata_missing = 0
    for row in expected_rows:
        row_data = row.asDict()
        expected = expected_by_coordinate[_expected_coordinate(row_data)]
        if str(row_data["raw_value"]) != str(expected["raw_value"]):
            payload_mismatches += 1
        if str(row_data["kafka_key"]) != str(expected["kafka_key"]):
            key_mismatches += 1
        if row_data["kafka_timestamp"] is None:
            metadata_missing += 1
    return len(matched_coordinates), payload_mismatches, key_mismatches, metadata_missing


def inspect_bronze_dataset(
    spark: Any,
    config: SparkBronzeConfig,
    *,
    expected_payloads: Sequence[str] | None = None,
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> BronzeInspectionSummary:
    from pyspark.sql import functions as spark_fn

    expected_payload_total = None if expected_payloads is None else len(set(expected_payloads))
    expected_coordinate_total = None if expected_records is None else len(expected_records)
    output_path = Path(container_path(config.bronze_output_path))
    if not output_path.exists():
        return _empty_summary(expected_payload_total, expected_coordinate_total)

    try:
        bronze_df = spark.read.parquet(str(output_path))
    except Exception:
        return _empty_summary(expected_payload_total, expected_coordinate_total)

    total_row_count = int(bronze_df.count())
    if total_row_count == 0:
        return _empty_summary(expected_payload_total, expected_coordinate_total)

    topics = tuple(
        row["kafka_topic"]
        for row in bronze_df.select("kafka_topic").distinct().orderBy("kafka_topic").collect()
    )
    partition_counts = {
        str(row["kafka_partition"]): int(row["count"])
        for row in bronze_df.groupBy("kafka_partition").count().orderBy("kafka_partition").collect()
    }
    offset_ranges = {
        str(row["kafka_partition"]): {
            "min_offset": int(row["min_offset"]),
            "max_offset": int(row["max_offset"]),
        }
        for row in bronze_df.groupBy("kafka_partition")
        .agg(
            spark_fn.min("kafka_offset").alias("min_offset"),
            spark_fn.max("kafka_offset").alias("max_offset"),
        )
        .orderBy("kafka_partition")
        .collect()
    }
    duplicate_coordinate_count = int(
        bronze_df.groupBy("kafka_topic", "kafka_partition", "kafka_offset")
        .count()
        .where(spark_fn.col("count") > 1)
        .count()
    )
    distinct_sha_count = None
    if "payload_sha256" in bronze_df.columns:
        distinct_sha_count = int(
            bronze_df.where(spark_fn.col("payload_sha256").isNotNull())
            .select("payload_sha256")
            .distinct()
            .count()
        )

    expected_match_count = None
    expected_payloads_present = None
    expected_key_mismatch_count = None
    expected_metadata_missing_count = None
    expected_payload_duplicate_rows = None
    if expected_payloads is not None:
        (
            expected_match_count,
            expected_payloads_present,
            expected_key_mismatch_count,
            expected_metadata_missing_count,
            expected_payload_duplicate_rows,
        ) = _inspect_expected_payloads(bronze_df, expected_payloads)

    expected_coordinate_match_count = None
    expected_coordinate_payload_mismatch_count = None
    expected_coordinate_key_mismatch_count = None
    expected_coordinate_metadata_missing_count = None
    if expected_records is not None:
        (
            expected_coordinate_match_count,
            expected_coordinate_payload_mismatch_count,
            expected_coordinate_key_mismatch_count,
            expected_coordinate_metadata_missing_count,
        ) = _inspect_expected_records(bronze_df, expected_records)

    return BronzeInspectionSummary(
        total_row_count=total_row_count,
        distinct_kafka_topics=topics,
        kafka_partition_counts=partition_counts,
        kafka_offset_ranges=offset_ranges,
        null_kafka_key_count=int(bronze_df.where(spark_fn.col("kafka_key").isNull()).count()),
        null_raw_value_count=int(bronze_df.where(spark_fn.col("raw_value").isNull()).count()),
        duplicate_kafka_coordinate_count=duplicate_coordinate_count,
        distinct_payload_sha256_count=distinct_sha_count,
        expected_payload_match_count=expected_match_count,
        expected_payload_total=expected_payload_total,
        expected_payloads_present=expected_payloads_present,
        expected_key_mismatch_count=expected_key_mismatch_count,
        expected_metadata_missing_count=expected_metadata_missing_count,
        expected_payload_duplicate_rows=expected_payload_duplicate_rows,
        expected_coordinate_total=expected_coordinate_total,
        expected_coordinate_match_count=expected_coordinate_match_count,
        expected_coordinate_payload_mismatch_count=expected_coordinate_payload_mismatch_count,
        expected_coordinate_key_mismatch_count=expected_coordinate_key_mismatch_count,
        expected_coordinate_metadata_missing_count=expected_coordinate_metadata_missing_count,
    )

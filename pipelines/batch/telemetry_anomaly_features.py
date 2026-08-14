"""Spark extraction of operational anomaly features from canonical Silver telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CONTAINER_WORKSPACE = PurePosixPath("/workspace")
EXPECTED_APPLICATION_NAME = "industrial-fleet-telemetry-anomaly-features"
EXPECTED_SPARK_VERSION = "4.0.4"
EXPECTED_MASTER = "local[2]"
EXPECTED_TIMEZONE = "UTC"
EXPECTED_SHUFFLE_PARTITIONS = 3
EXPECTED_OUTPUT_FORMAT = "json"
EXPECTED_SOURCE_PATH = "data/silver/telemetry"
EXPECTED_OUTPUT_PATH = "data/model_input/anomaly/telemetry"
EXPECTED_FEATURES = ("vibration_mm_s", "pressure_bar")
REQUIRED_SILVER_FIELD_TYPES = {
    "event_id": "string",
    "event_time": "timestamp",
    "machine_code": "string",
    "payload_sha256": "string",
    "pressure_bar": "double",
    "source_kafka_key": "string",
    "source_kafka_offset": "bigint",
    "source_kafka_partition": "int",
    "source_kafka_timestamp": "timestamp",
    "source_kafka_topic": "string",
    "vibration_mm_s": "double",
}


class TelemetryAnomalyFeatureExtractionError(ValueError):
    """Raised when anomaly feature extraction configuration or data is invalid."""


@dataclass(frozen=True)
class TelemetryAnomalyFeatureConfig:
    """Static Spark feature-extraction configuration."""

    application_name: str
    spark_version: str
    master: str
    source: str
    output: str
    output_format: str
    timezone: str
    shuffle_partitions: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class TelemetryAnomalyFeatureCounts:
    """Logical counts from an anomaly feature-export rebuild."""

    silver_row_count: int
    silver_distinct_event_id_count: int
    feature_row_count: int
    feature_distinct_event_id_count: int
    distinct_machine_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "distinct_machine_count": self.distinct_machine_count,
            "feature_distinct_event_id_count": self.feature_distinct_event_id_count,
            "feature_row_count": self.feature_row_count,
            "silver_distinct_event_id_count": self.silver_distinct_event_id_count,
            "silver_row_count": self.silver_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_feature_config() -> TelemetryAnomalyFeatureConfig:
    return TelemetryAnomalyFeatureConfig(
        application_name=EXPECTED_APPLICATION_NAME,
        spark_version=EXPECTED_SPARK_VERSION,
        master=EXPECTED_MASTER,
        source=EXPECTED_SOURCE_PATH,
        output=EXPECTED_OUTPUT_PATH,
        output_format=EXPECTED_OUTPUT_FORMAT,
        timezone=EXPECTED_TIMEZONE,
        shuffle_partitions=EXPECTED_SHUFFLE_PARTITIONS,
        features=EXPECTED_FEATURES,
    )


def container_path(relative_path: str) -> str:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise TelemetryAnomalyFeatureExtractionError(f"Unsafe relative path: {relative_path}")
    return str(CONTAINER_WORKSPACE / path)


def validate_feature_config(config: TelemetryAnomalyFeatureConfig) -> None:
    if config.application_name != EXPECTED_APPLICATION_NAME:
        raise TelemetryAnomalyFeatureExtractionError("Unexpected application name.")
    if config.spark_version != EXPECTED_SPARK_VERSION:
        raise TelemetryAnomalyFeatureExtractionError("Unexpected Spark version.")
    if config.master != EXPECTED_MASTER:
        raise TelemetryAnomalyFeatureExtractionError("Unexpected Spark master.")
    if config.source != EXPECTED_SOURCE_PATH:
        raise TelemetryAnomalyFeatureExtractionError("Source must be canonical Silver telemetry.")
    if config.output != EXPECTED_OUTPUT_PATH:
        raise TelemetryAnomalyFeatureExtractionError("Unexpected anomaly feature output path.")
    if config.output_format != EXPECTED_OUTPUT_FORMAT:
        raise TelemetryAnomalyFeatureExtractionError("Feature output format must be json.")
    if config.timezone != EXPECTED_TIMEZONE:
        raise TelemetryAnomalyFeatureExtractionError("Timezone must be UTC.")
    if config.shuffle_partitions != EXPECTED_SHUFFLE_PARTITIONS:
        raise TelemetryAnomalyFeatureExtractionError("shuffle_partitions must be 3.")
    if config.features != EXPECTED_FEATURES:
        raise TelemetryAnomalyFeatureExtractionError(
            "Anomaly feature extraction must use exactly vibration_mm_s and pressure_bar."
        )


def create_spark_session(config: TelemetryAnomalyFeatureConfig) -> Any:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(config.application_name)
        .master(config.master)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .config("spark.sql.session.timeZone", config.timezone)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def validate_silver_schema(silver_df: Any) -> None:
    actual_types = dict(silver_df.dtypes)
    missing = sorted(set(REQUIRED_SILVER_FIELD_TYPES) - set(actual_types))
    if missing:
        raise TelemetryAnomalyFeatureExtractionError(
            "Silver is missing anomaly field(s): " + ", ".join(missing)
        )
    mismatches = {
        field: (EXPECTED, actual_types[field])
        for field, EXPECTED in REQUIRED_SILVER_FIELD_TYPES.items()
        if actual_types[field] != EXPECTED
    }
    if mismatches:
        details = ", ".join(
            f"{field}: expected {expected}, found {actual}"
            for field, (expected, actual) in sorted(mismatches.items())
        )
        raise TelemetryAnomalyFeatureExtractionError("Silver type mismatch: " + details)


def read_silver_snapshot(spark: Any, config: TelemetryAnomalyFeatureConfig) -> Any:
    validate_feature_config(config)
    source_path = Path(container_path(config.source))
    if not source_path.exists():
        raise TelemetryAnomalyFeatureExtractionError(
            f"Canonical Silver does not exist: {config.source}"
        )
    silver_df = spark.read.parquet(str(source_path))
    validate_silver_schema(silver_df)
    return silver_df


def build_feature_records(silver_df: Any, config: TelemetryAnomalyFeatureConfig) -> Any:
    from pyspark.sql import functions as spark_fn

    validate_silver_schema(silver_df)
    return silver_df.select(
        "event_id",
        "machine_code",
        spark_fn.col("event_time").cast("string").alias("event_time"),
        spark_fn.col("vibration_mm_s").cast("double").alias("vibration_mm_s"),
        spark_fn.col("pressure_bar").cast("double").alias("pressure_bar"),
        "source_kafka_topic",
        "source_kafka_partition",
        "source_kafka_offset",
        spark_fn.col("source_kafka_timestamp").cast("string").alias("source_kafka_timestamp"),
        "source_kafka_key",
        "payload_sha256",
    ).orderBy(
        "event_time",
        "machine_code",
        "event_id",
        "source_kafka_timestamp",
        "source_kafka_topic",
        "source_kafka_partition",
        "source_kafka_offset",
        "source_kafka_key",
        "payload_sha256",
    )


def _distinct_count(df: Any, column_name: str) -> int:
    return int(df.select(column_name).distinct().count())


def validate_feature_records(silver_df: Any, feature_df: Any) -> TelemetryAnomalyFeatureCounts:
    silver_count = int(silver_df.count())
    feature_count = int(feature_df.count())
    silver_distinct = _distinct_count(silver_df, "event_id")
    feature_distinct = _distinct_count(feature_df, "event_id")
    if feature_count != silver_count:
        raise TelemetryAnomalyFeatureExtractionError(
            "Anomaly feature row count does not match Silver row count."
        )
    if silver_distinct != silver_count:
        raise TelemetryAnomalyFeatureExtractionError(
            "Canonical Silver event_id values are not unique."
        )
    if feature_distinct != feature_count:
        raise TelemetryAnomalyFeatureExtractionError(
            "Feature export event_id values are not unique."
        )
    return TelemetryAnomalyFeatureCounts(
        silver_row_count=silver_count,
        silver_distinct_event_id_count=silver_distinct,
        feature_row_count=feature_count,
        feature_distinct_event_id_count=feature_distinct,
        distinct_machine_count=_distinct_count(feature_df, "machine_code"),
    )


def write_feature_output(feature_df: Any, config: TelemetryAnomalyFeatureConfig) -> None:
    feature_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.output)
    )


def rebuild_feature_output(
    spark: Any,
    config: TelemetryAnomalyFeatureConfig | None = None,
) -> TelemetryAnomalyFeatureCounts:
    active_config = config or default_feature_config()
    silver_df = read_silver_snapshot(spark, active_config)
    feature_df = build_feature_records(silver_df, active_config)
    counts = validate_feature_records(silver_df, feature_df)
    write_feature_output(feature_df, active_config)
    return counts

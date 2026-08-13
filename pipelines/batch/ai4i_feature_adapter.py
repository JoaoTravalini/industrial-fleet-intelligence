"""Spark adapter from canonical Silver telemetry to AI4I model-input records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_RELATIVE_PATH = Path("pipelines") / "batch" / "ai4i_feature_adapter_config.json"
SUMMARY_RELATIVE_PATH = Path("reports") / "inference" / "ai4i_telemetry_bridge_summary.json"
CONTAINER_WORKSPACE = PurePosixPath("/workspace")

EXPECTED_ADAPTER_VERSION = "1.0"
EXPECTED_APPLICATION_NAME = "industrial-fleet-ai4i-feature-adapter"
EXPECTED_SPARK_VERSION = "4.0.4"
EXPECTED_MASTER = "local[2]"
EXPECTED_SOURCE_PATH = "data/silver/telemetry"
EXPECTED_OUTPUT_PATH = "data/model_input/ai4i/telemetry"
EXPECTED_OUTPUT_FORMAT = "json"
EXPECTED_TIMEZONE = "UTC"
EXPECTED_SHUFFLE_PARTITIONS = 3
EXPECTED_MODEL_INPUT_FEATURES = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
EXPECTED_MAPPING = {
    "product_quality_type": "Type",
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}
EXPECTED_EXCLUDED_CURRENT_MODEL_FIELDS = ("vibration_mm_s", "pressure_bar")
LINEAGE_FIELDS = (
    "source_kafka_topic",
    "source_kafka_partition",
    "source_kafka_offset",
    "source_kafka_timestamp",
    "source_kafka_key",
    "payload_sha256",
)
REQUIRED_SILVER_FIELD_TYPES = {
    "air_temperature_k": "double",
    "event_id": "string",
    "event_time": "timestamp",
    "machine_code": "string",
    "payload_sha256": "string",
    "pressure_bar": "double",
    "process_temperature_k": "double",
    "product_quality_type": "string",
    "rotational_speed_rpm": "int",
    "source_kafka_key": "string",
    "source_kafka_offset": "bigint",
    "source_kafka_partition": "int",
    "source_kafka_timestamp": "timestamp",
    "source_kafka_topic": "string",
    "tool_wear_min": "int",
    "torque_nm": "double",
    "vibration_mm_s": "double",
}


class AI4IFeatureAdapterConfigError(ValueError):
    """Raised when the AI4I telemetry adapter configuration is invalid."""


class AI4IFeatureAdapterValidationError(RuntimeError):
    """Raised when Silver or adapted model-input records fail validation."""


@dataclass(frozen=True)
class AI4IFeatureAdapterConfig:
    """Static Spark feature-adapter configuration."""

    adapter_version: str
    application_name: str
    spark_version: str
    master: str
    source: str
    output: str
    output_format: str
    timezone: str
    shuffle_partitions: int
    model_input_features: tuple[str, ...]
    mapping: dict[str, str]
    excluded_current_model_fields: tuple[str, ...]


@dataclass(frozen=True)
class AI4IAdapterCounts:
    """Logical counts from an AI4I feature-adapter rebuild."""

    silver_row_count: int
    silver_distinct_event_id_count: int
    adapter_row_count: int
    adapter_distinct_event_id_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "adapter_distinct_event_id_count": self.adapter_distinct_event_id_count,
            "adapter_row_count": self.adapter_row_count,
            "silver_distinct_event_id_count": self.silver_distinct_event_id_count,
            "silver_row_count": self.silver_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def require_text(raw_config: Mapping[str, Any], key: str) -> str:
    value = raw_config[key]
    if not isinstance(value, str) or not value.strip():
        raise AI4IFeatureAdapterConfigError(f"Adapter config key {key} must be text.")
    return value


def require_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AI4IFeatureAdapterConfigError(f"Adapter config key {key} must be an integer.")
    return value


def require_string_list(raw_config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw_config[key]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise AI4IFeatureAdapterConfigError(
            f"Adapter config key {key} must be a non-empty string list."
        )
    return tuple(value)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, str]:
    value = raw_config[key]
    if not isinstance(value, dict) or any(
        not isinstance(source, str)
        or not isinstance(target, str)
        or not source.strip()
        or not target.strip()
        for source, target in value.items()
    ):
        raise AI4IFeatureAdapterConfigError(
            f"Adapter config key {key} must be a string-to-string object."
        )
    return dict(value)


def validate_relative_path(value: str, field_name: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise AI4IFeatureAdapterConfigError(f"{field_name} must be a safe relative path.")


def validate_adapter_config(config: AI4IFeatureAdapterConfig) -> None:
    for field_name, value in (("source", config.source), ("output", config.output)):
        validate_relative_path(value, field_name)

    expected_text_values = (
        ("adapter_version", config.adapter_version, EXPECTED_ADAPTER_VERSION),
        ("application_name", config.application_name, EXPECTED_APPLICATION_NAME),
        ("spark_version", config.spark_version, EXPECTED_SPARK_VERSION),
        ("master", config.master, EXPECTED_MASTER),
        ("source", config.source, EXPECTED_SOURCE_PATH),
        ("output", config.output, EXPECTED_OUTPUT_PATH),
        ("output_format", config.output_format, EXPECTED_OUTPUT_FORMAT),
        ("timezone", config.timezone, EXPECTED_TIMEZONE),
    )
    for field_name, actual, expected in expected_text_values:
        if actual != expected:
            raise AI4IFeatureAdapterConfigError(f"{field_name} must be {expected}.")
    if config.shuffle_partitions != EXPECTED_SHUFFLE_PARTITIONS:
        raise AI4IFeatureAdapterConfigError("shuffle_partitions must be 3.")
    if config.model_input_features != EXPECTED_MODEL_INPUT_FEATURES:
        raise AI4IFeatureAdapterConfigError("model_input_features do not match the frozen model.")
    if config.mapping != EXPECTED_MAPPING:
        raise AI4IFeatureAdapterConfigError("mapping does not match the approved adapter policy.")
    if config.excluded_current_model_fields != EXPECTED_EXCLUDED_CURRENT_MODEL_FIELDS:
        raise AI4IFeatureAdapterConfigError(
            "excluded_current_model_fields must be vibration_mm_s and pressure_bar."
        )


def parse_adapter_config(raw_config: Mapping[str, Any]) -> AI4IFeatureAdapterConfig:
    required_keys = {
        "adapter_version",
        "application_name",
        "excluded_current_model_fields",
        "mapping",
        "master",
        "model_input_features",
        "output",
        "output_format",
        "shuffle_partitions",
        "source",
        "spark_version",
        "timezone",
    }
    actual_keys = set(raw_config)
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if missing:
        raise AI4IFeatureAdapterConfigError("Missing adapter config key(s): " + ", ".join(missing))
    if unknown:
        raise AI4IFeatureAdapterConfigError("Unknown adapter config key(s): " + ", ".join(unknown))

    config = AI4IFeatureAdapterConfig(
        adapter_version=require_text(raw_config, "adapter_version"),
        application_name=require_text(raw_config, "application_name"),
        spark_version=require_text(raw_config, "spark_version"),
        master=require_text(raw_config, "master"),
        source=require_text(raw_config, "source"),
        output=require_text(raw_config, "output"),
        output_format=require_text(raw_config, "output_format"),
        timezone=require_text(raw_config, "timezone"),
        shuffle_partitions=require_int(raw_config, "shuffle_partitions"),
        model_input_features=require_string_list(raw_config, "model_input_features"),
        mapping=require_mapping(raw_config, "mapping"),
        excluded_current_model_fields=require_string_list(
            raw_config,
            "excluded_current_model_fields",
        ),
    )
    validate_adapter_config(config)
    return config


def load_adapter_config(path: Path | None = None) -> AI4IFeatureAdapterConfig:
    config_file = path or config_path()
    try:
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AI4IFeatureAdapterConfigError(
            f"Adapter config file not found: {config_file}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AI4IFeatureAdapterConfigError(
            f"Adapter config file is not valid JSON: {config_file}"
        ) from exc
    if not isinstance(raw_config, dict):
        raise AI4IFeatureAdapterConfigError("Adapter config must be a JSON object.")
    return parse_adapter_config(raw_config)


def container_path(relative_path: str) -> str:
    validate_relative_path(relative_path, "container path")
    return str(CONTAINER_WORKSPACE / PurePosixPath(relative_path.replace("\\", "/")))


def adapt_silver_event_to_model_input(
    silver_event: Mapping[str, Any],
    config: AI4IFeatureAdapterConfig | None = None,
) -> dict[str, Any]:
    """Map one canonical Silver event into the frozen AI4I feature contract."""

    adapter_config = config or AI4IFeatureAdapterConfig(
        adapter_version=EXPECTED_ADAPTER_VERSION,
        application_name=EXPECTED_APPLICATION_NAME,
        spark_version=EXPECTED_SPARK_VERSION,
        master=EXPECTED_MASTER,
        source=EXPECTED_SOURCE_PATH,
        output=EXPECTED_OUTPUT_PATH,
        output_format=EXPECTED_OUTPUT_FORMAT,
        timezone=EXPECTED_TIMEZONE,
        shuffle_partitions=EXPECTED_SHUFFLE_PARTITIONS,
        model_input_features=EXPECTED_MODEL_INPUT_FEATURES,
        mapping=dict(EXPECTED_MAPPING),
        excluded_current_model_fields=EXPECTED_EXCLUDED_CURRENT_MODEL_FIELDS,
    )
    validate_adapter_config(adapter_config)
    source_by_target = {target: source for source, target in adapter_config.mapping.items()}
    missing_targets = sorted(set(adapter_config.model_input_features) - set(source_by_target))
    if missing_targets:
        raise AI4IFeatureAdapterValidationError(
            "Adapter mapping is missing target feature(s): " + ", ".join(missing_targets)
        )

    model_input: dict[str, Any] = {}
    for target_field in adapter_config.model_input_features:
        source_field = source_by_target[target_field]
        if source_field not in silver_event:
            raise AI4IFeatureAdapterValidationError(
                f"Silver event is missing source field {source_field}."
            )
        model_input[target_field] = silver_event[source_field]
    return model_input


def create_spark_session(config: AI4IFeatureAdapterConfig) -> Any:
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
    actual_types = {field.name: field.dataType.simpleString() for field in silver_df.schema.fields}
    missing = sorted(set(REQUIRED_SILVER_FIELD_TYPES) - set(actual_types))
    mismatched = sorted(
        field
        for field, expected_type in REQUIRED_SILVER_FIELD_TYPES.items()
        if field in actual_types and actual_types[field] != expected_type
    )
    if missing:
        raise AI4IFeatureAdapterValidationError(
            "Canonical Silver is missing required column(s): " + ", ".join(missing)
        )
    if mismatched:
        details = ", ".join(
            f"{field} expected {REQUIRED_SILVER_FIELD_TYPES[field]} got {actual_types[field]}"
            for field in mismatched
        )
        raise AI4IFeatureAdapterValidationError(
            "Canonical Silver has incompatible type(s): " + details
        )


def read_silver_snapshot(spark: Any, config: AI4IFeatureAdapterConfig) -> Any:
    source_path = Path(container_path(config.source))
    if not source_path.exists():
        raise AI4IFeatureAdapterValidationError(f"Canonical Silver does not exist: {config.source}")
    silver_df = spark.read.parquet(str(source_path))
    validate_silver_schema(silver_df)
    return silver_df


def build_adapter_records(silver_df: Any, config: AI4IFeatureAdapterConfig) -> Any:
    from pyspark.sql import functions as spark_fn

    validate_silver_schema(silver_df)
    model_input = spark_fn.struct(
        spark_fn.col("product_quality_type").alias("Type"),
        spark_fn.col("air_temperature_k").cast("double").alias("Air temperature [K]"),
        spark_fn.col("process_temperature_k").cast("double").alias("Process temperature [K]"),
        spark_fn.col("rotational_speed_rpm").cast("double").alias("Rotational speed [rpm]"),
        spark_fn.col("torque_nm").cast("double").alias("Torque [Nm]"),
        spark_fn.col("tool_wear_min").cast("double").alias("Tool wear [min]"),
    )
    source_lineage = spark_fn.struct(
        spark_fn.col("source_kafka_topic").alias("source_kafka_topic"),
        spark_fn.col("source_kafka_partition").alias("source_kafka_partition"),
        spark_fn.col("source_kafka_offset").alias("source_kafka_offset"),
        spark_fn.col("source_kafka_timestamp").cast("string").alias("source_kafka_timestamp"),
        spark_fn.col("source_kafka_key").alias("source_kafka_key"),
        spark_fn.col("payload_sha256").alias("payload_sha256"),
    )
    return silver_df.select(
        spark_fn.lit(config.adapter_version).alias("adapter_version"),
        "event_id",
        "machine_code",
        spark_fn.col("event_time").cast("string").alias("event_time"),
        model_input.alias("model_input"),
        source_lineage.alias("source_lineage"),
    ).orderBy(
        "event_time",
        "machine_code",
        "event_id",
        spark_fn.col("source_lineage.source_kafka_timestamp"),
        spark_fn.col("source_lineage.source_kafka_topic"),
        spark_fn.col("source_lineage.source_kafka_partition"),
        spark_fn.col("source_lineage.source_kafka_offset"),
    )


def _distinct_count(df: Any, column_name: str) -> int:
    return int(df.select(column_name).distinct().count())


def _model_input_field_names(adapter_df: Any) -> tuple[str, ...]:
    model_input_field = adapter_df.schema["model_input"]
    return tuple(field.name for field in model_input_field.dataType.fields)


def _model_input_type_mismatch_count(adapter_df: Any, silver_df: Any) -> int:
    from pyspark.sql import functions as spark_fn

    expected = silver_df.select("event_id", "product_quality_type")
    joined = adapter_df.select(
        "event_id",
        spark_fn.col("model_input.Type").alias("_adapter_type"),
    ).join(expected, on="event_id", how="inner")
    return int(
        joined.where(spark_fn.col("_adapter_type") != spark_fn.col("product_quality_type")).count()
    )


def _model_input_excluded_field_count(adapter_df: Any, config: AI4IFeatureAdapterConfig) -> int:
    field_names = set(_model_input_field_names(adapter_df))
    return len(field_names & set(config.excluded_current_model_fields))


def validate_adapter_records(
    silver_df: Any,
    adapter_df: Any,
    config: AI4IFeatureAdapterConfig,
) -> AI4IAdapterCounts:
    silver_row_count = int(silver_df.count())
    adapter_row_count = int(adapter_df.count())
    silver_distinct_event_id_count = _distinct_count(silver_df, "event_id")
    adapter_distinct_event_id_count = _distinct_count(adapter_df, "event_id")
    if adapter_row_count != silver_row_count:
        raise AI4IFeatureAdapterValidationError(
            "Adapter row count does not match Silver row count."
        )
    if silver_distinct_event_id_count != silver_row_count:
        raise AI4IFeatureAdapterValidationError("Canonical Silver event_id values are not unique.")
    if adapter_distinct_event_id_count != adapter_row_count:
        raise AI4IFeatureAdapterValidationError("Adapter event_id values are not unique.")
    if _model_input_field_names(adapter_df) != config.model_input_features:
        raise AI4IFeatureAdapterValidationError("Adapter model_input feature order is not exact.")
    if _model_input_excluded_field_count(adapter_df, config) != 0:
        raise AI4IFeatureAdapterValidationError("Excluded telemetry field entered model_input.")
    if _model_input_type_mismatch_count(adapter_df, silver_df) != 0:
        raise AI4IFeatureAdapterValidationError("product_quality_type was not mapped to Type.")
    return AI4IAdapterCounts(
        silver_row_count=silver_row_count,
        silver_distinct_event_id_count=silver_distinct_event_id_count,
        adapter_row_count=adapter_row_count,
        adapter_distinct_event_id_count=adapter_distinct_event_id_count,
    )


def write_adapter_output(adapter_df: Any, config: AI4IFeatureAdapterConfig) -> None:
    adapter_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.output)
    )


def rebuild_adapter_output(spark: Any, config: AI4IFeatureAdapterConfig) -> AI4IAdapterCounts:
    silver_df = read_silver_snapshot(spark, config)
    adapter_df = build_adapter_records(silver_df, config)
    counts = validate_adapter_records(silver_df, adapter_df, config)
    write_adapter_output(adapter_df, config)
    return counts


def build_static_bridge_summary(
    config: AI4IFeatureAdapterConfig,
    *,
    model_name: str,
    model_version: str,
    frozen_threshold: float,
    final_config_hash: str,
) -> dict[str, Any]:
    return {
        "adapter_version": config.adapter_version,
        "source_layer": "canonical Silver telemetry",
        "source_path": config.source,
        "adapter_output_path": config.output,
        "prediction_output_path": "data/predictions/ai4i/telemetry_predictions.jsonl",
        "model_input_features": list(config.model_input_features),
        "feature_mapping": dict(config.mapping),
        "excluded_current_model_fields": list(config.excluded_current_model_fields),
        "model_name": model_name,
        "model_version": model_version,
        "frozen_threshold": float(frozen_threshold),
        "final_config_hash": final_config_hash,
        "inference_mode": "deterministic local batch inference from adapted Silver events",
        "runtime_counts": "intentionally excluded from tracked summary",
        "ground_truth_policy": "No labels or performance evaluation are involved.",
    }


def write_static_bridge_summary(
    root: Path | None,
    config: AI4IFeatureAdapterConfig,
    *,
    model_name: str,
    model_version: str,
    frozen_threshold: float,
    final_config_hash: str,
) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_static_bridge_summary(
                config,
                model_name=model_name,
                model_version=model_version,
                frozen_threshold=frozen_threshold,
                final_config_hash=final_config_hash,
            ),
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

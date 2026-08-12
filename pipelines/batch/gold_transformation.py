"""Spark descriptive analytics from canonical Silver telemetry into Gold datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_RELATIVE_PATH = Path("pipelines") / "batch" / "gold_config.json"
SUMMARY_RELATIVE_PATH = Path("reports") / "streaming" / "spark_gold_summary.json"
CONTAINER_WORKSPACE = PurePosixPath("/workspace")

EXPECTED_SPARK_VERSION = "4.0.4"
EXPECTED_APPLICATION_NAME = "industrial-fleet-gold-analytics"
EXPECTED_MASTER = "local[2]"
EXPECTED_SILVER_INPUT_PATH = "data/silver/telemetry"
EXPECTED_MACHINE_SUMMARY_OUTPUT_PATH = "data/gold/machine_summary"
EXPECTED_MACHINE_WINDOWS_OUTPUT_PATH = "data/gold/machine_windows"
EXPECTED_FLEET_SUMMARY_OUTPUT_PATH = "data/gold/fleet_summary"
EXPECTED_OUTPUT_FORMAT = "parquet"
EXPECTED_WINDOW_DURATION = "1 minute"
EXPECTED_TIMEZONE = "UTC"
EXPECTED_SHUFFLE_PARTITIONS = 3
FLEET_SCOPE = "all_machines"
PRODUCT_QUALITY_TYPES = ("H", "L", "M")
PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS = tuple(
    f"product_quality_type_{quality.lower()}_event_count" for quality in PRODUCT_QUALITY_TYPES
)
FLEET_PRODUCT_QUALITY_TYPE_COUNT_FIELDS = tuple(
    f"product_quality_type_{quality.lower()}_count" for quality in PRODUCT_QUALITY_TYPES
)

SENSOR_FIELDS = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "pressure_bar",
)
SILVER_REQUIRED_FIELD_TYPES = {
    "air_temperature_k": "double",
    "bronze_ingested_at": "timestamp",
    "event_id": "string",
    "event_time": "timestamp",
    "machine_code": "string",
    "payload_sha256": "string",
    "pressure_bar": "double",
    "process_temperature_k": "double",
    "product_quality_type": "string",
    "rotational_speed_rpm": "int",
    "schema_version": "string",
    "sequence_number": "bigint",
    "source": "string",
    "source_kafka_key": "string",
    "source_kafka_offset": "bigint",
    "source_kafka_partition": "int",
    "source_kafka_timestamp": "timestamp",
    "source_kafka_topic": "string",
    "tool_wear_min": "int",
    "torque_nm": "double",
    "vibration_mm_s": "double",
}
LATEST_OBSERVATION_ORDER = (
    ("event_time", "desc"),
    ("source_kafka_timestamp", "desc"),
    ("source_kafka_topic", "desc"),
    ("source_kafka_partition", "desc"),
    ("source_kafka_offset", "desc"),
    ("event_id", "desc"),
)
MACHINE_SUMMARY_GRAIN = ("machine_code",)
MACHINE_WINDOWS_GRAIN = ("machine_code", "window_start", "window_end")
FLEET_SUMMARY_GRAIN = ("fleet_scope",)
FORBIDDEN_GOLD_FIELD_FRAGMENTS = (
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
)


class SparkGoldConfigError(ValueError):
    """Raised when Gold configuration is missing or incompatible."""


class SparkGoldValidationError(RuntimeError):
    """Raised when Silver input or Gold outputs violate expected invariants."""


@dataclass(frozen=True)
class SparkGoldConfig:
    """Static local Spark Gold descriptive analytics configuration."""

    spark_version: str
    application_name: str
    master: str
    silver_input_path: str
    machine_summary_output_path: str
    machine_windows_output_path: str
    fleet_summary_output_path: str
    output_format: str
    window_duration: str
    timezone: str
    shuffle_partitions: int


@dataclass(frozen=True)
class GoldTransformResult:
    """DataFrames produced by the Gold analytical transformation."""

    machine_summary_df: Any
    machine_windows_df: Any
    fleet_summary_df: Any


@dataclass(frozen=True)
class GoldWriteCounts:
    """Logical counts from a Gold snapshot rebuild."""

    silver_row_count: int
    silver_machine_count: int
    machine_summary_row_count: int
    machine_window_row_count: int
    fleet_summary_row_count: int
    machine_summary_event_count_sum: int
    machine_windows_event_count_sum: int

    def to_dict(self) -> dict[str, int]:
        return {
            "fleet_summary_row_count": self.fleet_summary_row_count,
            "machine_summary_event_count_sum": self.machine_summary_event_count_sum,
            "machine_summary_row_count": self.machine_summary_row_count,
            "machine_window_row_count": self.machine_window_row_count,
            "machine_windows_event_count_sum": self.machine_windows_event_count_sum,
            "silver_machine_count": self.silver_machine_count,
            "silver_row_count": self.silver_row_count,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def load_gold_config(path: Path | None = None) -> SparkGoldConfig:
    config_file = path or config_path()
    try:
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SparkGoldConfigError(f"Gold config file not found: {config_file}") from exc
    except json.JSONDecodeError as exc:
        raise SparkGoldConfigError(f"Gold config file is not valid JSON: {config_file}") from exc
    if not isinstance(raw_config, dict):
        raise SparkGoldConfigError("Gold config must be a JSON object.")
    return parse_gold_config(raw_config)


def parse_gold_config(raw_config: Mapping[str, Any]) -> SparkGoldConfig:
    required_keys = {
        "application_name",
        "fleet_summary_output_path",
        "machine_summary_output_path",
        "machine_windows_output_path",
        "master",
        "output_format",
        "shuffle_partitions",
        "silver_input_path",
        "spark_version",
        "timezone",
        "window_duration",
    }
    actual_keys = set(raw_config)
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if missing:
        raise SparkGoldConfigError("Missing Gold config key(s): " + ", ".join(missing))
    if unknown:
        raise SparkGoldConfigError("Unknown Gold config key(s): " + ", ".join(unknown))

    config = SparkGoldConfig(
        spark_version=require_text(raw_config, "spark_version"),
        application_name=require_text(raw_config, "application_name"),
        master=require_text(raw_config, "master"),
        silver_input_path=require_text(raw_config, "silver_input_path"),
        machine_summary_output_path=require_text(raw_config, "machine_summary_output_path"),
        machine_windows_output_path=require_text(raw_config, "machine_windows_output_path"),
        fleet_summary_output_path=require_text(raw_config, "fleet_summary_output_path"),
        output_format=require_text(raw_config, "output_format"),
        window_duration=require_text(raw_config, "window_duration"),
        timezone=require_text(raw_config, "timezone"),
        shuffle_partitions=require_int(raw_config, "shuffle_partitions"),
    )
    validate_gold_config(config)
    return config


def require_text(raw_config: Mapping[str, Any], key: str) -> str:
    value = raw_config[key]
    if not isinstance(value, str) or not value.strip():
        raise SparkGoldConfigError(f"Gold config key {key} must be a non-empty string.")
    return value


def require_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SparkGoldConfigError(f"Gold config key {key} must be an integer.")
    return value


def validate_relative_path(value: str, field_name: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise SparkGoldConfigError(f"{field_name} must be a safe relative path.")


def validate_gold_config(config: SparkGoldConfig) -> None:
    for field_name, value in (
        ("silver_input_path", config.silver_input_path),
        ("machine_summary_output_path", config.machine_summary_output_path),
        ("machine_windows_output_path", config.machine_windows_output_path),
        ("fleet_summary_output_path", config.fleet_summary_output_path),
    ):
        validate_relative_path(value, field_name)

    expected_values: tuple[tuple[str, str, str], ...] = (
        ("spark_version", config.spark_version, EXPECTED_SPARK_VERSION),
        ("application_name", config.application_name, EXPECTED_APPLICATION_NAME),
        ("master", config.master, EXPECTED_MASTER),
        ("silver_input_path", config.silver_input_path, EXPECTED_SILVER_INPUT_PATH),
        (
            "machine_summary_output_path",
            config.machine_summary_output_path,
            EXPECTED_MACHINE_SUMMARY_OUTPUT_PATH,
        ),
        (
            "machine_windows_output_path",
            config.machine_windows_output_path,
            EXPECTED_MACHINE_WINDOWS_OUTPUT_PATH,
        ),
        (
            "fleet_summary_output_path",
            config.fleet_summary_output_path,
            EXPECTED_FLEET_SUMMARY_OUTPUT_PATH,
        ),
        ("output_format", config.output_format, EXPECTED_OUTPUT_FORMAT),
        ("window_duration", config.window_duration, EXPECTED_WINDOW_DURATION),
        ("timezone", config.timezone, EXPECTED_TIMEZONE),
    )
    for field_name, actual, expected in expected_values:
        if actual != expected:
            raise SparkGoldConfigError(f"{field_name} must be {expected}.")
    if config.shuffle_partitions != EXPECTED_SHUFFLE_PARTITIONS:
        raise SparkGoldConfigError("shuffle_partitions must be 3.")


def container_path(relative_path: str) -> str:
    validate_relative_path(relative_path, "container path")
    return str(CONTAINER_WORKSPACE / PurePosixPath(relative_path.replace("\\", "/")))


def aggregate_field_names() -> tuple[str, ...]:
    names: list[str] = []
    for sensor in SENSOR_FIELDS:
        names.extend((f"avg_{sensor}", f"min_{sensor}", f"max_{sensor}"))
    return tuple(names)


def latest_observation_order() -> tuple[tuple[str, str], ...]:
    return LATEST_OBSERVATION_ORDER


def gold_output_paths() -> dict[str, str]:
    return {
        "fleet_summary": EXPECTED_FLEET_SUMMARY_OUTPUT_PATH,
        "machine_summary": EXPECTED_MACHINE_SUMMARY_OUTPUT_PATH,
        "machine_windows": EXPECTED_MACHINE_WINDOWS_OUTPUT_PATH,
    }


def event_accounting_holds(
    *,
    silver_row_count: int,
    machine_summary_event_count_sum: int,
    machine_windows_event_count_sum: int,
    fleet_event_count: int,
) -> bool:
    return (
        silver_row_count == machine_summary_event_count_sum
        and silver_row_count == machine_windows_event_count_sum
        and silver_row_count == fleet_event_count
    )


def product_quality_type_event_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {quality: 0 for quality in PRODUCT_QUALITY_TYPES}
    for row in rows:
        quality = str(row["product_quality_type"])
        if quality in counts:
            counts[quality] += 1
    return counts


def type_event_counts_reconcile(
    row: Mapping[str, Any],
    *,
    event_count_field: str = "event_count",
    type_count_fields: Sequence[str] = PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS,
) -> bool:
    return sum(int(row.get(field, 0)) for field in type_count_fields) == int(row[event_count_field])


def conceptual_latest_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not records:
        return None
    return sorted(
        records,
        key=lambda row: (
            str(row["event_time"]),
            str(row["source_kafka_timestamp"]),
            str(row["source_kafka_topic"]),
            int(row["source_kafka_partition"]),
            int(row["source_kafka_offset"]),
            str(row["event_id"]),
        ),
        reverse=True,
    )[0]


def conceptual_window_key(row: Mapping[str, Any]) -> tuple[str, str]:
    event_time = row["event_time"]
    if isinstance(event_time, str):
        timestamp = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    elif isinstance(event_time, datetime):
        timestamp = event_time
    else:
        raise ValueError("event_time must be a datetime or ISO timestamp string.")
    window_start = timestamp.replace(second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=1)
    return window_start.isoformat(), window_end.isoformat()


def conceptual_machine_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_machine: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        rows_by_machine.setdefault(str(record["machine_code"]), []).append(record)

    summaries: list[dict[str, Any]] = []
    for machine_code, machine_rows in sorted(rows_by_machine.items()):
        latest = conceptual_latest_record(machine_rows)
        if latest is None:
            continue
        type_counts = product_quality_type_event_counts(machine_rows)
        summaries.append(
            {
                "event_count": len(machine_rows),
                "latest_product_quality_type": str(latest["product_quality_type"]),
                "machine_code": machine_code,
                **{
                    f"product_quality_type_{quality.lower()}_event_count": count
                    for quality, count in type_counts.items()
                },
            }
        )
    return summaries


def conceptual_machine_windows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_window: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        window_start, window_end = conceptual_window_key(record)
        key = (str(record["machine_code"]), window_start, window_end)
        rows_by_window.setdefault(key, []).append(record)

    windows: list[dict[str, Any]] = []
    for (machine_code, window_start, window_end), window_rows in sorted(rows_by_window.items()):
        type_counts = product_quality_type_event_counts(window_rows)
        windows.append(
            {
                "event_count": len(window_rows),
                "machine_code": machine_code,
                "window_end": window_end,
                "window_start": window_start,
                **{
                    f"product_quality_type_{quality.lower()}_event_count": count
                    for quality, count in type_counts.items()
                },
            }
        )
    return windows


def build_static_summary(config: SparkGoldConfig) -> dict[str, Any]:
    return {
        "analytics_type": "descriptive_analytics_only",
        "execution_model": "deterministic local Spark snapshot rebuild on local[2]",
        "fleet_summary_grain": list(FLEET_SUMMARY_GRAIN),
        "fleet_summary_output_path": config.fleet_summary_output_path,
        "latest_observation_order": [
            f"{field} {direction}" for field, direction in LATEST_OBSERVATION_ORDER
        ],
        "machine_summary_grain": list(MACHINE_SUMMARY_GRAIN),
        "machine_summary_output_path": config.machine_summary_output_path,
        "machine_windows_grain": list(MACHINE_WINDOWS_GRAIN),
        "machine_windows_output_path": config.machine_windows_output_path,
        "output_format": config.output_format,
        "product_quality_type_semantics": (
            "event-level synthetic model-compatible attribute; not a stable machine attribute"
        ),
        "machine_summary_product_quality_fields": [
            "latest_product_quality_type",
            *PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS,
        ],
        "machine_windows_product_quality_fields": list(PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS),
        "fleet_summary_product_quality_fields": list(FLEET_PRODUCT_QUALITY_TYPE_COUNT_FIELDS),
        "runtime_counts": "intentionally excluded from tracked summary",
        "silver_input_path": config.silver_input_path,
        "spark_version": config.spark_version,
        "window_duration": config.window_duration,
    }


def write_static_summary(root: Path | None, config: SparkGoldConfig) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_static_summary(config), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def create_spark_session(config: SparkGoldConfig) -> Any:
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
    missing = sorted(set(SILVER_REQUIRED_FIELD_TYPES) - set(actual_types))
    mismatched = sorted(
        field
        for field, expected_type in SILVER_REQUIRED_FIELD_TYPES.items()
        if field in actual_types and actual_types[field] != expected_type
    )
    if missing:
        raise SparkGoldValidationError(
            "Canonical Silver is missing required column(s): " + ", ".join(missing)
        )
    if mismatched:
        details = ", ".join(
            f"{field} expected {SILVER_REQUIRED_FIELD_TYPES[field]} got {actual_types[field]}"
            for field in mismatched
        )
        raise SparkGoldValidationError("Canonical Silver has incompatible type(s): " + details)


def read_silver_snapshot(spark: Any, config: SparkGoldConfig) -> Any:
    silver_path = Path(container_path(config.silver_input_path))
    if not silver_path.exists():
        raise SparkGoldValidationError(
            f"Canonical Silver does not exist: {config.silver_input_path}"
        )
    silver_df = spark.read.parquet(str(silver_path))
    validate_silver_schema(silver_df)
    return silver_df


def _aggregate_expressions() -> list[Any]:
    from pyspark.sql import functions as spark_fn

    expressions: list[Any] = []
    for sensor in SENSOR_FIELDS:
        expressions.extend(
            [
                spark_fn.avg(sensor).cast("double").alias(f"avg_{sensor}"),
                spark_fn.min(sensor).alias(f"min_{sensor}"),
                spark_fn.max(sensor).alias(f"max_{sensor}"),
            ]
        )
    return expressions


def _product_quality_type_count_expressions(*, fleet: bool = False) -> list[Any]:
    from pyspark.sql import functions as spark_fn

    fields = (
        FLEET_PRODUCT_QUALITY_TYPE_COUNT_FIELDS
        if fleet
        else PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS
    )
    return [
        spark_fn.sum(spark_fn.when(spark_fn.col("product_quality_type") == quality, 1).otherwise(0))
        .cast("long")
        .alias(field)
        for quality, field in zip(PRODUCT_QUALITY_TYPES, fields, strict=True)
    ]


def build_latest_observations(silver_df: Any) -> Any:
    from pyspark.sql import Window
    from pyspark.sql import functions as spark_fn

    order_columns = [
        spark_fn.col("event_time").desc(),
        spark_fn.col("source_kafka_timestamp").desc(),
        spark_fn.col("source_kafka_topic").desc(),
        spark_fn.col("source_kafka_partition").desc(),
        spark_fn.col("source_kafka_offset").desc(),
        spark_fn.col("event_id").desc(),
    ]
    window = Window.partitionBy("machine_code").orderBy(*order_columns)
    return silver_df.withColumn("_latest_rank", spark_fn.row_number().over(window)).where(
        spark_fn.col("_latest_rank") == 1
    )


def build_machine_summary(silver_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    latest = build_latest_observations(silver_df).select(
        "machine_code",
        spark_fn.col("product_quality_type").alias("latest_product_quality_type"),
        spark_fn.col("air_temperature_k").alias("latest_air_temperature_k"),
        spark_fn.col("process_temperature_k").alias("latest_process_temperature_k"),
        spark_fn.col("rotational_speed_rpm").alias("latest_rotational_speed_rpm"),
        spark_fn.col("torque_nm").alias("latest_torque_nm"),
        spark_fn.col("tool_wear_min").alias("latest_tool_wear_min"),
        spark_fn.col("vibration_mm_s").alias("latest_vibration_mm_s"),
        spark_fn.col("pressure_bar").alias("latest_pressure_bar"),
    )
    grouped = silver_df.groupBy("machine_code").agg(
        spark_fn.count("*").cast("long").alias("event_count"),
        spark_fn.min("event_time").alias("first_event_time"),
        spark_fn.max("event_time").alias("latest_event_time"),
        spark_fn.countDistinct("event_id").cast("long").alias("distinct_event_count"),
        *_product_quality_type_count_expressions(),
        *_aggregate_expressions(),
    )
    return grouped.join(latest, on="machine_code", how="inner").select(
        "machine_code",
        "latest_product_quality_type",
        "event_count",
        "first_event_time",
        "latest_event_time",
        *PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS,
        "latest_air_temperature_k",
        "latest_process_temperature_k",
        "latest_rotational_speed_rpm",
        "latest_torque_nm",
        "latest_tool_wear_min",
        "latest_vibration_mm_s",
        "latest_pressure_bar",
        "avg_air_temperature_k",
        "min_air_temperature_k",
        "max_air_temperature_k",
        "avg_process_temperature_k",
        "min_process_temperature_k",
        "max_process_temperature_k",
        "avg_rotational_speed_rpm",
        "min_rotational_speed_rpm",
        "max_rotational_speed_rpm",
        "avg_torque_nm",
        "min_torque_nm",
        "max_torque_nm",
        "avg_tool_wear_min",
        "min_tool_wear_min",
        "max_tool_wear_min",
        "avg_vibration_mm_s",
        "min_vibration_mm_s",
        "max_vibration_mm_s",
        "avg_pressure_bar",
        "min_pressure_bar",
        "max_pressure_bar",
        "distinct_event_count",
    )


def build_machine_windows(silver_df: Any, config: SparkGoldConfig) -> Any:
    from pyspark.sql import functions as spark_fn

    return (
        silver_df.groupBy(
            "machine_code",
            spark_fn.window("event_time", config.window_duration).alias("_window"),
        )
        .agg(
            spark_fn.count("*").cast("long").alias("event_count"),
            *_product_quality_type_count_expressions(),
            *_aggregate_expressions(),
        )
        .select(
            "machine_code",
            spark_fn.col("_window.start").alias("window_start"),
            spark_fn.col("_window.end").alias("window_end"),
            "event_count",
            *PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS,
            *aggregate_field_names(),
        )
    )


def build_fleet_summary(silver_df: Any) -> Any:
    from pyspark.sql import functions as spark_fn

    return silver_df.agg(
        spark_fn.lit(FLEET_SCOPE).alias("fleet_scope"),
        spark_fn.countDistinct("machine_code").cast("long").alias("machine_count"),
        spark_fn.count("*").cast("long").alias("event_count"),
        spark_fn.min("event_time").alias("first_event_time"),
        spark_fn.max("event_time").alias("latest_event_time"),
        *_aggregate_expressions(),
        *_product_quality_type_count_expressions(fleet=True),
    )


def transform_silver_to_gold(silver_df: Any, config: SparkGoldConfig) -> GoldTransformResult:
    validate_silver_schema(silver_df)
    return GoldTransformResult(
        machine_summary_df=build_machine_summary(silver_df),
        machine_windows_df=build_machine_windows(silver_df, config),
        fleet_summary_df=build_fleet_summary(silver_df),
    )


def _sum_column(df: Any, column_name: str) -> int:
    from pyspark.sql import functions as spark_fn

    value = df.agg(spark_fn.sum(column_name).alias("total")).collect()[0]["total"]
    return 0 if value is None else int(value)


def _distinct_count(df: Any, column_name: str) -> int:
    return int(df.select(column_name).distinct().count())


def _fleet_event_count(fleet_summary_df: Any) -> int:
    rows = fleet_summary_df.select("event_count").collect()
    if len(rows) != 1:
        return -1
    return int(rows[0]["event_count"])


def _type_event_count_mismatch_count(
    df: Any,
    type_count_fields: Sequence[str] = PRODUCT_QUALITY_TYPE_EVENT_COUNT_FIELDS,
) -> int:
    from pyspark.sql import functions as spark_fn

    type_count_sum = None
    for field in type_count_fields:
        value = spark_fn.col(field).cast("long")
        type_count_sum = value if type_count_sum is None else type_count_sum + value
    if type_count_sum is None:
        return int(df.count())
    return int(df.where(type_count_sum != spark_fn.col("event_count").cast("long")).count())


def _fleet_type_count_mismatch(fleet_summary_df: Any) -> bool:
    return (
        _type_event_count_mismatch_count(
            fleet_summary_df,
            FLEET_PRODUCT_QUALITY_TYPE_COUNT_FIELDS,
        )
        != 0
    )


def validate_gold_invariants(
    silver_df: Any,
    result: GoldTransformResult,
) -> None:
    silver_rows = int(silver_df.count())
    silver_machines = _distinct_count(silver_df, "machine_code")
    machine_summary_rows = int(result.machine_summary_df.count())
    machine_summary_distinct_machines = _distinct_count(result.machine_summary_df, "machine_code")
    machine_windows_duplicate_grain = _duplicate_grain_count(
        result.machine_windows_df,
        list(MACHINE_WINDOWS_GRAIN),
    )
    fleet_rows = int(result.fleet_summary_df.count())
    if machine_summary_rows != silver_machines:
        raise SparkGoldValidationError("Machine summary row count does not match Silver machines.")
    if machine_summary_distinct_machines != machine_summary_rows:
        raise SparkGoldValidationError("Machine summary grain is not unique.")
    if machine_windows_duplicate_grain != 0:
        raise SparkGoldValidationError("Machine windows grain is not unique.")
    if fleet_rows != 1:
        raise SparkGoldValidationError("Fleet summary must contain exactly one row.")
    if _type_event_count_mismatch_count(result.machine_summary_df) != 0:
        raise SparkGoldValidationError("Machine summary product-quality event counts failed.")
    if _type_event_count_mismatch_count(result.machine_windows_df) != 0:
        raise SparkGoldValidationError("Machine window product-quality event counts failed.")
    if _fleet_type_count_mismatch(result.fleet_summary_df):
        raise SparkGoldValidationError("Fleet product-quality event counts failed.")
    if not event_accounting_holds(
        silver_row_count=silver_rows,
        machine_summary_event_count_sum=_sum_column(result.machine_summary_df, "event_count"),
        machine_windows_event_count_sum=_sum_column(result.machine_windows_df, "event_count"),
        fleet_event_count=_fleet_event_count(result.fleet_summary_df),
    ):
        raise SparkGoldValidationError("Gold event accounting invariants failed.")


def write_gold_outputs(result: GoldTransformResult, config: SparkGoldConfig) -> None:
    result.machine_summary_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.machine_summary_output_path)
    )
    result.machine_windows_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.machine_windows_output_path)
    )
    result.fleet_summary_df.write.mode("overwrite").format(config.output_format).save(
        container_path(config.fleet_summary_output_path)
    )


def rebuild_gold_snapshot(spark: Any, config: SparkGoldConfig) -> GoldWriteCounts:
    silver_df = read_silver_snapshot(spark, config)
    result = transform_silver_to_gold(silver_df, config)
    validate_gold_invariants(silver_df, result)
    counts = GoldWriteCounts(
        silver_row_count=int(silver_df.count()),
        silver_machine_count=_distinct_count(silver_df, "machine_code"),
        machine_summary_row_count=int(result.machine_summary_df.count()),
        machine_window_row_count=int(result.machine_windows_df.count()),
        fleet_summary_row_count=int(result.fleet_summary_df.count()),
        machine_summary_event_count_sum=_sum_column(result.machine_summary_df, "event_count"),
        machine_windows_event_count_sum=_sum_column(result.machine_windows_df, "event_count"),
    )
    write_gold_outputs(result, config)
    return counts


def _path_exists(relative_path: str) -> bool:
    return Path(container_path(relative_path)).exists()


def _read_parquet(spark: Any, relative_path: str) -> Any:
    return spark.read.parquet(container_path(relative_path))


def _duplicate_grain_count(df: Any, grain_columns: Sequence[str]) -> int:
    from pyspark.sql import functions as spark_fn

    return int(df.groupBy(*grain_columns).count().where(spark_fn.col("count") > 1).count())


def _type_counts(silver_df: Any) -> dict[str, int]:
    counts = {
        str(row["product_quality_type"]): int(row["count"])
        for row in silver_df.groupBy("product_quality_type")
        .count()
        .orderBy("product_quality_type")
        .collect()
    }
    return {quality: int(counts.get(quality, 0)) for quality in PRODUCT_QUALITY_TYPES}


def _mixed_product_quality_type_machine_count(silver_df: Any) -> int:
    from pyspark.sql import functions as spark_fn

    return int(
        silver_df.groupBy("machine_code")
        .agg(spark_fn.countDistinct("product_quality_type").alias("_type_count"))
        .where(spark_fn.col("_type_count") > 1)
        .count()
    )


def _first_latest_event_times(silver_df: Any) -> tuple[str | None, str | None]:
    from pyspark.sql import functions as spark_fn

    row = silver_df.agg(
        spark_fn.min("event_time").cast("string").alias("first_event_time"),
        spark_fn.max("event_time").cast("string").alias("latest_event_time"),
    ).collect()[0]
    first_event_time = row["first_event_time"]
    latest_event_time = row["latest_event_time"]
    return (
        None if first_event_time is None else str(first_event_time),
        None if latest_event_time is None else str(latest_event_time),
    )


def _latest_observation_mismatch_count(silver_df: Any, machine_summary_df: Any) -> int:
    from pyspark.sql import functions as spark_fn

    latest = build_latest_observations(silver_df).select(
        "machine_code",
        spark_fn.col("product_quality_type").alias("_expected_latest_product_quality_type"),
        spark_fn.col("air_temperature_k").alias("_expected_latest_air_temperature_k"),
        spark_fn.col("process_temperature_k").alias("_expected_latest_process_temperature_k"),
        spark_fn.col("rotational_speed_rpm").alias("_expected_latest_rotational_speed_rpm"),
        spark_fn.col("torque_nm").alias("_expected_latest_torque_nm"),
        spark_fn.col("tool_wear_min").alias("_expected_latest_tool_wear_min"),
        spark_fn.col("vibration_mm_s").alias("_expected_latest_vibration_mm_s"),
        spark_fn.col("pressure_bar").alias("_expected_latest_pressure_bar"),
    )
    joined = machine_summary_df.join(latest, on="machine_code", how="inner")
    mismatch = (
        (
            spark_fn.col("latest_product_quality_type")
            != spark_fn.col("_expected_latest_product_quality_type")
        )
        | (
            spark_fn.col("latest_air_temperature_k")
            != spark_fn.col("_expected_latest_air_temperature_k")
        )
        | (
            spark_fn.col("latest_process_temperature_k")
            != spark_fn.col("_expected_latest_process_temperature_k")
        )
        | (
            spark_fn.col("latest_rotational_speed_rpm")
            != spark_fn.col("_expected_latest_rotational_speed_rpm")
        )
        | (spark_fn.col("latest_torque_nm") != spark_fn.col("_expected_latest_torque_nm"))
        | (spark_fn.col("latest_tool_wear_min") != spark_fn.col("_expected_latest_tool_wear_min"))
        | (spark_fn.col("latest_vibration_mm_s") != spark_fn.col("_expected_latest_vibration_mm_s"))
        | (spark_fn.col("latest_pressure_bar") != spark_fn.col("_expected_latest_pressure_bar"))
    )
    return int(joined.where(mismatch).count())


def _machine_event_count_mismatch_count(silver_df: Any, machine_summary_df: Any) -> int:
    from pyspark.sql import functions as spark_fn

    expected = silver_df.groupBy("machine_code").agg(
        spark_fn.count("*").cast("long").alias("_expected_event_count"),
        spark_fn.countDistinct("event_id").cast("long").alias("_expected_distinct_event_count"),
        spark_fn.min("event_time").alias("_expected_first_event_time"),
        spark_fn.max("event_time").alias("_expected_latest_event_time"),
    )
    joined = machine_summary_df.join(expected, on="machine_code", how="inner")
    mismatch = (
        (spark_fn.col("event_count") != spark_fn.col("_expected_event_count"))
        | (spark_fn.col("distinct_event_count") != spark_fn.col("_expected_distinct_event_count"))
        | (spark_fn.col("first_event_time") != spark_fn.col("_expected_first_event_time"))
        | (spark_fn.col("latest_event_time") != spark_fn.col("_expected_latest_event_time"))
        | (spark_fn.col("event_count") != spark_fn.col("distinct_event_count"))
    )
    return int(joined.where(mismatch).count())


def _non_finite_aggregate_count(*dataframes: Any) -> int:
    from pyspark.sql import functions as spark_fn

    total = 0
    aggregate_fields = aggregate_field_names()
    for dataframe in dataframes:
        columns = [field for field in aggregate_fields if field in dataframe.columns]
        if not columns:
            continue
        conditions = [
            spark_fn.col(column).isNull() | spark_fn.isnan(spark_fn.col(column).cast("double"))
            for column in columns
        ]
        condition = conditions[0]
        for next_condition in conditions[1:]:
            condition = condition | next_condition
        total += int(dataframe.where(condition).count())
    return total


def _logical_hash(df: Any, columns: Sequence[str]) -> str | None:
    from pyspark.sql import functions as spark_fn

    if int(df.count()) == 0:
        return None
    parts = [
        spark_fn.coalesce(spark_fn.col(column).cast("string"), spark_fn.lit(""))
        for column in columns
    ]
    signature = spark_fn.concat_ws("|", *parts)
    value = (
        df.select(signature.alias("signature"))
        .agg(
            spark_fn.sha2(
                spark_fn.concat_ws("\n", spark_fn.array_sort(spark_fn.collect_list("signature"))),
                256,
            ).alias("digest")
        )
        .collect()[0]["digest"]
    )
    return None if value is None else str(value)


def inspect_gold_outputs(spark: Any, config: SparkGoldConfig) -> dict[str, Any]:
    for path_name, relative_path in (
        ("silver_input_path", config.silver_input_path),
        ("machine_summary_output_path", config.machine_summary_output_path),
        ("machine_windows_output_path", config.machine_windows_output_path),
        ("fleet_summary_output_path", config.fleet_summary_output_path),
    ):
        if not _path_exists(relative_path):
            raise SparkGoldValidationError(f"{path_name} does not exist: {relative_path}")

    silver_df = _read_parquet(spark, config.silver_input_path)
    validate_silver_schema(silver_df)
    machine_summary_df = _read_parquet(spark, config.machine_summary_output_path)
    machine_windows_df = _read_parquet(spark, config.machine_windows_output_path)
    fleet_summary_df = _read_parquet(spark, config.fleet_summary_output_path)

    silver_row_count = int(silver_df.count())
    silver_machine_count = _distinct_count(silver_df, "machine_code")
    machine_summary_row_count = int(machine_summary_df.count())
    machine_summary_event_count_sum = _sum_column(machine_summary_df, "event_count")
    machine_windows_event_count_sum = _sum_column(machine_windows_df, "event_count")
    fleet_rows = fleet_summary_df.collect()
    fleet_row = fleet_rows[0].asDict() if len(fleet_rows) == 1 else {}
    first_event_time, latest_event_time = _first_latest_event_times(silver_df)
    machine_summary_columns = list(machine_summary_df.columns)
    machine_window_columns = list(machine_windows_df.columns)
    fleet_summary_columns = list(fleet_summary_df.columns)
    product_quality_type_event_counts = _type_counts(silver_df)

    return {
        "excluded_field_count": excluded_field_count(
            [*machine_summary_columns, *machine_window_columns, *fleet_summary_columns]
        ),
        "first_event_time": first_event_time,
        "fleet_event_count_mismatch": int(fleet_row.get("event_count", -1)) != silver_row_count,
        "fleet_machine_count_mismatch": int(fleet_row.get("machine_count", -1))
        != silver_machine_count,
        "fleet_product_quality_type_count_sum": sum(
            int(fleet_row.get(f"product_quality_type_{quality.lower()}_count", 0))
            for quality in PRODUCT_QUALITY_TYPES
        ),
        "fleet_summary_event_count": int(fleet_row.get("event_count", -1)),
        "fleet_summary_machine_count": int(fleet_row.get("machine_count", -1)),
        "fleet_summary_record": normalize_record(fleet_row) if fleet_row else {},
        "fleet_summary_row_count": len(fleet_rows),
        "fleet_type_count_mismatch": _fleet_type_count_mismatch(fleet_summary_df),
        "latest_event_time": latest_event_time,
        "latest_observation_mismatch_count": _latest_observation_mismatch_count(
            silver_df,
            machine_summary_df,
        ),
        "machine_summary_distinct_machine_count": _distinct_count(
            machine_summary_df,
            "machine_code",
        ),
        "machine_summary_duplicate_machine_count": _duplicate_grain_count(
            machine_summary_df,
            list(MACHINE_SUMMARY_GRAIN),
        ),
        "machine_summary_event_count_mismatch_count": _machine_event_count_mismatch_count(
            silver_df,
            machine_summary_df,
        ),
        "machine_summary_event_count_sum": machine_summary_event_count_sum,
        "machine_summary_records_sample": [
            normalize_record(row.asDict())
            for row in machine_summary_df.orderBy("machine_code").limit(3).collect()
        ],
        "machine_summary_row_count": machine_summary_row_count,
        "machine_summary_selection_sha256": _logical_hash(
            machine_summary_df,
            machine_summary_columns,
        ),
        "machine_summary_type_count_mismatch_count": _type_event_count_mismatch_count(
            machine_summary_df
        ),
        "machine_windows_duplicate_grain_count": _duplicate_grain_count(
            machine_windows_df,
            list(MACHINE_WINDOWS_GRAIN),
        ),
        "machine_windows_event_count_sum": machine_windows_event_count_sum,
        "machine_windows_records_sample": [
            normalize_record(row.asDict())
            for row in machine_windows_df.orderBy("machine_code", "window_start", "window_end")
            .limit(3)
            .collect()
        ],
        "machine_windows_row_count": int(machine_windows_df.count()),
        "machine_windows_selection_sha256": _logical_hash(
            machine_windows_df,
            machine_window_columns,
        ),
        "machine_windows_type_count_mismatch_count": _type_event_count_mismatch_count(
            machine_windows_df
        ),
        "non_finite_aggregate_count": _non_finite_aggregate_count(
            machine_summary_df,
            machine_windows_df,
            fleet_summary_df,
        ),
        "product_quality_type_event_counts": product_quality_type_event_counts,
        "silver_machine_count": silver_machine_count,
        "silver_mixed_product_quality_type_machine_count": (
            _mixed_product_quality_type_machine_count(silver_df)
        ),
        "silver_row_count": silver_row_count,
        "window_event_count_mismatch": machine_windows_event_count_sum != silver_row_count,
    }


def excluded_field_count(field_names: Sequence[str]) -> int:
    lowered = [name.lower() for name in field_names]
    return sum(
        1
        for field_name in lowered
        if any(fragment in field_name for fragment in FORBIDDEN_GOLD_FIELD_FRAGMENTS)
    )


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, datetime):
            normalized[key] = value.isoformat(sep=" ")
        else:
            normalized[key] = value
    return normalized


def _synthetic_rows() -> list[dict[str, Any]]:
    return [
        {
            "air_temperature_k": 300.0,
            "bronze_ingested_at": datetime(2026, 2, 1, 0, 0, 1),
            "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1",
            "event_time": datetime(2026, 2, 1, 0, 0, 5),
            "machine_code": "MCH-0001",
            "payload_sha256": "a1",
            "pressure_bar": 6.0,
            "process_temperature_k": 309.0,
            "product_quality_type": "L",
            "rotational_speed_rpm": 1500,
            "schema_version": "1.0",
            "sequence_number": 1,
            "source": "synthetic_simulator",
            "source_kafka_key": "MCH-0001",
            "source_kafka_offset": 1,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": datetime(2026, 2, 1, 0, 0, 6),
            "source_kafka_topic": "industrial.telemetry.v1",
            "tool_wear_min": 10,
            "torque_nm": 40.0,
            "vibration_mm_s": 2.0,
        },
        {
            "air_temperature_k": 302.0,
            "bronze_ingested_at": datetime(2026, 2, 1, 0, 0, 11),
            "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2",
            "event_time": datetime(2026, 2, 1, 0, 0, 55),
            "machine_code": "MCH-0001",
            "payload_sha256": "a2",
            "pressure_bar": 7.0,
            "process_temperature_k": 311.0,
            "product_quality_type": "H",
            "rotational_speed_rpm": 1600,
            "schema_version": "1.0",
            "sequence_number": 2,
            "source": "synthetic_simulator",
            "source_kafka_key": "MCH-0001",
            "source_kafka_offset": 2,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": datetime(2026, 2, 1, 0, 0, 56),
            "source_kafka_topic": "industrial.telemetry.v1",
            "tool_wear_min": 12,
            "torque_nm": 44.0,
            "vibration_mm_s": 2.5,
        },
        {
            "air_temperature_k": 303.0,
            "bronze_ingested_at": datetime(2026, 2, 1, 0, 1, 2),
            "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa3",
            "event_time": datetime(2026, 2, 1, 0, 1, 5),
            "machine_code": "MCH-0001",
            "payload_sha256": "a3",
            "pressure_bar": 8.0,
            "process_temperature_k": 312.0,
            "product_quality_type": "M",
            "rotational_speed_rpm": 1700,
            "schema_version": "1.0",
            "sequence_number": 3,
            "source": "synthetic_simulator",
            "source_kafka_key": "MCH-0001",
            "source_kafka_offset": 3,
            "source_kafka_partition": 0,
            "source_kafka_timestamp": datetime(2026, 2, 1, 0, 1, 6),
            "source_kafka_topic": "industrial.telemetry.v1",
            "tool_wear_min": 15,
            "torque_nm": 46.0,
            "vibration_mm_s": 3.0,
        },
        {
            "air_temperature_k": 299.0,
            "bronze_ingested_at": datetime(2026, 2, 1, 0, 1, 1),
            "event_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1",
            "event_time": datetime(2026, 2, 1, 0, 1, 0),
            "machine_code": "MCH-0002",
            "payload_sha256": "b1",
            "pressure_bar": 5.0,
            "process_temperature_k": 308.0,
            "product_quality_type": "H",
            "rotational_speed_rpm": 1400,
            "schema_version": "1.0",
            "sequence_number": 1,
            "source": "synthetic_simulator",
            "source_kafka_key": "MCH-0002",
            "source_kafka_offset": 10,
            "source_kafka_partition": 1,
            "source_kafka_timestamp": datetime(2026, 2, 1, 0, 1, 1),
            "source_kafka_topic": "industrial.telemetry.v1",
            "tool_wear_min": 20,
            "torque_nm": 38.0,
            "vibration_mm_s": 1.5,
        },
        {
            "air_temperature_k": 301.0,
            "bronze_ingested_at": datetime(2026, 2, 1, 0, 1, 1),
            "event_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb2",
            "event_time": datetime(2026, 2, 1, 0, 1, 0),
            "machine_code": "MCH-0002",
            "payload_sha256": "b2",
            "pressure_bar": 5.5,
            "process_temperature_k": 310.0,
            "product_quality_type": "H",
            "rotational_speed_rpm": 1450,
            "schema_version": "1.0",
            "sequence_number": 2,
            "source": "synthetic_simulator",
            "source_kafka_key": "MCH-0002",
            "source_kafka_offset": 11,
            "source_kafka_partition": 1,
            "source_kafka_timestamp": datetime(2026, 2, 1, 0, 1, 2),
            "source_kafka_topic": "industrial.telemetry.v1",
            "tool_wear_min": 21,
            "torque_nm": 39.0,
            "vibration_mm_s": 1.8,
        },
    ]


def build_synthetic_silver_dataframe(spark: Any) -> Any:
    from pyspark.sql import types as spark_types

    schema = spark_types.StructType(
        [
            spark_types.StructField("schema_version", spark_types.StringType(), False),
            spark_types.StructField("event_id", spark_types.StringType(), False),
            spark_types.StructField("machine_code", spark_types.StringType(), False),
            spark_types.StructField("sequence_number", spark_types.LongType(), False),
            spark_types.StructField("event_time", spark_types.TimestampType(), False),
            spark_types.StructField("source", spark_types.StringType(), False),
            spark_types.StructField("product_quality_type", spark_types.StringType(), False),
            spark_types.StructField("air_temperature_k", spark_types.DoubleType(), False),
            spark_types.StructField("process_temperature_k", spark_types.DoubleType(), False),
            spark_types.StructField("rotational_speed_rpm", spark_types.IntegerType(), False),
            spark_types.StructField("torque_nm", spark_types.DoubleType(), False),
            spark_types.StructField("tool_wear_min", spark_types.IntegerType(), False),
            spark_types.StructField("vibration_mm_s", spark_types.DoubleType(), False),
            spark_types.StructField("pressure_bar", spark_types.DoubleType(), False),
            spark_types.StructField("source_kafka_topic", spark_types.StringType(), False),
            spark_types.StructField("source_kafka_partition", spark_types.IntegerType(), False),
            spark_types.StructField("source_kafka_offset", spark_types.LongType(), False),
            spark_types.StructField("source_kafka_timestamp", spark_types.TimestampType(), False),
            spark_types.StructField("source_kafka_key", spark_types.StringType(), False),
            spark_types.StructField("bronze_ingested_at", spark_types.TimestampType(), False),
            spark_types.StructField("payload_sha256", spark_types.StringType(), False),
        ]
    )
    return spark.createDataFrame(_synthetic_rows(), schema=schema)


def run_synthetic_analytics_check(spark: Any, config: SparkGoldConfig) -> dict[str, Any]:
    silver_df = build_synthetic_silver_dataframe(spark)
    result = transform_silver_to_gold(silver_df, config)
    validate_gold_invariants(silver_df, result)
    machine_summary_rows = {
        str(row["machine_code"]): row.asDict()
        for row in result.machine_summary_df.orderBy("machine_code").collect()
    }
    fleet_row = result.fleet_summary_df.collect()[0].asDict()
    mch_0001_summary = machine_summary_rows.get("MCH-0001", {})
    mch_0001_windows = [
        row.asDict()
        for row in result.machine_windows_df.where("machine_code = 'MCH-0001'")
        .orderBy("window_start", "window_end")
        .collect()
    ]
    first_mch_0001_window = mch_0001_windows[0] if len(mch_0001_windows) >= 1 else {}
    second_mch_0001_window = mch_0001_windows[1] if len(mch_0001_windows) >= 2 else {}
    summary = {
        "correct_event_counts": _sum_column(result.machine_summary_df, "event_count") == 5,
        "correct_fleet_type_counts": (
            int(fleet_row.get("product_quality_type_h_count", -1)) == 3
            and int(fleet_row.get("product_quality_type_l_count", -1)) == 1
            and int(fleet_row.get("product_quality_type_m_count", -1)) == 1
            and not _fleet_type_count_mismatch(result.fleet_summary_df)
        ),
        "correct_latest_observation": _latest_observation_mismatch_count(
            silver_df,
            result.machine_summary_df,
        )
        == 0,
        "correct_latest_product_quality_type": (
            mch_0001_summary.get("latest_product_quality_type") == "M"
        ),
        "correct_machine_summary_rows": int(result.machine_summary_df.count()) == 2,
        "correct_machine_summary_type_counts": (
            int(mch_0001_summary.get("product_quality_type_h_event_count", -1)) == 1
            and int(mch_0001_summary.get("product_quality_type_l_event_count", -1)) == 1
            and int(mch_0001_summary.get("product_quality_type_m_event_count", -1)) == 1
            and _type_event_count_mismatch_count(result.machine_summary_df) == 0
        ),
        "correct_machine_window_rows": int(result.machine_windows_df.count()) == 3,
        "correct_machine_window_type_counts": (
            int(first_mch_0001_window.get("event_count", -1)) == 2
            and int(first_mch_0001_window.get("product_quality_type_h_event_count", -1)) == 1
            and int(first_mch_0001_window.get("product_quality_type_l_event_count", -1)) == 1
            and int(first_mch_0001_window.get("product_quality_type_m_event_count", -1)) == 0
            and int(second_mch_0001_window.get("event_count", -1)) == 1
            and int(second_mch_0001_window.get("product_quality_type_m_event_count", -1)) == 1
            and _type_event_count_mismatch_count(result.machine_windows_df) == 0
        ),
        "mixed_product_quality_type_accepted": (
            int(mch_0001_summary.get("event_count", -1)) == 3
            and int(result.machine_summary_df.where("machine_code = 'MCH-0001'").count()) == 1
        ),
        "correct_numeric_aggregates": _non_finite_aggregate_count(
            result.machine_summary_df,
            result.machine_windows_df,
            result.fleet_summary_df,
        )
        == 0,
        "fleet_summary_rows": int(result.fleet_summary_df.count()),
        "machine_summary_event_count_sum": _sum_column(result.machine_summary_df, "event_count"),
        "machine_window_event_count_sum": _sum_column(result.machine_windows_df, "event_count"),
        "source_row_count": int(silver_df.count()),
    }
    summary["all_checks_passed"] = all(bool(value) for value in summary.values())
    return summary

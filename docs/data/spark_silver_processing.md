# Spark Silver Telemetry Processing

## Purpose

The Silver telemetry phase converts raw Bronze Kafka records into validated typed telemetry, a canonical deduplicated Silver dataset, an audit dataset for valid duplicate business events, and a quarantine dataset for invalid telemetry. It is a local portfolio data-engineering phase focused on contract enforcement, lineage, auditability, and deterministic rebuilds.

## Bronze to Silver Architecture

The implemented downstream flow is:

```text
Bronze raw Parquet -> Spark parsing and validation -> Silver outputs -> Gold descriptive analytics
```

Bronze remains the authoritative raw record layer. Silver reads `data/bronze/telemetry/` and writes generated Parquet datasets under `data/silver/`.

## Execution Model

Silver is a deterministic snapshot rebuild over the current Bronze Parquet data. It is not another Structured Streaming query. Kafka-to-Bronze already demonstrates streaming ingestion; this layer is intentionally a batch-style Spark transformation for local development, auditability, and repeatable validation.

The job runs inside the existing Docker Spark runtime using Spark `4.0.4`, `local[2]`, UTC session time, and `spark.sql.shuffle.partitions=3`. It does not require the Spark Kafka connector because it reads Parquet only.

## Explicit Telemetry Schema

Silver parses `raw_value` with an explicit Spark schema:

- `schema_version`: string
- `event_id`: string
- `machine_code`: string
- `sequence_number`: long
- `event_time`: timestamp after UTC parsing
- `source`: string
- `product_quality_type`: string
- `air_temperature_k`: double
- `process_temperature_k`: double
- `rotational_speed_rpm`: integer
- `torque_nm`: double
- `tool_wear_min`: integer
- `vibration_mm_s`: double
- `pressure_bar`: double

The canonical Silver dataset keeps `product_quality_type` as an event-level synthetic telemetry attribute. The same `machine_code` may validly appear with multiple `product_quality_type` values across different events. Silver does not map telemetry into AI4I model feature names yet.

## Contract Validation

Silver validates that each `raw_value` is a JSON object with exactly the expected telemetry fields. Missing required fields and unexpected fields are rejected. Validation uses Spark-native DataFrame expressions, explicit JSON parsing, key inspection, scalar token checks for type strictness, and deterministic rejection reasons.

Spark parsing can convert valid JSON into typed columns, while scalar token checks are used to keep booleans and quoted strings from being trusted as numeric telemetry. The current contract is a flat JSON object with scalar values, so this Spark-native approach is sufficient for the simulator contract without a Python UDF or an additional schema-validation library.

## Data Quality Rules

Silver validates:

- `schema_version` is exactly `1.0`.
- `event_id` is a UUID string.
- `machine_code` is `MCH-0001` through `MCH-0100`.
- `sequence_number` is at least `1`.
- `event_time` is a UTC timestamp ending in `Z`.
- `source` is exactly `synthetic_simulator`.
- `product_quality_type` is one of `L`, `M`, or `H`.
- Sensor values are finite and within the simulator guardrail ranges.
- `process_temperature_k` is greater than `air_temperature_k`.
- Kafka key equals `machine_code`.

Silver does not reject events because sequence numbers repeat across deterministic simulator replays. Global machine-sequence reconstruction belongs to later monitoring or analytics phases.

## Rejection Reasons

Invalid rows receive stable machine-readable reason identifiers. A row may have more than one reason. Reasons include malformed JSON, missing required fields, unexpected fields, invalid identity fields, invalid sensor fields, process-temperature invariant failure, and Kafka key mismatch.

## Quarantine

Invalid rows are written to:

```text
data/silver/quarantine/
```

Quarantine preserves Kafka topic, partition, offset, timestamp, key, raw payload, payload hash, Bronze ingestion timestamp, safely parsed identifying fields when available, and `rejection_reasons`. Invalid raw payloads are not discarded or corrected.

## Business Event Identity

Business event identity is:

```text
event_id
```

Silver canonical telemetry keeps one valid record per `event_id`.

## Kafka Record Identity

Kafka record identity is:

```text
topic + partition + offset
```

This is separate from business identity. Bronze preserves every Kafka record. The same `event_id` at different Kafka offsets is allowed in Bronze.

## event_id Deduplication

Valid records are deduplicated by `event_id`. A repeated valid `event_id` is not invalid telemetry. It is removed from the canonical Silver view and retained in the duplicate audit dataset.

## Canonical Record Selection

Within each `event_id`, the canonical record is selected deterministically by:

1. `source_kafka_timestamp` ascending
2. `source_kafka_topic` ascending
3. `source_kafka_partition` ascending
4. `source_kafka_offset` ascending
5. `payload_sha256` ascending when available

The row with rank `1` becomes canonical.

## Duplicate Audit Dataset

Valid non-canonical records are written to:

```text
data/silver/duplicates/
```

The duplicate audit dataset preserves typed telemetry fields, source Kafka lineage, `payload_sha256`, `duplicate_rank`, and the canonical source Kafka coordinate. It exists so deduplication remains auditable.

## Lineage

Canonical Silver telemetry includes:

- `source_kafka_topic`
- `source_kafka_partition`
- `source_kafka_offset`
- `source_kafka_timestamp`
- `source_kafka_key`
- `bronze_ingested_at`
- `payload_sha256`

The canonical dataset does not retain `raw_value`; Bronze is the raw payload source.

## Accounting Invariants

The Silver phase validates:

```text
Bronze rows = quarantine rows + valid pre-dedup rows
valid pre-dedup rows = canonical Silver rows + duplicate audit rows
Bronze rows = quarantine rows + canonical Silver rows + duplicate audit rows
```

These counts are derived from the actual local Bronze snapshot and are not hard-coded.

## Reproducibility

Silver configuration is tracked in `pipelines/streaming/silver_config.json`. Static architecture and policy are summarized in `reports/streaming/spark_silver_summary.json`. Runtime counts, timestamps, and container identifiers are intentionally excluded from the tracked summary.

Repeated Silver runs over the same unchanged Bronze snapshot should produce the same logical counts and canonical record selections. Binary Parquet file hashes are not used as a determinism requirement because Spark output file layout and metadata can vary.

## Limitations

This is a local portfolio/development architecture. It does not claim production orchestration, table transactions, Delta Lake, Iceberg, HDFS, cloud object storage, or incremental table maintenance. It also does not perform model inference, drift monitoring, anomaly detection, PostgreSQL telemetry writes, API serving, frontend rendering, GenAI behavior, or Databricks integration.

## Downstream Gold Layer

The implemented Gold layer derives descriptive machine summaries, one-minute machine windows, and a fleet-level summary from canonical Silver telemetry. ML inference and anomaly detection remain planned for later dedicated phases.

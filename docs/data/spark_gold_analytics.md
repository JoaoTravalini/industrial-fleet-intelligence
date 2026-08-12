# Spark Gold Fleet Analytics

## Purpose

The Gold telemetry phase creates business-ready descriptive analytics from canonical Silver telemetry. It summarizes accepted telemetry observations by machine, by fixed event-time windows, and across the full fleet snapshot.

## Medallion Position

The implemented local Medallion flow is:

```text
Bronze raw Kafka record fidelity -> Silver typed validated canonical telemetry -> Gold descriptive analytics
```

Bronze preserves raw Kafka records. Silver validates, quarantines, and deduplicates business events. Gold aggregates the accepted canonical Silver records.

## Input

Gold reads only:

```text
data/silver/telemetry/
```

It does not derive analytics from the Silver duplicate audit dataset or the Silver quarantine dataset.

## Execution Model

Gold is a deterministic Spark snapshot rebuild. It runs inside the existing local Spark Docker container using Spark `4.0.4`, `local[2]`, UTC session time, and `spark.sql.shuffle.partitions=3`.

Generated Gold datasets are overwritten on each successful rebuild. Gold is not a Structured Streaming query and does not connect to Kafka, PostgreSQL, or model artifacts.

## Gold Datasets

Gold writes exactly three generated Parquet datasets:

- `data/gold/machine_summary/`
- `data/gold/machine_windows/`
- `data/gold/fleet_summary/`

These are local runtime datasets and are Git-ignored.

## Machine Summary

`machine_summary` has exactly one row per `machine_code` present in canonical Silver. It includes event counts, first/latest event time, `latest_product_quality_type`, deterministic latest-observation telemetry, product-quality event counts, and min/avg/max aggregates for the numeric telemetry fields.

`distinct_event_count` should equal `event_count` because canonical Silver contains one record per valid business `event_id`.

The product-quality distribution columns are:

- `product_quality_type_h_event_count`
- `product_quality_type_l_event_count`
- `product_quality_type_m_event_count`

## Deterministic Latest Observation

Latest machine-observation fields are selected with a deterministic Spark window. Within each machine, rows are ordered by:

1. `event_time` descending
2. `source_kafka_timestamp` descending
3. `source_kafka_topic` descending
4. `source_kafka_partition` descending
5. `source_kafka_offset` descending
6. `event_id` descending

The row ranked `1` supplies `latest_product_quality_type` and the `latest_*` telemetry columns.

## Machine Time Windows

`machine_windows` uses Spark's built-in event-time tumbling window expression with a fixed `1 minute` duration. The grain is:

```text
machine_code + window_start + window_end
```

The Spark window struct is flattened into `window_start` and `window_end`. This snapshot job does not use watermarks, streaming state, rolling windows, sliding windows, or session windows.

Windows are not split by `product_quality_type`. Each machine/window row contains the product-quality event distribution for that window:

- `product_quality_type_h_event_count`
- `product_quality_type_l_event_count`
- `product_quality_type_m_event_count`

## Fleet Summary

`fleet_summary` contains exactly one row with:

```text
fleet_scope = "all_machines"
```

It includes fleet machine count, event count, first/latest event time, numeric min/avg/max aggregates, and product-quality event counts.

## Aggregation Semantics

All aggregates are descriptive statistics derived directly from canonical Silver telemetry. Numeric aggregate columns preserve Spark double precision in Parquet. Presentation formatting and rounding belong to later UI/API layers.

Product-quality counts in `fleet_summary` represent canonical Silver event counts by type, not machine counts.

## Accounting Invariants

Gold validates:

```text
Silver canonical rows = sum(machine_summary.event_count)
Silver canonical rows = sum(machine_windows.event_count)
Silver canonical rows = fleet_summary.event_count
Distinct Silver machines = machine_summary rows
Distinct Silver machines = fleet_summary.machine_count
Per-machine H + L + M event counts = machine_summary.event_count
Per-window H + L + M event counts = machine_windows.event_count
fleet H + L + M event counts = Silver canonical rows
```

## Product Quality Type Semantics

`machine_code` is the stable fictional equipment identity.

`product_quality_type` is an event-level synthetic model-compatible attribute designed for a future explicit mapping to the AI4I `Type` field. It is not an intrinsic stable attribute of a machine, and the same `machine_code` may validly appear with multiple product-quality values across different events.

`latest_product_quality_type` is the `product_quality_type` observed on the deterministic latest canonical telemetry event for a machine. It is selected from the same row that supplies the latest telemetry sensor values.

`product_quality_type_*_event_count` columns are event-distribution counts, not machine classifications. Gold must never document a machine such as `MCH-0001` as being an `H`, `L`, or `M` machine.

Operational `machine_type` and telemetry `product_quality_type` remain different concepts. Operational `machine_type` describes fictional equipment categories in PostgreSQL seed data; `product_quality_type` is a synthetic telemetry event attribute.

## Reproducibility

Gold configuration is tracked in `pipelines/batch/gold_config.json`. Static architecture and policy are summarized in `reports/streaming/spark_gold_summary.json`. Runtime row counts, rebuild timestamps, local paths, and container identifiers are intentionally excluded from the tracked summary.

Repeated Gold rebuilds over unchanged Silver should produce the same logical counts, aggregate values, and deterministic selection hashes. Raw Parquet file hashes are not used because Spark file layout and metadata can vary between writes.

## Limitations

Gold intentionally does not contain model predictions, anomaly scores, health scores, risk scores, operational statuses, maintenance recommendations, availability, downtime, or OEE. The current telemetry snapshot does not support those metrics truthfully.

## Future ML Integration

A later dedicated phase may map `product_quality_type` and telemetry measurements into a model-compatible feature adapter. This Gold phase does not perform inference and does not compare telemetry to the AI4I training distribution.

## Future API Consumption

A future API and dashboard can consume these curated datasets or equivalent operational projections. This phase does not implement FastAPI routes, frontend components, or serving logic.

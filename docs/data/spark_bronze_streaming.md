# Spark Structured Streaming Bronze Ingestion

## Purpose

This phase implements the local Kafka to Spark Structured Streaming to Bronze ingestion path for synthetic telemetry. Bronze is the raw streaming ingestion layer: it preserves Kafka records with ingestion metadata and does not apply business cleanup or modeling logic.

## Local Spark Runtime

Spark runs locally in Docker through the single `spark` Compose service. The service is a local execution container, not a Spark standalone cluster. It does not create a Spark master, Spark worker, History Server, Hadoop, HDFS, Hive metastore, Delta Lake, Iceberg, or external storage service.

## Spark Version

The pinned Spark runtime is Apache Spark `4.0.4` using the image `apache/spark:4.0.4-scala2.13-java17-python3-ubuntu`. Host-side project commands still use `.venv\Scripts\python.exe`, but PySpark execution is provided by the Docker image. The host project does not add `pyspark` as a Python dependency.

## Kafka Connector

Spark uses the official Structured Streaming Kafka connector coordinate:

```text
org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4
```

The connector is passed through `spark-submit --packages`, allowing Spark to resolve compatible transitive Kafka dependencies. Downloaded connector JARs and Ivy/Maven caches are runtime state and must not be committed. The wrapper directs Ivy to `/tmp/spark-ivy` inside the Spark container so package resolution does not write into the project. The first execution may take longer while Spark downloads the pinned connector dependencies.

## Streaming Architecture

The implemented development flow is:

```text
Telemetry Simulator -> Kafka -> Spark Structured Streaming -> Bronze Parquet -> Silver validation and deduplication -> Gold descriptive analytics
```

Spark uses `local[2]` execution inside the container. This demonstrates real Structured Streaming semantics without introducing unnecessary distributed master/worker infrastructure in this phase.

## Kafka Source

Inside Docker, Spark reads Kafka through `kafka:29092`. The source topic is `industrial.telemetry.v1`. For a brand-new query without a checkpoint, Spark starts from `earliest` offsets. Data loss is configured to fail clearly with `failOnDataLoss=true`.

## Bronze Schema

Bronze stores Kafka records with these columns:

- `kafka_topic`
- `kafka_partition`
- `kafka_offset`
- `kafka_timestamp`
- `kafka_key`
- `raw_value`
- `bronze_ingested_at`
- `payload_sha256`

The `payload_sha256` column is a SHA-256 integrity value derived from `raw_value`.

## Raw Payload Preservation

Kafka telemetry values are UTF-8 JSON. Spark casts the Kafka byte array to a string and stores it as `raw_value`. Bronze does not flatten telemetry sensor fields, validate business constraints, normalize units, quarantine malformed payloads, or rewrite the event. Parsing, validation, valid-record deduplication, duplicate auditing, and quarantine handling are implemented downstream in the Silver telemetry layer.

## Kafka Metadata

Spark preserves Kafka metadata from the source DataFrame: topic, partition, offset, timestamp, key, and value. These are renamed into explicit Bronze column names. Kafka metadata is not fabricated.

## Parquet Storage

Bronze is stored as local Parquet files at:

```text
data/bronze/telemetry/
```

This path is generated runtime data and is Git-ignored. The `data/bronze/.gitkeep` placeholder remains tracked to preserve repository structure.

## Checkpointing

Structured Streaming checkpoints are stored at:

```text
data/checkpoints/spark/bronze_telemetry/
```

Spark uses this checkpoint to preserve Kafka offset progress across repeated executions of the same query. Offsets are not stored manually elsewhere. Deleting the checkpoint intentionally changes replay behavior and may cause Spark to read earlier Kafka offsets again.

## Starting Offsets

The canonical pipeline uses `startingOffsets=earliest`. This applies when the query is brand new and no checkpoint exists. Once the checkpoint exists, Spark resumes from checkpointed offsets and `startingOffsets` no longer defines normal continuation behavior.

## Available-Now Execution

The local runner uses Spark Structured Streaming available-now mode. It processes all currently available Kafka records, writes them to Bronze, and then stops cleanly. This is the default development behavior for this phase. An endless streaming daemon is intentionally not implemented yet.

## Duplicate Event Semantics

Bronze does not deduplicate logical telemetry events. The simulator can publish the same business `event_id` again, and Kafka will assign a new offset. Bronze preserves both Kafka records.

## Kafka Coordinate Identity

Bronze uniqueness is based on the Kafka coordinate:

```text
topic + partition + offset
```

This is different from business event identity such as `event_id`. Duplicate Kafka coordinates indicate an invalid Bronze dataset. Duplicate `event_id` values at different Kafka offsets are valid Bronze records and are handled explicitly in the implemented Silver layer.

## Reproducibility

Spark configuration is tracked in `pipelines/streaming/spark_config.json`. Static architecture and configuration are summarized in `reports/streaming/spark_bronze_summary.json`. Runtime offsets, batch IDs, timestamps, container IDs, and record counts that change as Kafka grows are intentionally excluded from the tracked summary.

## Local Development

Start Kafka and Spark:

```powershell
docker compose up -d kafka spark
```

Run available-now ingestion from the host:

```powershell
.\.venv\Scripts\python.exe scripts/run_spark_bronze_docker.py
```

Inspect Bronze from inside Spark:

```powershell
docker compose exec -T spark /opt/spark/bin/spark-submit /workspace/scripts/inspect_spark_bronze.py
```

Run the end-to-end validator:

```powershell
.\.venv\Scripts\python.exe scripts/check_spark_bronze.py
```

## Limitations

This phase does not implement ML inference, anomaly detection, drift monitoring, PostgreSQL telemetry writes, FastAPI routes, frontend components, GenAI behavior, Databricks integration, HDFS, or cloud storage.

## Downstream Silver Layer

The implemented Silver layer parses `raw_value`, validates the telemetry contract and business rules, quarantines invalid records, deduplicates valid `event_id` values explicitly, and prepares typed canonical telemetry records for downstream analytics and future ML workflows. The implemented Gold layer derives descriptive machine, window, and fleet analytics from canonical Silver telemetry.

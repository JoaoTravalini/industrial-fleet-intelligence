# Local Kafka Streaming

This document describes the implemented local Apache Kafka infrastructure for deterministic synthetic telemetry streaming. It is local-first, uses only Docker Desktop with Linux containers, and does not require paid services, cloud accounts, ZooKeeper, Confluent Platform images, Bitnami images, Redpanda, Schema Registry, PostgreSQL telemetry writes, or model inference. Kafka now feeds the implemented local Spark Structured Streaming Bronze ingestion path.

## Implemented Scope

- Single-node Apache Kafka broker using the official JVM image `apache/kafka:4.3.1`.
- KRaft mode with one combined broker and controller, no ZooKeeper.
- Host access on `localhost:9092` through `127.0.0.1:${KAFKA_PORT:-9092}:9092`.
- Docker-network access for future Compose clients on `kafka:29092`.
- Controller listener on `29093` inside the container.
- Telemetry topic `industrial.telemetry.v1` with 3 partitions and replication factor 1.
- Topic auto-creation disabled in the broker configuration.
- Deterministic UTF-8 JSON payloads using telemetry schema version `1.0`.
- Message key policy: UTF-8 encoded `machine_code`, never `event_id` and never null.
- Idempotent topic setup through `scripts/setup_kafka.py`.
- Finite telemetry producer through `scripts/produce_telemetry_kafka.py`.
- Finite telemetry consumer through `scripts/consume_telemetry_kafka.py`.
- Integration validation through `scripts/check_kafka.py`.
- Downstream local Spark Structured Streaming ingestion into Bronze Parquet through `scripts/run_spark_bronze_docker.py` and `scripts/check_spark_bronze.py`.

## Configuration Files

Kafka runtime placeholders are documented in `.env.example`:

```text
KAFKA_PORT=9092
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TELEMETRY_TOPIC=industrial.telemetry.v1
```

The static streaming configuration is tracked in `services/streaming/kafka_config.json`. It contains only local development configuration and no secrets, credentials, host-specific absolute paths, runtime group IDs, offsets, or event IDs.

The deterministic integration summary is tracked at `reports/streaming/kafka_integration_summary.json`. It records static configuration only and deliberately excludes runtime consumer group IDs, offsets, timestamps, and smoke-run identifiers.

## Install The Streaming Client Dependency

Install declared project dependency groups only into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,mlops,explainability,streaming]"
```

The streaming group currently adds exactly `confluent-kafka==2.15.0`.

Confirm the installed client version:

```powershell
.\.venv\Scripts\python.exe -c "import confluent_kafka; print(confluent_kafka.__version__)"
```

## Start Kafka

Ensure Docker Desktop is running with Linux containers. From the repository root, start Kafka only:

```powershell
docker compose up -d kafka
```

Check service status:

```powershell
docker compose ps
```

The Compose health check uses Kafka's own CLI inside the container:

```text
/opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

## Create Or Verify The Topic

Run the topic setup script after the Kafka container is healthy:

```powershell
.\.venv\Scripts\python.exe scripts/setup_kafka.py
```

The script is idempotent. Running it again reuses the existing topic if it already has 3 partitions and replication factor 1. If an incompatible topic already exists, the script fails and does not mutate it.

## Validate The Kafka Integration

Run the integration validator:

```powershell
.\.venv\Scripts\python.exe scripts/check_kafka.py
```

The validator checks Compose configuration, container health, broker connectivity, topic shape, a deterministic smoke produce/consume batch, key-to-partition consistency, and per-machine ordering. It does not delete topics, compact data, or write telemetry into PostgreSQL.

## Produce Telemetry

Produce the canonical finite batch with the default simulator settings:

```powershell
.\.venv\Scripts\python.exe scripts/produce_telemetry_kafka.py
```

Produce a smaller deterministic batch:

```powershell
.\.venv\Scripts\python.exe scripts/produce_telemetry_kafka.py --machines 2 --events-per-machine 3 --seed 42 --start-time 2026-01-01T00:00:00Z --interval-seconds 5
```

The producer does not sleep between events. It emits a finite batch and reports broker, topic, attempted count, delivered count, failure count, machine range, and event time range.

## Consume Telemetry

Consume a finite sample from the beginning of the topic with an explicit group ID:

```powershell
.\.venv\Scripts\python.exe scripts/consume_telemetry_kafka.py --group-id manual-validation --from-beginning --max-messages 5 --timeout-seconds 10
```

Each consumed line is compact JSON with Kafka metadata under `kafka` and the telemetry payload under `event`. Consumer auto-commit is disabled.

## Logs And Lifecycle

View Kafka logs:

```powershell
docker compose logs kafka
```

Stop Kafka without deleting data:

```powershell
docker compose stop kafka
```

Start Kafka again:

```powershell
docker compose start kafka
```

Stop and remove containers while preserving named volumes:

```powershell
docker compose down
```

Delete local Kafka and PostgreSQL named volumes only when intentionally resetting all local service state:

```powershell
docker compose down -v
```

## Current Boundaries

Kafka currently transports complete synthetic telemetry events into the implemented downstream flow: Telemetry Simulator -> Kafka -> Spark Structured Streaming -> Bronze Parquet -> Silver telemetry -> Gold descriptive analytics. ML inference on Kafka messages, anomaly detection, drift monitoring, FastAPI routes, frontend components, Ollama/GenAI behavior, and Databricks integration are not implemented.

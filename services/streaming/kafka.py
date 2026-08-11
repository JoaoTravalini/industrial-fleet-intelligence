"""Kafka utilities for local synthetic telemetry streaming."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.simulator import telemetry

CONFIG_RELATIVE_PATH = Path("services") / "streaming" / "kafka_config.json"
SUMMARY_RELATIVE_PATH = Path("reports") / "streaming" / "kafka_integration_summary.json"
EXPECTED_TOPIC = "industrial.telemetry.v1"
EXPECTED_PARTITION_COUNT = 3
EXPECTED_REPLICATION_FACTOR = 1
EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_MESSAGE_KEY = "machine_code"
EXPECTED_SERIALIZATION = "utf-8 JSON"
EXPECTED_DOCKER_IMAGE = "apache/kafka:4.3.1"
EXPECTED_KAFKA_VERSION = "4.3.1"
EXPECTED_DEPLOYMENT_MODE = "single-node-kraft"
SMOKE_MACHINE_COUNT = 2
SMOKE_EVENTS_PER_MACHINE = 3


class KafkaConfigError(ValueError):
    """Raised when local Kafka configuration is missing or incompatible."""


class KafkaStreamingError(RuntimeError):
    """Raised when Kafka streaming validation or runtime operations fail."""


@dataclass(frozen=True)
class KafkaConfig:
    """Static local Kafka configuration for synthetic telemetry."""

    bootstrap_servers_host: str
    bootstrap_servers_docker: str
    telemetry_topic: str
    partition_count: int
    replication_factor: int
    message_key: str
    serialization: str
    schema_version: str
    docker_image: str
    kafka_version: str
    deployment_mode: str


@dataclass(frozen=True)
class TopicDescription:
    """Kafka topic shape relevant to this project."""

    name: str
    partitions: int
    replication_factor: int


@dataclass(frozen=True)
class DeliveryMetadata:
    """Metadata returned by Kafka after a successful delivery."""

    topic: str
    partition: int
    offset: int
    key: str
    event_id: str


@dataclass(frozen=True)
class ProduceResult:
    """Finite batch delivery result for synthetic telemetry events."""

    attempted: int
    delivered: int
    failed: int
    failures: tuple[str, ...]
    deliveries: tuple[DeliveryMetadata, ...]


@dataclass(frozen=True)
class KafkaMessageMetadata:
    """Kafka metadata kept separate from the telemetry event payload."""

    topic: str
    partition: int
    offset: int
    key: str


@dataclass(frozen=True)
class ConsumedTelemetryRecord:
    """Validated consumed telemetry event and its Kafka metadata."""

    kafka: KafkaMessageMetadata
    event: dict[str, Any]


@dataclass(frozen=True)
class ExpectedRecordValidation:
    """Summary of consumed-record validation against a deterministic expectation."""

    matched_event_count: int
    machine_partitions: dict[str, int]


class DeliveryTracker:
    """Collect delivery callbacks from confluent-kafka producer.flush()."""

    def __init__(self) -> None:
        self.deliveries: list[DeliveryMetadata] = []
        self.failures: list[str] = []

    def callback(self, error: Any, message: Any) -> None:
        if error is not None:
            self.failures.append(str(error))
            return

        key = decode_message_key(message.key())
        event = deserialize_telemetry_event(message.value())
        event_id = str(event["event_id"])
        self.deliveries.append(
            DeliveryMetadata(
                topic=str(message.topic()),
                partition=int(message.partition()),
                offset=int(message.offset()),
                key=key,
                event_id=event_id,
            )
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def load_kafka_config(path: Path | None = None) -> KafkaConfig:
    config_file = path or config_path()
    try:
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KafkaConfigError(f"Kafka config file not found: {config_file}") from exc
    except json.JSONDecodeError as exc:
        raise KafkaConfigError(f"Kafka config file is not valid JSON: {config_file}") from exc
    if not isinstance(raw_config, dict):
        raise KafkaConfigError("Kafka config must be a JSON object.")
    return parse_kafka_config(raw_config)


def parse_kafka_config(raw_config: Mapping[str, Any]) -> KafkaConfig:
    required_keys = {
        "bootstrap_servers_host",
        "bootstrap_servers_docker",
        "deployment_mode",
        "docker_image",
        "kafka_version",
        "message_key",
        "partition_count",
        "replication_factor",
        "schema_version",
        "serialization",
        "telemetry_topic",
    }
    actual_keys = set(raw_config)
    missing = sorted(required_keys - actual_keys)
    unknown = sorted(actual_keys - required_keys)
    if missing:
        raise KafkaConfigError("Missing Kafka config key(s): " + ", ".join(missing))
    if unknown:
        raise KafkaConfigError("Unknown Kafka config key(s): " + ", ".join(unknown))

    config = KafkaConfig(
        bootstrap_servers_host=require_text(raw_config, "bootstrap_servers_host"),
        bootstrap_servers_docker=require_text(raw_config, "bootstrap_servers_docker"),
        telemetry_topic=require_text(raw_config, "telemetry_topic"),
        partition_count=require_int(raw_config, "partition_count"),
        replication_factor=require_int(raw_config, "replication_factor"),
        message_key=require_text(raw_config, "message_key"),
        serialization=require_text(raw_config, "serialization"),
        schema_version=require_text(raw_config, "schema_version"),
        docker_image=require_text(raw_config, "docker_image"),
        kafka_version=require_text(raw_config, "kafka_version"),
        deployment_mode=require_text(raw_config, "deployment_mode"),
    )
    validate_kafka_config(config)
    return config


def require_text(raw_config: Mapping[str, Any], key: str) -> str:
    value = raw_config[key]
    if not isinstance(value, str) or not value.strip():
        raise KafkaConfigError(f"Kafka config key {key} must be a non-empty string.")
    return value


def require_int(raw_config: Mapping[str, Any], key: str) -> int:
    value = raw_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise KafkaConfigError(f"Kafka config key {key} must be an integer.")
    return value


def validate_kafka_config(config: KafkaConfig) -> None:
    if config.telemetry_topic != EXPECTED_TOPIC:
        raise KafkaConfigError(f"Kafka telemetry topic must be {EXPECTED_TOPIC}.")
    if config.partition_count != EXPECTED_PARTITION_COUNT:
        raise KafkaConfigError("Kafka telemetry topic must use exactly 3 partitions.")
    if config.replication_factor != EXPECTED_REPLICATION_FACTOR:
        raise KafkaConfigError("Kafka telemetry topic must use replication factor 1 locally.")
    if config.message_key != EXPECTED_MESSAGE_KEY:
        raise KafkaConfigError("Kafka telemetry message key must be machine_code.")
    if config.serialization != EXPECTED_SERIALIZATION:
        raise KafkaConfigError("Kafka telemetry serialization must be utf-8 JSON.")
    if config.schema_version != EXPECTED_SCHEMA_VERSION:
        raise KafkaConfigError("Kafka telemetry schema version must be 1.0.")
    if config.docker_image != EXPECTED_DOCKER_IMAGE:
        raise KafkaConfigError("Kafka Docker image must be apache/kafka:4.3.1.")
    if config.kafka_version != EXPECTED_KAFKA_VERSION:
        raise KafkaConfigError("Kafka version must be 4.3.1.")
    if config.deployment_mode != EXPECTED_DEPLOYMENT_MODE:
        raise KafkaConfigError("Kafka deployment mode must be single-node-kraft.")
    for name, value in (
        ("bootstrap_servers_host", config.bootstrap_servers_host),
        ("bootstrap_servers_docker", config.bootstrap_servers_docker),
    ):
        if "\\" in value or value.startswith("/"):
            raise KafkaConfigError(f"{name} must not be an absolute filesystem path.")


def build_integration_summary(config: KafkaConfig) -> dict[str, Any]:
    return {
        "bootstrap_servers_docker": config.bootstrap_servers_docker,
        "bootstrap_servers_host": config.bootstrap_servers_host,
        "deployment_mode": config.deployment_mode,
        "docker_image": config.docker_image,
        "kafka_version": config.kafka_version,
        "message_key_policy": "UTF-8 machine_code; never event_id or null",
        "partition_count": config.partition_count,
        "replication_factor": config.replication_factor,
        "serialization": config.serialization,
        "smoke_expected_event_count": SMOKE_MACHINE_COUNT * SMOKE_EVENTS_PER_MACHINE,
        "smoke_expected_events_per_machine": SMOKE_EVENTS_PER_MACHINE,
        "smoke_expected_machine_count": SMOKE_MACHINE_COUNT,
        "telemetry_schema_version": config.schema_version,
        "telemetry_topic": config.telemetry_topic,
    }


def write_integration_summary(root: Path | None, config: KafkaConfig) -> Path:
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(build_integration_summary(config), indent=2, sort_keys=False) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def create_admin_client(config: KafkaConfig, bootstrap_servers: str | None = None) -> Any:
    from confluent_kafka.admin import AdminClient

    return AdminClient(
        {
            "bootstrap.servers": bootstrap_servers or config.bootstrap_servers_host,
            "socket.timeout.ms": 10000,
            "request.timeout.ms": 15000,
            "broker.address.family": "v4",
        }
    )


def check_broker_connectivity(admin_client: Any, timeout_seconds: int = 10) -> None:
    try:
        admin_client.list_topics(timeout=timeout_seconds)
    except Exception as exc:
        raise KafkaStreamingError(f"Kafka broker connectivity failed: {exc}") from exc


def topic_error(topic_metadata: Any) -> Any:
    error = getattr(topic_metadata, "error", None)
    return error() if callable(error) else error


def get_topic_description(
    admin_client: Any,
    topic_name: str,
    timeout_seconds: int = 10,
) -> TopicDescription | None:
    try:
        metadata = admin_client.list_topics(topic=topic_name, timeout=timeout_seconds)
    except Exception as exc:
        raise KafkaStreamingError(f"Could not retrieve topic metadata: {exc}") from exc

    topic_metadata = metadata.topics.get(topic_name)
    if topic_metadata is None:
        return None

    error = topic_error(topic_metadata)
    if error is not None:
        text = str(error).lower()
        if "unknown topic" in text or "unknown_topic" in text:
            return None
        raise KafkaStreamingError(f"Topic metadata returned an error: {error}")

    partition_metadata = getattr(topic_metadata, "partitions", {})
    if not partition_metadata:
        raise KafkaStreamingError(f"Topic {topic_name} has no partition metadata.")

    replication_factors = {
        len(getattr(partition, "replicas", []) or []) for partition in partition_metadata.values()
    }
    if len(replication_factors) != 1:
        raise KafkaStreamingError(f"Topic {topic_name} has inconsistent replication factors.")

    return TopicDescription(
        name=topic_name,
        partitions=len(partition_metadata),
        replication_factor=replication_factors.pop(),
    )


def validate_topic_description(description: TopicDescription, config: KafkaConfig) -> None:
    if description.name != config.telemetry_topic:
        raise KafkaConfigError(
            f"Expected topic {config.telemetry_topic}, found {description.name}."
        )
    if description.partitions != config.partition_count:
        raise KafkaConfigError(
            f"Topic {description.name} has {description.partitions} partitions; "
            f"expected {config.partition_count}."
        )
    if description.replication_factor != config.replication_factor:
        raise KafkaConfigError(
            f"Topic {description.name} has replication factor {description.replication_factor}; "
            f"expected {config.replication_factor}."
        )


def create_telemetry_topic(
    admin_client: Any,
    config: KafkaConfig,
    timeout_seconds: int = 20,
) -> tuple[TopicDescription, str]:
    existing = get_topic_description(admin_client, config.telemetry_topic, timeout_seconds)
    if existing is not None:
        validate_topic_description(existing, config)
        return existing, "reused"

    from confluent_kafka.admin import NewTopic

    topic = NewTopic(
        topic=config.telemetry_topic,
        num_partitions=config.partition_count,
        replication_factor=config.replication_factor,
    )
    futures = admin_client.create_topics([topic], request_timeout=timeout_seconds)
    future = futures[config.telemetry_topic]
    try:
        future.result(timeout=timeout_seconds)
    except Exception as exc:
        text = str(exc).lower()
        if "already exists" not in text and "topic_exists" not in text:
            raise KafkaStreamingError(f"Could not create Kafka topic: {exc}") from exc

    created = get_topic_description(admin_client, config.telemetry_topic, timeout_seconds)
    if created is None:
        raise KafkaStreamingError(f"Kafka topic {config.telemetry_topic} was not created.")
    validate_topic_description(created, config)
    return created, "created"


def create_producer(config: KafkaConfig, bootstrap_servers: str | None = None) -> Any:
    from confluent_kafka import Producer

    return Producer(
        {
            "bootstrap.servers": bootstrap_servers or config.bootstrap_servers_host,
            "client.id": "industrial-fleet-telemetry-producer",
            "acks": "all",
            "broker.address.family": "v4",
        }
    )


def create_consumer(
    config: KafkaConfig,
    group_id: str,
    *,
    auto_offset_reset: str = "latest",
    bootstrap_servers: str | None = None,
) -> Any:
    if not group_id.strip():
        raise KafkaConfigError("Kafka consumer group.id must be a non-empty string.")
    if auto_offset_reset not in {"earliest", "latest"}:
        raise KafkaConfigError("auto.offset.reset must be earliest or latest.")

    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers or config.bootstrap_servers_host,
            "group.id": group_id,
            "client.id": "industrial-fleet-telemetry-consumer",
            "enable.auto.commit": False,
            "auto.offset.reset": auto_offset_reset,
            "broker.address.family": "v4",
        }
    )
    consumer.subscribe([config.telemetry_topic])
    return consumer


def encode_message_key(event: Mapping[str, Any]) -> bytes:
    machine_code = telemetry.validate_machine_code(event.get("machine_code"))
    return machine_code.encode("utf-8")


def decode_message_key(key: bytes | str | None) -> str:
    if key is None:
        raise KafkaStreamingError("Kafka telemetry message key must not be null.")
    if isinstance(key, str):
        decoded = key
    elif isinstance(key, bytes):
        try:
            decoded = key.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KafkaStreamingError("Kafka telemetry message key must be UTF-8.") from exc
    else:
        raise KafkaStreamingError("Kafka telemetry message key must be bytes or string.")
    return telemetry.validate_machine_code(decoded)


def validate_key_matches_event(key: bytes | str | None, event: Mapping[str, Any]) -> str:
    decoded_key = decode_message_key(key)
    machine_code = telemetry.validate_machine_code(event.get("machine_code"))
    if decoded_key != machine_code:
        raise KafkaStreamingError("Kafka message key must match telemetry machine_code.")
    return decoded_key


def serialize_telemetry_event(event: Mapping[str, Any]) -> bytes:
    return telemetry.serialize_event(event).encode("utf-8")


def deserialize_telemetry_event(payload: bytes | str | None) -> dict[str, Any]:
    if payload is None:
        raise KafkaStreamingError("Kafka telemetry payload must not be null.")
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KafkaStreamingError("Kafka telemetry payload must be UTF-8 JSON.") from exc
    else:
        raise KafkaStreamingError("Kafka telemetry payload must be bytes or string.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KafkaStreamingError("Kafka telemetry payload must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise KafkaStreamingError("Kafka telemetry payload must be a JSON object.")
    telemetry.validate_event(parsed)
    return parsed


def produce_telemetry_events(
    producer: Any,
    events: Sequence[Mapping[str, Any]],
    config: KafkaConfig,
    *,
    flush_timeout_seconds: int = 30,
) -> ProduceResult:
    tracker = DeliveryTracker()
    for event in events:
        key = encode_message_key(event)
        value = serialize_telemetry_event(event)
        try:
            producer.produce(
                config.telemetry_topic,
                key=key,
                value=value,
                on_delivery=tracker.callback,
            )
            producer.poll(0)
        except BufferError as exc:
            tracker.failures.append(f"Kafka producer buffer error: {exc}")
        except Exception as exc:
            raise KafkaStreamingError(f"Could not produce telemetry event: {exc}") from exc

    try:
        remaining = producer.flush(flush_timeout_seconds)
    except Exception as exc:
        raise KafkaStreamingError(f"Could not flush Kafka producer: {exc}") from exc
    if remaining:
        tracker.failures.append(f"{remaining} message(s) were not delivered before flush timeout.")

    return ProduceResult(
        attempted=len(events),
        delivered=len(tracker.deliveries),
        failed=len(tracker.failures),
        failures=tuple(tracker.failures),
        deliveries=tuple(tracker.deliveries),
    )


def consumed_record_from_message(message: Any) -> ConsumedTelemetryRecord:
    error = message.error()
    if error is not None:
        raise KafkaStreamingError(f"Kafka consumer returned an error: {error}")

    event = deserialize_telemetry_event(message.value())
    key = validate_key_matches_event(message.key(), event)
    metadata = KafkaMessageMetadata(
        topic=str(message.topic()),
        partition=int(message.partition()),
        offset=int(message.offset()),
        key=key,
    )
    return ConsumedTelemetryRecord(kafka=metadata, event=event)


def poll_consumed_records(
    consumer: Any,
    *,
    max_messages: int,
    timeout_seconds: float,
    close: bool = True,
) -> list[ConsumedTelemetryRecord]:
    if max_messages < 1:
        raise KafkaConfigError("max_messages must be positive.")
    if timeout_seconds <= 0:
        raise KafkaConfigError("timeout_seconds must be positive.")

    deadline = time.monotonic() + timeout_seconds
    records: list[ConsumedTelemetryRecord] = []
    try:
        while len(records) < max_messages and time.monotonic() < deadline:
            remaining = max(0.05, min(1.0, deadline - time.monotonic()))
            message = consumer.poll(remaining)
            if message is None:
                continue
            records.append(consumed_record_from_message(message))
    finally:
        if close:
            consumer.close()
    return records


def expected_events_by_id(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for event in events:
        telemetry.validate_event(event)
        event_id = str(event["event_id"])
        fingerprint = telemetry.serialize_event(event)
        if event_id in expected:
            if fingerprints[event_id] != fingerprint:
                raise KafkaStreamingError(
                    f"Expected telemetry event_id {event_id} appears with different payloads."
                )
            continue
        expected[event_id] = dict(event)
        fingerprints[event_id] = fingerprint
    if not expected:
        raise KafkaStreamingError("Expected telemetry events must not be empty.")
    return expected


def validate_partition_consistency(records: Sequence[ConsumedTelemetryRecord]) -> dict[str, int]:
    machine_partitions: dict[str, int] = {}
    for record in records:
        machine_code = telemetry.validate_machine_code(record.event.get("machine_code"))
        previous_partition = machine_partitions.get(machine_code)
        if previous_partition is None:
            machine_partitions[machine_code] = record.kafka.partition
        elif previous_partition != record.kafka.partition:
            raise KafkaStreamingError(
                f"Machine {machine_code} appeared on partitions "
                f"{previous_partition} and {record.kafka.partition}."
            )
    return machine_partitions


def validate_per_machine_order(records: Sequence[ConsumedTelemetryRecord]) -> None:
    seen_by_machine: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        machine_code = telemetry.validate_machine_code(record.event.get("machine_code"))
        sequence_number = int(record.event["sequence_number"])
        seen_by_machine.setdefault(machine_code, []).append((sequence_number, record.kafka.offset))

    for machine_code, sequence_offsets in seen_by_machine.items():
        sequence_numbers = [sequence for sequence, _offset in sequence_offsets]
        offsets = [offset for _sequence, offset in sequence_offsets]
        if sequence_numbers != sorted(sequence_numbers):
            raise KafkaStreamingError(f"Machine {machine_code} events were consumed out of order.")
        if offsets != sorted(offsets):
            raise KafkaStreamingError(f"Machine {machine_code} offsets were not monotonic.")


def validate_expected_records(
    records: Sequence[ConsumedTelemetryRecord],
    expected_events: Iterable[Mapping[str, Any]],
) -> ExpectedRecordValidation:
    expected = expected_events_by_id(expected_events)
    matched: dict[str, ConsumedTelemetryRecord] = {}
    matching_records: list[ConsumedTelemetryRecord] = []

    for record in records:
        telemetry.validate_event(record.event)
        validate_key_matches_event(record.kafka.key, record.event)
        event_id = str(record.event["event_id"])
        expected_event = expected.get(event_id)
        if expected_event is None:
            continue
        if record.event != expected_event:
            raise KafkaStreamingError(f"Telemetry event {event_id} does not match expectation.")
        if event_id in matched:
            continue
        matched[event_id] = record
        matching_records.append(record)

    missing = [event_id for event_id in expected if event_id not in matched]
    if missing:
        raise KafkaStreamingError("Missing expected telemetry event_id(s): " + ", ".join(missing))

    machine_partitions = validate_partition_consistency(matching_records)
    validate_per_machine_order(matching_records)
    return ExpectedRecordValidation(
        matched_event_count=len(matched),
        machine_partitions=machine_partitions,
    )


def consume_expected_records(
    consumer: Any,
    expected_events: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: float,
) -> list[ConsumedTelemetryRecord]:
    expected = expected_events_by_id(expected_events)
    deadline = time.monotonic() + timeout_seconds
    matched: dict[str, ConsumedTelemetryRecord] = {}
    records: list[ConsumedTelemetryRecord] = []
    try:
        while len(matched) < len(expected) and time.monotonic() < deadline:
            remaining = max(0.05, min(1.0, deadline - time.monotonic()))
            message = consumer.poll(remaining)
            if message is None:
                continue
            record = consumed_record_from_message(message)
            event_id = str(record.event["event_id"])
            if event_id not in expected or event_id in matched:
                continue
            matched[event_id] = record
            records.append(record)
    finally:
        consumer.close()

    validate_expected_records(records, expected.values())
    return records


def record_to_json_object(record: ConsumedTelemetryRecord) -> dict[str, Any]:
    return {
        "kafka": {
            "topic": record.kafka.topic,
            "partition": record.kafka.partition,
            "offset": record.kafka.offset,
            "key": record.kafka.key,
        },
        "event": record.event,
    }

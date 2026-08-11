from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from services.simulator import telemetry
from services.streaming import kafka


class FakeProducer:
    def __init__(self, *, delivery_error: object | None = None, flush_remaining: int = 0) -> None:
        self.delivery_error = delivery_error
        self.flush_remaining = flush_remaining
        self.produced: list[dict[str, object]] = []

    def produce(self, topic: str, *, key: bytes, value: bytes, on_delivery: object) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})
        message = FakeMessage(
            topic=topic, partition=1, offset=len(self.produced) - 1, key=key, value=value
        )
        on_delivery(self.delivery_error, message)

    def poll(self, timeout: float) -> None:
        assert timeout == 0

    def flush(self, timeout: int) -> int:
        assert timeout > 0
        return self.flush_remaining


class FakeMessage:
    def __init__(
        self,
        *,
        topic: str = kafka.EXPECTED_TOPIC,
        partition: int = 0,
        offset: int = 0,
        key: bytes | str | None = None,
        value: bytes | str | None = None,
        error: object | None = None,
    ) -> None:
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._key = key
        self._value = value
        self._error = error

    def error(self) -> object | None:
        return self._error

    def key(self) -> bytes | str | None:
        return self._key

    def value(self) -> bytes | str | None:
        return self._value

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage]) -> None:
        self.messages = list(messages)
        self.closed = False

    def poll(self, timeout: float) -> FakeMessage | None:
        assert timeout > 0
        if self.messages:
            return self.messages.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


def sample_events() -> list[dict[str, object]]:
    config = telemetry.SimulatorConfig(machine_count=2, events_per_machine=3)
    return telemetry.generate_events(config)


def sample_config() -> kafka.KafkaConfig:
    return kafka.parse_kafka_config(
        {
            "bootstrap_servers_docker": "kafka:29092",
            "bootstrap_servers_host": "localhost:9092",
            "deployment_mode": "single-node-kraft",
            "docker_image": "apache/kafka:4.3.1",
            "kafka_version": "4.3.1",
            "message_key": "machine_code",
            "partition_count": 3,
            "replication_factor": 1,
            "schema_version": "1.0",
            "serialization": "utf-8 JSON",
            "telemetry_topic": "industrial.telemetry.v1",
        }
    )


def record_for(
    event: dict[str, object], *, partition: int, offset: int
) -> kafka.ConsumedTelemetryRecord:
    machine_code = str(event["machine_code"])
    return kafka.ConsumedTelemetryRecord(
        kafka=kafka.KafkaMessageMetadata(
            topic=kafka.EXPECTED_TOPIC,
            partition=partition,
            offset=offset,
            key=machine_code,
        ),
        event=dict(event),
    )


def test_parse_kafka_config_accepts_expected_static_values() -> None:
    config = sample_config()

    assert config.telemetry_topic == "industrial.telemetry.v1"
    assert config.partition_count == 3
    assert config.replication_factor == 1
    assert config.message_key == "machine_code"
    assert config.docker_image == "apache/kafka:4.3.1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partition_count", 1, "3 partitions"),
        ("replication_factor", 2, "replication factor 1"),
        ("message_key", "event_id", "machine_code"),
        ("serialization", "json", "utf-8 JSON"),
    ],
)
def test_parse_kafka_config_rejects_incompatible_values(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = dict(sample_config().__dict__)
    raw[field] = value

    with pytest.raises(kafka.KafkaConfigError, match=message):
        kafka.parse_kafka_config(raw)


def test_validate_topic_description_requires_expected_shape() -> None:
    config = sample_config()
    kafka.validate_topic_description(
        kafka.TopicDescription(name=config.telemetry_topic, partitions=3, replication_factor=1),
        config,
    )

    with pytest.raises(kafka.KafkaConfigError, match="partitions"):
        kafka.validate_topic_description(
            kafka.TopicDescription(name=config.telemetry_topic, partitions=1, replication_factor=1),
            config,
        )


def test_telemetry_serialization_is_utf8_json_and_round_trips() -> None:
    event = sample_events()[0]
    payload = kafka.serialize_telemetry_event(event)

    assert isinstance(payload, bytes)
    assert json.loads(payload.decode("utf-8")) == event
    assert kafka.deserialize_telemetry_event(payload) == event


def test_message_key_is_machine_code_bytes() -> None:
    event = sample_events()[0]

    assert kafka.encode_message_key(event) == b"MCH-0001"
    assert kafka.validate_key_matches_event(b"MCH-0001", event) == "MCH-0001"

    with pytest.raises(kafka.KafkaStreamingError, match="must match"):
        kafka.validate_key_matches_event(b"MCH-0002", event)


@pytest.mark.parametrize("key", [None, b"not-utf8-\xff", 1])
def test_invalid_message_keys_are_rejected(key: object) -> None:
    with pytest.raises((kafka.KafkaStreamingError, telemetry.TelemetryValidationError)):
        kafka.decode_message_key(key)


def test_invalid_telemetry_is_rejected_before_production() -> None:
    config = sample_config()
    event = sample_events()[0]
    invalid_event = {**event, "source": "external"}

    with pytest.raises(telemetry.TelemetryValidationError, match="source"):
        kafka.produce_telemetry_events(FakeProducer(), [invalid_event], config)


def test_producer_delivery_callbacks_track_successes() -> None:
    config = sample_config()
    events = sample_events()[:2]
    producer = FakeProducer()

    result = kafka.produce_telemetry_events(producer, events, config)

    assert result.attempted == 2
    assert result.delivered == 2
    assert result.failed == 0
    assert [delivery.key for delivery in result.deliveries] == ["MCH-0001", "MCH-0002"]
    assert [item["topic"] for item in producer.produced] == [config.telemetry_topic] * 2


def test_producer_delivery_callbacks_track_failures() -> None:
    config = sample_config()
    result = kafka.produce_telemetry_events(
        FakeProducer(delivery_error=RuntimeError("delivery failed")),
        sample_events()[:1],
        config,
    )

    assert result.attempted == 1
    assert result.delivered == 0
    assert result.failed == 1
    assert "delivery failed" in result.failures[0]


def test_producer_flush_timeout_counts_as_failure() -> None:
    config = sample_config()
    result = kafka.produce_telemetry_events(
        FakeProducer(flush_remaining=1),
        sample_events()[:1],
        config,
    )

    assert result.failed == 1
    assert "not delivered" in result.failures[0]


def test_consumed_record_conversion_separates_kafka_metadata() -> None:
    event = sample_events()[0]
    message = FakeMessage(
        topic=kafka.EXPECTED_TOPIC,
        partition=2,
        offset=15,
        key=b"MCH-0001",
        value=kafka.serialize_telemetry_event(event),
    )

    record = kafka.consumed_record_from_message(message)

    assert record.kafka.topic == kafka.EXPECTED_TOPIC
    assert record.kafka.partition == 2
    assert record.kafka.offset == 15
    assert record.kafka.key == "MCH-0001"
    assert record.event == event


def test_consumer_validation_rejects_payload_key_mismatch() -> None:
    event = sample_events()[0]
    message = FakeMessage(key=b"MCH-0002", value=kafka.serialize_telemetry_event(event))

    with pytest.raises(kafka.KafkaStreamingError, match="must match"):
        kafka.consumed_record_from_message(message)


def test_poll_consumed_records_closes_consumer() -> None:
    event = sample_events()[0]
    consumer = FakeConsumer(
        [FakeMessage(key=b"MCH-0001", value=kafka.serialize_telemetry_event(event))]
    )

    records = kafka.poll_consumed_records(consumer, max_messages=1, timeout_seconds=1.0)

    assert len(records) == 1
    assert consumer.closed is True


def test_expected_event_matching_tolerates_unrelated_and_duplicate_records() -> None:
    events = sample_events()
    unrelated = sample_events()[0]
    unrelated = {**unrelated, "machine_code": "MCH-0003", "sequence_number": 1}
    unrelated["event_id"] = telemetry.deterministic_event_id(
        str(unrelated["machine_code"]),
        int(unrelated["sequence_number"]),
        str(unrelated["event_time"]),
    )
    records = [record_for(unrelated, partition=2, offset=0)]
    records.extend(
        record_for(event, partition=index % 2, offset=index + 1)
        for index, event in enumerate(events)
    )
    records.append(record_for(events[0], partition=0, offset=100))

    validation = kafka.validate_expected_records(records, events)

    assert validation.matched_event_count == len(events)
    assert validation.machine_partitions == {"MCH-0001": 0, "MCH-0002": 1}


def test_expected_events_reject_conflicting_duplicate_event_id() -> None:
    event = sample_events()[0]
    changed = {**event, "air_temperature_k": event["air_temperature_k"] + 0.001}

    with pytest.raises(kafka.KafkaStreamingError, match="different payloads"):
        kafka.expected_events_by_id([event, changed])


def test_partition_consistency_rejects_same_machine_on_multiple_partitions() -> None:
    events = sample_events()
    records = [
        record_for(events[0], partition=0, offset=0),
        record_for(events[2], partition=1, offset=1),
    ]

    with pytest.raises(kafka.KafkaStreamingError, match="appeared on partitions"):
        kafka.validate_partition_consistency(records)


def test_per_machine_order_rejects_sequence_regression() -> None:
    events = sample_events()
    records = [
        record_for(events[2], partition=0, offset=2),
        record_for(events[0], partition=0, offset=3),
    ]

    with pytest.raises(kafka.KafkaStreamingError, match="out of order"):
        kafka.validate_per_machine_order(records)


def test_per_machine_order_rejects_offset_regression() -> None:
    events = sample_events()
    records = [
        record_for(events[0], partition=0, offset=3),
        record_for(events[2], partition=0, offset=2),
    ]

    with pytest.raises(kafka.KafkaStreamingError, match="offsets"):
        kafka.validate_per_machine_order(records)


def test_consume_expected_records_ignores_stale_duplicates() -> None:
    events = sample_events()[:2]
    stale_duplicate = FakeMessage(
        key=b"MCH-0001",
        value=kafka.serialize_telemetry_event(events[0]),
        partition=0,
        offset=0,
    )
    expected_messages = [
        FakeMessage(
            key=str(event["machine_code"]).encode("utf-8"),
            value=kafka.serialize_telemetry_event(event),
            partition=index,
            offset=index + 1,
        )
        for index, event in enumerate(events)
    ]
    consumer = FakeConsumer([stale_duplicate, *expected_messages])

    records = kafka.consume_expected_records(consumer, events, timeout_seconds=1.0)

    assert len(records) == 2
    assert consumer.closed is True


def test_build_integration_summary_is_deterministic_and_runtime_free() -> None:
    config = sample_config()

    first = kafka.build_integration_summary(config)
    second = kafka.build_integration_summary(config)

    assert first == second
    rendered = json.dumps(first, sort_keys=True)
    assert '"event_id":' not in rendered
    assert '"offset":' not in rendered
    assert "group_id" not in rendered
    assert first["smoke_expected_event_count"] == 6


def test_get_topic_description_parses_metadata_shape() -> None:
    topic_metadata = SimpleNamespace(
        error=None,
        partitions={
            0: SimpleNamespace(replicas=[1]),
            1: SimpleNamespace(replicas=[1]),
            2: SimpleNamespace(replicas=[1]),
        },
    )
    metadata = SimpleNamespace(topics={kafka.EXPECTED_TOPIC: topic_metadata})
    admin_client = SimpleNamespace(list_topics=lambda topic, timeout: metadata)

    description = kafka.get_topic_description(admin_client, kafka.EXPECTED_TOPIC)

    assert description == kafka.TopicDescription(kafka.EXPECTED_TOPIC, 3, 1)

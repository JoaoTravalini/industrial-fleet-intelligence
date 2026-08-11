from __future__ import annotations

import json
from datetime import timedelta

import pytest

from services.simulator import telemetry


def small_config(seed: int = 42) -> telemetry.SimulatorConfig:
    return telemetry.SimulatorConfig(machine_count=3, events_per_machine=4, seed=seed)


def comparable_machine_state(state: telemetry.MachineState) -> dict[str, object]:
    return {
        "machine_code": state.machine_code,
        "product_quality_type": state.product_quality_type,
        "air_temperature_k": state.air_temperature_k,
        "process_temperature_k": state.process_temperature_k,
        "rotational_speed_rpm": state.rotational_speed_rpm,
        "torque_nm": state.torque_nm,
        "tool_wear_min": state.tool_wear_min,
        "vibration_mm_s": state.vibration_mm_s,
        "pressure_bar": state.pressure_bar,
        "sequence_number": state.sequence_number,
    }


def test_machine_code_validation() -> None:
    assert telemetry.validate_machine_code("MCH-0001") == "MCH-0001"
    assert telemetry.validate_machine_code("MCH-0100") == "MCH-0100"

    for invalid in ["MCH-0000", "MCH-0101", "machine-1", "MCH-1", 1]:
        with pytest.raises(telemetry.TelemetryValidationError):
            telemetry.validate_machine_code(invalid)


def test_machine_initialization_is_deterministic() -> None:
    first = telemetry.initialize_machine("MCH-0007", seed=42)
    second = telemetry.initialize_machine("MCH-0007", seed=42)
    other = telemetry.initialize_machine("MCH-0007", seed=43)

    assert comparable_machine_state(first) == comparable_machine_state(second)
    assert comparable_machine_state(first) != comparable_machine_state(other)


def test_identical_seed_produces_identical_events_and_jsonl_bytes() -> None:
    first = telemetry.generate_events(small_config(seed=42))
    second = telemetry.generate_events(small_config(seed=42))

    assert first == second
    assert (
        telemetry.serialize_events_jsonl(first).encode()
        == telemetry.serialize_events_jsonl(second).encode()
    )


def test_different_seed_produces_different_event_values() -> None:
    first = telemetry.generate_events(small_config(seed=42))
    second = telemetry.generate_events(small_config(seed=99))

    assert telemetry.serialize_events_jsonl(first) != telemetry.serialize_events_jsonl(second)
    assert first[0]["event_id"] == second[0]["event_id"]
    assert first[0]["air_temperature_k"] != second[0]["air_temperature_k"]


def test_deterministic_uuid_generation() -> None:
    event_time = "2026-01-01T00:00:00Z"

    first = telemetry.deterministic_event_id("MCH-0001", 1, event_time)
    second = telemetry.deterministic_event_id("MCH-0001", 1, event_time)
    different = telemetry.deterministic_event_id("MCH-0001", 2, event_time)

    assert first == second
    assert first != different
    assert len(first) == 36


def test_event_field_set_and_product_quality_values() -> None:
    events = telemetry.generate_events(small_config())

    assert set(events[0]) == set(telemetry.FIELD_ORDER)
    assert {event["product_quality_type"] for event in events}.issubset(
        set(telemetry.PRODUCT_QUALITY_TYPES)
    )


def test_numerical_bounds_and_process_temperature_invariant() -> None:
    events = telemetry.generate_events(small_config())

    for event in events:
        telemetry.validate_event(event)
        assert event["process_temperature_k"] > event["air_temperature_k"]
        for field, (lower, upper) in telemetry.SENSOR_BOUNDS.items():
            assert lower <= event[field] <= upper


def test_tool_wear_monotonicity_sequence_numbering_ordering_and_interval() -> None:
    config = small_config()
    events = telemetry.generate_events(config)

    telemetry.validate_event_batch(
        events,
        expected_machine_count=config.machine_count,
        expected_events_per_machine=config.events_per_machine,
        interval_seconds=config.interval_seconds,
        start_time=config.start_time,
        expected_machine_codes=["MCH-0001", "MCH-0002", "MCH-0003"],
    )

    assert [(event["event_time"], event["machine_code"]) for event in events] == sorted(
        (event["event_time"], event["machine_code"]) for event in events
    )
    by_machine = {
        machine_code: [event for event in events if event["machine_code"] == machine_code]
        for machine_code in ["MCH-0001", "MCH-0002", "MCH-0003"]
    }
    for machine_events in by_machine.values():
        assert [event["sequence_number"] for event in machine_events] == [1, 2, 3, 4]
        wear_values = [event["tool_wear_min"] for event in machine_events]
        assert wear_values == sorted(wear_values)
        timestamps = [
            telemetry.parse_utc_timestamp(event["event_time"]) for event in machine_events
        ]
        assert timestamps[1] - timestamps[0] == timedelta(seconds=config.interval_seconds)


def test_serialized_jsonl_is_compact_and_deterministic() -> None:
    events = telemetry.generate_events(
        telemetry.SimulatorConfig(machine_count=2, events_per_machine=2)
    )
    jsonl = telemetry.serialize_events_jsonl(events)

    assert jsonl == telemetry.serialize_events_jsonl(events)
    assert jsonl.endswith("\n")
    assert " " not in jsonl.splitlines()[0]
    assert json.loads(jsonl.splitlines()[0]) == events[0]


def test_batch_validation_rejects_duplicate_event_id() -> None:
    event = telemetry.generate_events(
        telemetry.SimulatorConfig(machine_count=1, events_per_machine=1)
    )[0]
    events = [event, dict(event)]

    with pytest.raises(telemetry.TelemetryValidationError, match="duplicate event_id"):
        telemetry.validate_event_batch(events)


def test_batch_validation_rejects_sequence_gap() -> None:
    events = telemetry.generate_events(
        telemetry.SimulatorConfig(machine_count=1, events_per_machine=2)
    )
    changed = {**events[1], "sequence_number": 3}
    changed["event_id"] = telemetry.deterministic_event_id(
        changed["machine_code"],
        changed["sequence_number"],
        changed["event_time"],
    )
    events[1] = changed

    with pytest.raises(telemetry.TelemetryValidationError, match="without gaps"):
        telemetry.validate_event_batch(events)


def test_invalid_event_rejection_boolean_numeric_and_unknown_field() -> None:
    event = telemetry.generate_events(
        telemetry.SimulatorConfig(machine_count=1, events_per_machine=1)
    )[0]
    invalid_source = {**event, "source": "other"}
    boolean_sensor = {**event, "air_temperature_k": True}
    unknown_field = {**event, "unexpected": 1}

    with pytest.raises(telemetry.TelemetryValidationError, match="source"):
        telemetry.validate_event(invalid_source)
    with pytest.raises(telemetry.TelemetryValidationError, match="not boolean"):
        telemetry.validate_event(boolean_sensor)
    with pytest.raises(telemetry.TelemetryValidationError, match="Unknown telemetry field"):
        telemetry.validate_event(unknown_field)


def test_model_target_prediction_and_anomaly_fields_are_excluded() -> None:
    event = telemetry.generate_events(
        telemetry.SimulatorConfig(machine_count=1, events_per_machine=1)
    )[0]

    assert set(event).isdisjoint(telemetry.MODEL_TARGET_AND_OUTPUT_FIELDS)
    with pytest.raises(telemetry.TelemetryValidationError, match="Unknown telemetry field"):
        telemetry.validate_event({**event, "failure_prediction": 1})


def test_summary_matches_events_and_sample_hash() -> None:
    config = telemetry.SimulatorConfig(machine_count=2, events_per_machine=2)
    events = telemetry.generate_events(config)
    jsonl_bytes = telemetry.serialize_events_jsonl(events).encode()
    summary = telemetry.build_summary(events, config, jsonl_bytes)

    assert summary["event_count"] == 4
    assert summary["machine_count"] == 2
    assert summary["product_quality_type_distribution"] == telemetry.product_quality_distribution(
        events
    )
    assert summary["sensor_min_max"] == telemetry.sensor_min_max(events)
    assert summary["sample_sha256"] == telemetry.sample_sha256(jsonl_bytes)

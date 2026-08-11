"""Deterministic synthetic industrial telemetry simulator."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SOURCE = "synthetic_simulator"
DEFAULT_RANDOM_SEED = 42
DEFAULT_START_TIME = "2026-01-01T00:00:00Z"
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_SAMPLE_MACHINE_COUNT = 10
DEFAULT_SAMPLE_EVENTS_PER_MACHINE = 10
MAX_MACHINE_COUNT = 100
MACHINE_CODE_PATTERN = re.compile(r"^MCH-(\d{4})$")
PRODUCT_QUALITY_TYPES = ("L", "M", "H")
FIELD_ORDER = (
    "schema_version",
    "event_id",
    "machine_code",
    "sequence_number",
    "event_time",
    "source",
    "product_quality_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "pressure_bar",
)
MODEL_TARGET_AND_OUTPUT_FIELDS = frozenset(
    {
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "failure_probability",
        "failure_prediction",
        "shap_values",
        "anomaly_label",
    }
)
SENSOR_BOUNDS = {
    "air_temperature_k": (294.0, 306.0),
    "process_temperature_k": (304.0, 315.0),
    "rotational_speed_rpm": (1000, 3000),
    "torque_nm": (0.0, 80.0),
    "tool_wear_min": (0, 300),
    "vibration_mm_s": (0.0, 15.0),
    "pressure_bar": (1.0, 12.0),
}
SAMPLE_RELATIVE_PATH = Path("data") / "sample" / "telemetry_events.jsonl"
SUMMARY_RELATIVE_PATH = Path("reports") / "telemetry" / "simulator_summary.json"


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for deterministic local telemetry generation."""

    machine_count: int = DEFAULT_SAMPLE_MACHINE_COUNT
    events_per_machine: int = DEFAULT_SAMPLE_EVENTS_PER_MACHINE
    seed: int = DEFAULT_RANDOM_SEED
    start_time: datetime = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS


@dataclass
class MachineState:
    """Mutable per-machine simulator state."""

    machine_code: str
    product_quality_type: str
    random: random.Random
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: int
    torque_nm: float
    tool_wear_min: int
    vibration_mm_s: float
    pressure_bar: float
    sequence_number: int = 0


class TelemetryValidationError(ValueError):
    """Raised when telemetry event validation fails."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SAMPLE_RELATIVE_PATH


def summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SUMMARY_RELATIVE_PATH


def parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TelemetryValidationError("Timestamp must be an ISO-8601 UTC string ending with Z.")
    try:
        timestamp = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise TelemetryValidationError(f"Invalid UTC timestamp: {value}") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise TelemetryValidationError("Timestamp must be timezone-aware UTC.")
    return timestamp.astimezone(UTC)


def format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise TelemetryValidationError("Timestamp must be timezone-aware UTC.")
    timestamp = value.astimezone(UTC)
    if timestamp.microsecond == 0:
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    timespec = "milliseconds" if timestamp.microsecond % 1000 == 0 else "microseconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


def parse_start_time(value: str) -> datetime:
    return parse_utc_timestamp(value)


def machine_code(number: int) -> str:
    if isinstance(number, bool) or not isinstance(number, int):
        raise TelemetryValidationError("Machine number must be an integer.")
    if number < 1 or number > MAX_MACHINE_COUNT:
        raise TelemetryValidationError("Machine number must be between 1 and 100.")
    return f"MCH-{number:04d}"


def validate_machine_code(code: Any) -> str:
    if not isinstance(code, str):
        raise TelemetryValidationError("machine_code must be a string.")
    match = MACHINE_CODE_PATTERN.fullmatch(code)
    if match is None:
        raise TelemetryValidationError("machine_code must use the MCH-XXXX format.")
    number = int(match.group(1))
    if number < 1 or number > MAX_MACHINE_COUNT:
        raise TelemetryValidationError(
            "machine_code must be in the MCH-0001 through MCH-0100 range."
        )
    return code


def validate_config(config: SimulatorConfig) -> None:
    if isinstance(config.machine_count, bool) or not isinstance(config.machine_count, int):
        raise TelemetryValidationError("machine_count must be an integer.")
    if config.machine_count < 1 or config.machine_count > MAX_MACHINE_COUNT:
        raise TelemetryValidationError("machine_count must be between 1 and 100.")
    if isinstance(config.events_per_machine, bool) or not isinstance(
        config.events_per_machine, int
    ):
        raise TelemetryValidationError("events_per_machine must be an integer.")
    if config.events_per_machine < 1:
        raise TelemetryValidationError("events_per_machine must be positive.")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TelemetryValidationError("seed must be an integer.")
    if not isinstance(config.interval_seconds, int | float) or isinstance(
        config.interval_seconds, bool
    ):
        raise TelemetryValidationError("interval_seconds must be numeric.")
    if not math.isfinite(float(config.interval_seconds)) or float(config.interval_seconds) <= 0:
        raise TelemetryValidationError("interval_seconds must be positive and finite.")
    format_utc_timestamp(config.start_time)


def stable_machine_seed(seed: int, code: str) -> int:
    digest = hashlib.sha256(f"{seed}:{code}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def rounded_sensor(value: float) -> float:
    if not math.isfinite(value):
        raise TelemetryValidationError("Sensor value must be finite.")
    return round(float(value), 3)


def initialize_machine(code: str, seed: int = DEFAULT_RANDOM_SEED) -> MachineState:
    validate_machine_code(code)
    generator = random.Random(stable_machine_seed(seed, code))
    air = generator.uniform(296.0, 303.0)
    process = clamp(air + generator.uniform(6.0, 8.5), 304.0, 315.0)
    return MachineState(
        machine_code=code,
        product_quality_type=generator.choice(PRODUCT_QUALITY_TYPES),
        random=generator,
        air_temperature_k=air,
        process_temperature_k=process,
        rotational_speed_rpm=generator.randint(1250, 2150),
        torque_nm=generator.uniform(20.0, 55.0),
        tool_wear_min=generator.randint(0, 90),
        vibration_mm_s=generator.uniform(0.8, 4.5),
        pressure_bar=generator.uniform(4.0, 9.0),
    )


def build_machine_states(config: SimulatorConfig) -> list[MachineState]:
    validate_config(config)
    return [
        initialize_machine(machine_code(index), config.seed)
        for index in range(1, config.machine_count + 1)
    ]


def deterministic_event_id(machine: str, sequence_number: int, event_time: str) -> str:
    stable_identity = f"{SCHEMA_VERSION}|{machine}|{sequence_number}|{event_time}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_identity))


def next_machine_event(state: MachineState, event_time: datetime) -> dict[str, Any]:
    state.sequence_number += 1
    sequence_number = state.sequence_number
    generator = state.random

    state.air_temperature_k = clamp(
        state.air_temperature_k + generator.uniform(-0.12, 0.12),
        *SENSOR_BOUNDS["air_temperature_k"],
    )
    minimum_process = max(SENSOR_BOUNDS["process_temperature_k"][0], state.air_temperature_k + 0.5)
    state.process_temperature_k = clamp(
        max(
            minimum_process,
            state.process_temperature_k + generator.uniform(-0.10, 0.14),
        ),
        minimum_process,
        SENSOR_BOUNDS["process_temperature_k"][1],
    )
    state.rotational_speed_rpm = int(
        clamp(
            state.rotational_speed_rpm + generator.randint(-28, 28),
            *SENSOR_BOUNDS["rotational_speed_rpm"],
        )
    )
    torque_drift = generator.uniform(-0.65, 0.65) + (1600 - state.rotational_speed_rpm) / 2500
    state.torque_nm = clamp(
        state.torque_nm + torque_drift,
        *SENSOR_BOUNDS["torque_nm"],
    )
    wear_increment = generator.choice((0, 0, 1, 1, 2))
    if state.torque_nm > 55 or state.vibration_mm_s > 7:
        wear_increment += 1
    state.tool_wear_min = int(
        clamp(
            state.tool_wear_min + wear_increment,
            *SENSOR_BOUNDS["tool_wear_min"],
        )
    )
    vibration_drift = generator.uniform(-0.08, 0.10) + max(state.torque_nm - 45.0, 0.0) / 250
    state.vibration_mm_s = clamp(
        state.vibration_mm_s + vibration_drift,
        *SENSOR_BOUNDS["vibration_mm_s"],
    )
    pressure_drift = generator.uniform(-0.05, 0.05) + (state.process_temperature_k - 309.0) / 700
    state.pressure_bar = clamp(
        state.pressure_bar + pressure_drift,
        *SENSOR_BOUNDS["pressure_bar"],
    )

    event_time_text = format_utc_timestamp(event_time)
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": deterministic_event_id(
            state.machine_code,
            sequence_number,
            event_time_text,
        ),
        "machine_code": state.machine_code,
        "sequence_number": sequence_number,
        "event_time": event_time_text,
        "source": SOURCE,
        "product_quality_type": state.product_quality_type,
        "air_temperature_k": rounded_sensor(state.air_temperature_k),
        "process_temperature_k": rounded_sensor(state.process_temperature_k),
        "rotational_speed_rpm": state.rotational_speed_rpm,
        "torque_nm": rounded_sensor(state.torque_nm),
        "tool_wear_min": state.tool_wear_min,
        "vibration_mm_s": rounded_sensor(state.vibration_mm_s),
        "pressure_bar": rounded_sensor(state.pressure_bar),
    }
    validate_event(event)
    return event


def generate_events(config: SimulatorConfig | None = None) -> list[dict[str, Any]]:
    active_config = config or SimulatorConfig()
    validate_config(active_config)
    states = build_machine_states(active_config)
    events: list[dict[str, Any]] = []
    interval = timedelta(seconds=float(active_config.interval_seconds))
    for timestamp_index in range(active_config.events_per_machine):
        event_time = active_config.start_time + interval * timestamp_index
        for state in states:
            events.append(next_machine_event(state, event_time))
    validate_event_batch(
        events,
        expected_machine_count=active_config.machine_count,
        expected_events_per_machine=active_config.events_per_machine,
        interval_seconds=active_config.interval_seconds,
        start_time=active_config.start_time,
        expected_machine_codes=[state.machine_code for state in states],
    )
    return events


def ordered_event(event: Mapping[str, Any]) -> dict[str, Any]:
    validate_event(event)
    return {field: event[field] for field in FIELD_ORDER}


def serialize_event(event: Mapping[str, Any]) -> str:
    return json.dumps(ordered_event(event), ensure_ascii=False, separators=(",", ":"))


def serialize_events_jsonl(events: Iterable[Mapping[str, Any]]) -> str:
    return "".join(serialize_event(event) + "\n" for event in events)


def write_jsonl(events: Sequence[Mapping[str, Any]], path: Path) -> bytes:
    content = serialize_events_jsonl(events).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TelemetryValidationError(f"Invalid JSON on line {line_number}.") from exc
            if not isinstance(data, dict):
                raise TelemetryValidationError(f"JSONL line {line_number} must contain an object.")
            validate_event(data)
            events.append(data)
    return events


def validate_exact_fields(event: Mapping[str, Any]) -> None:
    actual_fields = set(event.keys())
    expected_fields = set(FIELD_ORDER)
    missing = sorted(expected_fields - actual_fields)
    unknown = sorted(actual_fields - expected_fields)
    if missing:
        raise TelemetryValidationError("Missing telemetry field(s): " + ", ".join(missing))
    if unknown:
        raise TelemetryValidationError("Unknown telemetry field(s): " + ", ".join(unknown))
    if any(field in actual_fields for field in MODEL_TARGET_AND_OUTPUT_FIELDS):
        raise TelemetryValidationError(
            "Telemetry events must not contain target or model output fields."
        )


def validate_uuid(value: Any) -> None:
    if not isinstance(value, str):
        raise TelemetryValidationError("event_id must be a UUID string.")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise TelemetryValidationError("event_id must be a valid UUID string.") from exc
    if str(parsed) != value:
        raise TelemetryValidationError("event_id must use canonical UUID string formatting.")


def validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TelemetryValidationError(f"{name} must be an integer.")
    if value <= 0:
        raise TelemetryValidationError(f"{name} must be positive.")
    return int(value)


def validate_bounded_integer(name: str, value: Any, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TelemetryValidationError(f"{name} must be an integer.")
    if value < lower or value > upper:
        raise TelemetryValidationError(f"{name} must be between {lower} and {upper}.")
    return int(value)


def validate_bounded_number(name: str, value: Any, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TelemetryValidationError(f"{name} must be numeric and not boolean.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TelemetryValidationError(f"{name} must be finite.")
    if numeric < lower or numeric > upper:
        raise TelemetryValidationError(f"{name} must be between {lower} and {upper}.")
    return numeric


def validate_event(event: Mapping[str, Any]) -> None:
    if not isinstance(event, Mapping):
        raise TelemetryValidationError("Telemetry event must be a JSON object.")
    validate_exact_fields(event)
    if event["schema_version"] != SCHEMA_VERSION:
        raise TelemetryValidationError("schema_version must be exactly 1.0.")
    validate_uuid(event["event_id"])
    validate_machine_code(event["machine_code"])
    validate_positive_integer("sequence_number", event["sequence_number"])
    parse_utc_timestamp(event["event_time"])
    if event["source"] != SOURCE:
        raise TelemetryValidationError("source must be synthetic_simulator.")
    if event["product_quality_type"] not in PRODUCT_QUALITY_TYPES:
        raise TelemetryValidationError("product_quality_type must be one of L, M, or H.")

    air = validate_bounded_number("air_temperature_k", event["air_temperature_k"], 294.0, 306.0)
    process = validate_bounded_number(
        "process_temperature_k",
        event["process_temperature_k"],
        304.0,
        315.0,
    )
    if process <= air:
        raise TelemetryValidationError("process_temperature_k must be above air_temperature_k.")
    validate_bounded_integer(
        "rotational_speed_rpm",
        event["rotational_speed_rpm"],
        1000,
        3000,
    )
    validate_bounded_number("torque_nm", event["torque_nm"], 0.0, 80.0)
    validate_bounded_integer("tool_wear_min", event["tool_wear_min"], 0, 300)
    validate_bounded_number("vibration_mm_s", event["vibration_mm_s"], 0.0, 15.0)
    validate_bounded_number("pressure_bar", event["pressure_bar"], 1.0, 12.0)

    expected_event_id = deterministic_event_id(
        str(event["machine_code"]),
        int(event["sequence_number"]),
        str(event["event_time"]),
    )
    if event["event_id"] != expected_event_id:
        raise TelemetryValidationError("event_id does not match deterministic event identity.")


def validate_event_batch(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_machine_count: int | None = None,
    expected_events_per_machine: int | None = None,
    interval_seconds: float | None = None,
    start_time: datetime | None = None,
    expected_machine_codes: Sequence[str] | None = None,
) -> None:
    if not events:
        raise TelemetryValidationError("Telemetry batch must contain at least one event.")
    for event in events:
        validate_event(event)

    event_ids = [str(event["event_id"]) for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise TelemetryValidationError("Telemetry batch contains duplicate event_id values.")

    ordering_keys = [
        (parse_utc_timestamp(str(event["event_time"])), str(event["machine_code"]))
        for event in events
    ]
    if ordering_keys != sorted(ordering_keys):
        raise TelemetryValidationError(
            "Telemetry events must be ordered by timestamp then machine_code."
        )

    machine_codes = sorted({str(event["machine_code"]) for event in events})
    if expected_machine_codes is not None and machine_codes != sorted(expected_machine_codes):
        raise TelemetryValidationError("Telemetry batch machine codes do not match expectation.")
    if expected_machine_count is not None and len(machine_codes) != expected_machine_count:
        raise TelemetryValidationError("Telemetry batch machine count does not match expectation.")

    events_by_machine: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_machine[str(event["machine_code"])].append(event)

    expected_interval = None if interval_seconds is None else float(interval_seconds)
    for code in machine_codes:
        machine_events = events_by_machine[code]
        sequence_numbers = [int(event["sequence_number"]) for event in machine_events]
        expected_sequences = list(range(1, len(machine_events) + 1))
        if sequence_numbers != expected_sequences:
            raise TelemetryValidationError(
                f"Sequence numbers for {code} must start at 1 without gaps."
            )
        if (
            expected_events_per_machine is not None
            and len(machine_events) != expected_events_per_machine
        ):
            raise TelemetryValidationError(f"{code} event count does not match expectation.")

        previous_tool_wear = -1
        first_time = parse_utc_timestamp(str(machine_events[0]["event_time"]))
        if start_time is not None and first_time != start_time.astimezone(UTC):
            raise TelemetryValidationError(f"{code} first timestamp does not match start_time.")
        for index, event in enumerate(machine_events):
            current_wear = int(event["tool_wear_min"])
            if current_wear < previous_tool_wear:
                raise TelemetryValidationError(f"tool_wear_min decreases for {code}.")
            previous_tool_wear = current_wear
            if expected_interval is not None:
                actual_time = parse_utc_timestamp(str(event["event_time"]))
                expected_seconds = expected_interval * index
                actual_seconds = (actual_time - first_time).total_seconds()
                if abs(actual_seconds - expected_seconds) > 1e-9:
                    raise TelemetryValidationError(
                        f"{code} timestamps do not follow the expected interval."
                    )


def sample_sha256(jsonl_bytes: bytes) -> str:
    return hashlib.sha256(jsonl_bytes).hexdigest()


def sensor_min_max(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for field in SENSOR_BOUNDS:
        values = [event[field] for event in events]
        summary[field] = {"min": min(values), "max": max(values)}
    return summary


def product_quality_distribution(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(event["product_quality_type"]) for event in events)
    return {quality: int(counts.get(quality, 0)) for quality in PRODUCT_QUALITY_TYPES}


def build_summary(
    events: Sequence[Mapping[str, Any]],
    config: SimulatorConfig,
    jsonl_bytes: bytes,
) -> dict[str, Any]:
    validate_event_batch(
        events,
        expected_machine_count=config.machine_count,
        expected_events_per_machine=config.events_per_machine,
        interval_seconds=config.interval_seconds,
        start_time=config.start_time,
        expected_machine_codes=[
            machine_code(index) for index in range(1, config.machine_count + 1)
        ],
    )
    machine_codes = sorted({str(event["machine_code"]) for event in events})
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "synthetic_data_classification": "synthetic generated telemetry",
        "seed": int(config.seed),
        "machine_count": int(config.machine_count),
        "event_count": int(len(events)),
        "events_per_machine": int(config.events_per_machine),
        "start_time": format_utc_timestamp(config.start_time),
        "interval_seconds": float(config.interval_seconds),
        "first_event_time": str(events[0]["event_time"]),
        "last_event_time": str(events[-1]["event_time"]),
        "machine_code_range": {
            "first": machine_codes[0],
            "last": machine_codes[-1],
        },
        "product_quality_type_distribution": product_quality_distribution(events),
        "sensor_min_max": sensor_min_max(events),
        "sample_sha256": sample_sha256(jsonl_bytes),
    }


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def generate_sample(root: Path | None = None) -> dict[str, Any]:
    root_path = root or project_root()
    config = SimulatorConfig()
    events = generate_events(config)
    jsonl_bytes = write_jsonl(events, sample_path(root_path))
    summary = build_summary(events, config, jsonl_bytes)
    write_json(summary, summary_path(root_path))
    return summary

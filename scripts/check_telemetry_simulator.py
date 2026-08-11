"""Read-only validator for the canonical synthetic telemetry simulator sample."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.simulator import telemetry  # noqa: E402


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    mandatory: bool = True


@dataclass(frozen=True)
class ValidationReport:
    results: list[CheckResult]

    @property
    def is_valid(self) -> bool:
        return not any(result.status is Status.FAIL and result.mandatory for result in self.results)


def result(name: str, passed: bool, pass_message: str, fail_message: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=Status.PASS if passed else Status.FAIL,
        message=pass_message if passed else fail_message,
    )


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise telemetry.TelemetryValidationError("Simulator summary must contain a JSON object.")
    return data


def validate_all_event_schemas(events: list[dict[str, Any]]) -> tuple[bool, str]:
    try:
        for event in events:
            telemetry.validate_event(event)
    except telemetry.TelemetryValidationError as exc:
        return False, str(exc)
    return True, "All events match the telemetry contract."


def expected_machine_codes() -> list[str]:
    return [telemetry.machine_code(index) for index in range(1, 11)]


def per_machine_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {code: 0 for code in expected_machine_codes()}
    for event in events:
        code = str(event["machine_code"])
        counts[code] = counts.get(code, 0) + 1
    return counts


def summary_sensor_min_max_matches(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
) -> bool:
    return summary.get("sensor_min_max") == telemetry.sensor_min_max(events)


def summary_counts_match(events: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    counts = per_machine_counts(events)
    return (
        summary.get("schema_version") == telemetry.SCHEMA_VERSION
        and summary.get("source") == telemetry.SOURCE
        and summary.get("seed") == telemetry.DEFAULT_RANDOM_SEED
        and summary.get("machine_count") == 10
        and summary.get("event_count") == len(events) == 100
        and summary.get("events_per_machine") == 10
        and summary.get("start_time") == telemetry.DEFAULT_START_TIME
        and summary.get("interval_seconds") == telemetry.DEFAULT_INTERVAL_SECONDS
        and summary.get("machine_code_range") == {"first": "MCH-0001", "last": "MCH-0010"}
        and all(value == 10 for value in counts.values())
        and summary.get("product_quality_type_distribution")
        == telemetry.product_quality_distribution(events)
    )


def validate_sample() -> ValidationReport:
    results: list[CheckResult] = []
    sample_path = telemetry.sample_path(PROJECT_ROOT)
    summary_path = telemetry.summary_path(PROJECT_ROOT)
    results.extend(
        [
            result(
                "Telemetry Sample File",
                sample_path.exists() and sample_path.stat().st_size > 0,
                "Canonical telemetry JSONL sample exists.",
                "Canonical telemetry JSONL sample is missing or empty.",
            ),
            result(
                "Telemetry Summary File",
                summary_path.exists() and summary_path.stat().st_size > 0,
                "Telemetry simulator summary exists.",
                "Telemetry simulator summary is missing or empty.",
            ),
        ]
    )
    if any(item.status is Status.FAIL for item in results):
        return ValidationReport(results)

    try:
        events = telemetry.load_jsonl(sample_path)
        summary = load_summary(summary_path)
    except (OSError, json.JSONDecodeError, telemetry.TelemetryValidationError) as exc:
        results.append(CheckResult("Readable Sample Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    schema_ok, schema_message = validate_all_event_schemas(events)
    machine_codes = sorted({str(event["machine_code"]) for event in events})
    counts = per_machine_counts(events)
    event_ids = [str(event["event_id"]) for event in events]
    sources = {str(event["source"]) for event in events}
    quality_values = {str(event["product_quality_type"]) for event in events}
    no_model_outputs = all(
        set(event).isdisjoint(telemetry.MODEL_TARGET_AND_OUTPUT_FIELDS) for event in events
    )

    results.extend(
        [
            result(
                "Event Count",
                len(events) == 100,
                "Canonical sample contains exactly 100 events.",
                f"Expected 100 events, found {len(events)}.",
            ),
            result(
                "Machine Count",
                len(machine_codes) == 10,
                "Canonical sample contains exactly 10 machines.",
                f"Expected 10 machines, found {len(machine_codes)}.",
            ),
            result(
                "Machine Codes",
                machine_codes == expected_machine_codes(),
                "Canonical sample uses MCH-0001 through MCH-0010.",
                "Canonical sample machine codes are not MCH-0001 through MCH-0010.",
            ),
            result(
                "Events Per Machine",
                all(counts.get(code) == 10 for code in expected_machine_codes()),
                "Each canonical sample machine has exactly 10 events.",
                "One or more machines do not have exactly 10 events.",
            ),
            result(
                "Schema Validity",
                schema_ok,
                schema_message,
                schema_message,
            ),
            result(
                "Unique Event IDs",
                len(event_ids) == len(set(event_ids)),
                "Event IDs are unique.",
                "Duplicate event IDs were found.",
            ),
        ]
    )

    try:
        telemetry.validate_event_batch(
            events,
            expected_machine_count=10,
            expected_events_per_machine=10,
            interval_seconds=telemetry.DEFAULT_INTERVAL_SECONDS,
            start_time=telemetry.parse_start_time(telemetry.DEFAULT_START_TIME),
            expected_machine_codes=expected_machine_codes(),
        )
        sequence_valid = True
        sequence_message = (
            "Batch ordering, sequence numbers, timestamps, and wear monotonicity are valid."
        )
    except telemetry.TelemetryValidationError as exc:
        sequence_valid = False
        sequence_message = str(exc)
    results.append(
        result(
            "Sequence Validation",
            sequence_valid,
            sequence_message,
            sequence_message,
        )
    )

    results.extend(
        [
            result(
                "Timestamp Interval",
                sequence_valid,
                "Event timestamps follow the configured five-second interval.",
                "Event timestamps do not follow the configured interval.",
            ),
            result(
                "Source Values",
                sources == {telemetry.SOURCE},
                "All events use source synthetic_simulator.",
                "Unexpected source values were found.",
            ),
            result(
                "Product Quality Values",
                quality_values.issubset(set(telemetry.PRODUCT_QUALITY_TYPES)),
                "product_quality_type values are limited to L, M, and H.",
                "Unexpected product_quality_type values were found.",
            ),
            result(
                "Sensor Bounds",
                schema_ok,
                "All sensor values are finite and within simulator bounds.",
                "One or more sensor values violate simulator bounds.",
            ),
            result(
                "Process Temperature Invariant",
                all(
                    event["process_temperature_k"] > event["air_temperature_k"] for event in events
                ),
                "process_temperature_k is above air_temperature_k for every event.",
                "process_temperature_k is not above air_temperature_k for every event.",
            ),
            result(
                "Tool Wear Monotonicity",
                sequence_valid,
                "tool_wear_min never decreases per machine.",
                "tool_wear_min decreases for at least one machine.",
            ),
            result(
                "Summary Counts",
                summary_counts_match(events, summary),
                "Summary counts and distributions match the sample.",
                "Summary counts or distributions do not match the sample.",
            ),
            result(
                "Summary Sensor Min Max",
                summary_sensor_min_max_matches(events, summary),
                "Summary sensor min/max values match the sample.",
                "Summary sensor min/max values do not match the sample.",
            ),
            result(
                "Summary Sample SHA-256",
                summary.get("sample_sha256") == telemetry.sample_sha256(sample_path.read_bytes()),
                "Summary SHA-256 matches telemetry_events.jsonl.",
                "Summary SHA-256 does not match telemetry_events.jsonl.",
            ),
            result(
                "No Model Or Anomaly Labels",
                no_model_outputs,
                "Telemetry events contain no target, prediction, SHAP, or anomaly labels.",
                "Telemetry events contain target, prediction, SHAP, or anomaly labels.",
            ),
        ]
    )
    return ValidationReport(results)


def print_report(report: ValidationReport) -> None:
    for item in report.results:
        print(f"{item.status} {item.name}: {item.message}")
    pass_count = sum(1 for item in report.results if item.status is Status.PASS)
    warn_count = sum(1 for item in report.results if item.status is Status.WARN)
    fail_count = sum(1 for item in report.results if item.status is Status.FAIL)
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    report = validate_sample()
    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

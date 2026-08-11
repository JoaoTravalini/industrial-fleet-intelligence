"""End-to-end validator for Kafka to Spark Bronze ingestion."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_spark_bronze_docker  # noqa: E402
from services.simulator import telemetry  # noqa: E402
from services.streaming import kafka  # noqa: E402

KAFKA_CONTAINER = "industrial-fleet-kafka"
SMOKE_MACHINE_COUNT = 2
SMOKE_EVENTS_PER_MACHINE = 3
SMOKE_SEED = 2026
SMOKE_START_TIME = "2026-02-01T00:00:00Z"
SMOKE_TIMEOUT_SECONDS = 900
INSPECT_TIMEOUT_SECONDS = 300


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


def docker_inspect_health(container_name: str) -> tuple[bool, str | None, str | None]:
    result = run_spark_bronze_docker.run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container_name,
        ],
        timeout=30,
    )
    if not result.succeeded:
        return False, None, run_spark_bronze_docker.command_failure_message(result)
    running, health = run_spark_bronze_docker.parse_container_health(result.output)
    return running, health, None


def check_kafka_health() -> CheckResult:
    running, health, error = docker_inspect_health(KAFKA_CONTAINER)
    if error:
        return CheckResult("Kafka Health", Status.FAIL, error)
    if not running:
        return CheckResult("Kafka Health", Status.FAIL, "Kafka container is not running.")
    if health != "healthy":
        return CheckResult("Kafka Health", Status.FAIL, f"Kafka health status is {health}.")
    return CheckResult("Kafka Health", Status.PASS, "Kafka container is running and healthy.")


def check_spark_health() -> CheckResult:
    ok, message = run_spark_bronze_docker.verify_spark_container()
    status = Status.PASS if ok else Status.FAIL
    return CheckResult("Spark Health", status, message)


def check_kafka_topic() -> CheckResult:
    try:
        kafka_config = kafka.load_kafka_config()
        admin_client = kafka.create_admin_client(kafka_config)
        kafka.check_broker_connectivity(admin_client)
        description = kafka.get_topic_description(admin_client, kafka_config.telemetry_topic)
        if description is None:
            return CheckResult(
                "Kafka Topic",
                Status.FAIL,
                f"Topic {kafka_config.telemetry_topic} does not exist.",
            )
        kafka.validate_topic_description(description, kafka_config)
        message = (
            f"Topic has {description.partitions} partitions "
            f"and RF {description.replication_factor}."
        )
        return CheckResult(
            "Kafka Topic",
            Status.PASS,
            message,
        )
    except Exception as exc:
        return CheckResult("Kafka Topic", Status.FAIL, str(exc))


def build_smoke_events() -> list[dict[str, Any]]:
    config = telemetry.SimulatorConfig(
        machine_count=SMOKE_MACHINE_COUNT,
        events_per_machine=SMOKE_EVENTS_PER_MACHINE,
        seed=SMOKE_SEED,
        start_time=telemetry.parse_start_time(SMOKE_START_TIME),
        interval_seconds=telemetry.DEFAULT_INTERVAL_SECONDS,
    )
    return telemetry.generate_events(config)


def produce_smoke_events(
    events: Sequence[dict[str, Any]],
) -> tuple[CheckResult, list[dict[str, Any]], list[str]]:
    kafka_config = kafka.load_kafka_config()
    producer = kafka.create_producer(kafka_config)
    result = kafka.produce_telemetry_events(producer, events, kafka_config)
    if result.failures or result.delivered != len(events):
        failures = "; ".join(result.failures) if result.failures else "delivery count mismatch"
        return CheckResult("Smoke Produce", Status.FAIL, failures), [], []

    payload_by_event_id = {
        str(event["event_id"]): telemetry.serialize_event(event) for event in events
    }
    expected_records = [
        {
            "kafka_key": delivery.key,
            "kafka_offset": delivery.offset,
            "kafka_partition": delivery.partition,
            "kafka_topic": delivery.topic,
            "raw_value": payload_by_event_id[delivery.event_id],
        }
        for delivery in result.deliveries
    ]
    expected_payloads = [payload_by_event_id[str(event["event_id"])] for event in events]
    return (
        CheckResult("Smoke Produce", Status.PASS, "Produced 6 deterministic telemetry events."),
        expected_records,
        expected_payloads,
    )


def run_bronze_ingestion() -> CheckResult:
    result = run_spark_bronze_docker.run_spark_bronze(timeout=SMOKE_TIMEOUT_SECONDS)
    if not result.succeeded:
        return CheckResult(
            "Bronze Ingestion",
            Status.FAIL,
            run_spark_bronze_docker.command_failure_message(result),
        )
    return CheckResult("Bronze Ingestion", Status.PASS, "Spark available-now ingestion completed.")


def build_inspection_command(
    expected_records: Sequence[dict[str, Any]],
    expected_payloads: Sequence[str],
) -> list[str]:
    return [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_bronze_docker.SPARK_SERVICE,
        run_spark_bronze_docker.SPARK_SUBMIT,
        "/workspace/scripts/inspect_spark_bronze.py",
        "--expected-payloads-json",
        json.dumps(list(expected_payloads), separators=(",", ":")),
        "--expected-records-json",
        json.dumps(list(expected_records), separators=(",", ":")),
    ]


def parse_inspection_summary(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Bronze inspection did not return a JSON summary.")


def inspect_bronze(
    expected_records: Sequence[dict[str, Any]],
    expected_payloads: Sequence[str],
) -> tuple[CheckResult, dict[str, Any] | None]:
    command = build_inspection_command(expected_records, expected_payloads)
    result = run_spark_bronze_docker.run_command(command, timeout=INSPECT_TIMEOUT_SECONDS)
    if not result.succeeded:
        return (
            CheckResult(
                "Bronze Inspection",
                Status.FAIL,
                run_spark_bronze_docker.command_failure_message(result),
            ),
            None,
        )
    try:
        summary = parse_inspection_summary(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        return CheckResult("Bronze Inspection", Status.FAIL, str(exc)), None
    return (
        CheckResult(
            "Bronze Inspection",
            Status.PASS,
            f"Bronze contains {summary['total_row_count']} row(s).",
        ),
        summary,
    )


def validate_bronze_summary(summary: Mapping[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    if summary.get("expected_payload_match_count") == 6 and summary.get(
        "expected_payloads_present"
    ):
        results.append(
            CheckResult("Expected Payloads", Status.PASS, "All six raw payloads are present.")
        )
    else:
        results.append(
            CheckResult("Expected Payloads", Status.FAIL, "Not all six raw payloads are present.")
        )

    if summary.get("expected_coordinate_match_count") == 6:
        results.append(
            CheckResult(
                "Expected Coordinates",
                Status.PASS,
                "All six produced Kafka coordinates are present.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Expected Coordinates",
                Status.FAIL,
                "Not all produced Kafka coordinates are present.",
            )
        )

    if summary.get("expected_coordinate_payload_mismatch_count") == 0:
        results.append(
            CheckResult("Raw Payload Preservation", Status.PASS, "Raw payloads match exactly.")
        )
    else:
        results.append(
            CheckResult("Raw Payload Preservation", Status.FAIL, "Raw payload mismatch found.")
        )

    if (
        summary.get("expected_key_mismatch_count") == 0
        and summary.get("expected_coordinate_key_mismatch_count") == 0
    ):
        results.append(
            CheckResult("Kafka Key Preservation", Status.PASS, "Kafka keys match machine_code.")
        )
    else:
        results.append(
            CheckResult("Kafka Key Preservation", Status.FAIL, "Kafka key mismatch found.")
        )

    if (
        summary.get("expected_metadata_missing_count") == 0
        and summary.get("expected_coordinate_metadata_missing_count") == 0
    ):
        results.append(
            CheckResult(
                "Kafka Metadata", Status.PASS, "Kafka topic/partition/offset metadata exists."
            )
        )
    else:
        results.append(
            CheckResult("Kafka Metadata", Status.FAIL, "Expected Kafka metadata is missing.")
        )

    if summary.get("duplicate_kafka_coordinate_count") == 0:
        results.append(
            CheckResult(
                "Kafka Coordinate Uniqueness", Status.PASS, "No duplicate Kafka coordinates."
            )
        )
    else:
        results.append(
            CheckResult(
                "Kafka Coordinate Uniqueness", Status.FAIL, "Duplicate Kafka coordinates found."
            )
        )

    duplicate_rows = int(summary.get("expected_payload_duplicate_rows") or 0)
    duplicate_message = (
        f"Expected payload duplicate row count is {duplicate_rows}; Bronze preserves Kafka records."
    )
    results.append(
        CheckResult(
            "Event ID Duplicate Policy",
            Status.PASS,
            duplicate_message,
        )
    )
    if int(summary.get("null_kafka_key_count") or 0) > 0:
        results.append(
            CheckResult(
                "Null Kafka Keys",
                Status.WARN,
                "Bronze contains null Kafka keys from existing records.",
                mandatory=False,
            )
        )
    if int(summary.get("null_raw_value_count") or 0) > 0:
        results.append(
            CheckResult(
                "Null Raw Values",
                Status.WARN,
                "Bronze contains null raw values from existing records.",
                mandatory=False,
            )
        )
    return results


def run_checks() -> tuple[list[CheckResult], dict[str, Any] | None]:
    results = [check_kafka_health(), check_spark_health(), check_kafka_topic()]
    if any(result.status is Status.FAIL and result.mandatory for result in results):
        return results, None

    events = build_smoke_events()
    produce_result, expected_records, expected_payloads = produce_smoke_events(events)
    results.append(produce_result)
    if produce_result.status is Status.FAIL:
        return results, None

    ingestion_result = run_bronze_ingestion()
    results.append(ingestion_result)
    if ingestion_result.status is Status.FAIL:
        return results, None

    inspection_result, summary = inspect_bronze(expected_records, expected_payloads)
    results.append(inspection_result)
    if summary is not None:
        results.extend(validate_bronze_summary(summary))
    return results, summary


def print_results(results: Sequence[CheckResult], summary: Mapping[str, Any] | None) -> None:
    print("Industrial Fleet Intelligence Platform Spark Bronze validation")
    print()
    for result in results:
        print(f"{result.status.value} {result.name}: {result.message}")
    if summary is not None:
        print()
        print("Bronze inspection summary:")
        print(json.dumps(summary, indent=2, sort_keys=True))
    mandatory_failures = [
        result for result in results if result.status is Status.FAIL and result.mandatory
    ]
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    pass_count = sum(1 for result in results if result.status is Status.PASS)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {len(mandatory_failures)} FAIL")


def main() -> int:
    results, summary = run_checks()
    print_results(results, summary)
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

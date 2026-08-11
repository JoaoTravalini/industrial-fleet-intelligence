"""Validate local Kafka infrastructure and deterministic telemetry streaming."""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.simulator import telemetry  # noqa: E402
from services.streaming import kafka  # noqa: E402

SERVICE_NAME = "kafka"
CONTAINER_NAME = "industrial-fleet-kafka"
DEFAULT_TIMEOUT_SECONDS = 20
SMOKE_TIMEOUT_SECONDS = 45.0


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.error

    @property
    def output(self) -> str:
        parts = [normalize_output(part.strip()) for part in (self.stdout, self.stderr) if part]
        return "\n".join(part for part in parts if part).strip()


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    mandatory: bool = True


def normalize_output(text: str) -> str:
    return text.replace("\x00", "")


def run_command(args: Sequence[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(tuple(args), None, error="command not found")
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        return CommandResult(
            tuple(args),
            None,
            stdout=stdout or "",
            stderr=stderr or "",
            error=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return CommandResult(tuple(args), None, error=str(exc))

    return CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)


def command_failure_message(result: CommandResult) -> str:
    if result.error:
        return result.error
    if result.output:
        return f"command exited with code {result.returncode}: {result.output.splitlines()[0]}"
    return f"command exited with code {result.returncode}"


def check_compose_service() -> CheckResult:
    result = run_command(["docker", "compose", "config", "--services"])
    if not result.succeeded:
        return CheckResult(
            "Compose Service",
            Status.FAIL,
            f"Could not read Compose services: {command_failure_message(result)}",
        )
    services = [line.strip() for line in normalize_output(result.output).splitlines()]
    if SERVICE_NAME in services:
        return CheckResult("Compose Service", Status.PASS, "Kafka service is defined.")
    return CheckResult("Compose Service", Status.FAIL, "Compose service 'kafka' was not found.")


def parse_inspect_state(output: str) -> tuple[bool, str | None]:
    value = normalize_output(output).strip().lower()
    if not value:
        return False, None
    running_text, _, health = value.partition("|")
    return running_text == "true", health or None


def check_container_state() -> CheckResult:
    result = run_command(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            CONTAINER_NAME,
        ]
    )
    if not result.succeeded:
        return CheckResult(
            "Container State",
            Status.FAIL,
            f"Could not inspect Kafka container: {command_failure_message(result)}",
        )
    running, health = parse_inspect_state(result.output)
    if not running:
        return CheckResult("Container State", Status.FAIL, "Kafka container is not running.")
    if health == "healthy":
        return CheckResult(
            "Container State", Status.PASS, "Kafka container is running and healthy."
        )
    return CheckResult(
        "Container State",
        Status.FAIL,
        f"Kafka container is running but health status is {health or 'unknown'}.",
    )


def check_broker_and_topic(config: kafka.KafkaConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        admin_client = kafka.create_admin_client(config)
        kafka.check_broker_connectivity(admin_client)
        results.append(
            CheckResult(
                "Broker Connectivity",
                Status.PASS,
                f"Connected to Kafka at {config.bootstrap_servers_host}.",
            )
        )
        description = kafka.get_topic_description(admin_client, config.telemetry_topic)
        if description is None:
            results.append(
                CheckResult(
                    "Topic Configuration",
                    Status.FAIL,
                    f"Kafka topic {config.telemetry_topic} does not exist.",
                )
            )
        else:
            kafka.validate_topic_description(description, config)
            results.append(
                CheckResult(
                    "Topic Configuration",
                    Status.PASS,
                    "Topic has "
                    f"{description.partitions} partitions and RF "
                    f"{description.replication_factor}.",
                )
            )
    except (ImportError, kafka.KafkaConfigError, kafka.KafkaStreamingError) as exc:
        results.append(CheckResult("Broker/Topic", Status.FAIL, str(exc)))
    return results


def smoke_events() -> list[dict[str, object]]:
    config = telemetry.SimulatorConfig(
        machine_count=kafka.SMOKE_MACHINE_COUNT,
        events_per_machine=kafka.SMOKE_EVENTS_PER_MACHINE,
        seed=telemetry.DEFAULT_RANDOM_SEED,
        start_time=telemetry.parse_start_time(telemetry.DEFAULT_START_TIME),
        interval_seconds=telemetry.DEFAULT_INTERVAL_SECONDS,
    )
    return telemetry.generate_events(config)


def check_smoke_flow(config: kafka.KafkaConfig) -> list[CheckResult]:
    events = smoke_events()
    group_id = f"industrial-fleet-kafka-smoke-{uuid.uuid4()}"
    results: list[CheckResult] = []
    try:
        producer = kafka.create_producer(config)
        produce_result = kafka.produce_telemetry_events(producer, events, config)
        if produce_result.failures:
            results.append(
                CheckResult(
                    "Smoke Produce",
                    Status.FAIL,
                    "; ".join(produce_result.failures),
                )
            )
            return results
        results.append(
            CheckResult(
                "Smoke Produce",
                Status.PASS,
                f"Produced {produce_result.delivered} deterministic telemetry events.",
            )
        )

        consumer = kafka.create_consumer(config, group_id=group_id, auto_offset_reset="earliest")
        records = kafka.consume_expected_records(
            consumer,
            events,
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
        )
        validation = kafka.validate_expected_records(records, events)
        results.append(
            CheckResult(
                "Smoke Consume",
                Status.PASS,
                f"Consumed and matched {validation.matched_event_count} expected events.",
            )
        )
        results.append(
            CheckResult(
                "Partition Policy",
                Status.PASS,
                "Each machine key stayed on one partition with monotonic per-machine order.",
            )
        )
    except (
        ImportError,
        telemetry.TelemetryValidationError,
        kafka.KafkaConfigError,
        kafka.KafkaStreamingError,
    ) as exc:
        results.append(CheckResult("Smoke Flow", Status.FAIL, str(exc)))
    return results


def check_summary_report(config: kafka.KafkaConfig) -> CheckResult:
    path = kafka.write_integration_summary(PROJECT_ROOT, config)
    try:
        content = path.read_text(encoding="utf-8")
        loaded = kafka.build_integration_summary(config)
        if content != json_dumps_report(loaded):
            return CheckResult(
                "Summary Report", Status.FAIL, "Kafka summary report is not deterministic."
            )
    except OSError as exc:
        return CheckResult("Summary Report", Status.FAIL, f"Could not read summary report: {exc}")
    return CheckResult(
        "Summary Report",
        Status.PASS,
        "Tracked Kafka integration summary contains static configuration only.",
    )


def json_dumps_report(report: dict[str, object]) -> str:
    import json

    return json.dumps(report, indent=2, sort_keys=False) + "\n"


def run_checks() -> list[CheckResult]:
    results = [check_compose_service(), check_container_state()]
    try:
        config = kafka.load_kafka_config()
    except kafka.KafkaConfigError as exc:
        results.append(CheckResult("Kafka Config", Status.FAIL, str(exc)))
        return results

    results.append(CheckResult("Kafka Config", Status.PASS, "Kafka config file is valid."))
    results.extend(check_broker_and_topic(config))
    if not any(result.status is Status.FAIL for result in results):
        results.extend(check_smoke_flow(config))
    results.append(check_summary_report(config))
    return results


def print_results(results: Sequence[CheckResult]) -> None:
    for result in results:
        print(f"{result.status.value} {result.name}: {result.message}")
    mandatory_failures = [
        result for result in results if result.status is Status.FAIL and result.mandatory
    ]
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    print("Summary:")
    print(f"PASS: {sum(1 for result in results if result.status is Status.PASS)}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {len(mandatory_failures)}")


def main() -> int:
    results = run_checks()
    print_results(results)
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

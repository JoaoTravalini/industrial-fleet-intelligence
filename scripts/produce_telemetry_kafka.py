"""Produce a finite deterministic synthetic telemetry batch to Kafka."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.simulator import telemetry  # noqa: E402
from services.streaming import kafka  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce deterministic synthetic telemetry events to local Kafka."
    )
    parser.add_argument("--machines", type=int, default=telemetry.DEFAULT_SAMPLE_MACHINE_COUNT)
    parser.add_argument(
        "--events-per-machine",
        type=int,
        default=telemetry.DEFAULT_SAMPLE_EVENTS_PER_MACHINE,
    )
    parser.add_argument("--seed", type=int, default=telemetry.DEFAULT_RANDOM_SEED)
    parser.add_argument("--start-time", default=telemetry.DEFAULT_START_TIME)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=telemetry.DEFAULT_INTERVAL_SECONDS,
    )
    return parser.parse_args()


def build_simulator_config(args: argparse.Namespace) -> telemetry.SimulatorConfig:
    return telemetry.SimulatorConfig(
        machine_count=args.machines,
        events_per_machine=args.events_per_machine,
        seed=args.seed,
        start_time=telemetry.parse_start_time(args.start_time),
        interval_seconds=args.interval_seconds,
    )


def main() -> int:
    args = parse_args()
    try:
        kafka_config = kafka.load_kafka_config()
        simulator_config = build_simulator_config(args)
        events = telemetry.generate_events(simulator_config)
        producer = kafka.create_producer(kafka_config)
        result = kafka.produce_telemetry_events(producer, events, kafka_config)
        machine_codes = sorted({str(event["machine_code"]) for event in events})
        first_time = str(events[0]["event_time"])
        last_time = str(events[-1]["event_time"])

        print(f"Broker: {kafka_config.bootstrap_servers_host}")
        print(f"Topic: {kafka_config.telemetry_topic}")
        print(f"Events attempted: {result.attempted}")
        print(f"Events delivered: {result.delivered}")
        print(f"Delivery failures: {result.failed}")
        print(f"Machine range: {machine_codes[0]} to {machine_codes[-1]}")
        print(f"Event time range: {first_time} to {last_time}")
        if result.failures:
            for failure in result.failures:
                print(f"FAIL {failure}", file=sys.stderr)
            return 1
    except (
        ImportError,
        ValueError,
        telemetry.TelemetryValidationError,
        kafka.KafkaConfigError,
        kafka.KafkaStreamingError,
    ) as exc:
        print(f"FAIL Kafka telemetry production failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Consume a finite batch of local Kafka telemetry records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.streaming import kafka  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume a finite number of validated telemetry events from local Kafka."
    )
    parser.add_argument("--max-messages", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--from-beginning", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auto_offset_reset = "earliest" if args.from_beginning else "latest"
    try:
        config = kafka.load_kafka_config()
        consumer = kafka.create_consumer(
            config,
            group_id=args.group_id,
            auto_offset_reset=auto_offset_reset,
        )
        records = kafka.poll_consumed_records(
            consumer,
            max_messages=args.max_messages,
            timeout_seconds=args.timeout_seconds,
        )
        for record in records:
            print(json.dumps(kafka.record_to_json_object(record), separators=(",", ":")))
        print(f"Consumed records: {len(records)}", file=sys.stderr)
    except (ImportError, kafka.KafkaConfigError, kafka.KafkaStreamingError) as exc:
        print(f"FAIL Kafka telemetry consumption failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

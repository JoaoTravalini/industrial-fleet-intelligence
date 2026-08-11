"""Create and verify the local Kafka telemetry topic."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.streaming import kafka  # noqa: E402


def main() -> int:
    try:
        config = kafka.load_kafka_config()
        admin_client = kafka.create_admin_client(config)
        kafka.check_broker_connectivity(admin_client)
        print(f"PASS Kafka broker connectivity verified: {config.bootstrap_servers_host}")

        description, action = kafka.create_telemetry_topic(admin_client, config)
        kafka.write_integration_summary(PROJECT_ROOT, config)
        print(f"PASS Kafka topic ready: {description.name}")
        print(f"Topic action: {action}")
        print(f"Partitions: {description.partitions}")
        print(f"Replication factor: {description.replication_factor}")
        print("Summary: " + kafka.summary_path(PROJECT_ROOT).relative_to(PROJECT_ROOT).as_posix())
    except (ImportError, kafka.KafkaConfigError, kafka.KafkaStreamingError) as exc:
        print(f"FAIL Kafka topic setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

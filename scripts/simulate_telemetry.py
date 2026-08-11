"""Run deterministic local synthetic telemetry simulation without Kafka."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.simulator import telemetry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic telemetry JSONL."
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
        "--interval-seconds", type=float, default=telemetry.DEFAULT_INTERVAL_SECONDS
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> telemetry.SimulatorConfig:
    start_time = telemetry.parse_start_time(args.start_time)
    return telemetry.SimulatorConfig(
        machine_count=args.machines,
        events_per_machine=args.events_per_machine,
        seed=args.seed,
        start_time=start_time,
        interval_seconds=args.interval_seconds,
    )


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        config = build_config(args)
        events = telemetry.generate_events(config)
        content = telemetry.serialize_events_jsonl(events)
        if args.output is None:
            print(content, end="")
        else:
            output_path = resolve_output(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            print(
                "PASS telemetry simulation written to "
                + output_path.relative_to(PROJECT_ROOT).as_posix()
            )
            print(f"Events: {len(events)}")
            print(f"Machines: {config.machine_count}")
    except (OSError, telemetry.TelemetryValidationError, ValueError) as exc:
        print(f"FAIL telemetry simulation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

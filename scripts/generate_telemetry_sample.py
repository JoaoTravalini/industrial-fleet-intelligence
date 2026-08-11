"""Generate the tracked canonical synthetic telemetry sample."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.simulator import telemetry  # noqa: E402


def main() -> int:
    try:
        summary = telemetry.generate_sample(PROJECT_ROOT)
    except (OSError, telemetry.TelemetryValidationError, ValueError) as exc:
        print(f"FAIL telemetry sample generation failed: {exc}", file=sys.stderr)
        return 1

    print("PASS canonical telemetry sample generated.")
    print(f"Events: {summary['event_count']}")
    print(f"Machines: {summary['machine_count']}")
    print(f"Events per machine: {summary['events_per_machine']}")
    print(f"Sample SHA-256: {summary['sample_sha256']}")
    print(f"Output: {telemetry.sample_path(PROJECT_ROOT).relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Summary: {telemetry.summary_path(PROJECT_ROOT).relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

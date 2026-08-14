"""Package the frozen operational telemetry anomaly model from canonical Silver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomaly import telemetry_detector  # noqa: E402
from scripts import run_spark_telemetry_anomaly_features_docker  # noqa: E402


def command_error(
    result: run_spark_telemetry_anomaly_features_docker.CommandResult,
) -> RuntimeError:
    return RuntimeError(run_spark_telemetry_anomaly_features_docker.command_failure_message(result))


def export_feature_records() -> None:
    ok, message = run_spark_telemetry_anomaly_features_docker.verify_spark_container()
    if not ok:
        raise RuntimeError(message)
    result = run_spark_telemetry_anomaly_features_docker.run_spark_telemetry_anomaly_features()
    if not result.succeeded:
        raise command_error(result)


def main() -> int:
    print("Industrial Fleet Intelligence Platform telemetry anomaly model packaging")
    print()
    try:
        export_feature_records()
        records = telemetry_detector.load_feature_records_from_export(PROJECT_ROOT)
        summary = telemetry_detector.package_anomaly_artifact(records, root=PROJECT_ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score canonical Silver telemetry with the trusted anomaly model artifact."""

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
    print("Industrial Fleet Intelligence Platform telemetry anomaly scoring")
    print()
    try:
        if not telemetry_detector.artifact_path(PROJECT_ROOT).exists():
            raise FileNotFoundError(
                "Packaged telemetry anomaly artifact is missing. Run "
                ".\\.venv\\Scripts\\python.exe scripts\\package_telemetry_anomaly_model.py first."
            )
        export_feature_records()
        summary = telemetry_detector.run_scoring_pipeline(root=PROJECT_ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

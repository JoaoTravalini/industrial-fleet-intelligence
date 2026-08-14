"""Build the frozen deterministic drift reference profiles."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.monitoring import drift  # noqa: E402
from scripts import run_spark_telemetry_anomaly_features_docker  # noqa: E402


def command_error(
    result: run_spark_telemetry_anomaly_features_docker.CommandResult,
) -> RuntimeError:
    return RuntimeError(run_spark_telemetry_anomaly_features_docker.command_failure_message(result))


def export_anomaly_reference_features() -> None:
    ok, message = run_spark_telemetry_anomaly_features_docker.verify_spark_container()
    if not ok:
        raise RuntimeError(message)
    result = run_spark_telemetry_anomaly_features_docker.run_spark_telemetry_anomaly_features()
    if not result.succeeded:
        raise command_error(result)


def build_and_write() -> tuple[dict[str, object], str]:
    config = drift.load_config(PROJECT_ROOT)
    export_anomaly_reference_features()
    first_profile = drift.build_reference_profile(root=PROJECT_ROOT, config=config)
    first_bytes = drift.deterministic_bytes(first_profile)
    profile_path = drift.write_reference_profile(first_profile, PROJECT_ROOT)
    first_hash = drift.file_sha256(profile_path)

    second_profile = drift.build_reference_profile(root=PROJECT_ROOT, config=config)
    second_bytes = drift.deterministic_bytes(second_profile)
    if first_bytes != second_bytes:
        raise drift.DriftMonitoringError("Reference profile is not byte deterministic.")
    drift.write_reference_profile(second_profile, PROJECT_ROOT)
    second_hash = drift.file_sha256(profile_path)
    if first_hash != second_hash:
        raise drift.DriftMonitoringError("Reference profile hash changed on immediate rebuild.")

    drift.write_static_summary(PROJECT_ROOT)
    return second_profile, second_hash


def main() -> int:
    print("Industrial Fleet Intelligence Platform drift reference profile builder")
    print()
    try:
        profile, profile_hash = build_and_write()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    ai4i = profile[drift.AI4I_SCOPE]
    anomaly = profile[drift.ANOMALY_SCOPE]
    print(f"AI4I reference count: {ai4i['reference_row_count']}")
    print("AI4I features: " + ", ".join(feature["feature_name"] for feature in ai4i["features"]))
    print(f"Anomaly reference count: {anomaly['reference_row_count']}")
    print(
        "Anomaly features: " + ", ".join(feature["feature_name"] for feature in anomaly["features"])
    )
    print(f"Reference profile SHA-256: {profile_hash}")
    print("PASS Drift reference profile written deterministically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

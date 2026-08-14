"""Calculate deterministic current input-data drift diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.monitoring import drift  # noqa: E402
from scripts import (  # noqa: E402
    run_spark_ai4i_adapter_docker,
    run_spark_telemetry_anomaly_features_docker,
)


def command_error(result) -> RuntimeError:
    return RuntimeError(
        result.error or result.output or f"command exited with code {result.returncode}"
    )


def rebuild_current_ai4i_adapter() -> None:
    ok, message = run_spark_ai4i_adapter_docker.verify_spark_container()
    if not ok:
        raise RuntimeError(message)
    result = run_spark_ai4i_adapter_docker.run_spark_ai4i_adapter()
    if not result.succeeded:
        raise command_error(result)


def rebuild_current_anomaly_features() -> None:
    ok, message = run_spark_telemetry_anomaly_features_docker.verify_spark_container()
    if not ok:
        raise RuntimeError(message)
    result = run_spark_telemetry_anomaly_features_docker.run_spark_telemetry_anomaly_features()
    if not result.succeeded:
        raise command_error(result)


def build_current_report() -> dict[str, object]:
    config = drift.load_config(PROJECT_ROOT)
    reference_profile = drift.load_reference_profile(PROJECT_ROOT, config)
    reference_hash = drift.file_sha256(drift.reference_profile_path(PROJECT_ROOT))

    rebuild_current_ai4i_adapter()
    rebuild_current_anomaly_features()

    ai4i_records = drift.ai4i_current_records_from_adapter(PROJECT_ROOT)
    anomaly_records = drift.anomaly_current_records_from_export(PROJECT_ROOT)
    report = drift.build_drift_report(
        reference_profile=reference_profile,
        reference_profile_sha256=reference_hash,
        ai4i_current_records=ai4i_records,
        anomaly_current_records=anomaly_records,
        config=config,
    )
    drift.write_drift_report(report, PROJECT_ROOT)
    return report


def print_scope_summary(title: str, scope: dict[str, object]) -> None:
    print(f"{title} current rows: {scope['current_record_count']}")
    print(f"{title} overall status: {scope['overall_status']}")
    for feature in scope["features"]:
        print(f"  {feature['feature_name']}: PSI {float(feature['psi']):.6f}, {feature['status']}")


def main() -> int:
    print("Industrial Fleet Intelligence Platform data drift monitor")
    print()
    try:
        report = build_current_report()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print_scope_summary("AI4I", report[drift.AI4I_SCOPE])
    print()
    print_scope_summary("Anomaly", report[drift.ANOMALY_SCOPE])
    print()
    print(f"Reference profile SHA-256: {report['reference_profile_sha256']}")
    print(f"Runtime report: {drift.DRIFT_REPORT_RELATIVE_PATH.as_posix()}")
    print("PASS Data drift report written deterministically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

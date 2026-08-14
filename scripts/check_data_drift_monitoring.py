"""Integration validator for deterministic data drift monitoring."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomaly import telemetry_detector  # noqa: E402
from ml.inference import ai4i_predictor  # noqa: E402
from ml.monitoring import drift  # noqa: E402
from scripts import (  # noqa: E402
    apply_migrations,
    check_postgres,
    check_schema,
    inspect_drift_state,
    monitor_data_drift,
    persist_drift_report,
)
from services.database import drift_monitoring  # noqa: E402


class Status(StrEnum):
    """Validation status values printed by the drift checker."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    mandatory: bool = True


@dataclass(frozen=True)
class ValidationRun:
    results: list[CheckResult]
    report: dict[str, Any] | None
    first_persistence: drift_monitoring.PersistenceSummary | None
    second_persistence: drift_monitoring.PersistenceSummary | None
    state: drift_monitoring.DriftStateSummary | None


def pass_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.PASS, message)


def warn_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.WARN, message, mandatory=False)


def fail_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.FAIL, message)


def aggregate_external_results(name: str, failed: bool, pass_message: str) -> CheckResult:
    if failed:
        return fail_result(name, "One or more required checks failed.")
    return pass_result(name, pass_message)


def query_count(sql: str) -> int:
    result = apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise RuntimeError(apply_migrations.command_failure_message(result))
    return drift_monitoring.parse_count_output(result.output)


def current_table_counts() -> dict[str, int]:
    return {
        "alerts": query_count("SELECT count(*) FROM alerts;"),
        "anomalies": query_count("SELECT count(*) FROM anomalies;"),
        "machine_health": query_count("SELECT count(*) FROM machine_health;"),
        "model_predictions": query_count("SELECT count(*) FROM model_predictions;"),
    }


def current_artifact_hashes() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    ai4i_path = ai4i_predictor.artifact_path(PROJECT_ROOT)
    anomaly_path = telemetry_detector.artifact_path(PROJECT_ROOT)
    hashes["ai4i_model_artifact_sha256"] = (
        ai4i_predictor.file_sha256(ai4i_path) if ai4i_path.exists() else None
    )
    hashes["anomaly_model_artifact_sha256"] = (
        telemetry_detector.file_sha256(anomaly_path) if anomaly_path.exists() else None
    )
    hashes["ai4i_final_config_sha256"] = drift.file_sha256(
        PROJECT_ROOT / ai4i_predictor.ai4i_final_evaluation.CONFIG_RELATIVE_PATH
    )
    hashes["anomaly_config_sha256"] = drift.file_sha256(
        telemetry_detector.config_path(PROJECT_ROOT)
    )
    return hashes


def validate_config_and_reference() -> list[CheckResult]:
    results: list[CheckResult] = []
    config = drift.load_config(PROJECT_ROOT)
    drift.validate_config(config)
    results.append(pass_result("Drift Config", "Static drift monitoring config is valid."))

    profile = drift.load_reference_profile(PROJECT_ROOT, config)
    results.append(pass_result("Reference Profile", "Frozen drift reference profile is valid."))
    ai4i_identity = profile[drift.AI4I_SCOPE]["reference_identity"]
    if drift.FORBIDDEN_AI4I_SOURCE_PATH in ai4i_identity["source_paths"]:
        results.append(fail_result("AI4I Test Guard", "AI4I test.csv was used as a source path."))
    elif drift.FORBIDDEN_AI4I_SOURCE_PATH in ai4i_identity["forbidden_source_paths"]:
        results.append(pass_result("AI4I Test Guard", "AI4I reference forbids test.csv."))
    else:
        results.append(fail_result("AI4I Test Guard", "AI4I test.csv is not explicitly forbidden."))

    ai4i_features = tuple(
        feature["feature_name"] for feature in profile[drift.AI4I_SCOPE]["features"]
    )
    anomaly_features = tuple(
        feature["feature_name"] for feature in profile[drift.ANOMALY_SCOPE]["features"]
    )
    results.append(
        pass_result("AI4I Drift Features", "Exact six frozen AI4I inputs are monitored.")
        if ai4i_features == drift.AI4I_FEATURES
        else fail_result("AI4I Drift Features", "AI4I drift feature contract changed.")
    )
    results.append(
        pass_result("Anomaly Drift Features", "Exact vibration/pressure inputs are monitored.")
        if anomaly_features == drift.ANOMALY_FEATURES
        else fail_result("Anomaly Drift Features", "Anomaly drift feature contract changed.")
    )
    return results


def validate_report(report: Mapping[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    config = drift.load_config(PROJECT_ROOT)
    drift.validate_drift_report(report, config)
    results.append(pass_result("Runtime Report", "Runtime drift report is structurally valid."))

    finite_psi = True
    band_matches = True
    for scope_name in (drift.AI4I_SCOPE, drift.ANOMALY_SCOPE):
        for metric in report[scope_name]["features"]:
            psi = float(metric["psi"])
            finite_psi = finite_psi and math.isfinite(psi) and psi >= 0
            band_matches = band_matches and metric["status"] == drift.status_for_psi(psi, config)
    results.append(
        pass_result("PSI Values", "All PSI values are finite and non-negative.")
        if finite_psi
        else fail_result("PSI Values", "A PSI value is not finite or is negative.")
    )
    results.append(
        pass_result("Monitoring Bands", "All feature statuses match PSI thresholds.")
        if band_matches
        else fail_result("Monitoring Bands", "A status does not match PSI thresholds.")
    )

    hashes_valid = all(
        drift.HASH_PATTERN.fullmatch(str(value)) is not None
        for value in (
            report["reference_profile_sha256"],
            report[drift.AI4I_SCOPE]["current_data_hash"],
            report[drift.ANOMALY_SCOPE]["current_data_hash"],
        )
    )
    results.append(
        pass_result("Current Hashes", "Reference and current population hashes are valid.")
        if hashes_valid
        else fail_result("Current Hashes", "A current population hash is invalid.")
    )

    identities = [
        (scope_name, metric["feature_name"])
        for scope_name in (drift.AI4I_SCOPE, drift.ANOMALY_SCOPE)
        for metric in report[scope_name]["features"]
    ]
    results.append(
        pass_result("Metric Identities", "Runtime feature metric identities are unique.")
        if len(identities) == len(set(identities))
        else fail_result("Metric Identities", "Runtime feature metric identities are duplicated.")
    )
    return results


def validate_report_determinism() -> tuple[list[CheckResult], dict[str, Any]]:
    first = monitor_data_drift.build_current_report()
    first_bytes = drift.drift_report_path(PROJECT_ROOT).read_bytes()
    second = monitor_data_drift.build_current_report()
    second_bytes = drift.drift_report_path(PROJECT_ROOT).read_bytes()
    result = (
        pass_result("Report Determinism", "Repeated monitoring produced identical bytes.")
        if first_bytes == second_bytes and first == second
        else fail_result("Report Determinism", "Repeated monitoring changed the report bytes.")
    )
    return [result], second


def validate_persisted_report(report: Mapping[str, Any]) -> list[CheckResult]:
    existing = persist_drift_report.load_existing_snapshots(dict(report))
    reuse = drift_monitoring.summarize_report_reuse(report, existing)
    expected_metrics = len(drift_monitoring.metric_values(report))
    return [
        pass_result("Persisted Snapshot", "Current drift snapshot exists and matches report.")
        if reuse.existing_identical_snapshots_reused == 1 and not reuse.conflicts
        else fail_result("Persisted Snapshot", "Persisted snapshot differs from runtime report."),
        pass_result("Persisted Metrics", "All feature metrics exist and match report.")
        if reuse.existing_identical_feature_metrics_reused == expected_metrics
        and not reuse.conflicts
        else fail_result("Persisted Metrics", "A persisted feature metric differs or is missing."),
    ]


def validate_unchanged_counts(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> list[CheckResult]:
    return [
        pass_result("Alerts Unchanged", "Drift monitoring did not create alerts.")
        if before["alerts"] == after["alerts"]
        else fail_result("Alerts Unchanged", "Alert row count changed."),
        pass_result("Predictions Unchanged", "AI4I prediction row count is unchanged.")
        if before["model_predictions"] == after["model_predictions"]
        else fail_result("Predictions Unchanged", "AI4I prediction row count changed."),
        pass_result("Anomaly Audit Unchanged", "Anomaly audit row count is unchanged.")
        if before["anomalies"] == after["anomalies"]
        else fail_result("Anomaly Audit Unchanged", "Anomaly audit row count changed."),
        pass_result("Machine Health Unchanged", "machine_health row count is unchanged.")
        if before["machine_health"] == after["machine_health"]
        else fail_result("Machine Health Unchanged", "machine_health row count changed."),
    ]


def validate_unchanged_hashes(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> list[CheckResult]:
    return [
        pass_result("AI4I Model Unchanged", "AI4I model artifact/config hashes are unchanged.")
        if before["ai4i_model_artifact_sha256"] == after["ai4i_model_artifact_sha256"]
        and before["ai4i_final_config_sha256"] == after["ai4i_final_config_sha256"]
        else fail_result("AI4I Model Unchanged", "AI4I model artifact or config hash changed."),
        pass_result(
            "Anomaly Model Unchanged",
            "Anomaly model artifact/config hashes are unchanged.",
        )
        if before["anomaly_model_artifact_sha256"] == after["anomaly_model_artifact_sha256"]
        and before["anomaly_config_sha256"] == after["anomaly_config_sha256"]
        else fail_result(
            "Anomaly Model Unchanged",
            "Anomaly model artifact or config hash changed.",
        ),
    ]


def validate_state(state: drift_monitoring.DriftStateSummary) -> list[CheckResult]:
    return [
        pass_result("DB State", f"{state.snapshot_count} drift snapshot(s) stored.")
        if state.snapshot_count >= 1
        else fail_result("DB State", "No drift snapshots are stored."),
        pass_result("Snapshot Identity Unique", "No duplicate drift snapshot identity exists.")
        if state.duplicate_snapshot_identity_count == 0
        else fail_result("Snapshot Identity Unique", "Duplicate drift snapshot identity found."),
        pass_result("Metric Identity Unique", "No duplicate feature metric identity exists.")
        if state.duplicate_feature_metric_identity_count == 0
        else fail_result("Metric Identity Unique", "Duplicate feature metric identity found."),
    ]


def run_checks() -> ValidationRun:
    results: list[CheckResult] = []

    postgres_results = check_postgres.run_checks()
    postgres_failed = check_postgres.exit_code_for(postgres_results) != 0
    results.append(
        aggregate_external_results(
            "PostgreSQL Validator",
            postgres_failed,
            "Existing PostgreSQL validator passed.",
        )
    )
    if postgres_failed:
        return ValidationRun(results, None, None, None, None)

    migration_results = apply_migrations.run_migrations()
    migration_failed = apply_migrations.exit_code_for(migration_results) != 0
    results.append(
        aggregate_external_results(
            "Migration Runner",
            migration_failed,
            "Database migrations are current.",
        )
    )
    if migration_failed:
        return ValidationRun(results, None, None, None, None)

    schema_results = check_schema.run_checks()
    schema_failed = check_schema.exit_code_for(schema_results) != 0
    results.append(
        aggregate_external_results(
            "Schema Validator",
            schema_failed,
            "Schema validator passed with drift monitoring migration applied.",
        )
    )
    if schema_failed:
        return ValidationRun(results, None, None, None, None)

    results.extend(validate_config_and_reference())
    if any(result.status is Status.FAIL for result in results):
        return ValidationRun(results, None, None, None, None)

    counts_before = current_table_counts()
    hashes_before = current_artifact_hashes()

    determinism_results, report = validate_report_determinism()
    results.extend(determinism_results)
    results.extend(validate_report(report))
    if any(result.status is Status.FAIL for result in results):
        return ValidationRun(results, report, None, None, None)

    first = persist_drift_report.persist_drift_report().summary
    results.append(
        pass_result(
            "First Persistence",
            f"Inserted {first.new_snapshots_inserted} snapshot(s) and "
            f"{first.new_feature_metrics_inserted} metric row(s).",
        )
    )
    results.extend(validate_persisted_report(report))

    second = persist_drift_report.persist_drift_report().summary
    idempotent = (
        second.new_snapshots_inserted == 0
        and second.new_feature_metrics_inserted == 0
        and second.conflicts == 0
    )
    results.append(
        pass_result("Repeated Persistence", "Second persistence run was idempotent.")
        if idempotent
        else fail_result("Repeated Persistence", "Second persistence run inserted or conflicted.")
    )

    state = inspect_drift_state.inspect_state().summary
    results.extend(validate_state(state))

    counts_after = current_table_counts()
    hashes_after = current_artifact_hashes()
    results.extend(validate_unchanged_counts(counts_before, counts_after))
    results.extend(validate_unchanged_hashes(hashes_before, hashes_after))

    if report[drift.ANOMALY_SCOPE]["overall_status"] == "stable":
        results.append(
            warn_result(
                "Anomaly Drift",
                "Anomaly current inputs are stable; this is expected when Silver equals "
                "the frozen baseline.",
            )
        )

    return ValidationRun(results, report, first, second, state)


def print_report(run: ValidationRun) -> None:
    print("Industrial Fleet Intelligence Platform data drift monitoring validation")
    print()

    name_width = max(len(result.name) for result in run.results)
    for result in run.results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")

    pass_count = sum(1 for result in run.results if result.status is Status.PASS)
    warn_count = sum(1 for result in run.results if result.status is Status.WARN)
    fail_count = sum(1 for result in run.results if result.status is Status.FAIL)

    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    if run.report is not None:
        ai4i = run.report[drift.AI4I_SCOPE]
        anomaly = run.report[drift.ANOMALY_SCOPE]
        print()
        print(f"AI4I current rows: {ai4i['current_record_count']}")
        print(f"AI4I overall status: {ai4i['overall_status']}")
        print(f"Anomaly current rows: {anomaly['current_record_count']}")
        print(f"Anomaly overall status: {anomaly['overall_status']}")


def exit_code_for(results: Sequence[CheckResult]) -> int:
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


def main() -> int:
    try:
        run = run_checks()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform data drift monitoring validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(run)
    return exit_code_for(run.results)


if __name__ == "__main__":
    raise SystemExit(main())

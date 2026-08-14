"""Integration validator for operational telemetry anomaly detection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomaly import telemetry_detector  # noqa: E402
from scripts import (  # noqa: E402
    check_postgres,
    check_schema,
    inspect_telemetry_anomaly_state,
    package_telemetry_anomaly_model,
    persist_telemetry_anomalies,
    score_telemetry_anomalies,
)
from services.database import telemetry_anomalies  # noqa: E402


class Status(StrEnum):
    """Validation status values printed by the anomaly checker."""

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
    scoring_summary: telemetry_detector.TelemetryAnomalyScoringSummary | None
    first_persistence: telemetry_anomalies.PersistenceSummary | None
    second_persistence: telemetry_anomalies.PersistenceSummary | None
    final_state: telemetry_anomalies.AnomalyStateSummary | None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-if-missing",
        action="store_true",
        help="Package the telemetry anomaly artifact if it is missing before validation.",
    )
    return parser.parse_args()


def query_count(sql: str) -> int:
    result = persist_telemetry_anomalies.apply_migrations.run_psql_query(sql)
    if not result.succeeded:
        raise RuntimeError(
            persist_telemetry_anomalies.apply_migrations.command_failure_message(result)
        )
    return telemetry_anomalies.parse_count_output(result.output)


def validate_config_and_artifact(
    *,
    package_if_missing: bool,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    config = telemetry_detector.load_config(PROJECT_ROOT)
    telemetry_detector.validate_config(config)
    results.append(pass_result("Anomaly Config", "Config uses exact vibration/pressure contract."))
    if set(config.features) & {
        "product_quality_type",
        "failure_probability",
        "failure_prediction",
    }:
        results.append(
            fail_result("AI4I Feature Separation", "AI4I fields entered anomaly config.")
        )
    else:
        results.append(
            pass_result("AI4I Feature Separation", "No AI4I output fields are features.")
        )

    if not telemetry_detector.artifact_path(PROJECT_ROOT).exists():
        if package_if_missing:
            package_telemetry_anomaly_model.export_feature_records()
            records = telemetry_detector.load_feature_records_from_export(PROJECT_ROOT)
            telemetry_detector.package_anomaly_artifact(
                records,
                root=PROJECT_ROOT,
            )
            results.append(pass_result("Artifact Packaging", "Artifact was packaged on request."))
        else:
            results.append(
                fail_result(
                    "Trusted Artifact",
                    "Artifact is missing. Run package_telemetry_anomaly_model.py first.",
                )
            )
            return results
    telemetry_detector.load_trusted_artifact(PROJECT_ROOT)
    results.append(pass_result("Trusted Artifact", "Artifact metadata and SHA-256 passed."))
    return results


def validate_runtime_output(
    records: Sequence[dict[str, object]],
    features: Sequence[telemetry_detector.FeatureRecord],
    artifact: telemetry_detector.TrustedAnomalyArtifact,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    validated = telemetry_detector.validate_anomaly_output_records(records)
    results.append(pass_result("Runtime Anomaly Count", f"Loaded {len(validated)} anomaly rows."))
    if len(validated) == len(features):
        results.append(
            pass_result("Silver Count Match", "Scored row count equals Silver row count.")
        )
    else:
        results.append(fail_result("Silver Count Match", "Scored row count differs from Silver."))
    if len({record["event_id"] for record in validated}) == len(validated):
        results.append(pass_result("Event Identity", "Runtime event IDs are unique."))
    else:
        results.append(fail_result("Event Identity", "Runtime event IDs are duplicated."))
    scores_finite = all(0 <= float(record["anomaly_score"]) <= 1 for record in validated)
    results.append(
        pass_result("Score Bounds", "All anomaly scores are finite and bounded.")
        if scores_finite
        else fail_result("Score Bounds", "An anomaly score is outside [0, 1].")
    )
    expected = telemetry_detector.score_feature_records(features, artifact)
    by_event_id = {item.record["event_id"]: item.record for item in expected}
    flags_match = all(
        bool(record["anomaly_flag"]) == bool(by_event_id[record["event_id"]]["anomaly_flag"])
        for record in validated
    )
    results.append(
        pass_result("Model Decisions", "Runtime flags match IsolationForest decisions.")
        if flags_match
        else fail_result("Model Decisions", "A runtime flag mismatches model decision."),
    )
    forbidden_terms = {
        "failure_probability",
        "failure_prediction",
        "Machine failure",
        "ground_truth",
        "shap",
    }
    forbidden_found = any(set(record) & forbidden_terms for record in validated)
    results.append(
        pass_result("No Supervised Outputs", "No labels, SHAP, or failure predictions are present.")
        if not forbidden_found
        else fail_result("No Supervised Outputs", "A forbidden supervised field is present."),
    )
    return results


def validate_persistence(
    records: list[telemetry_anomalies.AnomalyRecord],
    machine_ids_by_code: dict[str, int],
) -> list[CheckResult]:
    existing_rows = persist_telemetry_anomalies.load_existing_anomalies(records)
    existing_by_identity = {row.record.identity.as_tuple(): row for row in existing_rows}
    reuse = telemetry_anomalies.summarize_anomaly_reuse(
        records,
        existing_by_identity,
        machine_ids_by_code,
    )
    return [
        pass_result("Persisted Anomaly Presence", "Every runtime anomaly row exists.")
        if len(existing_rows) == len(records)
        else fail_result("Persisted Anomaly Presence", "A runtime anomaly row is missing."),
        pass_result("Persisted Anomaly Values", "Persisted anomaly rows match runtime JSONL.")
        if reuse.existing_identical_records == len(records) and not reuse.conflicts
        else fail_result("Persisted Anomaly Values", "A persisted anomaly value differs."),
        pass_result("Machine References", "All anomaly machines resolve to machines rows.")
        if len(machine_ids_by_code) == len({record.machine_code for record in records})
        else fail_result("Machine References", "A machine reference did not resolve."),
    ]


def run_checks(*, package_if_missing: bool = False) -> ValidationRun:
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

    schema_results = check_schema.run_checks()
    schema_failed = check_schema.exit_code_for(schema_results) != 0
    results.append(
        aggregate_external_results(
            "Schema Validator",
            schema_failed,
            "Schema validator passed with anomaly persistence migration applied.",
        )
    )
    if schema_failed:
        return ValidationRun(results, None, None, None, None)

    results.extend(validate_config_and_artifact(package_if_missing=package_if_missing))
    if any(result.status is Status.FAIL for result in results):
        return ValidationRun(results, None, None, None, None)

    artifact = telemetry_detector.load_trusted_artifact(PROJECT_ROOT)
    score_telemetry_anomalies.export_feature_records()
    features = telemetry_detector.load_feature_records_from_export(PROJECT_ROOT)
    first_summary = telemetry_detector.run_scoring_pipeline(root=PROJECT_ROOT)
    first_bytes = telemetry_detector.anomaly_output_path(PROJECT_ROOT).read_bytes()
    runtime_records = telemetry_detector.read_anomalies_jsonl(
        telemetry_detector.anomaly_output_path(PROJECT_ROOT)
    )
    results.extend(validate_runtime_output(runtime_records, features, artifact))

    alerts_before = query_count("SELECT count(*) FROM alerts;")
    ai4i_predictions_before = query_count("SELECT count(*) FROM model_predictions;")
    machine_health_before = query_count("SELECT count(*) FROM machine_health;")

    first = persist_telemetry_anomalies.persist_anomalies()
    results.append(
        pass_result(
            "First Persistence",
            f"Inserted {first.summary.new_anomaly_rows_inserted} new anomaly row(s).",
        )
    )
    results.extend(validate_persistence(first.records, first.machine_ids_by_code))

    first_state = inspect_telemetry_anomaly_state.inspect_state()
    duplicate_ok = first_state.summary.duplicate_anomaly_identity_count == 0
    machine_reference_ok = first_state.summary.machine_reference_mismatch_count == 0
    results.append(
        pass_result("Stable Identity Uniqueness", "No duplicate anomaly identities exist.")
        if duplicate_ok
        else fail_result("Stable Identity Uniqueness", "Duplicate anomaly identities exist."),
    )
    results.append(
        pass_result("Machine Reference Integrity", "No machine-reference mismatches exist.")
        if machine_reference_ok
        else fail_result("Machine Reference Integrity", "Machine-reference mismatch found."),
    )

    before_second_count = first_state.summary.anomaly_row_count
    second = persist_telemetry_anomalies.persist_anomalies()
    second_state = inspect_telemetry_anomaly_state.inspect_state()
    second_idempotent = (
        second.summary.new_anomaly_rows_inserted == 0
        and second.summary.existing_identical_anomalies_reused == len(first.records)
        and second_state.summary.anomaly_row_count == before_second_count
        and second_state.summary.duplicate_anomaly_identity_count == 0
    )
    results.append(
        pass_result("Second Persistence Idempotency", "Second run reused existing rows.")
        if second_idempotent
        else fail_result("Second Persistence Idempotency", "Second run was not idempotent."),
    )

    second_summary = telemetry_detector.run_scoring_pipeline(root=PROJECT_ROOT)
    second_bytes = telemetry_detector.anomaly_output_path(PROJECT_ROOT).read_bytes()
    results.append(
        pass_result("Scoring Determinism", "Second scoring run reproduced identical JSONL bytes.")
        if second_bytes == first_bytes
        else fail_result("Scoring Determinism", "Second scoring run changed the JSONL output."),
    )

    alerts_after = query_count("SELECT count(*) FROM alerts;")
    ai4i_predictions_after = query_count("SELECT count(*) FROM model_predictions;")
    machine_health_after = query_count("SELECT count(*) FROM machine_health;")
    results.append(
        pass_result("Alerts Unchanged", "No alerts were created.")
        if alerts_before == alerts_after
        else fail_result("Alerts Unchanged", "Alert count changed."),
    )
    ai4i_unchanged = (
        ai4i_predictions_before == ai4i_predictions_after
        and machine_health_before == machine_health_after
    )
    results.append(
        pass_result(
            "AI4I State Unchanged",
            "model_predictions and machine_health counts unchanged.",
        )
        if ai4i_unchanged
        else fail_result("AI4I State Unchanged", "AI4I persistence state changed."),
    )
    if second_summary.output_sha256 != first_summary.output_sha256:
        results.append(warn_result("Output Hash", "Second scoring summary hash differed."))
    return ValidationRun(
        results,
        first_summary,
        first.summary,
        second.summary,
        second_state.summary,
    )


def print_results(run: ValidationRun) -> None:
    print("Industrial Fleet Intelligence Platform telemetry anomaly detection validation")
    print()
    for result in run.results:
        print(f"{result.status.value} {result.name}: {result.message}")
    if run.scoring_summary is not None:
        print()
        print("Scoring summary:")
        print(json.dumps(run.scoring_summary.to_dict(), indent=2, sort_keys=True))
    if run.first_persistence is not None:
        print()
        print("First persistence summary:")
        print(json.dumps(run.first_persistence.to_dict(), indent=2, sort_keys=True))
    if run.second_persistence is not None:
        print()
        print("Second persistence summary:")
        print(json.dumps(run.second_persistence.to_dict(), indent=2, sort_keys=True))
    if run.final_state is not None:
        print()
        print("Final persisted anomaly state:")
        print(json.dumps(run.final_state.to_dict(), indent=2, sort_keys=True))
    pass_count = sum(1 for result in run.results if result.status is Status.PASS)
    warn_count = sum(1 for result in run.results if result.status is Status.WARN)
    fail_count = sum(
        1 for result in run.results if result.status is Status.FAIL and result.mandatory
    )
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    args = parse_args()
    try:
        run = run_checks(package_if_missing=args.package_if_missing)
    except (OSError, RuntimeError, ValueError) as exc:
        run = ValidationRun([fail_result("Validator", str(exc))], None, None, None, None)
    print_results(run)
    has_failure = any(result.status is Status.FAIL and result.mandatory for result in run.results)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

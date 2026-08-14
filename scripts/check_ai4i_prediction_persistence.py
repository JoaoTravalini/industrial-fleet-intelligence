"""Integration validator for AI4I telemetry prediction persistence."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import (  # noqa: E402
    apply_migrations,
    check_postgres,
    check_schema,
    inspect_ai4i_prediction_state,
    persist_ai4i_predictions,
)
from services.database import ai4i_predictions  # noqa: E402


class Status(StrEnum):
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
    first_persistence: ai4i_predictions.PersistenceSummary | None
    second_persistence: ai4i_predictions.PersistenceSummary | None
    final_state: ai4i_predictions.PredictionStateSummary | None


def pass_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.PASS, message)


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
    return ai4i_predictions.parse_count_output(result.output)


def validate_runtime_records(
    records: Sequence[ai4i_predictions.PredictionRecord],
) -> list[CheckResult]:
    probabilities_ok = all(0 <= record.failure_probability <= 1 for record in records)
    threshold_ok = all(
        record.failure_prediction == (record.failure_probability >= record.frozen_threshold)
        for record in records
    )
    identity_ok = (
        len(
            {
                (record.model_name, record.model_version, record.final_config_hash)
                for record in records
            }
        )
        == 1
    )
    lineage_ok = all(
        record.source_kafka_topic
        and record.source_kafka_partition >= 0
        and record.source_kafka_offset >= 0
        and record.source_kafka_timestamp
        and record.source_kafka_key
        and record.payload_sha256
        for record in records
    )
    return [
        pass_result("Runtime Prediction Count", f"Loaded {len(records)} prediction record(s)."),
        pass_result("Runtime Probability Bounds", "All runtime probabilities are in [0, 1].")
        if probabilities_ok
        else fail_result("Runtime Probability Bounds", "A runtime probability is outside [0, 1]."),
        pass_result("Runtime Threshold Decisions", "All runtime decisions follow the threshold.")
        if threshold_ok
        else fail_result("Runtime Threshold Decisions", "A runtime decision mismatches threshold."),
        pass_result("Runtime Model Identity", "Runtime records use one model identity/config.")
        if identity_ok
        else fail_result("Runtime Model Identity", "Runtime records mix model identities."),
        pass_result("Runtime Source Lineage", "Runtime records preserve Kafka/source lineage.")
        if lineage_ok
        else fail_result("Runtime Source Lineage", "A runtime record has incomplete lineage."),
    ]


def validate_persisted_records(
    records: list[ai4i_predictions.PredictionRecord],
    machine_ids_by_code: dict[str, int],
) -> list[CheckResult]:
    existing_rows = persist_ai4i_predictions.load_existing_predictions(records)
    existing_by_identity = {row.record.identity.as_tuple(): row for row in existing_rows}
    reuse = ai4i_predictions.summarize_prediction_reuse(
        records,
        existing_by_identity,
        machine_ids_by_code,
    )
    all_present = len(existing_rows) == len(records)
    all_match = reuse.existing_identical_records == len(records) and not reuse.conflicts
    return [
        pass_result("Persisted Prediction Presence", "Every runtime prediction exists.")
        if all_present
        else fail_result("Persisted Prediction Presence", "A runtime prediction is missing."),
        pass_result("Persisted Prediction Values", "Persisted values match runtime JSONL.")
        if all_match
        else fail_result("Persisted Prediction Values", "A persisted prediction value differs."),
        pass_result("Machine References", "All predicted machines resolve to machines rows.")
        if len(machine_ids_by_code) == len({record.machine_code for record in records})
        else fail_result("Machine References", "A predicted machine did not resolve."),
    ]


def validate_source_guards() -> CheckResult:
    guarded_files = [
        PROJECT_ROOT / "services" / "database" / "ai4i_predictions.py",
        PROJECT_ROOT / "scripts" / "persist_ai4i_predictions.py",
        PROJECT_ROOT / "scripts" / "inspect_ai4i_prediction_state.py",
        PROJECT_ROOT / "scripts" / "check_ai4i_prediction_persistence.py",
    ]
    forbidden_terms = [
        "load_" + "predictor",
        "predict_" + "batch",
        "predict_" + "proba",
        "final_model" + ".joblib",
        "." + "fit(",
        "test" + ".csv",
        "Tree" + "Explainer",
        "Isolation" + "Forest",
        "INSERT INTO " + "alerts",
        "INSERT INTO " + "anomalies",
        "TRUN" + "CATE",
        "DELETE FROM " + "model_predictions",
        "DELETE FROM " + "machine_health",
        "psy" + "copg",
        "sql" + "alchemy",
        "async" + "pg",
        "ale" + "mbic",
    ]
    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in source:
                return fail_result("Source Guards", f"{term} found in {path.name}.")
    return pass_result(
        "Source Guards",
        "No model execution, alerts, anomalies, or DB clients found.",
    )


def validate_no_alerts_or_anomalies() -> list[CheckResult]:
    alerts = query_count("SELECT count(*) FROM alerts;")
    anomalies = query_count("SELECT count(*) FROM anomalies;")
    return [
        pass_result("Alerts Table", "No alerts were created.")
        if alerts == 0
        else fail_result("Alerts Table", f"Found {alerts} alert row(s)."),
        pass_result("Anomalies Table", "No anomalies were created.")
        if anomalies == 0
        else fail_result("Anomalies Table", f"Found {anomalies} anomaly row(s)."),
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
        return ValidationRun(results, None, None, None)

    schema_results = check_schema.run_checks()
    schema_failed = check_schema.exit_code_for(schema_results) != 0
    results.append(
        aggregate_external_results(
            "Schema Validator",
            schema_failed,
            "Schema validator passed with AI4I persistence migration applied.",
        )
    )
    if schema_failed:
        return ValidationRun(results, None, None, None)

    prediction_path = ai4i_predictions.prediction_output_path(PROJECT_ROOT)
    records = ai4i_predictions.load_prediction_records(prediction_path)
    results.extend(validate_runtime_records(records))

    first = persist_ai4i_predictions.persist_predictions()
    results.append(
        pass_result(
            "First Persistence",
            f"Inserted {first.summary.new_prediction_rows_inserted} new prediction row(s).",
        )
    )
    results.extend(validate_persisted_records(first.records, first.machine_ids_by_code))

    first_state = inspect_ai4i_prediction_state.inspect_state(first.records)
    duplicate_ok = first_state.summary.duplicate_prediction_business_identity_count == 0
    projection_ok = (
        first_state.summary.machine_health_prediction_mismatch_count == 0
        and first_state.summary.machine_health_latest_event_mismatch_count == 0
    )
    results.append(
        pass_result("Stable Identity Uniqueness", "No duplicate prediction identities exist.")
        if duplicate_ok
        else fail_result("Stable Identity Uniqueness", "Duplicate prediction identities exist.")
    )
    results.append(
        pass_result("machine_health Latest Projection", "Latest projections match predictions.")
        if projection_ok
        else fail_result("machine_health Latest Projection", "Latest projection mismatch found.")
    )

    before_second_count = first_state.summary.prediction_row_count
    second = persist_ai4i_predictions.persist_predictions()
    second_state = inspect_ai4i_prediction_state.inspect_state(first.records)
    second_idempotent = (
        second.summary.new_prediction_rows_inserted == 0
        and second.summary.existing_identical_predictions_reused == len(first.records)
        and second_state.summary.prediction_row_count == before_second_count
        and second_state.summary.duplicate_prediction_business_identity_count == 0
    )
    results.append(
        pass_result("Second Persistence Idempotency", "Second run reused existing rows.")
        if second_idempotent
        else fail_result("Second Persistence Idempotency", "Second run was not idempotent."),
    )
    second_projection_ok = (
        second_state.summary.machine_health_prediction_mismatch_count == 0
        and second_state.summary.machine_health_latest_event_mismatch_count == 0
    )
    results.append(
        pass_result("Projection Regression Protection", "machine_health did not regress.")
        if second_projection_ok
        else fail_result(
            "Projection Regression Protection",
            "machine_health projection regressed.",
        ),
    )

    results.extend(validate_no_alerts_or_anomalies())
    results.append(validate_source_guards())
    return ValidationRun(results, first.summary, second.summary, second_state.summary)


def print_results(run: ValidationRun) -> None:
    print("Industrial Fleet Intelligence Platform AI4I prediction persistence validation")
    print()
    for result in run.results:
        print(f"{result.status.value} {result.name}: {result.message}")
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
        print("Final persisted prediction state:")
        print(json.dumps(run.final_state.to_dict(), indent=2, sort_keys=True))
    pass_count = sum(1 for result in run.results if result.status is Status.PASS)
    warn_count = sum(1 for result in run.results if result.status is Status.WARN)
    fail_count = sum(
        1 for result in run.results if result.status is Status.FAIL and result.mandatory
    )
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    try:
        run = run_checks()
    except (OSError, RuntimeError, ValueError) as exc:
        run = ValidationRun([fail_result("Validator", str(exc))], None, None, None)
    print_results(run)
    has_failure = any(result.status is Status.FAIL and result.mandatory for result in run.results)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

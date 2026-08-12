"""End-to-end validator for Spark Bronze-to-Silver telemetry processing."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.streaming import silver_transformation  # noqa: E402
from scripts import run_spark_silver_docker  # noqa: E402

INSPECT_TIMEOUT_SECONDS = 300
SILVER_TIMEOUT_SECONDS = 900


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


def check_spark_health() -> CheckResult:
    ok, message = run_spark_silver_docker.verify_spark_container()
    status = Status.PASS if ok else Status.FAIL
    return CheckResult("Spark Health", status, message)


def check_bronze_exists() -> CheckResult:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_silver_docker.SPARK_SERVICE,
        "test",
        "-d",
        silver_transformation.container_path(silver_transformation.EXPECTED_BRONZE_INPUT_PATH),
    ]
    result = run_spark_silver_docker.run_command(command, timeout=30)
    if not result.succeeded:
        return CheckResult(
            "Bronze Dataset",
            Status.FAIL,
            "Bronze telemetry dataset is not available inside the Spark container.",
        )
    return CheckResult("Bronze Dataset", Status.PASS, "Bronze telemetry dataset exists.")


def build_inspection_command(*, synthetic_rules_check: bool = False) -> list[str]:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_silver_docker.SPARK_SERVICE,
        run_spark_silver_docker.SPARK_SUBMIT,
        "/workspace/scripts/inspect_spark_silver.py",
    ]
    if synthetic_rules_check:
        command.append("--synthetic-rules-check")
    return command


def parse_json_summary(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Silver inspection did not return a JSON summary.")


def run_inspection(
    *, synthetic_rules_check: bool = False
) -> tuple[CheckResult, dict[str, Any] | None]:
    command = build_inspection_command(synthetic_rules_check=synthetic_rules_check)
    label = "Synthetic Rules" if synthetic_rules_check else "Silver Inspection"
    result = run_spark_silver_docker.run_command(command, timeout=INSPECT_TIMEOUT_SECONDS)
    if not result.succeeded:
        return (
            CheckResult(
                label, Status.FAIL, run_spark_silver_docker.command_failure_message(result)
            ),
            None,
        )
    try:
        summary = parse_json_summary(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        return CheckResult(label, Status.FAIL, str(exc)), None
    return CheckResult(label, Status.PASS, "Inspection completed."), summary


def run_silver_transformation(label: str) -> CheckResult:
    result = run_spark_silver_docker.run_spark_silver(timeout=SILVER_TIMEOUT_SECONDS)
    if not result.succeeded:
        return CheckResult(
            label, Status.FAIL, run_spark_silver_docker.command_failure_message(result)
        )
    return CheckResult(label, Status.PASS, "Silver snapshot rebuild completed.")


def validate_synthetic_summary(summary: Mapping[str, Any]) -> list[CheckResult]:
    checks = [
        (
            "Synthetic Accepted",
            summary.get("canonical_valid_count") == 1,
            "One valid record became canonical.",
            "Synthetic valid record was not accepted as canonical.",
        ),
        (
            "Synthetic Duplicate",
            summary.get("valid_duplicate_count") == 1
            and summary.get("valid_duplicate_audited") is True,
            "Repeated event_id was audited as a valid duplicate.",
            "Repeated event_id was not audited as a valid duplicate.",
        ),
        (
            "Synthetic Quarantine Count",
            summary.get("quarantine_count") == 3,
            "Three invalid records were quarantined.",
            "Synthetic quarantine count did not match expectation.",
        ),
        (
            "Malformed JSON",
            summary.get("malformed_json_quarantined") is True,
            "Malformed JSON was quarantined.",
            "Malformed JSON was not quarantined.",
        ),
        (
            "Key Mismatch",
            summary.get("kafka_key_mismatch_quarantined") is True,
            "Kafka key mismatch was quarantined.",
            "Kafka key mismatch was not quarantined.",
        ),
        (
            "Invalid Sensor",
            summary.get("invalid_sensor_quarantined") is True,
            "Out-of-range telemetry was quarantined.",
            "Out-of-range telemetry was not quarantined.",
        ),
    ]
    return [
        CheckResult(
            name, Status.PASS if passed else Status.FAIL, ok_message if passed else fail_message
        )
        for name, passed, ok_message, fail_message in checks
    ]


def validate_accounting(summary: Mapping[str, Any]) -> CheckResult:
    counts_are_valid = silver_transformation.accounting_invariants_hold(
        bronze_row_count=int(summary.get("bronze_row_count") or 0),
        valid_pre_dedup_row_count=int(summary.get("valid_pre_dedup_row_count") or 0),
        canonical_silver_row_count=int(summary.get("canonical_silver_row_count") or 0),
        duplicate_audit_row_count=int(summary.get("duplicate_audit_row_count") or 0),
        quarantine_row_count=int(summary.get("quarantine_row_count") or 0),
    )
    flag_is_valid = summary.get("accounting_invariants_hold") is True
    if counts_are_valid and flag_is_valid:
        return CheckResult("Accounting", Status.PASS, "Silver accounting invariants hold.")
    return CheckResult("Accounting", Status.FAIL, "Silver accounting invariants failed.")


def validate_real_summary(summary: Mapping[str, Any]) -> list[CheckResult]:
    results = [validate_accounting(summary)]
    canonical_count = int(summary.get("canonical_silver_row_count") or 0)
    distinct_event_ids = int(summary.get("silver_distinct_event_id_count") or 0)
    duplicate_event_ids = int(summary.get("silver_duplicate_event_id_count") or 0)
    if canonical_count == distinct_event_ids and duplicate_event_ids == 0:
        results.append(
            CheckResult("event_id Uniqueness", Status.PASS, "Canonical event_id values are unique.")
        )
    else:
        results.append(
            CheckResult(
                "event_id Uniqueness",
                Status.FAIL,
                "Canonical Silver contains repeated event_id values.",
            )
        )

    if int(summary.get("quarantine_canonical_coordinate_overlap_count") or 0) == 0:
        results.append(
            CheckResult(
                "Quarantine Separation",
                Status.PASS,
                "No quarantined Kafka coordinate entered canonical Silver.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Quarantine Separation",
                Status.FAIL,
                "A quarantined Kafka coordinate appeared in canonical Silver.",
            )
        )

    if int(summary.get("duplicate_quarantine_coordinate_overlap_count") or 0) == 0:
        results.append(
            CheckResult(
                "Duplicate Separation",
                Status.PASS,
                "Valid duplicate coordinates are not placed in quarantine.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Duplicate Separation",
                Status.FAIL,
                "A duplicate audit coordinate appeared in quarantine.",
            )
        )

    null_lineage = sum(
        int(summary.get(key) or 0)
        for key in (
            "null_source_kafka_topic_count",
            "null_source_kafka_partition_count",
            "null_source_kafka_offset_count",
        )
    )
    if null_lineage == 0:
        results.append(CheckResult("Lineage", Status.PASS, "Canonical Kafka lineage is populated."))
    else:
        results.append(
            CheckResult("Lineage", Status.FAIL, "Canonical Kafka lineage contains nulls.")
        )

    expected_types = silver_transformation.telemetry_field_types()
    actual_types = summary.get("silver_field_types") or {}
    if all(actual_types.get(field) == expected for field, expected in expected_types.items()):
        results.append(CheckResult("Typed Schema", Status.PASS, "Canonical telemetry types match."))
    else:
        results.append(
            CheckResult("Typed Schema", Status.FAIL, "Canonical telemetry types differ.")
        )

    forbidden_fields = {
        "Machine " + "failure",
        "failure_" + "probability",
        "failure_" + "prediction",
        "S" + "HAP",
        "an" + "omaly_label",
    }
    lower_forbidden = {field.lower() for field in forbidden_fields}
    schema_fields = {str(field).lower() for field in actual_types}
    if schema_fields.isdisjoint(lower_forbidden):
        results.append(CheckResult("Excluded Fields", Status.PASS, "No predictive fields exist."))
    else:
        results.append(CheckResult("Excluded Fields", Status.FAIL, "Predictive fields exist."))

    if int(summary.get("quarantine_row_count") or 0) == 0:
        results.append(
            CheckResult(
                "Real Quarantine",
                Status.PASS,
                "Real Bronze snapshot produced zero quarantined rows; this is normal.",
            )
        )
    return results


def determinism_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "accounting_invariants_hold",
        "bronze_row_count",
        "canonical_selection_sha256",
        "canonical_silver_row_count",
        "duplicate_audit_row_count",
        "duplicate_rank_max",
        "event_time_max",
        "event_time_min",
        "machine_count",
        "product_quality_type_counts",
        "quarantine_row_count",
        "rejection_reason_counts",
        "silver_distinct_event_id_count",
        "silver_duplicate_event_id_count",
        "valid_pre_dedup_row_count",
    )
    return {key: summary.get(key) for key in keys}


def validate_determinism(first: Mapping[str, Any], second: Mapping[str, Any]) -> CheckResult:
    if determinism_projection(first) == determinism_projection(second):
        return CheckResult(
            "Second Run Determinism",
            Status.PASS,
            "Logical counts and canonical selections are unchanged.",
        )
    return CheckResult(
        "Second Run Determinism",
        Status.FAIL,
        "Logical Silver outputs changed between identical Bronze snapshot rebuilds.",
    )


def run_checks() -> tuple[list[CheckResult], dict[str, Any] | None, dict[str, Any] | None]:
    results = [check_spark_health(), check_bronze_exists()]
    if any(result.status is Status.FAIL and result.mandatory for result in results):
        return results, None, None

    synthetic_result, synthetic_summary = run_inspection(synthetic_rules_check=True)
    results.append(synthetic_result)
    if synthetic_summary is not None:
        results.extend(validate_synthetic_summary(synthetic_summary))
    else:
        return results, None, None

    first_run = run_silver_transformation("Silver Rebuild")
    results.append(first_run)
    if first_run.status is Status.FAIL:
        return results, synthetic_summary, None

    first_inspection, first_summary = run_inspection()
    results.append(first_inspection)
    if first_summary is None:
        return results, synthetic_summary, None
    results.extend(validate_real_summary(first_summary))

    second_run = run_silver_transformation("Silver Rebuild Repeat")
    results.append(second_run)
    if second_run.status is Status.FAIL:
        return results, synthetic_summary, first_summary

    second_inspection, second_summary = run_inspection()
    results.append(second_inspection)
    if second_summary is not None:
        results.append(validate_determinism(first_summary, second_summary))
    return results, synthetic_summary, second_summary or first_summary


def print_results(
    results: Sequence[CheckResult],
    synthetic_summary: Mapping[str, Any] | None,
    silver_summary: Mapping[str, Any] | None,
) -> None:
    print("Industrial Fleet Intelligence Platform Spark Silver validation")
    print()
    for result in results:
        print(f"{result.status.value} {result.name}: {result.message}")
    if synthetic_summary is not None:
        print()
        print("Synthetic rules summary:")
        print(json.dumps(synthetic_summary, indent=2, sort_keys=True))
    if silver_summary is not None:
        print()
        print("Silver inspection summary:")
        print(json.dumps(silver_summary, indent=2, sort_keys=True))
    mandatory_failures = [
        result for result in results if result.status is Status.FAIL and result.mandatory
    ]
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    pass_count = sum(1 for result in results if result.status is Status.PASS)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {len(mandatory_failures)} FAIL")


def main() -> int:
    results, synthetic_summary, silver_summary = run_checks()
    print_results(results, synthetic_summary, silver_summary)
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

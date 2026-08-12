"""End-to-end validator for Spark Gold descriptive analytics processing."""

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

from pipelines.batch import gold_transformation  # noqa: E402
from scripts import run_spark_gold_docker  # noqa: E402

INSPECT_TIMEOUT_SECONDS = 300
GOLD_TIMEOUT_SECONDS = 900


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
    ok, message = run_spark_gold_docker.verify_spark_container()
    status = Status.PASS if ok else Status.FAIL
    return CheckResult("Spark Health", status, message)


def check_silver_exists() -> CheckResult:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_gold_docker.SPARK_SERVICE,
        "test",
        "-d",
        gold_transformation.container_path(gold_transformation.EXPECTED_SILVER_INPUT_PATH),
    ]
    result = run_spark_gold_docker.run_command(command, timeout=30)
    if not result.succeeded:
        return CheckResult(
            "Canonical Silver",
            Status.FAIL,
            "Canonical Silver telemetry dataset is not available inside the Spark container.",
        )
    return CheckResult("Canonical Silver", Status.PASS, "Canonical Silver telemetry exists.")


def check_gold_config() -> CheckResult:
    try:
        gold_transformation.load_gold_config()
    except gold_transformation.SparkGoldConfigError as exc:
        return CheckResult("Gold Config", Status.FAIL, str(exc))
    return CheckResult("Gold Config", Status.PASS, "Gold configuration is valid.")


def build_inspection_command(*, synthetic_analytics_check: bool = False) -> list[str]:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_gold_docker.SPARK_SERVICE,
        run_spark_gold_docker.SPARK_SUBMIT,
        "/workspace/scripts/inspect_spark_gold.py",
    ]
    if synthetic_analytics_check:
        command.append("--synthetic-analytics-check")
    return command


def parse_json_summary(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Gold inspection did not return a JSON summary.")


def run_inspection(
    *,
    synthetic_analytics_check: bool = False,
) -> tuple[CheckResult, dict[str, Any] | None]:
    command = build_inspection_command(synthetic_analytics_check=synthetic_analytics_check)
    label = "Synthetic Analytics" if synthetic_analytics_check else "Gold Inspection"
    result = run_spark_gold_docker.run_command(command, timeout=INSPECT_TIMEOUT_SECONDS)
    if not result.succeeded:
        return (
            CheckResult(label, Status.FAIL, run_spark_gold_docker.command_failure_message(result)),
            None,
        )
    try:
        summary = parse_json_summary(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        return CheckResult(label, Status.FAIL, str(exc)), None
    return CheckResult(label, Status.PASS, "Inspection completed."), summary


def run_gold_transformation(label: str) -> CheckResult:
    result = run_spark_gold_docker.run_spark_gold(timeout=GOLD_TIMEOUT_SECONDS)
    if not result.succeeded:
        return CheckResult(
            label, Status.FAIL, run_spark_gold_docker.command_failure_message(result)
        )
    return CheckResult(label, Status.PASS, "Gold snapshot rebuild completed.")


def validate_synthetic_summary(summary: Mapping[str, Any]) -> list[CheckResult]:
    checks = [
        (
            "Synthetic Rows",
            summary.get("correct_machine_summary_rows") is True,
            "One machine summary row per synthetic machine.",
            "Synthetic machine summary grain failed.",
        ),
        (
            "Synthetic Windows",
            summary.get("correct_machine_window_rows") is True,
            "Synthetic events were grouped into expected one-minute windows.",
            "Synthetic machine-window grouping failed.",
        ),
        (
            "Synthetic Latest",
            summary.get("correct_latest_observation") is True,
            "Deterministic latest observation logic passed.",
            "Deterministic latest observation logic failed.",
        ),
        (
            "Synthetic Latest Product Type",
            summary.get("correct_latest_product_quality_type") is True,
            "Latest product-quality type comes from the deterministic latest row.",
            "Latest product-quality type selection failed.",
        ),
        (
            "Synthetic Mixed Types",
            summary.get("mixed_product_quality_type_accepted") is True,
            "Mixed product-quality events for one machine are accepted.",
            "Mixed product-quality events were not accepted.",
        ),
        (
            "Synthetic Machine Type Counts",
            summary.get("correct_machine_summary_type_counts") is True,
            "Synthetic machine type event counts reconcile.",
            "Synthetic machine type event counts failed.",
        ),
        (
            "Synthetic Window Type Counts",
            summary.get("correct_machine_window_type_counts") is True,
            "Synthetic window type event counts reconcile.",
            "Synthetic window type event counts failed.",
        ),
        (
            "Synthetic Fleet Type Counts",
            summary.get("correct_fleet_type_counts") is True,
            "Synthetic fleet type event counts reconcile.",
            "Synthetic fleet type event counts failed.",
        ),
        (
            "Synthetic Accounting",
            summary.get("machine_window_event_count_sum") == summary.get("source_row_count"),
            "Synthetic window counts account for all source rows.",
            "Synthetic window accounting failed.",
        ),
        (
            "Synthetic Fleet",
            summary.get("fleet_summary_rows") == 1,
            "Synthetic fleet summary has one row.",
            "Synthetic fleet summary row count failed.",
        ),
    ]
    return [
        CheckResult(
            name,
            Status.PASS if passed else Status.FAIL,
            ok_message if passed else fail_message,
        )
        for name, passed, ok_message, fail_message in checks
    ]


def validate_real_summary(summary: Mapping[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    silver_rows = int(summary.get("silver_row_count") or 0)
    silver_machines = int(summary.get("silver_machine_count") or 0)
    machine_summary_rows = int(summary.get("machine_summary_row_count") or 0)
    machine_summary_sum = int(summary.get("machine_summary_event_count_sum") or 0)
    machine_window_sum = int(summary.get("machine_windows_event_count_sum") or 0)
    fleet_event_count = int(summary.get("fleet_summary_event_count") or 0)
    fleet_machine_count = int(summary.get("fleet_summary_machine_count") or 0)

    if gold_transformation.event_accounting_holds(
        silver_row_count=silver_rows,
        machine_summary_event_count_sum=machine_summary_sum,
        machine_windows_event_count_sum=machine_window_sum,
        fleet_event_count=fleet_event_count,
    ):
        results.append(CheckResult("Accounting", Status.PASS, "Gold event accounting holds."))
    else:
        results.append(CheckResult("Accounting", Status.FAIL, "Gold event accounting failed."))

    if (
        machine_summary_rows == silver_machines
        and int(summary.get("machine_summary_duplicate_machine_count") or 0) == 0
    ):
        results.append(CheckResult("Machine Summary Grain", Status.PASS, "One row per machine."))
    else:
        results.append(
            CheckResult("Machine Summary Grain", Status.FAIL, "Machine summary grain failed.")
        )

    if int(summary.get("machine_windows_duplicate_grain_count") or 0) == 0:
        results.append(
            CheckResult("Machine Window Grain", Status.PASS, "Machine window grain is unique.")
        )
    else:
        results.append(
            CheckResult("Machine Window Grain", Status.FAIL, "Machine window grain failed.")
        )

    if int(summary.get("fleet_summary_row_count") or 0) == 1:
        results.append(CheckResult("Fleet Row Count", Status.PASS, "Fleet summary has one row."))
    else:
        results.append(
            CheckResult("Fleet Row Count", Status.FAIL, "Fleet summary row count failed.")
        )

    if int(summary.get("latest_observation_mismatch_count") or 0) == 0:
        results.append(
            CheckResult("Latest Observation", Status.PASS, "Latest observations are deterministic.")
        )
    else:
        results.append(
            CheckResult("Latest Observation", Status.FAIL, "Latest observation mismatch found.")
        )

    if int(summary.get("machine_summary_event_count_mismatch_count") or 0) == 0:
        results.append(
            CheckResult("Machine Counts", Status.PASS, "Machine event counts match Silver.")
        )
    else:
        results.append(
            CheckResult("Machine Counts", Status.FAIL, "Machine event count mismatch found.")
        )

    type_sum = int(summary.get("fleet_product_quality_type_count_sum") or 0)
    if type_sum == silver_rows and fleet_machine_count == silver_machines:
        results.append(CheckResult("Fleet Counts", Status.PASS, "Fleet counts match Silver."))
    else:
        results.append(CheckResult("Fleet Counts", Status.FAIL, "Fleet count mismatch found."))

    if int(summary.get("machine_summary_type_count_mismatch_count") or 0) == 0:
        results.append(
            CheckResult(
                "Machine Type Counts",
                Status.PASS,
                "Machine product-quality event counts reconcile.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Machine Type Counts",
                Status.FAIL,
                "Machine product-quality event counts failed.",
            )
        )

    if int(summary.get("machine_windows_type_count_mismatch_count") or 0) == 0:
        results.append(
            CheckResult(
                "Window Type Counts",
                Status.PASS,
                "Machine-window product-quality event counts reconcile.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Window Type Counts",
                Status.FAIL,
                "Machine-window product-quality event counts failed.",
            )
        )

    if bool(summary.get("fleet_type_count_mismatch")):
        results.append(
            CheckResult("Fleet Type Counts", Status.FAIL, "Fleet type event counts failed.")
        )
    else:
        results.append(
            CheckResult("Fleet Type Counts", Status.PASS, "Fleet type event counts reconcile.")
        )

    mixed_machine_count = int(summary.get("silver_mixed_product_quality_type_machine_count") or 0)
    if mixed_machine_count > 0:
        results.append(
            CheckResult(
                "Mixed Type Histories",
                Status.PASS,
                f"Gold accepted {mixed_machine_count} machine(s) with mixed event-level types.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Mixed Type Histories",
                Status.WARN,
                "Current Silver has no mixed type histories; "
                "synthetic validation covers this case.",
                mandatory=False,
            )
        )

    if int(summary.get("non_finite_aggregate_count") or 0) == 0:
        results.append(CheckResult("Finite Aggregates", Status.PASS, "Aggregates are finite."))
    else:
        results.append(CheckResult("Finite Aggregates", Status.FAIL, "Non-finite aggregate found."))

    if int(summary.get("excluded_field_count") or 0) == 0:
        results.append(CheckResult("Excluded Fields", Status.PASS, "No predictive fields exist."))
    else:
        results.append(CheckResult("Excluded Fields", Status.FAIL, "Forbidden field found."))

    if bool(summary.get("fleet_event_count_mismatch")) or bool(
        summary.get("fleet_machine_count_mismatch")
    ):
        results.append(CheckResult("Fleet Invariants", Status.FAIL, "Fleet invariants failed."))
    else:
        results.append(CheckResult("Fleet Invariants", Status.PASS, "Fleet invariants hold."))

    if bool(summary.get("window_event_count_mismatch")):
        results.append(CheckResult("Window Accounting", Status.FAIL, "Window accounting failed."))
    else:
        results.append(CheckResult("Window Accounting", Status.PASS, "Window accounting holds."))

    return results


def determinism_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "first_event_time",
        "fleet_summary_event_count",
        "fleet_summary_machine_count",
        "fleet_type_count_mismatch",
        "latest_event_time",
        "machine_summary_event_count_sum",
        "machine_summary_type_count_mismatch_count",
        "machine_summary_row_count",
        "machine_summary_selection_sha256",
        "machine_windows_event_count_sum",
        "machine_windows_row_count",
        "machine_windows_selection_sha256",
        "machine_windows_type_count_mismatch_count",
        "product_quality_type_event_counts",
        "silver_machine_count",
        "silver_mixed_product_quality_type_machine_count",
        "silver_row_count",
    )
    return {key: summary.get(key) for key in keys}


def validate_determinism(first: Mapping[str, Any], second: Mapping[str, Any]) -> CheckResult:
    if determinism_projection(first) == determinism_projection(second):
        return CheckResult(
            "Second Run Determinism",
            Status.PASS,
            "Logical counts, aggregates, and selections are unchanged.",
        )
    return CheckResult(
        "Second Run Determinism",
        Status.FAIL,
        "Logical Gold outputs changed between identical Silver snapshot rebuilds.",
    )


def run_checks() -> tuple[list[CheckResult], dict[str, Any] | None, dict[str, Any] | None]:
    results = [check_spark_health(), check_silver_exists(), check_gold_config()]
    if any(result.status is Status.FAIL and result.mandatory for result in results):
        return results, None, None

    synthetic_result, synthetic_summary = run_inspection(synthetic_analytics_check=True)
    results.append(synthetic_result)
    if synthetic_summary is not None:
        results.extend(validate_synthetic_summary(synthetic_summary))
    else:
        return results, None, None

    first_run = run_gold_transformation("Gold Rebuild")
    results.append(first_run)
    if first_run.status is Status.FAIL:
        return results, synthetic_summary, None

    first_inspection, first_summary = run_inspection()
    results.append(first_inspection)
    if first_summary is None:
        return results, synthetic_summary, None
    results.extend(validate_real_summary(first_summary))

    second_run = run_gold_transformation("Gold Rebuild Repeat")
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
    gold_summary: Mapping[str, Any] | None,
) -> None:
    print("Industrial Fleet Intelligence Platform Spark Gold validation")
    print()
    for result in results:
        print(f"{result.status.value} {result.name}: {result.message}")
    if synthetic_summary is not None:
        print()
        print("Synthetic analytics summary:")
        print(json.dumps(synthetic_summary, indent=2, sort_keys=True))
    if gold_summary is not None:
        print()
        print("Gold inspection summary:")
        print(json.dumps(gold_summary, indent=2, sort_keys=True))
    mandatory_failures = [
        result for result in results if result.status is Status.FAIL and result.mandatory
    ]
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    pass_count = sum(1 for result in results if result.status is Status.PASS)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {len(mandatory_failures)} FAIL")


def main() -> int:
    results, synthetic_summary, gold_summary = run_checks()
    print_results(results, synthetic_summary, gold_summary)
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

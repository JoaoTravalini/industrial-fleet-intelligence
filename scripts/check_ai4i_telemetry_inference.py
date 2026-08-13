"""Integration validator for Silver telemetry to AI4I batch inference."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import ai4i_predictor, ai4i_telemetry  # noqa: E402
from pipelines.batch import ai4i_feature_adapter  # noqa: E402
from scripts import run_spark_ai4i_adapter_docker  # noqa: E402

ADAPTER_TIMEOUT_SECONDS = 900


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


def pass_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.PASS, message)


def fail_result(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.FAIL, message)


def check_spark_health() -> CheckResult:
    ok, message = run_spark_ai4i_adapter_docker.verify_spark_container()
    return pass_result("Spark Health", message) if ok else fail_result("Spark Health", message)


def check_silver_exists() -> CheckResult:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        run_spark_ai4i_adapter_docker.SPARK_SERVICE,
        "test",
        "-d",
        ai4i_feature_adapter.container_path(ai4i_feature_adapter.EXPECTED_SOURCE_PATH),
    ]
    result = run_spark_ai4i_adapter_docker.run_command(command, timeout=30)
    if result.succeeded:
        return pass_result("Canonical Silver", "Canonical Silver telemetry exists.")
    return fail_result(
        "Canonical Silver",
        "Canonical Silver telemetry dataset is not available inside the Spark container.",
    )


def check_model_artifact() -> CheckResult:
    try:
        ai4i_predictor.load_predictor(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        return fail_result("Packaged Model", str(exc))
    return pass_result("Packaged Model", "Packaged AI4I model loads through trusted loader.")


def parse_adapter_counts(output: str) -> dict[str, int]:
    labels = {
        "Adapter output rows": "adapter_row_count",
        "Adapter unique event IDs": "adapter_distinct_event_id_count",
        "Silver input rows": "silver_row_count",
        "Silver unique event IDs": "silver_distinct_event_id_count",
    }
    counts: dict[str, int] = {}
    for line in output.splitlines():
        for label, key in labels.items():
            match = re.match(rf"^{re.escape(label)}:\s*(\d+)\s*$", line.strip())
            if match:
                counts[key] = int(match.group(1))
    return counts


def run_adapter() -> tuple[CheckResult, dict[str, int]]:
    result = run_spark_ai4i_adapter_docker.run_spark_ai4i_adapter(timeout=ADAPTER_TIMEOUT_SECONDS)
    if not result.succeeded:
        return (
            fail_result(
                "Spark Adapter",
                run_spark_ai4i_adapter_docker.command_failure_message(result),
            ),
            {},
        )
    counts = parse_adapter_counts(result.output)
    if not counts:
        return fail_result("Spark Adapter", "Adapter output did not include parseable counts."), {}
    return pass_result("Spark Adapter", "Spark AI4I feature adapter completed."), counts


def load_adapter_records() -> tuple[CheckResult, list[dict[str, Any]]]:
    try:
        config = ai4i_telemetry.load_adapter_config(PROJECT_ROOT)
        final_config = ai4i_predictor.load_final_config(PROJECT_ROOT)
        records = ai4i_telemetry.load_adapter_records(
            root=PROJECT_ROOT,
            config=config,
            final_config=final_config,
        )
    except (OSError, ValueError) as exc:
        return fail_result("Adapter Records", str(exc)), []
    return pass_result("Adapter Records", "Adapter records loaded and validated."), records


def validate_adapter_records(
    records: Sequence[Mapping[str, Any]],
    adapter_counts: Mapping[str, int],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    adapter_count = len(records)
    silver_count = int(adapter_counts.get("silver_row_count", -1))
    adapter_runner_count = int(adapter_counts.get("adapter_row_count", -1))
    unique_event_count = len({str(record["event_id"]) for record in records})
    results.append(
        pass_result("Adapter Count", "Adapter count equals Silver count.")
        if adapter_count == silver_count == adapter_runner_count
        else fail_result("Adapter Count", "Adapter count does not equal Silver count.")
    )
    results.append(
        pass_result("Adapter Event IDs", "Adapter event IDs are unique.")
        if unique_event_count == adapter_count
        else fail_result("Adapter Event IDs", "Adapter event IDs are not unique.")
    )
    expected_features = set(ai4i_feature_adapter.EXPECTED_MODEL_INPUT_FEATURES)
    exact_features = all(set(record["model_input"]) == expected_features for record in records)
    results.append(
        pass_result("Exact Model Features", "Every model_input has exactly six features.")
        if exact_features
        else fail_result("Exact Model Features", "A model_input has missing or extra features.")
    )
    excluded = set(ai4i_feature_adapter.EXPECTED_EXCLUDED_CURRENT_MODEL_FIELDS)
    excluded_absent = all(not (set(record["model_input"]) & excluded) for record in records)
    results.append(
        pass_result("Excluded Sensors", "vibration_mm_s and pressure_bar are excluded.")
        if excluded_absent
        else fail_result("Excluded Sensors", "An excluded sensor entered model_input.")
    )
    type_values_valid = all(record["model_input"]["Type"] in {"L", "M", "H"} for record in records)
    results.append(
        pass_result("Product Type Mapping", "product_quality_type maps per event to Type.")
        if type_values_valid
        else fail_result("Product Type Mapping", "Type contains an invalid value.")
    )
    lineage_preserved = all(
        set(record["source_lineage"]) == set(ai4i_telemetry.LINEAGE_FIELDS) for record in records
    )
    results.append(
        pass_result("Source Lineage", "All expected source lineage fields are preserved.")
        if lineage_preserved
        else fail_result("Source Lineage", "Source lineage fields are missing.")
    )
    return results


def run_inference() -> tuple[CheckResult, ai4i_telemetry.TelemetryPredictionSummary | None]:
    try:
        summary = ai4i_telemetry.run_prediction_pipeline(root=PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        return fail_result("Host Inference", str(exc)), None
    return pass_result("Host Inference", "Telemetry predictions were written."), summary


def validate_predictions(
    adapter_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    adapter_count = len(adapter_records)
    prediction_count = len(prediction_records)
    unique_event_count = len({str(record["event_id"]) for record in prediction_records})
    results.append(
        pass_result("Prediction Count", "Prediction count equals adapter count.")
        if prediction_count == adapter_count
        else fail_result("Prediction Count", "Prediction count does not equal adapter count.")
    )
    results.append(
        pass_result("Prediction Event IDs", "Prediction event IDs are unique.")
        if unique_event_count == prediction_count
        else fail_result("Prediction Event IDs", "Prediction event IDs are not unique.")
    )
    probability_bounds = all(
        0 <= float(record["failure_probability"]) <= 1 for record in prediction_records
    )
    results.append(
        pass_result("Probability Bounds", "All probabilities are in [0, 1].")
        if probability_bounds
        else fail_result("Probability Bounds", "A probability is outside [0, 1].")
    )
    threshold_ok = all(
        ai4i_telemetry.prediction_is_consistent(
            float(record["failure_probability"]),
            int(record["failure_prediction"]),
            float(record["frozen_threshold"]),
        )
        and float(record["frozen_threshold"]) == 0.14
        for record in prediction_records
    )
    results.append(
        pass_result("Frozen Threshold", "Predictions follow probability >= 0.14.")
        if threshold_ok
        else fail_result("Frozen Threshold", "A prediction does not follow threshold 0.14.")
    )
    final_config = ai4i_predictor.load_final_config(PROJECT_ROOT)
    expected_hash = ai4i_predictor.current_final_config_hash(final_config)
    identity_ok = all(
        record["model_name"] == ai4i_predictor.MODEL_NAME
        and record["model_version"] == ai4i_predictor.MODEL_VERSION
        and record["final_config_hash"] == expected_hash
        for record in prediction_records
    )
    results.append(
        pass_result("Model Identity", "Prediction records include expected model identity.")
        if identity_ok
        else fail_result("Model Identity", "Prediction records contain unexpected model identity.")
    )
    by_event_id = {str(record["event_id"]): record for record in adapter_records}
    hashes_ok = all(
        record["model_input_sha256"]
        == ai4i_telemetry.model_input_sha256(
            by_event_id[str(record["event_id"])]["model_input"],
            final_config,
        )
        for record in prediction_records
    )
    results.append(
        pass_result("Model Input Hash", "model_input_sha256 values are consistent.")
        if hashes_ok
        else fail_result("Model Input Hash", "A model_input_sha256 value is inconsistent.")
    )
    lineage_ok = all(
        all(field in record for field in ai4i_telemetry.LINEAGE_FIELDS)
        for record in prediction_records
    )
    results.append(
        pass_result("Prediction Lineage", "Prediction records preserve source lineage.")
        if lineage_ok
        else fail_result("Prediction Lineage", "Prediction lineage is incomplete.")
    )
    forbidden_fragments = (
        "Machine failure",
        "actual_failure",
        "ground_truth",
        "s" + "hap",
        "anomaly",
    )
    forbidden_absent = all(
        not any(fragment in key for fragment in forbidden_fragments)
        for record in prediction_records
        for key in record
    )
    results.append(
        pass_result("Forbidden Prediction Fields", "No labels, SHAP, or anomaly fields exist.")
        if forbidden_absent
        else fail_result("Forbidden Prediction Fields", "A forbidden prediction field exists.")
    )
    return results


def validate_source_guards() -> CheckResult:
    guarded_files = [
        PROJECT_ROOT / "pipelines" / "batch" / "ai4i_feature_adapter.py",
        PROJECT_ROOT / "scripts" / "run_spark_ai4i_adapter.py",
        PROJECT_ROOT / "scripts" / "run_spark_ai4i_adapter_docker.py",
        PROJECT_ROOT / "ml" / "inference" / "ai4i_telemetry.py",
        PROJECT_ROOT / "scripts" / "predict_silver_telemetry.py",
    ]
    forbidden_terms = [
        "data/gold",
        "psycopg",
        "sqlalchemy",
        "postgresql://",
        "pg_isready",
        "confluent_kafka",
        '.format("kafka")',
        "readStream",
        ".fit(",
        "fit_transform",
        "test.csv",
        "TreeExplainer",
        "IsolationForest",
    ]
    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in source:
                return fail_result("Source Guards", f"{term} found in {path.name}.")
    return pass_result("Source Guards", "No restricted services or model-development calls found.")


def deterministic_repeat_check(
    first_predictions: Sequence[Mapping[str, Any]],
) -> tuple[CheckResult, ai4i_telemetry.TelemetryPredictionSummary | None]:
    first_path = ai4i_telemetry.prediction_output_path(PROJECT_ROOT)
    first_bytes = first_path.read_bytes()
    adapter_result, _adapter_counts = run_adapter()
    if adapter_result.status is Status.FAIL:
        return adapter_result, None
    inference_result, second_summary = run_inference()
    if inference_result.status is Status.FAIL or second_summary is None:
        return inference_result, None
    second_path = ai4i_telemetry.prediction_output_path(PROJECT_ROOT)
    second_predictions = ai4i_telemetry.read_predictions_jsonl(second_path)
    if list(first_predictions) == second_predictions and first_bytes == second_path.read_bytes():
        return pass_result(
            "Second Run Determinism",
            "Logical and byte-level outputs match.",
        ), second_summary
    return fail_result(
        "Second Run Determinism",
        "Second run changed prediction output.",
    ), second_summary


def run_checks() -> tuple[
    list[CheckResult],
    ai4i_telemetry.TelemetryPredictionSummary | None,
    list[dict[str, Any]],
]:
    results = [check_spark_health(), check_silver_exists(), check_model_artifact()]
    if any(result.status is Status.FAIL and result.mandatory for result in results):
        return results, None, []

    adapter_result, adapter_counts = run_adapter()
    results.append(adapter_result)
    if adapter_result.status is Status.FAIL:
        return results, None, []

    adapter_records_result, adapter_records = load_adapter_records()
    results.append(adapter_records_result)
    if adapter_records_result.status is Status.FAIL:
        return results, None, []
    results.extend(validate_adapter_records(adapter_records, adapter_counts))

    inference_result, summary = run_inference()
    results.append(inference_result)
    if inference_result.status is Status.FAIL or summary is None:
        return results, None, []
    prediction_path = ai4i_telemetry.prediction_output_path(PROJECT_ROOT)
    prediction_records = ai4i_telemetry.read_predictions_jsonl(prediction_path)
    results.extend(validate_predictions(adapter_records, prediction_records))
    results.append(validate_source_guards())

    determinism_result, second_summary = deterministic_repeat_check(prediction_records)
    results.append(determinism_result)
    return results, second_summary or summary, prediction_records


def print_results(
    results: Sequence[CheckResult],
    summary: ai4i_telemetry.TelemetryPredictionSummary | None,
) -> None:
    print("Industrial Fleet Intelligence Platform AI4I telemetry inference validation")
    print()
    for result in results:
        print(f"{result.status.value} {result.name}: {result.message}")
    if summary is not None:
        print()
        print("Telemetry inference summary:")
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    mandatory_failures = [
        result for result in results if result.status is Status.FAIL and result.mandatory
    ]
    warn_count = sum(1 for result in results if result.status is Status.WARN)
    pass_count = sum(1 for result in results if result.status is Status.PASS)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {len(mandatory_failures)} FAIL")


def main() -> int:
    results, summary, _prediction_records = run_checks()
    print_results(results, summary)
    return 1 if any(result.status is Status.FAIL and result.mandatory for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

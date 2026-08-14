"""Validate operational AI4I SHAP materialization, persistence, and API access."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.config import get_settings  # noqa: E402
from apps.api.main import app  # noqa: E402
from apps.api.repositories.platform import PlatformRepository  # noqa: E402
from ml.explainability import ai4i_shap, ai4i_telemetry_shap  # noqa: E402
from ml.inference import ai4i_telemetry  # noqa: E402
from scripts.inspect_ai4i_explanation_state import inspect_state  # noqa: E402
from scripts.persist_ai4i_explanations import persist_explanations  # noqa: E402
from services.database import ai4i_explanations  # noqa: E402


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
class ValidationContext:
    predictor: Any | None = None
    generation: ai4i_telemetry_shap.OperationalExplainabilityResult | None = None
    first_persistence: Any | None = None
    second_persistence: Any | None = None
    inspection: Any | None = None
    counts_before: dict[str, int] | None = None
    counts_after: dict[str, int] | None = None
    representative_api_response: dict[str, Any] | None = None


def result(name: str, passed: bool, pass_message: str, fail_message: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=Status.PASS if passed else Status.FAIL,
        message=pass_message if passed else fail_message,
    )


def safe_check(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        return CheckResult(name, Status.FAIL, f"Unexpected error: {exc}")


def source_guard_violations(paths: Sequence[Path], tokens: Sequence[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        display_path = path.relative_to(PROJECT_ROOT).as_posix()
        for token in tokens:
            if token in text:
                violations.append(f"{display_path} contains {token}")
    return violations


def explanation_features_are_valid(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
) -> bool:
    expected = list(ai4i_telemetry_shap.EXPECTED_SEMANTIC_FEATURES)
    return all(
        [item.feature_name for item in record.feature_contributions] == expected
        for record in records
    )


def explanation_values_are_finite(
    records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
) -> bool:
    for record in records:
        values = [
            record.base_value,
            record.model_output_value,
            record.contribution_sum,
            record.additivity_error,
            *[item.shap_value for item in record.feature_contributions],
        ]
        if not all(math.isfinite(value) for value in values):
            return False
    return True


def runtime_matches_persisted(
    runtime_records: Sequence[ai4i_telemetry_shap.ExplanationRecord],
    persisted_rows: Sequence[ai4i_explanations.ExistingExplanationRow],
) -> bool:
    runtime_by_event = {record.event_id: record.to_dict() for record in runtime_records}
    persisted_by_event = {row.record.event_id: row.record.to_dict() for row in persisted_rows}
    return runtime_by_event == persisted_by_event


def unchanged_non_explanation_counts(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> bool:
    protected = ("model_predictions", "anomalies", "drift_snapshots", "drift_feature_metrics")
    return all(before.get(key) == after.get(key) for key in protected)


def api_source_guard() -> CheckResult:
    paths = list((PROJECT_ROOT / "apps" / "api").rglob("*.py"))
    tokens = (
        "final_model.joblib",
        "load_predictor",
        "predict_batch",
        "predict_proba",
        "TreeExplainer",
        "telemetry_explanations.jsonl",
        "data/explanations",
    )
    violations = source_guard_violations(paths, tokens)
    return result(
        "API Source Guard",
        not violations,
        "API source does not load models, calculate SHAP, or read explanation JSONL.",
        "; ".join(violations),
    )


def run_validation() -> tuple[list[CheckResult], ValidationContext]:
    results: list[CheckResult] = []
    repository = PlatformRepository(get_settings())
    counts_before = repository.protected_state_counts()
    predictor = None
    generation = None
    first_persistence = None
    second_persistence = None
    inspection = None
    representative_response: dict[str, Any] | None = None

    try:
        predictor = ai4i_shap.load_trusted_predictor(PROJECT_ROOT)
        results.append(
            CheckResult(
                "Packaged Model Integrity",
                Status.PASS,
                f"{predictor.model_name} {predictor.model_version} loaded with trusted metadata.",
            )
        )
    except (OSError, ValueError) as exc:
        results.append(CheckResult("Packaged Model Integrity", Status.FAIL, str(exc)))
        return results, ValidationContext(counts_before=counts_before)

    try:
        prediction_records = ai4i_telemetry_shap.load_prediction_records(
            ai4i_telemetry.prediction_output_path(PROJECT_ROOT)
        )
        results.append(
            CheckResult(
                "Operational Predictions",
                Status.PASS,
                f"{len(prediction_records)} runtime prediction record(s) loaded.",
            )
        )
    except (OSError, ValueError) as exc:
        results.append(CheckResult("Operational Predictions", Status.FAIL, str(exc)))
        return results, ValidationContext(predictor=predictor, counts_before=counts_before)

    try:
        adapter_records = ai4i_telemetry.load_adapter_records(
            root=PROJECT_ROOT,
            final_config=predictor.final_config,
        )
        results.append(
            CheckResult(
                "Adapter Records",
                Status.PASS,
                f"{len(adapter_records)} canonical adapter record(s) loaded.",
            )
        )
    except (OSError, ValueError) as exc:
        results.append(CheckResult("Adapter Records", Status.FAIL, str(exc)))
        return results, ValidationContext(predictor=predictor, counts_before=counts_before)

    try:
        generation = ai4i_telemetry_shap.run_operational_explainability(PROJECT_ROOT)
        output_sha = generation.summary.output_sha256
        second_generation = ai4i_telemetry_shap.run_operational_explainability(PROJECT_ROOT)
        deterministic = output_sha == second_generation.summary.output_sha256
        results.extend(
            [
                CheckResult(
                    "Explanation Generation",
                    Status.PASS,
                    (
                        f"{generation.summary.explanation_record_count} explanation "
                        "record(s) generated."
                    ),
                ),
                result(
                    "Count Alignment",
                    generation.summary.explanation_record_count
                    == generation.summary.prediction_record_count,
                    "Explanation count matches prediction count.",
                    "Explanation count does not match prediction count.",
                ),
                result(
                    "Event Identity Alignment",
                    {item.event_id for item in generation.prediction_records}
                    == {item.event_id for item in generation.explanation_records},
                    "Explanation event identities match prediction event identities.",
                    "Explanation event identities differ from prediction event identities.",
                ),
                result(
                    "Model Input Hash Alignment",
                    all(
                        prediction.model_input_sha256 == explanation.model_input_sha256
                        for prediction, explanation in zip(
                            generation.prediction_records,
                            generation.explanation_records,
                            strict=True,
                        )
                    ),
                    "Each explanation keeps the prediction model_input_sha256.",
                    "A model_input_sha256 mismatch was found.",
                ),
                result(
                    "Semantic Feature Contract",
                    explanation_features_are_valid(generation.explanation_records),
                    "Every explanation exposes exactly the six semantic AI4I features.",
                    "An explanation does not expose the exact six semantic features.",
                ),
                result(
                    "Finite SHAP Values",
                    explanation_values_are_finite(generation.explanation_records),
                    "All SHAP values and numeric explanation fields are finite.",
                    "Non-finite explanation value found.",
                ),
                result(
                    "Additivity",
                    generation.summary.max_additivity_error <= ai4i_shap.ADDITIVITY_TOLERANCE,
                    (
                        "Maximum additivity error "
                        f"{generation.summary.max_additivity_error} is within tolerance."
                    ),
                    "Maximum additivity error exceeds tolerance.",
                ),
                result(
                    "Deterministic Output",
                    deterministic,
                    f"Repeated generation kept SHA-256 {output_sha}.",
                    "Repeated generation changed the explanation output SHA-256.",
                ),
            ]
        )
    except (OSError, ValueError) as exc:
        results.append(CheckResult("Explanation Generation", Status.FAIL, str(exc)))
        return results, ValidationContext(predictor=predictor, counts_before=counts_before)

    try:
        first_persistence = persist_explanations()
        second_persistence = persist_explanations()
        inspection = inspect_state(generation.explanation_records)
        results.extend(
            [
                CheckResult(
                    "Explanation Persistence",
                    Status.PASS,
                    (
                        f"Inserted {first_persistence.summary.explanation_rows_inserted}; "
                        "reused "
                        f"{first_persistence.summary.existing_identical_explanations_reused}."
                    ),
                ),
                result(
                    "Persistence Idempotency",
                    second_persistence.summary.explanation_rows_inserted == 0,
                    (
                        "Second persistence reused "
                        f"{second_persistence.summary.existing_identical_explanations_reused} "
                        "row(s)."
                    ),
                    "Second persistence inserted duplicate explanation rows.",
                ),
                result(
                    "Persisted Values",
                    runtime_matches_persisted(
                        generation.explanation_records,
                        inspection.explanation_rows,
                    ),
                    "Persisted explanation values match runtime JSONL.",
                    "Persisted explanation values differ from runtime JSONL.",
                ),
                result(
                    "Stable Identity Uniqueness",
                    inspection.summary.duplicate_explanation_identity_count == 0,
                    "No duplicate explanation stable identities exist.",
                    "Duplicate explanation stable identities were found.",
                ),
            ]
        )
    except (OSError, RuntimeError, ValueError) as exc:
        results.append(CheckResult("Explanation Persistence", Status.FAIL, str(exc)))
        return results, ValidationContext(
            predictor=predictor,
            generation=generation,
            counts_before=counts_before,
        )

    try:
        first_record = generation.explanation_records[0]
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/machines/"
                f"{first_record.machine_code}/predictions/{first_record.event_id}/explanation"
            )
            if response.status_code == 200:
                representative_response = response.json()
            unknown_response = client.get(
                "/api/v1/machines/MCH-0001/predictions/"
                "00000000-0000-4000-8000-000000000404/explanation"
            )
        results.extend(
            [
                result(
                    "API Explanation Endpoint",
                    response.status_code == 200
                    and len(representative_response.get("feature_contributions", [])) == 6
                    if representative_response
                    else False,
                    "Explanation endpoint returned a persisted six-feature response.",
                    f"Expected 200 from explanation endpoint, got {response.status_code}.",
                ),
                result(
                    "API Explanation 404",
                    unknown_response.status_code == 404,
                    "Unknown explanation identity returns 404.",
                    f"Expected 404 for unknown explanation, got {unknown_response.status_code}.",
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        results.append(CheckResult("API Explanation Endpoint", Status.FAIL, str(exc)))

    results.append(api_source_guard())
    counts_after = repository.protected_state_counts()
    results.append(
        result(
            "Protected State Counts",
            unchanged_non_explanation_counts(counts_before, counts_after),
            "Prediction, anomaly, drift, and alert counts were not changed.",
            "A protected non-explanation state count changed.",
        )
    )

    return results, ValidationContext(
        predictor=predictor,
        generation=generation,
        first_persistence=first_persistence,
        second_persistence=second_persistence,
        inspection=inspection,
        counts_before=counts_before,
        counts_after=counts_after,
        representative_api_response=representative_response,
    )


def print_report(results: Sequence[CheckResult], context: ValidationContext) -> None:
    print("Industrial Fleet Intelligence Platform operational AI4I explainability validation")
    print()
    name_width = max(len(result.name) for result in results)
    for item in results:
        print(f"{item.status.value:<4} {item.name:<{name_width}} {item.message}")
    pass_count = sum(1 for item in results if item.status is Status.PASS)
    warn_count = sum(1 for item in results if item.status is Status.WARN)
    fail_count = sum(1 for item in results if item.status is Status.FAIL)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    if context.generation is not None:
        print()
        print(json.dumps(context.generation.summary.to_dict(), indent=2, sort_keys=True))


def exit_code_for(results: Sequence[CheckResult]) -> int:
    return 1 if any(item.status is Status.FAIL and item.mandatory for item in results) else 0


def main() -> int:
    results, context = run_validation()
    print_report(results, context)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only validator for AI4I SHAP explainability artifacts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability import ai4i_shap  # noqa: E402
from ml.inference import ai4i_predictor  # noqa: E402


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
class ValidationReport:
    results: list[CheckResult]

    @property
    def is_valid(self) -> bool:
        return not any(result.status is Status.FAIL and result.mandatory for result in self.results)


def result(name: str, passed: bool, pass_message: str, fail_message: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=Status.PASS if passed else Status.FAIL,
        message=pass_message if passed else fail_message,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return data


def source_guard_tokens() -> tuple[str, ...]:
    return ("test.csv", ".fit(", ".fit_transform(")


def source_guard_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            display_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = path.name
        for token in source_guard_tokens():
            if token in text:
                violations.append(f"{display_path} contains {token}")
    return violations


def required_artifact_paths() -> dict[str, Path]:
    paths = ai4i_shap.plot_paths(PROJECT_ROOT)
    return {
        "Packaged Model": ai4i_predictor.artifact_path(PROJECT_ROOT),
        "Explainability Summary": ai4i_shap.summary_path(PROJECT_ROOT),
        "Transformed Importance": ai4i_shap.transformed_importance_path(PROJECT_ROOT),
        "Grouped Importance": ai4i_shap.grouped_importance_path(PROJECT_ROOT),
        "Local Explanations": ai4i_shap.local_explanations_path(PROJECT_ROOT),
        "Sample Explanations": ai4i_shap.sample_explanations_path(PROJECT_ROOT),
        "Global Importance Plot": paths["global_importance"],
        "Beeswarm Plot": paths["beeswarm"],
        "Low Risk Waterfall": paths["low_risk"],
        "Threshold Near Waterfall": paths["threshold_near"],
        "High Risk Waterfall": paths["high_risk"],
    }


def validate_importance_report(
    path: Path,
    *,
    expected_features: list[str] | None = None,
    expected_count: int | None = None,
) -> tuple[list[CheckResult], pd.DataFrame]:
    results: list[CheckResult] = []
    frame = pd.read_csv(path)
    expected_columns = ["feature", "mean_absolute_shap", "rank"]
    results.append(
        result(
            f"{path.name} Columns",
            list(frame.columns) == expected_columns,
            "Importance report has expected columns.",
            "Importance report columns are not expected.",
        )
    )
    if expected_count is not None:
        results.append(
            result(
                f"{path.name} Row Count",
                len(frame) == expected_count,
                f"Importance report contains {expected_count} rows.",
                f"Expected {expected_count} rows, found {len(frame)}.",
            )
        )
    if expected_features is not None:
        results.append(
            result(
                f"{path.name} Features",
                frame["feature"].tolist() == expected_features,
                "Grouped importance contains the expected conceptual features.",
                "Grouped importance features are not expected.",
            )
        )
    finite_non_negative = bool(
        np.isfinite(frame["mean_absolute_shap"].to_numpy(dtype=float)).all()
        and (frame["mean_absolute_shap"] >= 0).all()
    )
    results.append(
        result(
            f"{path.name} Values",
            finite_non_negative,
            "Importance values are finite and non-negative.",
            "Importance values must be finite and non-negative.",
        )
    )
    ranks = frame["rank"].astype(int).tolist()
    results.append(
        result(
            f"{path.name} Ranks",
            ranks == list(range(1, len(frame) + 1)) and len(ranks) == len(set(ranks)),
            "Ranks are unique and consecutive.",
            "Ranks are not unique and consecutive.",
        )
    )
    expected_order = frame.sort_values(
        ["mean_absolute_shap", "feature"],
        ascending=[False, True],
        kind="mergesort",
    )["feature"].tolist()
    results.append(
        result(
            f"{path.name} Sorting",
            frame["feature"].tolist() == expected_order,
            "Importance rows are sorted deterministically.",
            "Importance rows are not sorted by descending attribution with feature tie-breaks.",
        )
    )
    return results, frame


def contribution_features(payload: Any) -> set[str]:
    features: set[str] = set()
    if isinstance(payload, dict):
        if isinstance(payload.get("feature"), str):
            features.add(payload["feature"])
        for value in payload.values():
            features.update(contribution_features(value))
    elif isinstance(payload, list):
        for item in payload:
            features.update(contribution_features(item))
    return features


def local_case_by_name(local_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = local_payload.get("cases", [])
    if not isinstance(cases, list):
        return {}
    return {str(case.get("case_name")): case for case in cases if isinstance(case, dict)}


def probability_predictions_are_consistent(cases: list[dict[str, Any]], threshold: float) -> bool:
    return all(
        int(case["failure_prediction"]) == int(float(case["failure_probability"]) >= threshold)
        for case in cases
    )


def max_reported_additivity_error(*payloads: dict[str, Any]) -> float:
    errors: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "additivity_check_error" in value:
                errors.append(float(value["additivity_check_error"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for payload in payloads:
        collect(payload)
    return max(errors) if errors else float("inf")


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    paths = required_artifact_paths()
    for name, path in paths.items():
        results.append(
            result(
                name,
                path.exists() and path.stat().st_size > 0,
                f"{path.relative_to(PROJECT_ROOT).as_posix()} exists.",
                f"{path.relative_to(PROJECT_ROOT).as_posix()} is missing or empty.",
            )
        )
    if any(item.status is Status.FAIL for item in results):
        return ValidationReport(results)

    try:
        predictor = ai4i_shap.load_trusted_predictor(PROJECT_ROOT)
        summary = load_json(ai4i_shap.summary_path(PROJECT_ROOT))
        local_payload = load_json(ai4i_shap.local_explanations_path(PROJECT_ROOT))
        sample_payload = load_json(ai4i_shap.sample_explanations_path(PROJECT_ROOT))
        transformed_results, transformed_frame = validate_importance_report(
            ai4i_shap.transformed_importance_path(PROJECT_ROOT),
            expected_count=8,
        )
        grouped_results, grouped_frame = validate_importance_report(
            ai4i_shap.grouped_importance_path(PROJECT_ROOT),
            expected_features=[
                row["feature"]
                for row in sorted(
                    pd.read_csv(ai4i_shap.grouped_importance_path(PROJECT_ROOT)).to_dict("records"),
                    key=lambda item: (-float(item["mean_absolute_shap"]), item["feature"]),
                )
            ],
            expected_count=6,
        )
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        results.append(CheckResult("Readable Explainability Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    results.extend(transformed_results)
    results.extend(grouped_results)
    expected_grouped = set(ai4i_shap.expected_grouped_features(predictor.final_config))
    results.append(
        result(
            "Grouped Conceptual Feature Set",
            set(grouped_frame["feature"].tolist()) == expected_grouped and len(grouped_frame) == 6,
            "Grouped report contains exactly the six original conceptual features.",
            "Grouped report does not contain exactly the six original conceptual features.",
        )
    )
    results.extend(
        [
            result(
                "SHAP Version",
                shap.__version__ == ai4i_shap.EXPECTED_SHAP_VERSION
                and summary.get("shap_version") == ai4i_shap.EXPECTED_SHAP_VERSION,
                "SHAP version is expected.",
                "SHAP version is not expected.",
            ),
            result(
                "Model Identity",
                summary.get("model_name") == ai4i_predictor.MODEL_NAME
                and summary.get("model_version") == ai4i_predictor.MODEL_VERSION
                and summary.get("final_config_hash") == predictor.final_config_hash,
                "Summary model identity matches the packaged model.",
                "Summary model identity does not match the packaged model.",
            ),
            result(
                "Summary Flags",
                summary.get("test_data_used") is False and summary.get("model_retrained") is False,
                "Summary records test_data_used=false and model_retrained=false.",
                "Summary flags must record test_data_used=false and model_retrained=false.",
            ),
            result(
                "Global Sample Size",
                summary.get("global_sample_size") == ai4i_shap.GLOBAL_SAMPLE_SIZE,
                "Global explanation sample size is exactly 1000.",
                "Global explanation sample size must be exactly 1000.",
            ),
            result(
                "Feature Counts",
                summary.get("transformed_feature_count") == len(transformed_frame)
                and summary.get("grouped_feature_count") == len(grouped_frame),
                "Summary feature counts match generated reports.",
                "Summary feature counts do not match generated reports.",
            ),
            result(
                "Frozen Threshold",
                float(predictor.decision_threshold) == 0.14,
                "Frozen threshold remains 0.14.",
                "Frozen threshold is not 0.14.",
            ),
        ]
    )

    case_map = local_case_by_name(local_payload)
    expected_cases = set(ai4i_shap.REPRESENTATIVE_CASE_NAMES)
    results.append(
        result(
            "Representative Cases",
            set(case_map) == expected_cases and len(case_map) == 3,
            "Local report contains exactly low_risk, threshold_near, and high_risk.",
            "Local report does not contain the expected three representative cases.",
        )
    )
    if set(case_map) == expected_cases:
        low = float(case_map["low_risk"]["failure_probability"])
        near = float(case_map["threshold_near"]["failure_probability"])
        high = float(case_map["high_risk"]["failure_probability"])
        local_cases = [case_map[name] for name in ai4i_shap.REPRESENTATIVE_CASE_NAMES]
        results.extend(
            [
                result(
                    "Representative Probability Ordering",
                    low <= near <= high,
                    "Representative probabilities are ordered low <= threshold_near <= high.",
                    "Representative probabilities are not ordered as expected.",
                ),
                result(
                    "Representative Probability Bounds",
                    all(0 <= float(case["failure_probability"]) <= 1 for case in local_cases),
                    "Representative probabilities are within [0, 1].",
                    "Representative probabilities are outside [0, 1].",
                ),
                result(
                    "Representative Threshold Predictions",
                    probability_predictions_are_consistent(local_cases, 0.14),
                    "Representative predictions follow probability >= 0.14.",
                    "Representative predictions do not follow probability >= 0.14.",
                ),
            ]
        )
        try:
            development_df, development_features, _modeling_config = (
                ai4i_shap.load_development_data(PROJECT_ROOT)
            )
            development_probabilities = predictor.pipeline.predict_proba(development_features)[
                :, ai4i_shap.POSITIVE_CLASS
            ]
            max_probability = float(np.max(development_probabilities))
            selected = ai4i_shap.select_representative_cases(
                development_df,
                development_probabilities,
                threshold=float(predictor.decision_threshold),
                traceability_field=str(predictor.final_config["traceability_field"]),
            )
            expected_by_name = {case.case_name: case for case in selected}
            results.append(
                result(
                    "High Risk Representative",
                    abs(high - max_probability) <= 5e-7
                    and int(case_map["high_risk"]["source_udi"])
                    == expected_by_name["high_risk"].source_udi,
                    "High-risk case is the highest-probability development observation.",
                    "High-risk case is not the expected highest-probability "
                    + "development observation.",
                )
            )
        except (OSError, ValueError) as exc:
            results.append(CheckResult("Representative Recalculation", Status.FAIL, str(exc)))

    sample_records = sample_payload.get("sample_explanations", [])
    results.extend(
        [
            result(
                "Sample Explanation Count",
                isinstance(sample_records, list) and len(sample_records) == 3,
                "Sample explanation report contains exactly three records.",
                "Sample explanation report must contain exactly three records.",
            ),
            result(
                "Sample Probability Bounds",
                isinstance(sample_records, list)
                and all(
                    0 <= float(record["failure_probability"]) <= 1 for record in sample_records
                ),
                "Sample probabilities are within [0, 1].",
                "Sample probabilities are outside [0, 1].",
            ),
            result(
                "Sample Threshold Predictions",
                isinstance(sample_records, list)
                and probability_predictions_are_consistent(sample_records, 0.14),
                "Sample predictions follow probability >= 0.14.",
                "Sample predictions do not follow probability >= 0.14.",
            ),
        ]
    )

    max_error = max(
        float(summary.get("max_observed_additivity_error", float("inf"))),
        max_reported_additivity_error(local_payload, sample_payload),
    )
    results.append(
        result(
            "Additivity Errors",
            max_error <= ai4i_shap.ADDITIVITY_TOLERANCE,
            "Reported additivity errors are within tolerance.",
            "Reported additivity errors exceed tolerance.",
        )
    )

    local_text = json.dumps(local_payload, sort_keys=True)
    forbidden_features = {
        predictor.final_config["target"],
        "UDI",
        "Product ID",
        *predictor.final_config["excluded_leakage_sensitive_fields"],
    }
    all_feature_names = set(transformed_frame["feature"].tolist()) | set(
        grouped_frame["feature"].tolist()
    )
    all_feature_names.update(contribution_features(local_payload))
    all_feature_names.update(contribution_features(sample_payload))
    results.extend(
        [
            result(
                "Local Target Exclusion",
                str(predictor.final_config["target"]) not in local_text,
                "Local explanations do not contain the target label.",
                "Local explanations contain the target label.",
            ),
            result(
                "Leakage-Sensitive Feature Exclusion",
                not (all_feature_names & forbidden_features),
                "Explainability artifacts exclude leakage-sensitive and identifier features.",
                "Explainability artifacts contain leakage-sensitive or identifier features.",
            ),
        ]
    )

    source_violations = source_guard_violations(
        [
            PROJECT_ROOT / "ml" / "explainability" / "ai4i_shap.py",
            PROJECT_ROOT / "scripts" / "explain_ai4i_model.py",
        ]
    )
    results.append(
        result(
            "Explainability Source Guard",
            not source_violations,
            "Explainability sources avoid restricted split references and model-fitting calls.",
            "; ".join(source_violations),
        )
    )
    return ValidationReport(results)


def print_report(report: ValidationReport) -> None:
    for item in report.results:
        print(f"{item.status} {item.name}: {item.message}")
    pass_count = sum(1 for item in report.results if item.status is Status.PASS)
    warn_count = sum(1 for item in report.results if item.status is Status.WARN)
    fail_count = sum(1 for item in report.results if item.status is Status.FAIL)
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    report = validate_artifacts()
    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

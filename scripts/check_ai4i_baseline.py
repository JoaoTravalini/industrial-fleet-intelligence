"""Read-only validator for AI4I baseline classification artifacts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from ml.training import ai4i_baseline  # noqa: E402


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
        name, Status.PASS if passed else Status.FAIL, pass_message if passed else fail_message
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_test_metrics(value: Any, parent_key: str = "") -> bool:
    allowed_test_keys = {"test_data_used", "test_set_status"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in allowed_test_keys and (
                key == "test_metrics" or key.startswith("test_") or key.endswith("_test")
            ):
                return True
            if contains_test_metrics(child, key):
                return True
    elif isinstance(value, list):
        return any(contains_test_metrics(item, parent_key) for item in value)
    return False


def artifact_paths() -> dict[str, Path]:
    paths = {
        "baseline_metrics": ai4i_baseline.baseline_metrics_path(),
        "validation_predictions": ai4i_baseline.baseline_predictions_path(),
        "logistic_coefficients": ai4i_baseline.logistic_coefficients_path(),
    }
    paths.update(ai4i_baseline.baseline_plot_paths())
    return paths


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    config = ai4i_modeling.load_modeling_config()
    split_summary = ai4i_modeling.load_split_summary()
    expected_validation_rows = int(split_summary["split_rows"]["validation"])

    paths = artifact_paths()
    missing = []
    for name, path in paths.items():
        exists = path.exists() and (path.stat().st_size > 0 if path.suffix == ".png" else True)
        if not exists:
            missing.append(name)
        results.append(
            result(
                f"{name} Artifact",
                exists,
                "Expected artifact exists.",
                f"Expected artifact is missing or empty: {path.name}",
            )
        )
    if missing:
        return ValidationReport(results)

    try:
        metrics = load_json(paths["baseline_metrics"])
        predictions = pd.read_csv(paths["validation_predictions"])
        coefficients = pd.read_csv(paths["logistic_coefficients"])
        validation_df = pd.read_csv(ai4i_baseline.validation_path())
        assignments = pd.read_csv(ai4i_modeling.generated_artifact_paths()["split_assignments"])
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        results.append(CheckResult("Readable Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    required_prediction_columns = [
        "source_udi",
        "target",
        "dummy_probability",
        "dummy_prediction",
        "logistic_probability",
        "logistic_prediction",
    ]
    results.append(
        result(
            "Prediction Schema",
            list(predictions.columns) == required_prediction_columns,
            "Prediction CSV contains the expected columns only.",
            "Prediction CSV schema is not the expected validation-output schema.",
        )
    )
    results.append(
        result(
            "Prediction Row Count",
            len(predictions) == expected_validation_rows == 1500,
            "Prediction CSV contains exactly 1500 validation rows.",
            f"Expected 1500 validation rows, found {len(predictions)}.",
        )
    )
    results.append(
        result(
            "Prediction source_udi Uniqueness",
            predictions["source_udi"].is_unique,
            "source_udi is unique in validation predictions.",
            "source_udi contains duplicate values in validation predictions.",
        )
    )

    validation_udis = set(assignments.loc[assignments["split"] == "validation", "source_udi"])
    train_udis = set(assignments.loc[assignments["split"] == "train", "source_udi"])
    locked_test_udis = set(assignments.loc[assignments["split"] == "test", "source_udi"])
    prediction_udis = set(predictions["source_udi"])
    validation_file_udis = set(validation_df["source_udi"])
    results.append(
        result(
            "Authoritative Validation Coverage",
            prediction_udis == validation_udis == validation_file_udis,
            "Prediction source_udi values exactly match the validation split.",
            "Prediction source_udi values do not match the authoritative validation split.",
        )
    )
    results.append(
        result(
            "No Train Rows In Predictions",
            not (prediction_udis & train_udis),
            "No train source_udi values appear in validation predictions.",
            "Train source_udi values appear in validation predictions.",
        )
    )
    results.append(
        result(
            "No Locked Test Rows In Predictions",
            not (prediction_udis & locked_test_udis),
            "No locked test source_udi values appear in validation predictions.",
            "Locked test source_udi values appear in validation predictions.",
        )
    )

    results.append(
        result(
            "Target Values",
            set(predictions["target"].unique()).issubset({0, 1}),
            "Target values are binary.",
            "Target values must contain only 0 and 1.",
        )
    )
    for column in ["dummy_prediction", "logistic_prediction"]:
        results.append(
            result(
                f"{column} Values",
                set(predictions[column].unique()).issubset({0, 1}),
                f"{column} contains only 0 and 1.",
                f"{column} contains values outside 0 and 1.",
            )
        )
    for column in ["dummy_probability", "logistic_probability"]:
        results.append(
            result(
                f"{column} Bounds",
                predictions[column].between(0, 1, inclusive="both").all(),
                f"{column} values are within [0, 1].",
                f"{column} contains probabilities outside [0, 1].",
            )
        )
    for model_key in ["dummy_classifier", "logistic_regression"]:
        metric_block = metrics.get(model_key)
        required_present = isinstance(metric_block, dict) and all(
            key in metric_block for key in ai4i_baseline.REQUIRED_METRIC_KEYS
        )
        results.append(
            result(
                f"{model_key} Metrics",
                required_present,
                "All required validation metrics are present.",
                "One or more required validation metrics are missing.",
            )
        )
        if isinstance(metric_block, dict) and "confusion_matrix" in metric_block:
            matrix = metric_block["confusion_matrix"]
            total = sum(sum(int(value) for value in row) for row in matrix)
            results.append(
                result(
                    f"{model_key} Confusion Matrix Total",
                    total == expected_validation_rows,
                    "Confusion matrix total equals 1500 validation rows.",
                    f"Confusion matrix total is {total}, expected {expected_validation_rows}.",
                )
            )

    data_block = metrics.get("data", {})
    expected_features = ai4i_baseline.predictive_feature_columns(config)
    results.append(
        result(
            "Feature Policy",
            data_block.get("predictive_feature_list") == expected_features,
            "Metrics JSON feature list matches modeling configuration.",
            "Metrics JSON feature list does not match modeling configuration.",
        )
    )
    results.append(
        result(
            "Excluded Identifier Policy",
            data_block.get("excluded_identifiers") == list(config.excluded_identifiers),
            "Excluded identifiers match modeling configuration.",
            "Excluded identifiers do not match modeling configuration.",
        )
    )
    results.append(
        result(
            "Excluded Leakage Policy",
            data_block.get("excluded_leakage_sensitive_fields")
            == list(config.excluded_leakage_sensitive_columns),
            "Excluded leakage-sensitive fields match modeling configuration.",
            "Excluded leakage-sensitive fields do not match modeling configuration.",
        )
    )
    results.append(
        result(
            "Test Data Flag",
            data_block.get("test_data_used") is False,
            "Metrics JSON explicitly records that test data was not used.",
            "Metrics JSON does not clearly record that test data was not used.",
        )
    )
    results.append(
        result(
            "No Test Metrics",
            not contains_test_metrics(metrics),
            "No test metrics are present in baseline_metrics.json.",
            "Test-like metric keys are present in baseline_metrics.json.",
        )
    )

    required_coefficient_columns = ["feature", "coefficient", "absolute_coefficient"]
    results.append(
        result(
            "Coefficient Schema",
            list(coefficients.columns) == required_coefficient_columns,
            "Coefficient report contains the expected columns only.",
            "Coefficient report schema is not expected.",
        )
    )
    coefficient_features = set(coefficients["feature"].astype(str))
    forbidden_exact = {
        "source_udi",
        "UDI",
        *config.excluded_identifiers,
        *config.excluded_leakage_sensitive_columns,
    }
    forbidden_found = sorted(coefficient_features & forbidden_exact)
    results.append(
        result(
            "Coefficient Leakage Check",
            not forbidden_found,
            "No leakage-sensitive or identifier feature appears in coefficients.",
            "Forbidden coefficient feature(s): " + ", ".join(forbidden_found),
        )
    )
    return ValidationReport(results)


def print_report(report: ValidationReport) -> None:
    print("Industrial Fleet Intelligence Platform AI4I baseline artifact validation")
    print("TEST SET STATUS: LOCKED / NOT USED")
    print()
    name_width = max(len(item.name) for item in report.results)
    for item in report.results:
        print(f"{item.status.value:<4} {item.name:<{name_width}} {item.message}")
    pass_count = sum(1 for item in report.results if item.status is Status.PASS)
    warn_count = sum(1 for item in report.results if item.status is Status.WARN)
    fail_count = sum(1 for item in report.results if item.status is Status.FAIL)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    try:
        report = validate_artifacts()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print("Industrial Fleet Intelligence Platform AI4I baseline artifact validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Baseline artifact validation failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform AI4I baseline artifact validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

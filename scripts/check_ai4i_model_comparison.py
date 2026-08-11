"""Read-only validator for AI4I model-comparison artifacts."""

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
from ml.training import ai4i_baseline, ai4i_imbalance, ai4i_model_comparison  # noqa: E402


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
        name,
        Status.PASS if passed else Status.FAIL,
        pass_message if passed else fail_message,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_test_metrics(value: Any) -> bool:
    allowed = {"test_data_used", "test_set_status"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in allowed and (
                key == "test_metrics" or key.startswith("test_") or key.endswith("_test")
            ):
                return True
            if contains_test_metrics(child):
                return True
    if isinstance(value, list):
        return any(contains_test_metrics(item) for item in value)
    return False


def artifact_paths() -> dict[str, Path]:
    paths = {
        "model_comparison_metrics": ai4i_model_comparison.comparison_metrics_path(),
        "oof_predictions": ai4i_model_comparison.oof_predictions_path(),
        "threshold_analysis": ai4i_model_comparison.threshold_analysis_path(),
        "validation_predictions": ai4i_model_comparison.comparison_validation_predictions_path(),
        "markdown_report": ai4i_model_comparison.comparison_doc_path(),
    }
    paths.update(ai4i_model_comparison.comparison_plot_paths())
    return paths


def matrix_total(matrix: list[list[int]]) -> int:
    return sum(sum(int(value) for value in row) for row in matrix)


def rounded_thresholds(values: pd.Series) -> list[float]:
    return [round(float(value), 2) for value in values.tolist()]


def floats_close(left: float | None, right: float | None, tolerance: float = 0.000001) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    config = ai4i_modeling.load_modeling_config()
    split_summary = ai4i_modeling.load_split_summary()
    expected_features = ai4i_baseline.predictive_feature_columns(config)
    expected_train_rows = int(split_summary["split_rows"]["train"])
    expected_validation_rows = int(split_summary["split_rows"]["validation"])

    paths = artifact_paths()
    missing: list[str] = []
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
        metrics = load_json(paths["model_comparison_metrics"])
        previous_metrics = load_json(ai4i_imbalance.imbalance_metrics_path())
        oof_predictions = pd.read_csv(paths["oof_predictions"])
        threshold_analysis = pd.read_csv(paths["threshold_analysis"])
        validation_predictions = pd.read_csv(paths["validation_predictions"])
        train_df = pd.read_csv(ai4i_baseline.train_path())
        validation_df = pd.read_csv(ai4i_baseline.validation_path())
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        results.append(CheckResult("Readable Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    expected_oof_columns = [
        "source_udi",
        "target",
        "standard_logistic_probability",
        "random_forest_probability",
        "xgboost_probability",
    ]
    results.append(
        result(
            "OOF Schema",
            list(oof_predictions.columns) == expected_oof_columns,
            "OOF prediction schema is expected.",
            "OOF prediction schema is not expected.",
        )
    )
    results.append(
        result(
            "OOF Row Count",
            len(oof_predictions) == expected_train_rows == 7000,
            "OOF predictions contain exactly 7000 training observations.",
            f"Expected 7000 OOF rows, found {len(oof_predictions)}.",
        )
    )
    results.append(
        result(
            "OOF source_udi Uniqueness",
            oof_predictions["source_udi"].is_unique,
            "OOF source_udi values are unique.",
            "OOF source_udi values contain duplicates.",
        )
    )
    train_udis = set(train_df["source_udi"])
    validation_udis = set(validation_df["source_udi"])
    oof_udis = set(oof_predictions["source_udi"])
    results.append(
        result(
            "OOF Training Coverage",
            oof_udis == train_udis,
            "OOF source_udi values exactly match training split.",
            "OOF source_udi values do not match the training split.",
        )
    )
    for column in [
        "standard_logistic_probability",
        "random_forest_probability",
        "xgboost_probability",
    ]:
        results.append(
            result(
                f"{column} Bounds",
                oof_predictions[column].between(0, 1, inclusive="both").all(),
                f"{column} values are within [0, 1].",
                f"{column} contains values outside [0, 1].",
            )
        )
    results.append(
        result(
            "Threshold Analysis Schema",
            list(threshold_analysis.columns) == ai4i_model_comparison.REQUIRED_THRESHOLD_COLUMNS,
            "Threshold analysis schema is expected.",
            "Threshold analysis schema is not expected.",
        )
    )
    expected_thresholds = list(ai4i_model_comparison.THRESHOLDS)
    threshold_failures: list[str] = []
    for model_name in ai4i_model_comparison.MODEL_NAMES:
        subset = threshold_analysis[threshold_analysis["model"] == model_name]
        if rounded_thresholds(subset["threshold"]) != expected_thresholds:
            threshold_failures.append(model_name)
    results.append(
        result(
            "Threshold Grid",
            not threshold_failures
            and set(threshold_analysis["model"]) == set(ai4i_model_comparison.MODEL_NAMES)
            and len(threshold_analysis) == len(ai4i_model_comparison.MODEL_NAMES) * 99,
            "Threshold analysis covers all three models with thresholds 0.01 through 0.99.",
            "Threshold grid is incomplete for: " + ", ".join(threshold_failures),
        )
    )

    recomputed_oof_metrics = ai4i_model_comparison.build_oof_model_metrics(oof_predictions)
    recomputed_candidates = ai4i_model_comparison.build_threshold_candidates(threshold_analysis)
    recomputed_selection = ai4i_model_comparison.select_development_candidate(
        recomputed_oof_metrics,
        recomputed_candidates,
    )
    results.append(
        result(
            "Threshold Candidate Consistency",
            metrics.get("threshold_candidates") == recomputed_candidates,
            "Threshold candidates match saved threshold analysis.",
            "Threshold candidates do not match saved threshold analysis.",
        )
    )
    saved_policy = metrics.get("candidate_selection_policy", {})
    results.append(
        result(
            "Candidate Selection Policy",
            saved_policy == recomputed_selection,
            "Selected development candidate follows the documented train OOF AP policy.",
            "Selected development candidate does not follow the train OOF AP policy.",
        )
    )
    selected_model = metrics.get("selected_model")
    selected_threshold = metrics.get("selected_threshold")
    selected_max_f2 = recomputed_candidates.get(str(selected_model), {}).get("max_f2", {})
    results.append(
        result(
            "Selected Threshold Source",
            selected_threshold == selected_max_f2.get("threshold"),
            "Selected threshold matches the selected model train-derived max-F2 threshold.",
            "Selected threshold does not match the selected model max-F2 threshold.",
        )
    )

    previous_standard = previous_metrics["train_oof_results"]["standard_logistic"]
    current_standard = recomputed_oof_metrics["standard_logistic"]
    logistic_reference_ok = floats_close(
        current_standard["average_precision"],
        previous_standard["average_precision"],
    ) and floats_close(current_standard["roc_auc"], previous_standard["roc_auc"])
    results.append(
        result(
            "Logistic Reference Regression",
            logistic_reference_ok,
            "Standard Logistic OOF AP and ROC-AUC reproduce the prior imbalance experiment.",
            "Standard Logistic OOF AP or ROC-AUC materially differs from the prior experiment.",
        )
    )

    xgb_config = metrics.get("model_configurations", {}).get("xgboost", {})
    expected_xgb_config = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
        "device": "cpu",
    }
    xgb_fixed_ok = all(xgb_config.get(key) == value for key, value in expected_xgb_config.items())
    results.append(
        result(
            "XGBoost Fixed Configuration",
            xgb_fixed_ok,
            "XGBoost fixed CPU configuration is recorded as expected.",
            "XGBoost fixed CPU configuration is not recorded as expected.",
        )
    )
    folds = ai4i_model_comparison.make_shared_fold_assignments(train_df, config)
    expected_fold_weights = ai4i_model_comparison.calculate_xgboost_fold_scale_pos_weights(
        train_df,
        config,
        folds,
    )
    results.append(
        result(
            "XGBoost scale_pos_weight Policy",
            xgb_config.get("oof_fold_scale_pos_weights") == expected_fold_weights
            and floats_close(
                xgb_config.get("full_train_scale_pos_weight_if_selected"),
                ai4i_model_comparison.calculate_scale_pos_weight(train_df[config.target_column]),
            ),
            "XGBoost scale_pos_weight values match training-label-only calculations.",
            "XGBoost scale_pos_weight values do not match training-label-only calculations.",
        )
    )
    expected_validation_columns = [
        "source_udi",
        "target",
        "probability",
        "prediction_threshold_0_5",
        "prediction_selected_threshold",
    ]
    results.append(
        result(
            "Validation Prediction Schema",
            list(validation_predictions.columns) == expected_validation_columns,
            "Validation prediction schema is expected.",
            "Validation prediction schema is not expected.",
        )
    )
    prediction_udis = set(validation_predictions["source_udi"])
    results.append(
        result(
            "Validation Prediction Row Count",
            len(validation_predictions) == expected_validation_rows == 1500,
            "Validation predictions contain exactly 1500 observations.",
            f"Expected 1500 validation predictions, found {len(validation_predictions)}.",
        )
    )
    results.append(
        result(
            "Validation Coverage",
            prediction_udis == validation_udis,
            "Validation predictions exactly match the authoritative validation split.",
            "Validation predictions do not match the authoritative validation split.",
        )
    )
    results.append(
        result(
            "No Train Rows In Validation Artifact",
            not (prediction_udis & train_udis),
            "No train source_udi values appear in validation predictions.",
            "Train source_udi values appear in validation predictions.",
        )
    )
    results.append(
        result(
            "Validation Probability Bounds",
            validation_predictions["probability"].between(0, 1, inclusive="both").all(),
            "Validation probabilities are within [0, 1].",
            "Validation probabilities contain values outside [0, 1].",
        )
    )
    for column in ["target", "prediction_threshold_0_5", "prediction_selected_threshold"]:
        results.append(
            result(
                f"{column} Binary Values",
                set(validation_predictions[column].unique()).issubset({0, 1}),
                f"{column} contains only 0 and 1.",
                f"{column} contains values outside 0 and 1.",
            )
        )

    results.append(
        result(
            "Feature Policy",
            metrics.get("feature_policy", {}).get("predictive_feature_list") == expected_features,
            "Feature policy matches modeling configuration.",
            "Feature policy does not match modeling configuration.",
        )
    )
    forbidden_features = {
        "source_udi",
        "UDI",
        *config.excluded_identifiers,
        *config.excluded_leakage_sensitive_columns,
    }
    configured_features = set(metrics.get("feature_policy", {}).get("predictive_feature_list", []))
    results.append(
        result(
            "Leakage Feature Check",
            not (configured_features & forbidden_features),
            "No leakage-sensitive or identifier feature appears in the feature policy.",
            "Forbidden feature(s) found: "
            + ", ".join(sorted(configured_features & forbidden_features)),
        )
    )
    results.append(
        result(
            "Test Data Flag",
            metrics.get("data", {}).get("test_data_used") is False,
            "Metrics JSON explicitly records that test data was not used.",
            "Metrics JSON does not clearly record that test data was not used.",
        )
    )
    results.append(
        result(
            "No Test Metrics",
            not contains_test_metrics(metrics),
            "No test metrics are present in model_comparison_metrics.json.",
            "Test-like metric keys are present in model_comparison_metrics.json.",
        )
    )

    matrix_failures: list[str] = []
    for model_name in ai4i_model_comparison.MODEL_NAMES:
        model_metrics = metrics["train_oof_results"][model_name]
        if matrix_total(model_metrics["threshold_0_5"]["confusion_matrix"]) != expected_train_rows:
            matrix_failures.append(f"{model_name} train_oof threshold_0_5")
        for candidate_name in ["max_f1", "max_f2"]:
            candidate = metrics["threshold_candidates"][model_name][candidate_name]
            if (
                int(candidate["true_positive"])
                + int(candidate["false_positive"])
                + int(candidate["true_negative"])
                + int(candidate["false_negative"])
                != expected_train_rows
            ):
                matrix_failures.append(f"{model_name} {candidate_name}")
    validation_results = metrics["validation_results"]
    for key in ["threshold_0_5", "selected_threshold"]:
        if matrix_total(validation_results[key]["confusion_matrix"]) != expected_validation_rows:
            matrix_failures.append(f"validation {key}")
    results.append(
        result(
            "Confusion Matrix Totals",
            not matrix_failures,
            "All confusion matrices sum to the expected split row counts.",
            "Bad confusion matrix totals: " + ", ".join(matrix_failures),
        )
    )
    return ValidationReport(results)


def print_report(report: ValidationReport) -> None:
    print("Industrial Fleet Intelligence Platform AI4I model comparison artifact validation")
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
        print("Industrial Fleet Intelligence Platform AI4I model comparison artifact validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Model comparison artifact validation failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform AI4I model comparison artifact validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

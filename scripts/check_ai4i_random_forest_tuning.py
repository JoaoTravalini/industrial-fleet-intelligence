"""Read-only validator for AI4I Random Forest tuning artifacts."""

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
from ml.training import ai4i_baseline, ai4i_random_forest_tuning  # noqa: E402


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


def contains_locked_test_metrics(value: Any) -> bool:
    allowed = {"test_data_used", "test_set_status"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in allowed and (
                key == "test_metrics" or key.startswith("locked_test") or key.endswith("_test")
            ):
                return True
            if contains_locked_test_metrics(child):
                return True
    if isinstance(value, list):
        return any(contains_locked_test_metrics(item) for item in value)
    return False


def artifact_paths() -> dict[str, Path]:
    paths = {
        "tuning_metrics": ai4i_random_forest_tuning.tuning_metrics_path(),
        "nested_oof_predictions": ai4i_random_forest_tuning.nested_oof_predictions_path(),
        "outer_folds": ai4i_random_forest_tuning.outer_folds_path(),
        "grid_results": ai4i_random_forest_tuning.grid_results_path(),
        "threshold_analysis": ai4i_random_forest_tuning.threshold_analysis_path(),
        "validation_predictions": ai4i_random_forest_tuning.validation_predictions_path(),
        "markdown_report": ai4i_random_forest_tuning.tuning_doc_path(),
    }
    paths.update(ai4i_random_forest_tuning.tuning_plot_paths())
    return paths


def matrix_total(matrix: list[list[int]]) -> int:
    return sum(sum(int(value) for value in row) for row in matrix)


def rounded_thresholds(values: pd.Series) -> list[float]:
    return [round(float(value), 2) for value in values.tolist()]


def parameter_key(n_estimators: Any, max_depth: Any, min_samples_leaf: Any) -> tuple[int, str, int]:
    return (
        int(n_estimators),
        ai4i_random_forest_tuning.format_max_depth(max_depth),
        int(min_samples_leaf),
    )


def serialized_key(params: dict[str, Any]) -> tuple[int, str, int]:
    return parameter_key(params["n_estimators"], params["max_depth"], params["min_samples_leaf"])


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    config = ai4i_modeling.load_modeling_config()
    expected_features = ai4i_baseline.predictive_feature_columns(config)
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
        metrics = load_json(paths["tuning_metrics"])
        nested_oof = pd.read_csv(paths["nested_oof_predictions"])
        outer_folds = pd.read_csv(paths["outer_folds"], keep_default_na=False)
        grid_results = pd.read_csv(paths["grid_results"], keep_default_na=False)
        threshold_analysis = pd.read_csv(paths["threshold_analysis"])
        validation_predictions = pd.read_csv(paths["validation_predictions"])
        train_df = pd.read_csv(ai4i_baseline.train_path())
        validation_df = pd.read_csv(ai4i_baseline.validation_path())
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        results.append(CheckResult("Readable Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    expected_train_rows = len(train_df)
    expected_validation_rows = len(validation_df)
    train_udis = set(train_df[config.derived_traceability_field])
    validation_udis = set(validation_df[config.derived_traceability_field])

    results.append(
        result(
            "Nested OOF Schema",
            list(nested_oof.columns) == ai4i_random_forest_tuning.NESTED_OOF_COLUMNS,
            "Nested OOF prediction schema is expected.",
            "Nested OOF prediction schema is not expected.",
        )
    )
    results.append(
        result(
            "Nested OOF Row Count",
            len(nested_oof) == expected_train_rows == 7000,
            "Nested OOF predictions contain exactly 7000 training observations.",
            f"Expected 7000 nested OOF rows, found {len(nested_oof)}.",
        )
    )
    results.append(
        result(
            "Nested OOF source_udi Uniqueness",
            nested_oof["source_udi"].is_unique,
            "Nested OOF source_udi values are unique.",
            "Nested OOF source_udi values contain duplicates.",
        )
    )
    results.append(
        result(
            "Nested OOF Training Coverage",
            set(nested_oof["source_udi"]) == train_udis,
            "Nested OOF source_udi values exactly match training split.",
            "Nested OOF source_udi values do not match training split.",
        )
    )
    results.append(
        result(
            "Nested OOF Outer Fold Values",
            set(nested_oof["outer_fold"].unique()) == {1, 2, 3, 4, 5},
            "Nested OOF rows identify exactly five outer folds.",
            "Nested OOF outer_fold values are not exactly 1 through 5.",
        )
    )
    results.append(
        result(
            "Nested OOF Probability Bounds",
            nested_oof["probability"].between(0, 1, inclusive="both").all(),
            "Nested OOF probabilities are within [0, 1].",
            "Nested OOF probabilities contain values outside [0, 1].",
        )
    )
    allowed_keys = ai4i_random_forest_tuning.allowed_parameter_keys()
    outer_keys = {
        parameter_key(
            row.selected_n_estimators,
            row.selected_max_depth,
            row.selected_min_samples_leaf,
        )
        for row in outer_folds.itertuples(index=False)
    }
    results.append(
        result(
            "Outer Fold Schema",
            list(outer_folds.columns) == ai4i_random_forest_tuning.OUTER_FOLD_COLUMNS,
            "Outer-fold search result schema is expected.",
            "Outer-fold search result schema is not expected.",
        )
    )
    results.append(
        result(
            "Outer Fold Count",
            len(outer_folds) == 5 and set(outer_folds["outer_fold"]) == {1, 2, 3, 4, 5},
            "Outer-fold search results contain exactly five folds.",
            "Outer-fold search results do not contain exactly five folds.",
        )
    )
    results.append(
        result(
            "Outer Fold Selected Parameters",
            outer_keys.issubset(allowed_keys),
            "Each outer-fold selected configuration belongs to the allowed grid.",
            "An outer-fold selected configuration is outside the allowed grid.",
        )
    )

    fold_count_failures: list[str] = []
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    expected_fold_counts: dict[int, tuple[int, int, int, int]] = {}
    for fold_number, (training_index, holdout_index) in enumerate(
        ai4i_random_forest_tuning.make_outer_cv().split(features, target), start=1
    ):
        expected_fold_counts[fold_number] = (
            len(training_index),
            len(holdout_index),
            int(target.iloc[training_index].sum()),
            int(target.iloc[holdout_index].sum()),
        )
    for row in outer_folds.itertuples(index=False):
        expected = expected_fold_counts[int(row.outer_fold)]
        actual = (
            int(row.training_rows),
            int(row.holdout_rows),
            int(row.training_positive_count),
            int(row.holdout_positive_count),
        )
        if actual != expected:
            fold_count_failures.append(f"outer_fold {row.outer_fold}")
    results.append(
        result(
            "Outer Fold Row Counts",
            not fold_count_failures,
            "Outer-fold row and positive counts match the train-only outer CV splits.",
            "Bad outer-fold count(s): " + ", ".join(fold_count_failures),
        )
    )

    grid_keys = {
        parameter_key(row.n_estimators, row.max_depth, row.min_samples_leaf)
        for row in grid_results.itertuples(index=False)
    }
    results.append(
        result(
            "Full Grid Search Schema",
            list(grid_results.columns) == ai4i_random_forest_tuning.GRID_RESULT_COLUMNS,
            "Full grid search schema is expected.",
            "Full grid search schema is not expected.",
        )
    )
    results.append(
        result(
            "Full Grid Search Configurations",
            len(grid_results) == 8 and grid_keys == allowed_keys,
            "Full grid search contains exactly the eight allowed parameter combinations.",
            "Full grid search does not contain exactly the eight allowed parameter combinations.",
        )
    )
    best_params = metrics.get("full_train_grid_search", {}).get("best_hyperparameters", {})
    results.append(
        result(
            "Full Train Best Parameters",
            bool(best_params) and serialized_key(best_params) in allowed_keys,
            "Full-train best parameters belong to the allowed grid.",
            "Full-train best parameters are missing or outside the allowed grid.",
        )
    )
    rank_one = grid_results[grid_results["rank_test_average_precision"] == 1]
    best_summary = metrics.get("full_train_grid_search", {})
    best_stats_ok = (
        len(rank_one) >= 1
        and ai4i_random_forest_tuning._rounded_float(
            float(rank_one.iloc[0]["mean_test_average_precision"])
        )
        == best_summary.get("best_mean_average_precision")
        and ai4i_random_forest_tuning._rounded_float(
            float(rank_one.iloc[0]["std_test_average_precision"])
        )
        == best_summary.get("best_std_average_precision")
    )
    results.append(
        result(
            "Full Train Best Score",
            best_stats_ok,
            "Full-train best AP and standard deviation match the ranked grid results.",
            "Full-train best AP or standard deviation does not match grid results.",
        )
    )

    results.append(
        result(
            "Threshold Analysis Schema",
            list(threshold_analysis.columns) == ai4i_random_forest_tuning.THRESHOLD_COLUMNS,
            "Threshold analysis schema is expected.",
            "Threshold analysis schema is not expected.",
        )
    )
    results.append(
        result(
            "Threshold Grid",
            rounded_thresholds(threshold_analysis["threshold"])
            == list(ai4i_random_forest_tuning.THRESHOLDS),
            "Threshold analysis covers thresholds 0.01 through 0.99.",
            "Threshold analysis does not cover the expected threshold grid.",
        )
    )
    recomputed_oof_metrics = ai4i_random_forest_tuning.build_nested_oof_metrics(nested_oof)
    recomputed_candidates = ai4i_random_forest_tuning.build_threshold_candidates(threshold_analysis)
    fixed_reference = ai4i_random_forest_tuning.load_fixed_random_forest_reference()
    recomputed_promotion = ai4i_random_forest_tuning.select_promotion_candidate(
        fixed_reference,
        recomputed_oof_metrics,
        recomputed_candidates,
    )
    results.append(
        result(
            "Nested OOF Metric Consistency",
            metrics.get("tuned_nested_oof_results") == recomputed_oof_metrics,
            "Nested OOF metrics match saved predictions.",
            "Nested OOF metrics do not match saved predictions.",
        )
    )
    results.append(
        result(
            "Threshold Candidate Consistency",
            metrics.get("threshold_candidates") == recomputed_candidates,
            "Threshold candidates match saved threshold analysis.",
            "Threshold candidates do not match saved threshold analysis.",
        )
    )
    results.append(
        result(
            "Promotion Policy",
            metrics.get("promotion_policy") == recomputed_promotion,
            "Promotion decision follows the documented train-only policy.",
            "Promotion decision does not follow the documented train-only policy.",
        )
    )
    selected_candidate = metrics.get("selected_candidate")
    if selected_candidate == "tuned_random_forest":
        expected_threshold = recomputed_candidates["max_f2"]["threshold"]
    else:
        expected_threshold = fixed_reference["max_f2_threshold"]["threshold"]
    results.append(
        result(
            "Selected Threshold Source",
            metrics.get("selected_threshold") == expected_threshold,
            "Selected threshold comes from the correct train-derived source.",
            "Selected threshold does not match the expected train-derived source.",
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
            "Validation predictions do not match the validation split.",
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
            "No Locked Test Metrics",
            not contains_locked_test_metrics(metrics),
            "No locked-test metrics are present in random_forest_tuning_metrics.json.",
            "Locked-test-like metric keys are present in random_forest_tuning_metrics.json.",
        )
    )

    matrix_failures: list[str] = []
    if (
        matrix_total(metrics["tuned_nested_oof_results"]["threshold_0_5"]["confusion_matrix"])
        != expected_train_rows
    ):
        matrix_failures.append("tuned nested_oof threshold_0_5")
    for candidate_name in ["max_f1", "max_f2"]:
        candidate = metrics["threshold_candidates"][candidate_name]
        total = (
            int(candidate["true_positive"])
            + int(candidate["false_positive"])
            + int(candidate["true_negative"])
            + int(candidate["false_negative"])
        )
        if total != expected_train_rows:
            matrix_failures.append(candidate_name)
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
    print("Industrial Fleet Intelligence Platform AI4I Random Forest tuning validation")
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
        print("Industrial Fleet Intelligence Platform AI4I Random Forest tuning validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Random Forest tuning artifact validation failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform AI4I Random Forest tuning validation")
        print("TEST SET STATUS: LOCKED / NOT USED")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

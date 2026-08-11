"""Read-only validator for AI4I final holdout evaluation artifacts."""

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

from ml.evaluation import ai4i_final_evaluation  # noqa: E402
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
        name=name,
        status=Status.PASS if passed else Status.FAIL,
        message=pass_message if passed else fail_message,
    )


def matrix_total(matrix: list[list[int]]) -> int:
    return sum(sum(int(value) for value in row) for row in matrix)


def contains_test_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"test_metrics", "final_test_metrics", "holdout_test_metrics"}:
                return True
            if key_text == "test_performance_included" and child is not False:
                return True
            if contains_test_metrics(child):
                return True
    if isinstance(value, list):
        return any(contains_test_metrics(item) for item in value)
    return False


def contains_adaptive_test_selection_field(value: Any) -> bool:
    adaptive_keys = {
        "test_selected_threshold",
        "test_best_threshold",
        "best_test_threshold",
        "threshold_selected_from_test",
        "test_tuned_parameters",
        "test_model_selection",
        "test_feature_selection",
        "selected_threshold_from_test",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in adaptive_keys:
                return True
            if contains_adaptive_test_selection_field(child):
                return True
    if isinstance(value, list):
        return any(contains_adaptive_test_selection_field(item) for item in value)
    return False


def final_evaluation_source_has_adaptive_search() -> bool:
    source = ai4i_final_evaluation.__file__
    if source is None:
        return True
    text = Path(source).read_text(encoding="utf-8")
    forbidden_tokens = [
        "GridSearchCV",
        "RandomizedSearchCV",
        "cross_val_score",
        "cross_validate",
        "StratifiedKFold",
        "KFold",
    ]
    return any(token in text for token in forbidden_tokens)


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    paths = {
        "frozen_config": ai4i_final_evaluation.final_config_path(PROJECT_ROOT),
        "final_decision": ai4i_final_evaluation.final_model_decision_path(PROJECT_ROOT),
        "final_predictions": ai4i_final_evaluation.final_test_predictions_path(PROJECT_ROOT),
        "final_metrics": ai4i_final_evaluation.final_test_metrics_path(PROJECT_ROOT),
        "markdown_report": ai4i_final_evaluation.final_doc_path(PROJECT_ROOT),
    }
    paths.update(ai4i_final_evaluation.final_plot_paths(PROJECT_ROOT))

    missing: list[str] = []
    for name, path in paths.items():
        exists = path.exists() and (path.stat().st_size > 0 if path.suffix == ".png" else True)
        if not exists:
            missing.append(name)
        results.append(
            result(
                f"{name} Artifact",
                exists,
                "Expected final evaluation artifact exists.",
                f"Expected artifact is missing or empty: {path.name}",
            )
        )
    if missing:
        return ValidationReport(results)

    try:
        modeling_config = ai4i_modeling.load_modeling_config(
            ai4i_modeling.config_path(PROJECT_ROOT)
        )
        split_summary = ai4i_final_evaluation.load_split_summary(PROJECT_ROOT)
        comparison_metrics = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.model_comparison_metrics_path(PROJECT_ROOT)
        )
        tuning_metrics = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.random_forest_tuning_metrics_path(PROJECT_ROOT)
        )
        final_config = ai4i_final_evaluation.load_final_model_config(paths["frozen_config"])
        decision = ai4i_final_evaluation.load_json(paths["final_decision"])
        metrics = ai4i_final_evaluation.load_json(paths["final_metrics"])
        predictions = pd.read_csv(paths["final_predictions"])
        train_df, validation_df = ai4i_baseline.load_training_and_validation_frames(PROJECT_ROOT)
        test_df = ai4i_final_evaluation.load_test_frame(PROJECT_ROOT)
    except (OSError, json.JSONDecodeError, ValueError, pd.errors.ParserError) as exc:
        results.append(CheckResult("Readable Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    try:
        ai4i_final_evaluation.validate_final_model_config(
            final_config,
            modeling_config,
            comparison_metrics,
            tuning_metrics,
        )
        config_valid = True
    except ValueError as exc:
        config_valid = False
        config_error = str(exc)
    else:
        config_error = ""

    config_hash = ai4i_final_evaluation.final_config_hash(final_config)
    feature_policy = ai4i_final_evaluation.feature_policy_from_config(modeling_config)
    expected_validation_reference = ai4i_final_evaluation.previous_validation_reference_metrics(
        tuning_metrics
    )
    expected_deltas = ai4i_final_evaluation.calculate_test_minus_validation_deltas(
        metrics["previous_validation_reference_metrics"],
        metrics["test_metrics"],
    )
    train_udis = set(train_df[modeling_config.derived_traceability_field].tolist())
    validation_udis = set(validation_df[modeling_config.derived_traceability_field].tolist())
    test_udis = set(test_df[modeling_config.derived_traceability_field].tolist())
    prediction_udis = (
        set(predictions["source_udi"].tolist()) if "source_udi" in predictions else set()
    )

    results.extend(
        [
            result(
                "Frozen Config Validity",
                config_valid,
                "Frozen final model config matches development decisions.",
                config_error,
            ),
            result(
                "Final Config Hash In Metrics",
                metrics.get("final_model_configuration_hash") == config_hash,
                "Metrics artifact records the frozen final config hash.",
                "Metrics artifact config hash does not match the frozen config.",
            ),
            result(
                "Final Config Hash In Decision",
                decision.get("final_decision", {}).get("configuration_hash") == config_hash,
                "Decision artifact records the frozen final config hash.",
                "Decision artifact config hash does not match the frozen config.",
            ),
            result(
                "Frozen Threshold",
                final_config.get("decision_threshold")
                == ai4i_final_evaluation.FROZEN_DECISION_THRESHOLD,
                "Frozen threshold is exactly 0.14.",
                "Frozen threshold is not 0.14.",
            ),
            result(
                "Tuning Promotion Result",
                tuning_metrics["promotion_policy"]["selected_candidate"] == "fixed_random_forest",
                "Tuning promotion result retained fixed_random_forest.",
                "Tuning promotion result did not retain fixed_random_forest.",
            ),
            result(
                "Prediction Schema",
                list(predictions.columns) == ai4i_final_evaluation.EXPECTED_PREDICTION_COLUMNS,
                "Final prediction schema is expected.",
                "Final prediction schema is not expected.",
            ),
            result(
                "Prediction Test Coverage",
                len(predictions) == len(test_df)
                and prediction_udis == test_udis
                and len(predictions) == ai4i_baseline.expected_split_rows(split_summary, "test"),
                "Final predictions contain exactly the authoritative test observations.",
                "Final predictions do not match the authoritative test split.",
            ),
            result(
                "Prediction source_udi Uniqueness",
                predictions["source_udi"].is_unique,
                "Final prediction source_udi values are unique.",
                "Final prediction source_udi values contain duplicates.",
            ),
            result(
                "No Train Or Validation source_udi In Predictions",
                prediction_udis.isdisjoint(train_udis)
                and prediction_udis.isdisjoint(validation_udis),
                "Final predictions contain no train or validation source_udi values.",
                "Final predictions overlap train or validation source_udi values.",
            ),
            result(
                "Probability Bounds",
                predictions["probability"].between(0, 1, inclusive="both").all(),
                "Final prediction probabilities are within [0, 1].",
                "Final prediction probabilities contain values outside [0, 1].",
            ),
            result(
                "Target Binary",
                set(predictions["target"].unique().tolist()).issubset({0, 1}),
                "Final prediction targets are binary.",
                "Final prediction targets contain non-binary values.",
            ),
            result(
                "Prediction Values Binary",
                set(predictions["prediction_threshold_0_5"].unique().tolist()).issubset({0, 1})
                and set(predictions["prediction_threshold_0_14"].unique().tolist()).issubset(
                    {0, 1}
                ),
                "Final prediction columns are binary.",
                "Final prediction columns contain non-binary values.",
            ),
        ]
    )

    expected_metric_sections = {"threshold_independent", "threshold_0_5", "threshold_0_14"}
    test_metrics = metrics.get("test_metrics", {})
    results.append(
        result(
            "Metric Sections",
            set(test_metrics.keys()) == expected_metric_sections,
            (
                "Final metrics contain threshold-independent, threshold 0.5, "
                "and threshold 0.14 sections."
            ),
            "Final metrics do not contain the expected metric sections.",
        )
    )
    for section_name in ["threshold_0_5", "threshold_0_14"]:
        matrix = test_metrics.get(section_name, {}).get("confusion_matrix", [])
        results.append(
            result(
                f"{section_name} Confusion Matrix Total",
                matrix_total(matrix) == len(test_df),
                f"{section_name} confusion matrix totals match test row count.",
                f"{section_name} confusion matrix totals do not match test row count.",
            )
        )

    results.extend(
        [
            result(
                "Validation Reference Metrics",
                metrics.get("previous_validation_reference_metrics")
                == expected_validation_reference,
                "Previous validation reference metrics match tuning artifacts.",
                "Previous validation reference metrics do not match tuning artifacts.",
            ),
            result(
                "Test Minus Validation Deltas",
                metrics.get("test_minus_validation_deltas") == expected_deltas,
                "Test-minus-validation deltas are internally consistent.",
                "Test-minus-validation deltas are not internally consistent.",
            ),
            result(
                "Decision Artifact Excludes Test Metrics",
                not contains_test_metrics(decision),
                "Final decision artifact contains no final test metrics.",
                "Final decision artifact contains final test metrics.",
            ),
            result(
                "Feature Policy",
                final_config.get("predictive_features") == feature_policy["predictive_features"]
                and metrics.get("feature_policy") == feature_policy,
                "Final feature policy matches modeling config.",
                "Final feature policy does not match modeling config.",
            ),
            result(
                "Leakage-Sensitive Features Excluded",
                set(final_config.get("predictive_features", [])).isdisjoint(
                    set(modeling_config.excluded_leakage_sensitive_columns)
                ),
                "Leakage-sensitive fields are absent from predictive features.",
                "A leakage-sensitive field appears in predictive features.",
            ),
            result(
                "No Adaptive Test Selection Fields",
                not contains_adaptive_test_selection_field(metrics),
                "Metrics contain no adaptive test-selection fields.",
                "Metrics contain an adaptive test-selection field.",
            ),
            result(
                "Final Evaluation Source Guard",
                not final_evaluation_source_has_adaptive_search(),
                "Final evaluation module contains no CV/search selection code.",
                "Final evaluation module contains adaptive search or CV code.",
            ),
            result(
                "Test Decision Guard",
                metrics.get("no_model_decision_changed_using_test_data") is True,
                "Metrics explicitly state no model decision changed using test data.",
                "Metrics do not state that test data left model decisions unchanged.",
            ),
        ]
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

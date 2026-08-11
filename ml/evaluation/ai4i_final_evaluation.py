"""Final holdout evaluation for the frozen AI4I classifier."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

CONFIG_RELATIVE_PATH = Path("ml") / "config" / "ai4i_final_model.json"
TEST_RELATIVE_PATH = Path("data") / "processed" / "ai4i" / "test.csv"
REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
FINAL_ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "final_evaluation"
FINAL_DOC_RELATIVE_PATH = Path("docs") / "ml" / "ai4i_final_evaluation.md"
MODEL_COMPARISON_METRICS_FILENAME = "model_comparison_metrics.json"
RANDOM_FOREST_TUNING_METRICS_FILENAME = "random_forest_tuning_metrics.json"
BASELINE_METRICS_FILENAME = "baseline_metrics.json"
IMBALANCE_METRICS_FILENAME = "imbalance_strategy_metrics.json"
FINAL_TEST_PREDICTIONS_FILENAME = "final_test_predictions.csv"
FINAL_TEST_METRICS_FILENAME = "final_test_metrics.json"
FINAL_MODEL_DECISION_FILENAME = "final_model_decision.json"
FROZEN_MODEL_FAMILY = "RandomForestClassifier"
FROZEN_DECISION_THRESHOLD = 0.14
REFERENCE_THRESHOLD = 0.5
RANDOM_SEED = 42
POSITIVE_CLASS = 1
FROZEN_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
EXPECTED_PREDICTION_COLUMNS = [
    "source_udi",
    "target",
    "probability",
    "prediction_threshold_0_5",
    "prediction_threshold_0_14",
]
TEST_UNLOCK_MESSAGE = "TEST SET STATUS: UNLOCKED FOR FINAL HOLDOUT EVALUATION"
FROZEN_SPEC_MESSAGE = "FINAL MODEL SPECIFICATION WAS FROZEN BEFORE TEST EVALUATION"


@dataclass(frozen=True)
class FinalEvaluationArtifacts:
    """Paths produced by the final holdout evaluation."""

    predictions_csv: Path
    metrics_json: Path
    decision_json: Path
    markdown_report: Path
    plot_paths: list[Path]


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Complete final holdout evaluation result."""

    final_config: dict[str, Any]
    final_config_hash: str
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    decision: dict[str, Any]
    artifacts: FinalEvaluationArtifacts


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def final_config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def test_path(root: Path | None = None) -> Path:
    return (root or project_root()) / TEST_RELATIVE_PATH


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def final_assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / FINAL_ASSETS_RELATIVE_DIR


def final_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / FINAL_DOC_RELATIVE_PATH


def model_comparison_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / MODEL_COMPARISON_METRICS_FILENAME


def random_forest_tuning_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / RANDOM_FOREST_TUNING_METRICS_FILENAME


def baseline_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / BASELINE_METRICS_FILENAME


def imbalance_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / IMBALANCE_METRICS_FILENAME


def final_test_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / FINAL_TEST_PREDICTIONS_FILENAME


def final_test_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / FINAL_TEST_METRICS_FILENAME


def final_model_decision_path(root: Path | None = None) -> Path:
    return reports_directory(root) / FINAL_MODEL_DECISION_FILENAME


def final_plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets = final_assets_directory(root)
    return {
        "test_confusion_matrix_threshold_0_14": assets / "test_confusion_matrix_threshold_0_14.png",
        "test_precision_recall_curve": assets / "test_precision_recall_curve.png",
        "test_roc_curve": assets / "test_roc_curve.png",
        "validation_vs_test_metrics": assets / "validation_vs_test_metrics.png",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}.")
    return data


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _rounded_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def final_config_hash(final_config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(final_config)).hexdigest()


def load_final_model_config(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or final_config_path())


def serialized_hyperparameters(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": "None" if params["max_depth"] is None else int(params["max_depth"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": str(params["max_features"]),
        "class_weight": str(params["class_weight"]),
        "random_state": int(params["random_state"]),
        "n_jobs": int(params["n_jobs"]),
    }


def feature_policy_from_config(config: ai4i_modeling.ModelingConfig) -> dict[str, Any]:
    return {
        "predictive_features": ai4i_baseline.predictive_feature_columns(config),
        "categorical_features": list(config.categorical_features),
        "numerical_features": list(config.numerical_features),
        "traceability_field": config.derived_traceability_field,
        "target": config.target_column,
        "excluded_identifier_fields": list(config.excluded_identifiers),
        "excluded_leakage_sensitive_fields": list(config.excluded_leakage_sensitive_columns),
    }


def preprocessing_policy() -> dict[str, str]:
    return {
        "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
        "numerical": "Five numerical features -> passthrough",
        "fit_policy": "Fit inside the pipeline on train + validation only; never fit on test.",
    }


def expected_final_model_config(
    modeling_config: ai4i_modeling.ModelingConfig,
    comparison_metrics: Mapping[str, Any],
    tuning_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_policy = comparison_metrics["candidate_selection_policy"]
    promotion_policy = tuning_metrics["promotion_policy"]
    ap_delta = promotion_policy["average_precision_delta"]
    required_delta = promotion_policy["promotion_delta_required"]
    return {
        "model_family": FROZEN_MODEL_FAMILY,
        "hyperparameters": dict(FROZEN_HYPERPARAMETERS),
        "decision_threshold": FROZEN_DECISION_THRESHOLD,
        "random_seed": RANDOM_SEED,
        **feature_policy_from_config(modeling_config),
        "preprocessing_policy": preprocessing_policy(),
        "training_data_policy": "train + validation",
        "final_evaluation_data_policy": "test only",
        "retention_reason": (
            "Fixed Random Forest was retained because tuned nested-OOF Average Precision "
            f"improved by {ap_delta} versus the fixed RF OOF reference, below the "
            f"predefined promotion minimum of {required_delta}."
        ),
        "selection_chain_summary": {
            "model_comparison_selected_model": comparison_policy["selected_model"],
            "model_comparison_threshold": comparison_policy["selected_threshold"],
            "tuning_selected_candidate": promotion_policy["selected_candidate"],
            "tuning_selected_threshold": promotion_policy["selected_threshold"],
            "validation_may_change_selection": False,
        },
        "source_development_artifacts": [
            MODEL_COMPARISON_METRICS_FILENAME,
            RANDOM_FOREST_TUNING_METRICS_FILENAME,
        ],
    }


def _contains_disallowed_config_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"timestamp", "created_at", "updated_at", "username", "user"}:
                return True
            if key_text in {"test_metrics", "test_performance", "test_results"}:
                return True
            if _contains_disallowed_config_value(child):
                return True
    elif isinstance(value, list):
        return any(_contains_disallowed_config_value(item) for item in value)
    elif isinstance(value, str):
        normalized = value.replace("/", "\\")
        return ":\\" in normalized or "\\users\\" in normalized.lower()
    return False


def validate_final_model_config(
    final_config: Mapping[str, Any],
    modeling_config: ai4i_modeling.ModelingConfig,
    comparison_metrics: Mapping[str, Any],
    tuning_metrics: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    expected_features = feature_policy_from_config(modeling_config)

    if final_config.get("model_family") != FROZEN_MODEL_FAMILY:
        errors.append("Final model family must be RandomForestClassifier.")
    if final_config.get("hyperparameters") != FROZEN_HYPERPARAMETERS:
        errors.append("Final Random Forest hyperparameters do not match the frozen policy.")
    if final_config.get("decision_threshold") != FROZEN_DECISION_THRESHOLD:
        errors.append("Final decision threshold must be 0.14.")
    if final_config.get("random_seed") != RANDOM_SEED:
        errors.append("Final random seed must be 42.")

    for key, expected in expected_features.items():
        if final_config.get(key) != expected:
            errors.append(f"Final feature policy field `{key}` does not match modeling config.")
    if final_config.get("preprocessing_policy") != preprocessing_policy():
        errors.append("Final preprocessing policy does not match the Random Forest policy.")
    if final_config.get("training_data_policy") != "train + validation":
        errors.append("Final training data policy must be train + validation.")
    if final_config.get("final_evaluation_data_policy") != "test only":
        errors.append("Final evaluation data policy must be test only.")

    comparison_policy = comparison_metrics.get("candidate_selection_policy", {})
    if comparison_policy.get("selected_model") != "random_forest":
        errors.append("Model comparison artifact must select random_forest.")
    if comparison_policy.get("selected_threshold") != FROZEN_DECISION_THRESHOLD:
        errors.append("Model comparison artifact must select threshold 0.14.")
    if comparison_policy.get("validation_may_change_selection") is not False:
        errors.append("Model comparison selection must not depend on validation.")

    promotion_policy = tuning_metrics.get("promotion_policy", {})
    if promotion_policy.get("selected_candidate") != "fixed_random_forest":
        errors.append("RF tuning promotion policy must retain fixed_random_forest.")
    if promotion_policy.get("selected_threshold") != FROZEN_DECISION_THRESHOLD:
        errors.append("RF tuning promotion policy must retain threshold 0.14.")
    if promotion_policy.get("validation_may_change_selection") is not False:
        errors.append("RF tuning selection must not depend on validation.")
    if float(promotion_policy.get("average_precision_delta", 1.0)) >= float(
        promotion_policy.get("promotion_delta_required", 0.0)
    ):
        errors.append("RF tuning AP delta no longer supports retaining fixed Random Forest.")

    forbidden_features = set(ai4i_baseline.forbidden_model_feature_columns(modeling_config))
    predictive_features = set(final_config.get("predictive_features", []))
    forbidden_used = sorted(predictive_features & forbidden_features)
    if forbidden_used:
        errors.append(
            "Forbidden predictive feature(s) in final config: " + ", ".join(forbidden_used)
        )
    if _contains_disallowed_config_value(final_config):
        errors.append(
            "Final config must not contain test metrics, timestamps, users, or absolute paths."
        )

    if errors:
        raise ValueError("Invalid frozen AI4I final model configuration: " + " ".join(errors))


def build_tree_preprocessor(config: ai4i_modeling.ModelingConfig) -> ColumnTransformer:
    ai4i_baseline.validate_feature_policy(config)
    categorical_pipeline = Pipeline(
        steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(config.categorical_features)),
            ("numerical", "passthrough", list(config.numerical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_frozen_random_forest_pipeline(
    config: ai4i_modeling.ModelingConfig,
    final_config: Mapping[str, Any] | None = None,
) -> Pipeline:
    params = dict((final_config or {"hyperparameters": FROZEN_HYPERPARAMETERS})["hyperparameters"])
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor(config)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=params["max_depth"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=str(params["max_features"]),
                    class_weight=str(params["class_weight"]),
                    random_state=int(params["random_state"]),
                    n_jobs=int(params["n_jobs"]),
                ),
            ),
        ]
    )


def load_test_frame(root: Path | None = None) -> pd.DataFrame:
    return ai4i_baseline.load_split_frame(test_path(root), "test")


def load_split_summary(root: Path | None = None) -> dict[str, Any]:
    return load_json(ai4i_modeling.split_summary_path(root))


def validate_split_separation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> None:
    source_column = config.derived_traceability_field
    train_udis = set(train_df[source_column].tolist())
    validation_udis = set(validation_df[source_column].tolist())
    test_udis = set(test_df[source_column].tolist())
    if train_udis & validation_udis:
        raise ValueError("Train and validation source_udi values must not overlap.")
    if train_udis & test_udis:
        raise ValueError("Train and test source_udi values must not overlap.")
    if validation_udis & test_udis:
        raise ValueError("Validation and test source_udi values must not overlap.")


def validate_final_evaluation_inputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
) -> None:
    ai4i_baseline.validate_training_inputs(train_df, validation_df, config, split_summary)
    ai4i_baseline.validate_split_frame(
        test_df,
        config,
        "test",
        ai4i_baseline.expected_split_rows(split_summary, "test"),
    )
    validate_split_separation(train_df, validation_df, test_df, config)


def combine_development_training_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> pd.DataFrame:
    validate_split_separation(train_df, validation_df, validation_df.iloc[0:0].copy(), config)
    combined = pd.concat([train_df, validation_df], ignore_index=True)
    if combined[config.derived_traceability_field].duplicated().any():
        raise ValueError("Combined development data contains duplicated source_udi values.")
    return combined.sort_values(config.derived_traceability_field, kind="mergesort").reset_index(
        drop=True
    )


def fit_final_pipeline(
    development_df: pd.DataFrame,
    modeling_config: ai4i_modeling.ModelingConfig,
    final_config: Mapping[str, Any],
) -> Pipeline:
    features, target = ai4i_baseline.extract_features_and_target(development_df, modeling_config)
    pipeline = build_frozen_random_forest_pipeline(modeling_config, final_config)
    pipeline.fit(features, target.astype(int))
    return pipeline


def threshold_predictions(probabilities: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) >= float(threshold)).astype(int)


def classification_metrics_at_threshold(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(target).astype(int)
    y_probability = np.asarray(probabilities, dtype=float)
    y_prediction = threshold_predictions(y_probability, threshold)
    matrix = confusion_matrix(y_true, y_prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = [
        int(value) for value in matrix.ravel()
    ]
    return {
        "threshold": _rounded_float(threshold),
        "accuracy": _rounded_float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": _rounded_float(balanced_accuracy_score(y_true, y_prediction)),
        "precision": _rounded_float(
            precision_score(y_true, y_prediction, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "recall": _rounded_float(
            recall_score(y_true, y_prediction, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "f1": _rounded_float(
            f1_score(y_true, y_prediction, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "f2": _rounded_float(
            fbeta_score(y_true, y_prediction, beta=2, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "predicted_positive_count": int(y_prediction.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def ranking_metrics(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, float | None]:
    y_true = np.asarray(target).astype(int)
    y_probability = np.asarray(probabilities, dtype=float)
    return {
        "average_precision": _rounded_float(average_precision_score(y_true, y_probability)),
        "roc_auc": _rounded_float(roc_auc_score(y_true, y_probability)),
    }


def evaluate_frozen_pipeline(
    pipeline: Pipeline,
    frozen_threshold: float,
    test_features: pd.DataFrame,
    test_target: pd.Series | np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = np.asarray(
        pipeline.predict_proba(test_features)[:, POSITIVE_CLASS], dtype=float
    )
    metrics = {
        "threshold_independent": ranking_metrics(test_target, probabilities),
        "threshold_0_5": classification_metrics_at_threshold(
            test_target, probabilities, REFERENCE_THRESHOLD
        ),
        "threshold_0_14": classification_metrics_at_threshold(
            test_target, probabilities, frozen_threshold
        ),
    }
    return metrics, probabilities


def build_prediction_report(
    test_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    probabilities: Sequence[float] | np.ndarray,
    frozen_threshold: float,
) -> pd.DataFrame:
    y_probability = np.asarray(probabilities, dtype=float)
    report = pd.DataFrame(
        {
            "source_udi": test_df[config.derived_traceability_field].astype(int).to_numpy(),
            "target": test_df[config.target_column].astype(int).to_numpy(),
            "probability": y_probability,
            "prediction_threshold_0_5": threshold_predictions(
                y_probability, REFERENCE_THRESHOLD
            ).astype(int),
            "prediction_threshold_0_14": threshold_predictions(
                y_probability, frozen_threshold
            ).astype(int),
        },
        columns=EXPECTED_PREDICTION_COLUMNS,
    )
    return report.sort_values("source_udi", kind="mergesort").reset_index(drop=True)


def previous_validation_reference_metrics(tuning_metrics: Mapping[str, Any]) -> dict[str, Any]:
    validation = tuning_metrics["validation_results"]
    return {
        "threshold_independent": dict(validation["threshold_independent"]),
        "threshold_0_5": dict(validation["threshold_0_5"]),
        "threshold_0_14": dict(validation["selected_threshold"]),
        "selection_uses_validation": bool(validation.get("selection_uses_validation", False)),
    }


def calculate_test_minus_validation_deltas(
    validation_reference: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    pairs = {
        "average_precision": ("threshold_independent", "average_precision"),
        "roc_auc": ("threshold_independent", "roc_auc"),
        "precision_at_0_14": ("threshold_0_14", "precision"),
        "recall_at_0_14": ("threshold_0_14", "recall"),
        "f1_at_0_14": ("threshold_0_14", "f1"),
        "f2_at_0_14": ("threshold_0_14", "f2"),
    }
    deltas: dict[str, float | None] = {}
    for name, (section, metric_name) in pairs.items():
        validation_value = validation_reference[section][metric_name]
        test_value = test_metrics[section][metric_name]
        if validation_value is None or test_value is None:
            deltas[name] = None
        else:
            deltas[name] = _rounded_float(float(test_value) - float(validation_value))
    return deltas


def build_final_metrics_summary(
    final_config: Mapping[str, Any],
    config_hash: str,
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    validation_reference: Mapping[str, Any],
    deltas: Mapping[str, float | None],
    modeling_config: ai4i_modeling.ModelingConfig,
) -> dict[str, Any]:
    return {
        "final_model_configuration_hash": config_hash,
        "model_family": final_config["model_family"],
        "hyperparameters": dict(final_config["hyperparameters"]),
        "feature_policy": feature_policy_from_config(modeling_config),
        "preprocessing_policy": preprocessing_policy(),
        "training_data_policy": final_config["training_data_policy"],
        "development_training_row_count": int(len(development_df)),
        "development_positive_count": int(development_df[modeling_config.target_column].sum()),
        "test_row_count": int(len(test_df)),
        "test_positive_count": int(test_df[modeling_config.target_column].sum()),
        "frozen_threshold": final_config["decision_threshold"],
        "test_metrics": dict(test_metrics),
        "previous_validation_reference_metrics": dict(validation_reference),
        "test_minus_validation_deltas": dict(deltas),
        "no_model_decision_changed_using_test_data": True,
        "test_set_usage_policy": {
            "allowed": [
                "final predictions",
                "final metrics",
                "final descriptive evaluation plots",
            ],
            "not_allowed": [
                "model selection",
                "hyperparameter tuning",
                "preprocessing fitting",
                "threshold selection",
                "feature engineering",
                "feature selection",
            ],
        },
    }


def _load_optional_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def build_final_model_decision(
    final_config: Mapping[str, Any],
    config_hash: str,
    baseline_metrics: Mapping[str, Any],
    imbalance_metrics: Mapping[str, Any],
    comparison_metrics: Mapping[str, Any],
    tuning_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_policy = comparison_metrics["candidate_selection_policy"]
    promotion_policy = tuning_metrics["promotion_policy"]
    fixed_reference = tuning_metrics["fixed_random_forest_reference"]
    tuned_nested = tuning_metrics["tuned_nested_oof_results"]
    baseline_logistic = baseline_metrics.get("logistic_regression", {})
    imbalance_policy = imbalance_metrics.get("candidate_selection_policy", {})
    comparison_ap = comparison_policy["model_average_precision"]
    return {
        "purpose": (
            "Evidence that the final model and operating threshold were chosen before final "
            "holdout test evaluation."
        ),
        "selection_chain": [
            {
                "stage": "Logistic Regression baseline",
                "source_artifact": BASELINE_METRICS_FILENAME,
                "summary": (
                    "Standard Logistic Regression established the first validation baseline."
                ),
                "average_precision": baseline_logistic.get("average_precision"),
                "roc_auc": baseline_logistic.get("roc_auc"),
            },
            {
                "stage": "Logistic imbalance and threshold experiment",
                "source_artifact": IMBALANCE_METRICS_FILENAME,
                "selected_model": imbalance_policy.get("selected_model"),
                "selected_threshold": imbalance_policy.get("selected_threshold"),
                "threshold_source": imbalance_policy.get("threshold_source"),
                "selection_uses_validation": bool(
                    imbalance_policy.get("validation_may_change_selection", True)
                ),
            },
            {
                "stage": "Non-linear comparison",
                "source_artifact": MODEL_COMPARISON_METRICS_FILENAME,
                "candidates": ["standard_logistic", "random_forest", "xgboost"],
                "primary_metric": comparison_policy["primary_metric"],
            },
            {
                "stage": "Random Forest vs XGBoost AP comparison",
                "source_artifact": MODEL_COMPARISON_METRICS_FILENAME,
                "random_forest_average_precision": comparison_ap["random_forest"],
                "xgboost_average_precision": comparison_ap["xgboost"],
                "ap_tie_tolerance": comparison_policy["ap_tie_tolerance"],
            },
            {
                "stage": "Simplicity tie-break decision",
                "source_artifact": MODEL_COMPARISON_METRICS_FILENAME,
                "selected_model": comparison_policy["selected_model"],
                "selected_threshold": comparison_policy["selected_threshold"],
                "reason": comparison_policy["reason"],
                "selection_uses_validation": bool(
                    comparison_policy["validation_may_change_selection"]
                ),
            },
            {
                "stage": "Targeted Random Forest tuning",
                "source_artifact": RANDOM_FOREST_TUNING_METRICS_FILENAME,
                "fixed_random_forest_average_precision": fixed_reference["average_precision"],
                "tuned_nested_average_precision": tuned_nested["average_precision"],
                "promotion_delta_required": promotion_policy["promotion_delta_required"],
                "average_precision_delta": promotion_policy["average_precision_delta"],
            },
            {
                "stage": "Tuning promotion decision",
                "source_artifact": RANDOM_FOREST_TUNING_METRICS_FILENAME,
                "selected_candidate": promotion_policy["selected_candidate"],
                "selected_threshold": promotion_policy["selected_threshold"],
                "reason": promotion_policy["reason"],
                "selection_uses_validation": bool(
                    promotion_policy["validation_may_change_selection"]
                ),
            },
            {
                "stage": "Final fixed Random Forest configuration",
                "model_family": final_config["model_family"],
                "hyperparameters": dict(final_config["hyperparameters"]),
            },
            {
                "stage": "Frozen threshold",
                "decision_threshold": final_config["decision_threshold"],
                "threshold_source": "train-only OOF development predictions",
            },
            {
                "stage": "Final configuration hash",
                "final_model_configuration_hash": config_hash,
            },
        ],
        "final_decision": {
            "model_family": final_config["model_family"],
            "selected_candidate": promotion_policy["selected_candidate"],
            "decision_threshold": final_config["decision_threshold"],
            "configuration_hash": config_hash,
            "test_performance_included": False,
        },
    }


def plot_confusion_matrix(matrix: Sequence[Sequence[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title("Test Confusion Matrix at Threshold 0.14")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["No failure", "Failure"])
    ax.set_yticks([0, 1], labels=["No failure", "Failure"])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, str(values[row, column]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(target).astype(int)
    y_probability = np.asarray(probabilities, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(recall, precision, color="#1f77b4", linewidth=2)
    ax.axhline(prevalence, color="#8c564b", linestyle="--", linewidth=1.5, label="Prevalence")
    ax.set_title("Final Test Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(target).astype(int)
    y_probability = np.asarray(probabilities, dtype=float)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(false_positive_rate, true_positive_rate, color="#2ca02c", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax.set_title("Final Test ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_validation_vs_test_metrics(
    validation_reference: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["precision", "recall", "f1", "f2"]
    validation_values = [validation_reference["threshold_0_14"][metric] for metric in metric_names]
    test_values = [test_metrics["threshold_0_14"][metric] for metric in metric_names]
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(positions - width / 2, validation_values, width, label="Validation", color="#4e79a7")
    ax.bar(positions + width / 2, test_values, width, label="Test", color="#f28e2b")
    ax.set_title("Validation vs Test at Frozen Threshold 0.14")
    ax.set_ylabel("Metric value")
    ax.set_xticks(positions, labels=["Precision", "Recall", "F1", "F2"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_final_plots(
    metrics: Mapping[str, Any],
    validation_reference: Mapping[str, Any],
    target: pd.Series,
    probabilities: np.ndarray,
    root: Path | None = None,
) -> list[Path]:
    paths = final_plot_paths(root)
    plot_confusion_matrix(
        metrics["threshold_0_14"]["confusion_matrix"],
        paths["test_confusion_matrix_threshold_0_14"],
    )
    plot_precision_recall_curve(target, probabilities, paths["test_precision_recall_curve"])
    plot_roc_curve(target, probabilities, paths["test_roc_curve"])
    plot_validation_vs_test_metrics(
        validation_reference,
        metrics,
        paths["validation_vs_test_metrics"],
    )
    return list(paths.values())


def render_markdown_report(metrics: Mapping[str, Any]) -> str:
    threshold_05 = metrics["test_metrics"]["threshold_0_5"]
    threshold_014 = metrics["test_metrics"]["threshold_0_14"]
    ranking = metrics["test_metrics"]["threshold_independent"]
    deltas = metrics["test_minus_validation_deltas"]
    matrix = threshold_014["confusion_matrix"]
    return "\n".join(
        [
            "# AI4I Final Holdout Evaluation",
            "",
            "## Development Process",
            "The leakage-safe AI4I development process completed baseline modeling, imbalance "
            "analysis, train-only threshold development, non-linear model-family comparison, and "
            "targeted Random Forest tuning before the test split was opened.",
            "",
            "## Frozen Model Decision",
            "The final specification was frozen before final holdout evaluation. The selected "
            "model remains the fixed Random Forest retained by the predefined tuning promotion "
            "policy.",
            "",
            "## Final Random Forest",
            "The final classifier is `RandomForestClassifier` with `n_estimators=300`, "
            '`max_depth=None`, `min_samples_leaf=1`, `max_features="sqrt"`, '
            '`class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=1`.',
            "",
            "## Frozen Decision Threshold",
            "The frozen operating threshold is 0.14. It was selected from train-only OOF "
            "development predictions and was not selected from the test split.",
            "",
            "## Train + Validation Refit",
            f"The frozen pipeline was fitted once on {metrics['development_training_row_count']} "
            f"combined train + validation rows with {metrics['development_positive_count']} "
            "positive labels. The test split was not used for preprocessing fitting.",
            "",
            "## Locked Test Protocol",
            f"The test split was opened only for final evaluation, containing "
            f"{metrics['test_row_count']} rows with {metrics['test_positive_count']} positive "
            "labels. No model choice, feature policy, hyperparameter, preprocessing step, or "
            "threshold was changed after viewing test results.",
            "",
            "## Final Test Metrics",
            f"Average Precision: {ranking['average_precision']}. ROC-AUC: {ranking['roc_auc']}.",
            "",
            "## Confusion Matrix",
            f"At threshold 0.14, the confusion matrix is `{matrix}` with precision "
            f"{threshold_014['precision']}, recall {threshold_014['recall']}, F1 "
            f"{threshold_014['f1']}, and F2 {threshold_014['f2']}.",
            "",
            "## Precision / Recall Trade-off",
            f"At threshold 0.5, precision is {threshold_05['precision']} and recall is "
            f"{threshold_05['recall']}. At the frozen threshold 0.14, precision is "
            f"{threshold_014['precision']} and recall is {threshold_014['recall']}. Threshold "
            "0.5 is reported only as a reference.",
            "",
            "## Validation vs Test",
            "Previous validation metrics are loaded from tracked development artifacts. The "
            f"test-minus-validation deltas are AP {deltas['average_precision']}, ROC-AUC "
            f"{deltas['roc_auc']}, precision@0.14 {deltas['precision_at_0_14']}, "
            f"recall@0.14 {deltas['recall_at_0_14']}, F1@0.14 {deltas['f1_at_0_14']}, "
            f"and F2@0.14 {deltas['f2_at_0_14']}. These differences are descriptive holdout "
            "variation only.",
            "",
            "## Interpretation",
            "The final evaluation provides a reproducible holdout estimate for this public "
            "synthetic dataset. It does not establish production readiness or real-world "
            "industrial generalization.",
            "",
            "## Limitations",
            "AI4I is synthetic and public. The result should not be interpreted causally, and it "
            "does not represent any specific manufacturer, fleet, or industrial operating site.",
            "",
            "## Next Steps",
            "Later phases may add model persistence, MLflow tracking, SHAP explainability, local "
            "serving, and dashboard integration without revisiting the final holdout test split "
            "for model selection.",
        ]
    )


def write_markdown_report(markdown: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown + "\n", encoding="utf-8")


def run_final_evaluation(
    root: Path | None = None,
    on_after_frozen_fit: Callable[[], None] | None = None,
) -> FinalEvaluationResult:
    root_path = root or project_root()
    modeling_config = ai4i_modeling.load_modeling_config(ai4i_modeling.config_path(root_path))
    split_summary = load_split_summary(root_path)
    comparison_metrics = load_json(model_comparison_metrics_path(root_path))
    tuning_metrics = load_json(random_forest_tuning_metrics_path(root_path))
    baseline_metrics = _load_optional_json(baseline_metrics_path(root_path))
    imbalance_metrics = _load_optional_json(imbalance_metrics_path(root_path))
    final_config = load_final_model_config(final_config_path(root_path))
    validate_final_model_config(final_config, modeling_config, comparison_metrics, tuning_metrics)
    config_hash = final_config_hash(final_config)

    train_df, validation_df = ai4i_baseline.load_training_and_validation_frames(root_path)
    ai4i_baseline.validate_training_inputs(train_df, validation_df, modeling_config, split_summary)
    development_df = combine_development_training_data(train_df, validation_df, modeling_config)
    pipeline = fit_final_pipeline(development_df, modeling_config, final_config)
    if on_after_frozen_fit is not None:
        on_after_frozen_fit()

    test_df = load_test_frame(root_path)
    validate_final_evaluation_inputs(
        train_df,
        validation_df,
        test_df,
        modeling_config,
        split_summary,
    )
    test_features, test_target = ai4i_baseline.extract_features_and_target(test_df, modeling_config)
    test_metrics, probabilities = evaluate_frozen_pipeline(
        pipeline,
        float(final_config["decision_threshold"]),
        test_features,
        test_target,
    )
    predictions = build_prediction_report(
        test_df,
        modeling_config,
        probabilities,
        float(final_config["decision_threshold"]),
    )
    validation_reference = previous_validation_reference_metrics(tuning_metrics)
    deltas = calculate_test_minus_validation_deltas(validation_reference, test_metrics)
    metrics = build_final_metrics_summary(
        final_config,
        config_hash,
        development_df,
        test_df,
        test_metrics,
        validation_reference,
        deltas,
        modeling_config,
    )
    decision = build_final_model_decision(
        final_config,
        config_hash,
        baseline_metrics,
        imbalance_metrics,
        comparison_metrics,
        tuning_metrics,
    )

    predictions_path = final_test_predictions_path(root_path)
    metrics_path = final_test_metrics_path(root_path)
    decision_path = final_model_decision_path(root_path)
    plot_paths = write_final_plots(
        test_metrics, validation_reference, test_target, probabilities, root_path
    )
    doc_path = final_doc_path(root_path)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    write_json(metrics, metrics_path)
    write_json(decision, decision_path)
    write_markdown_report(render_markdown_report(metrics), doc_path)

    return FinalEvaluationResult(
        final_config=final_config,
        final_config_hash=config_hash,
        metrics=metrics,
        predictions=predictions,
        decision=decision,
        artifacts=FinalEvaluationArtifacts(
            predictions_csv=predictions_path,
            metrics_json=metrics_path,
            decision_json=decision_path,
            markdown_report=doc_path,
            plot_paths=plot_paths,
        ),
    )

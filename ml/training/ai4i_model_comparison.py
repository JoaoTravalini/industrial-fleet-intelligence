"""Train-only AI4I model-family comparison utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
COMPARISON_ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "model_comparison"
COMPARISON_DOC_RELATIVE_PATH = Path("docs") / "ml" / "ai4i_model_comparison.md"
OOF_PREDICTIONS_FILENAME = "model_comparison_train_oof_predictions.csv"
THRESHOLD_ANALYSIS_FILENAME = "model_comparison_threshold_analysis.csv"
COMPARISON_METRICS_FILENAME = "model_comparison_metrics.json"
VALIDATION_PREDICTIONS_FILENAME = "model_comparison_validation_predictions.csv"
MODEL_NAMES = ("standard_logistic", "random_forest", "xgboost")
MODEL_DISPLAY_NAMES = {
    "standard_logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
SIMPLICITY_ORDER = MODEL_NAMES
CV_SPLITS = 5
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = ai4i_baseline.LOGISTIC_MAX_ITER
RANDOM_FOREST_N_ESTIMATORS = 300
XGBOOST_N_ESTIMATORS = 300
XGBOOST_MAX_DEPTH = 4
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_SUBSAMPLE = 0.9
XGBOOST_COLSAMPLE_BYTREE = 0.9
THRESHOLDS = tuple(round(value / 100, 2) for value in range(1, 100))
RECALL_TARGET = 0.70
AP_TIE_TOLERANCE = 0.01
TEST_SET_STATUS = "LOCKED / NOT USED"
REQUIRED_THRESHOLD_COLUMNS = [
    "model",
    "threshold",
    "precision",
    "recall",
    "f1",
    "f2",
    "balanced_accuracy",
    "predicted_positive_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
]


@dataclass(frozen=True)
class ModelComparisonArtifacts:
    """Paths produced by a model-family comparison run."""

    oof_predictions_csv: Path
    threshold_analysis_csv: Path
    metrics_json: Path
    validation_predictions_csv: Path
    markdown_report: Path
    plot_paths: list[Path]


@dataclass(frozen=True)
class ModelComparisonResult:
    """Full result returned by a model-family comparison experiment."""

    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    threshold_analysis: pd.DataFrame
    validation_predictions: pd.DataFrame
    artifacts: ModelComparisonArtifacts


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def comparison_assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / COMPARISON_ASSETS_RELATIVE_DIR


def comparison_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / COMPARISON_DOC_RELATIVE_PATH


def oof_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / OOF_PREDICTIONS_FILENAME


def threshold_analysis_path(root: Path | None = None) -> Path:
    return reports_directory(root) / THRESHOLD_ANALYSIS_FILENAME


def comparison_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / COMPARISON_METRICS_FILENAME


def comparison_validation_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / VALIDATION_PREDICTIONS_FILENAME


def comparison_plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets = comparison_assets_directory(root)
    return {
        "train_oof_precision_recall_comparison": assets
        / "train_oof_precision_recall_comparison.png",
        "train_oof_roc_comparison": assets / "train_oof_roc_comparison.png",
        "train_oof_average_precision": assets / "train_oof_average_precision.png",
        "train_oof_f2_at_selected_threshold": assets / "train_oof_f2_at_selected_threshold.png",
        "validation_selected_model_confusion_matrix": assets
        / "validation_selected_model_confusion_matrix.png",
        "validation_selected_model_precision_recall": assets
        / "validation_selected_model_precision_recall.png",
    }


def _rounded_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def make_stratified_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)


def build_logistic_preprocessor(config: ai4i_modeling.ModelingConfig) -> ColumnTransformer:
    ai4i_baseline.validate_feature_policy(config)
    categorical_pipeline = Pipeline(
        steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    numerical_pipeline = Pipeline(steps=[("standard_scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(config.categorical_features)),
            ("numerical", numerical_pipeline, list(config.numerical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


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


def calculate_scale_pos_weight(target: pd.Series | np.ndarray) -> float:
    y = np.asarray(target).astype(int)
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("XGBoost scale_pos_weight requires both positive and negative labels.")
    return negative_count / positive_count


def build_standard_logistic_pipeline(config: ai4i_modeling.ModelingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_logistic_preprocessor(config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=RANDOM_SEED,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(config: ai4i_modeling.ModelingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor(config)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def xgboost_fixed_parameters(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "n_estimators": XGBOOST_N_ESTIMATORS,
        "max_depth": XGBOOST_MAX_DEPTH,
        "learning_rate": XGBOOST_LEARNING_RATE,
        "subsample": XGBOOST_SUBSAMPLE,
        "colsample_bytree": XGBOOST_COLSAMPLE_BYTREE,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "tree_method": "hist",
        "device": "cpu",
        "scale_pos_weight": float(scale_pos_weight),
    }


def build_xgboost_pipeline(
    config: ai4i_modeling.ModelingConfig,
    scale_pos_weight: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor(config)),
            ("classifier", XGBClassifier(**xgboost_fixed_parameters(scale_pos_weight))),
        ]
    )


def build_model_pipeline(
    config: ai4i_modeling.ModelingConfig,
    model_name: str,
    training_target: pd.Series | np.ndarray | None = None,
) -> Pipeline:
    if model_name == "standard_logistic":
        return build_standard_logistic_pipeline(config)
    if model_name == "random_forest":
        return build_random_forest_pipeline(config)
    if model_name == "xgboost":
        if training_target is None:
            raise ValueError("XGBoost pipeline construction requires training labels.")
        return build_xgboost_pipeline(config, calculate_scale_pos_weight(training_target))
    raise ValueError(f"Unknown model family: {model_name}")


def build_model_pipelines(
    config: ai4i_modeling.ModelingConfig,
    training_target: pd.Series | np.ndarray | None = None,
) -> dict[str, Pipeline]:
    return {
        model_name: build_model_pipeline(config, model_name, training_target)
        for model_name in MODEL_NAMES
    }


def make_shared_fold_assignments(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    cv: StratifiedKFold | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    splitter = cv or make_stratified_kfold()
    return [
        (np.asarray(train_index, dtype=int), np.asarray(holdout_index, dtype=int))
        for train_index, holdout_index in splitter.split(features, target)
    ]


def calculate_xgboost_fold_scale_pos_weights(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    _, target = ai4i_baseline.extract_features_and_target(train_df, config)
    return [
        _rounded_float(calculate_scale_pos_weight(target.iloc[train_index]))
        for train_index, _ in folds
    ]


def generate_oof_probabilities_for_model(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    model_name: str,
    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    fold_assignments = list(folds or make_shared_fold_assignments(train_df, config))
    probabilities = np.full(len(train_df), np.nan, dtype=float)

    for train_index, holdout_index in fold_assignments:
        fold_target = target.iloc[train_index]
        fold_pipeline = build_model_pipeline(config, model_name, fold_target)
        fold_pipeline.fit(features.iloc[train_index], fold_target)
        probabilities[holdout_index] = fold_pipeline.predict_proba(features.iloc[holdout_index])[
            :, ai4i_baseline.POSITIVE_CLASS
        ]

    if np.isnan(probabilities).any():
        raise ValueError(f"OOF probabilities were not generated for every row: {model_name}")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"OOF probabilities outside [0, 1] for model: {model_name}")
    return probabilities


def generate_oof_predictions(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    fold_assignments = list(folds or make_shared_fold_assignments(train_df, config))
    result = pd.DataFrame(
        {
            "source_udi": train_df[config.derived_traceability_field].astype(int),
            "target": train_df[config.target_column].astype(int),
        }
    )
    for model_name in MODEL_NAMES:
        result[f"{model_name}_probability"] = generate_oof_probabilities_for_model(
            train_df,
            config,
            model_name,
            fold_assignments,
        )
    return result.sort_values("source_udi", kind="mergesort").reset_index(drop=True)


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def classification_metrics_at_threshold(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(target).astype(int)
    y_probability = np.asarray(probabilities).astype(float)
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
            precision_score(y_true, y_prediction, pos_label=1, zero_division=0)
        ),
        "recall": _rounded_float(recall_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "f1": _rounded_float(f1_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "f2": _rounded_float(
            fbeta_score(y_true, y_prediction, beta=2, pos_label=1, zero_division=0)
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
    y_probability = np.asarray(probabilities).astype(float)
    return {
        "roc_auc": _rounded_float(roc_auc_score(y_true, y_probability)),
        "average_precision": _rounded_float(average_precision_score(y_true, y_probability)),
    }


def build_threshold_analysis(
    oof_predictions: pd.DataFrame,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = oof_predictions["target"]
    for model_name in MODEL_NAMES:
        probabilities = oof_predictions[f"{model_name}_probability"]
        for threshold in thresholds:
            metrics = classification_metrics_at_threshold(target, probabilities, threshold)
            rows.append(
                {
                    "model": model_name,
                    "threshold": metrics["threshold"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "f2": metrics["f2"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "predicted_positive_count": metrics["predicted_positive_count"],
                    "true_positive": metrics["true_positive"],
                    "false_positive": metrics["false_positive"],
                    "true_negative": metrics["true_negative"],
                    "false_negative": metrics["false_negative"],
                }
            )
    return pd.DataFrame(rows, columns=REQUIRED_THRESHOLD_COLUMNS).sort_values(
        ["model", "threshold"], kind="mergesort"
    )


def threshold_row_to_dict(row: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            output[key] = _rounded_float(value)
        else:
            output[key] = value
    return output


def select_threshold_by_metric(
    threshold_analysis: pd.DataFrame,
    model_name: str,
    metric: str,
) -> dict[str, Any]:
    subset = threshold_analysis[threshold_analysis["model"] == model_name]
    if subset.empty:
        raise ValueError(f"No threshold rows found for model: {model_name}")
    best = subset.sort_values(
        [metric, "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return threshold_row_to_dict(best)


def select_recall_candidate(
    threshold_analysis: pd.DataFrame,
    model_name: str,
    minimum_recall: float = RECALL_TARGET,
) -> dict[str, Any] | None:
    subset = threshold_analysis[
        (threshold_analysis["model"] == model_name)
        & (threshold_analysis["recall"] >= minimum_recall)
    ]
    if subset.empty:
        return None
    best = subset.sort_values(
        ["precision", "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return threshold_row_to_dict(best)


def build_threshold_candidates(threshold_analysis: pd.DataFrame) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        recall_candidate = select_recall_candidate(threshold_analysis, model_name)
        candidates[model_name] = {
            "max_f1": select_threshold_by_metric(threshold_analysis, model_name, "f1"),
            "max_f2": select_threshold_by_metric(threshold_analysis, model_name, "f2"),
            "recall_70": recall_candidate
            if recall_candidate is not None
            else {
                "available": False,
                "reason": "No threshold in the deterministic grid achieved recall >= 0.70.",
            },
        }
    return candidates


def build_oof_model_metrics(oof_predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    target = oof_predictions["target"]
    for model_name in MODEL_NAMES:
        probabilities = oof_predictions[f"{model_name}_probability"]
        metrics[model_name] = {
            **ranking_metrics(target, probabilities),
            "threshold_0_5": classification_metrics_at_threshold(target, probabilities, 0.5),
        }
    return metrics


def select_development_candidate(
    oof_metrics: Mapping[str, Mapping[str, Any]],
    threshold_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ap_scores = {
        model_name: float(oof_metrics[model_name]["average_precision"])
        for model_name in MODEL_NAMES
    }
    best_ap = max(ap_scores.values())
    best_models = [
        model_name
        for model_name in SIMPLICITY_ORDER
        if best_ap - ap_scores[model_name] < AP_TIE_TOLERANCE
    ]
    selected_model = best_models[0]
    selected_threshold = float(threshold_candidates[selected_model]["max_f2"]["threshold"])
    highest_ap_model = max(MODEL_NAMES, key=lambda name: ap_scores[name])

    if selected_model == highest_ap_model:
        reason = (
            f"{MODEL_DISPLAY_NAMES[selected_model]} has the highest train OOF "
            "Average Precision outside the simplicity tie tolerance."
        )
    else:
        reason = (
            f"{MODEL_DISPLAY_NAMES[selected_model]} is within 0.01 train OOF AP of the best "
            "candidate and is simpler under the predefined order."
        )

    return {
        "policy": [
            "Compare standard_logistic, random_forest, and xgboost by TRAIN OOF Average Precision.",
            "If the best model and another candidate differ in AP by less than 0.01, "
            "prefer the simpler model.",
            "The explicit simplicity order is standard_logistic, random_forest, xgboost.",
            "For the selected model, use its TRAIN OOF max-F2 threshold.",
        ],
        "primary_metric": "train_oof_average_precision",
        "simplicity_order": list(SIMPLICITY_ORDER),
        "ap_tie_tolerance": AP_TIE_TOLERANCE,
        "model_average_precision": {
            name: _rounded_float(score) for name, score in ap_scores.items()
        },
        "highest_ap_model": highest_ap_model,
        "models_within_tie_tolerance": best_models,
        "selected_model": selected_model,
        "selected_threshold": _rounded_float(selected_threshold),
        "reason": reason,
        "threshold_source": "train_oof_max_f2",
        "validation_may_change_selection": False,
    }


def fit_selected_pipeline(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    model_name: str,
) -> Pipeline:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    pipeline = build_model_pipeline(config, model_name, target)
    pipeline.fit(features, target)
    return pipeline


def predict_probabilities(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> np.ndarray:
    features, _ = ai4i_baseline.extract_features_and_target(frame, config)
    probabilities = pipeline.predict_proba(features)[:, ai4i_baseline.POSITIVE_CLASS]
    return np.asarray(probabilities, dtype=float)


def create_validation_predictions(
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> pd.DataFrame:
    target = validation_df[config.target_column].astype(int)
    return (
        pd.DataFrame(
            {
                "source_udi": validation_df[config.derived_traceability_field].astype(int),
                "target": target,
                "probability": probabilities.astype(float),
                "prediction_threshold_0_5": threshold_predictions(probabilities, 0.5).astype(int),
                "prediction_selected_threshold": threshold_predictions(
                    probabilities, selected_threshold
                ).astype(int),
            }
        )
        .sort_values("source_udi", kind="mergesort")
        .reset_index(drop=True)
    )


def evaluate_validation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    selected_model: str,
    selected_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    selected_pipeline = fit_selected_pipeline(train_df, config, selected_model)
    probabilities = predict_probabilities(selected_pipeline, validation_df, config)
    target = validation_df[config.target_column].astype(int)
    threshold_0_5 = classification_metrics_at_threshold(target, probabilities, 0.5)
    selected_threshold_metrics = classification_metrics_at_threshold(
        target, probabilities, selected_threshold
    )
    metrics = {
        "threshold_independent": ranking_metrics(target, probabilities),
        "threshold_0_5": threshold_0_5,
        "selected_threshold": selected_threshold_metrics,
    }
    predictions = create_validation_predictions(
        validation_df,
        config,
        probabilities,
        selected_threshold,
    )
    return metrics, predictions, probabilities


def model_configurations(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    _, target = ai4i_baseline.extract_features_and_target(train_df, config)
    xgb_fold_weights = calculate_xgboost_fold_scale_pos_weights(train_df, config, folds)
    return {
        "standard_logistic": {
            "family": "sklearn.linear_model.LogisticRegression",
            "max_iter": LOGISTIC_MAX_ITER,
            "class_weight": None,
            "random_state": RANDOM_SEED,
        },
        "random_forest": {
            "family": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        },
        "xgboost": {
            "family": "xgboost.XGBClassifier",
            "xgboost_version": xgboost.__version__,
            "n_estimators": XGBOOST_N_ESTIMATORS,
            "max_depth": XGBOOST_MAX_DEPTH,
            "learning_rate": XGBOOST_LEARNING_RATE,
            "subsample": XGBOOST_SUBSAMPLE,
            "colsample_bytree": XGBOOST_COLSAMPLE_BYTREE,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "tree_method": "hist",
            "device": "cpu",
            "scale_pos_weight_policy": (
                "Calculated from the labels available to each fitted model only."
            ),
            "oof_fold_scale_pos_weights": xgb_fold_weights,
            "full_train_scale_pos_weight_if_selected": _rounded_float(
                calculate_scale_pos_weight(target)
            ),
        },
    }


def build_metrics_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    oof_metrics: Mapping[str, Mapping[str, Any]],
    threshold_candidates: Mapping[str, Mapping[str, Any]],
    selected_candidate: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "AI4I non-linear model comparison",
            "objective": "Binary classification of Machine failure",
            "phase": "train-oof model-family comparison with validation-only evaluation",
            "test_set_status": TEST_SET_STATUS,
        },
        "data": {
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "training_positive_count": int(train_df[config.target_column].sum()),
            "validation_positive_count": int(validation_df[config.target_column].sum()),
            "validation_positive_percentage": _rounded_float(
                float(validation_df[config.target_column].mean() * 100)
            ),
            "splits_used": ["train", "validation"],
            "test_data_used": False,
        },
        "cv_configuration": {
            "method": "StratifiedKFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_SEED,
            "shared_folds_for_all_models": True,
            "probability_policy": (
                "Each training observation receives exactly one out-of-fold probability from "
                "a pipeline not fitted on that observation."
            ),
        },
        "feature_policy": {
            "predictive_feature_list": ai4i_baseline.predictive_feature_columns(config),
            "categorical_features": list(config.categorical_features),
            "numerical_features": list(config.numerical_features),
            "traceability_field": config.derived_traceability_field,
            "excluded_identifiers": list(config.excluded_identifiers),
            "excluded_leakage_sensitive_fields": list(config.excluded_leakage_sensitive_columns),
        },
        "preprocessing": {
            "fit_policy": (
                "Preprocessing is fitted inside each fold or final pipeline on its "
                "corresponding training rows only."
            ),
            "standard_logistic": {
                "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
                "numerical": "Five process variables -> StandardScaler()",
            },
            "random_forest": {
                "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
                "numerical": "Five process variables -> passthrough",
            },
            "xgboost": {
                "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
                "numerical": "Five process variables -> passthrough",
            },
        },
        "model_configurations": model_configurations(train_df, config, folds),
        "train_oof_results": {
            model_name: dict(oof_metrics[model_name]) for model_name in MODEL_NAMES
        },
        "threshold_grid": {
            "minimum": min(THRESHOLDS),
            "maximum": max(THRESHOLDS),
            "step": 0.01,
            "selection_uses_accuracy": False,
        },
        "threshold_candidates": dict(threshold_candidates),
        "candidate_selection_policy": dict(selected_candidate),
        "selected_model": selected_candidate["selected_model"],
        "selected_threshold": selected_candidate["selected_threshold"],
        "validation_results": {
            "threshold_independent": dict(validation_metrics["threshold_independent"]),
            "threshold_0_5": dict(validation_metrics["threshold_0_5"]),
            "selected_threshold": dict(validation_metrics["selected_threshold"]),
            "selection_uses_validation": False,
        },
        "f2_note": (
            "F2 gives recall more weight than precision, but it is exploratory and not "
            "a confirmed business cost function."
        ),
    }


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_train_oof_precision_recall_comparison(
    oof_predictions: pd.DataFrame,
    path: Path,
) -> None:
    target = oof_predictions["target"]
    _, axis = plt.subplots(figsize=(7, 5))
    for model_name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(
            target,
            oof_predictions[f"{model_name}_probability"],
        )
        axis.plot(recall, precision, label=MODEL_DISPLAY_NAMES[model_name])
    axis.axhline(float(target.mean()), linestyle="--", label="Positive prevalence")
    axis.set_title("AI4I Train OOF Precision-Recall Comparison")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def plot_train_oof_roc_comparison(oof_predictions: pd.DataFrame, path: Path) -> None:
    target = oof_predictions["target"]
    _, axis = plt.subplots(figsize=(7, 5))
    for model_name in MODEL_NAMES:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            target,
            oof_predictions[f"{model_name}_probability"],
        )
        axis.plot(false_positive_rate, true_positive_rate, label=MODEL_DISPLAY_NAMES[model_name])
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier reference")
    axis.set_title("AI4I Train OOF ROC Comparison")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    _save_figure(path)


def plot_train_oof_average_precision(
    oof_metrics: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    labels = [MODEL_DISPLAY_NAMES[name] for name in MODEL_NAMES]
    values = [float(oof_metrics[name]["average_precision"]) for name in MODEL_NAMES]
    _, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, values)
    axis.set_title("AI4I Train OOF Average Precision")
    axis.set_xlabel("Model family")
    axis.set_ylabel("Average Precision")
    axis.set_ylim(0, max(values) * 1.15 if values else 1)
    _save_figure(path)


def plot_train_oof_f2_at_selected_threshold(
    threshold_candidates: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    labels = [MODEL_DISPLAY_NAMES[name] for name in MODEL_NAMES]
    values = [float(threshold_candidates[name]["max_f2"]["f2"]) for name in MODEL_NAMES]
    _, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, values)
    axis.set_title("AI4I Train OOF F2 at Max-F2 Threshold")
    axis.set_xlabel("Model family")
    axis.set_ylabel("F2")
    axis.set_ylim(0, max(values) * 1.15 if values else 1)
    _save_figure(path)


def plot_validation_selected_model_confusion_matrix(
    matrix: list[list[int]],
    selected_model: str,
    selected_threshold: float,
    path: Path,
) -> None:
    values = np.array(matrix, dtype=int)
    _, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(values)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(
        f"{MODEL_DISPLAY_NAMES[selected_model]} Validation Confusion Matrix "
        f"({selected_threshold:.2f})"
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks([0, 1], labels=["No failure", "Failure"])
    axis.set_yticks([0, 1], labels=["No failure", "Failure"])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{labels[row, column]}\n{values[row, column]}",
                ha="center",
                va="center",
            )
    _save_figure(path)


def plot_validation_selected_model_precision_recall(
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    probabilities: np.ndarray,
    selected_model: str,
    path: Path,
) -> None:
    target = validation_df[config.target_column].astype(int)
    precision, recall, _ = precision_recall_curve(target, probabilities)
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label=MODEL_DISPLAY_NAMES[selected_model])
    axis.axhline(float(target.mean()), linestyle="--", label="Validation prevalence")
    axis.set_title("AI4I Validation Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def create_plots(
    oof_predictions: pd.DataFrame,
    metrics: Mapping[str, Any],
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    validation_probabilities: np.ndarray,
    root: Path | None = None,
) -> list[Path]:
    paths = comparison_plot_paths(root)
    selected_model = str(metrics["selected_model"])
    selected_threshold = float(metrics["selected_threshold"])
    plot_train_oof_precision_recall_comparison(
        oof_predictions,
        paths["train_oof_precision_recall_comparison"],
    )
    plot_train_oof_roc_comparison(oof_predictions, paths["train_oof_roc_comparison"])
    plot_train_oof_average_precision(
        metrics["train_oof_results"],
        paths["train_oof_average_precision"],
    )
    plot_train_oof_f2_at_selected_threshold(
        metrics["threshold_candidates"],
        paths["train_oof_f2_at_selected_threshold"],
    )
    plot_validation_selected_model_confusion_matrix(
        metrics["validation_results"]["selected_threshold"]["confusion_matrix"],
        selected_model,
        selected_threshold,
        paths["validation_selected_model_confusion_matrix"],
    )
    plot_validation_selected_model_precision_recall(
        validation_df,
        config,
        validation_probabilities,
        selected_model,
        paths["validation_selected_model_precision_recall"],
    )
    return [paths[name] for name in sorted(paths)]


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def render_markdown(metrics: Mapping[str, Any]) -> str:
    train_results = metrics["train_oof_results"]
    candidates = metrics["threshold_candidates"]
    selected_policy = metrics["candidate_selection_policy"]
    validation = metrics["validation_results"]
    table_lines = [
        "| Model | OOF AP | OOF ROC-AUC | Max-F2 threshold | Precision | Recall | F2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in MODEL_NAMES:
        max_f2 = candidates[model_name]["max_f2"]
        table_lines.append(
            f"| {MODEL_DISPLAY_NAMES[model_name]} | "
            f"{train_results[model_name]['average_precision']} | "
            f"{train_results[model_name]['roc_auc']} | "
            f"{max_f2['threshold']} | {max_f2['precision']} | "
            f"{max_f2['recall']} | {max_f2['f2']} |"
        )
    validation_05 = validation["threshold_0_5"]
    validation_selected = validation["selected_threshold"]
    selected_model = str(metrics["selected_model"])
    selected_threshold = metrics["selected_threshold"]
    return "\n".join(
        [
            "# AI4I Non-Linear Model Comparison",
            "",
            "## Objective",
            "This phase compares the existing standard Logistic Regression reference with fixed "
            "Random Forest and XGBoost baselines for binary classification of `Machine failure`.",
            "",
            "## Experimental Protocol",
            "The comparison uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` "
            "on the training split only. All three model families use identical fold assignments. "
            "Each out-of-fold probability comes from a pipeline whose preprocessing and model "
            "parameters were fitted without that held-out row.",
            "",
            "## Locked Test Set",
            "The locked test split was not read, counted, predicted, evaluated, or used for "
            "candidate selection. Final test evaluation is reserved for a later phase.",
            "",
            "## Feature Policy",
            "All models receive the same predictive information: `Type`, `Air temperature [K]`, "
            "`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and "
            "`Tool wear [min]`. `source_udi` is traceability-only. `Product ID`, `TWF`, "
            "`HDF`, `PWF`, `OSF`, and `RNF` are excluded.",
            "",
            "## Logistic Regression Reference",
            "The reference model keeps the same standard Logistic Regression configuration used in "
            "the imbalance phase: no class weighting, `max_iter=1000`, and training-fold-only "
            "standardization for numerical variables.",
            "",
            "## Random Forest",
            "The Random Forest baseline uses a fixed configuration with `n_estimators=300`, "
            '`class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=1`. No '
            "hyperparameter search or validation-based tuning is performed.",
            "",
            "## XGBoost",
            "The XGBoost baseline uses a conservative CPU-only fixed configuration with "
            "`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, "
            '`colsample_bytree=0.9`, `objective="binary:logistic"`, '
            '`eval_metric="logloss"`, `tree_method="hist"`, `device="cpu"`, '
            "`random_state=42`, and `n_jobs=1`. `scale_pos_weight` is calculated only from "
            "the labels available to each fitted training fold, or from full train only if "
            "XGBoost is the selected final validation candidate.",
            "",
            "## Train OOF Results",
            *table_lines,
            "",
            "## Threshold Strategy",
            "Thresholds from 0.01 through 0.99 are evaluated on train OOF probabilities only. "
            "Each model is associated with its own train-derived max-F2 threshold. F2 is used as "
            "an exploratory recall-weighted operating-point summary, not as a confirmed business "
            "cost function.",
            "",
            "## Model Selection Policy",
            "The predefined selection policy chooses the highest train OOF Average Precision. If "
            "another candidate is within 0.01 AP of the best candidate, the simpler model is "
            "preferred in this order: `standard_logistic`, `random_forest`, `xgboost`. "
            "Validation metrics cannot change the selected model or selected threshold.",
            f"Selected model: `{selected_model}`. Selected threshold: {selected_threshold}. "
            f"Reason: {selected_policy['reason']}",
            "",
            "## Validation Evaluation",
            f"Validation AP: {validation['threshold_independent']['average_precision']}. "
            f"Validation ROC-AUC: {validation['threshold_independent']['roc_auc']}.",
            f"At threshold 0.5, precision {validation_05['precision']}, recall "
            f"{validation_05['recall']}, F1 {validation_05['f1']}, F2 {validation_05['f2']}.",
            f"At the train-selected threshold, precision {validation_selected['precision']}, "
            f"recall {validation_selected['recall']}, F1 {validation_selected['f1']}, "
            f"F2 {validation_selected['f2']}.",
            "",
            "## Complexity vs Performance",
            "Non-linear models are evaluated because they can represent interactions that a linear "
            "model may miss. Complexity is only justified when train-only development metrics show "
            "a measurable ranking improvement large enough to overcome the simplicity tie policy.",
            "",
            "## Limitations",
            "This is a fixed-configuration development comparison, not production readiness. It "
            "does not claim performance on real industrial equipment, does not persist a model, "
            "and does not provide feature-importance or explainability conclusions.",
            "",
            "## Next Steps",
            "Later phases may add hyperparameter tuning, final model selection, locked test "
            "evaluation, model persistence, MLflow tracking, and SHAP explainability.",
            "",
        ]
    )


def write_artifacts(
    oof_predictions: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    metrics: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    validation_probabilities: np.ndarray,
    root: Path | None = None,
) -> ModelComparisonArtifacts:
    oof_path = oof_predictions_path(root)
    threshold_path = threshold_analysis_path(root)
    metrics_path = comparison_metrics_path(root)
    validation_predictions_path = comparison_validation_predictions_path(root)
    markdown_path = comparison_doc_path(root)

    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_csv(oof_path, index=False, float_format="%.10f")
    threshold_analysis.to_csv(threshold_path, index=False, float_format="%.6f")
    write_json(metrics, metrics_path)
    validation_predictions.to_csv(validation_predictions_path, index=False, float_format="%.10f")
    plot_paths = create_plots(
        oof_predictions,
        metrics,
        validation_df,
        config,
        validation_probabilities,
        root,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(metrics), encoding="utf-8")
    return ModelComparisonArtifacts(
        oof_predictions_csv=oof_path,
        threshold_analysis_csv=threshold_path,
        metrics_json=metrics_path,
        validation_predictions_csv=validation_predictions_path,
        markdown_report=markdown_path,
        plot_paths=plot_paths,
    )


def run_model_comparison_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
    root: Path | None = None,
) -> ModelComparisonResult:
    ai4i_baseline.validate_training_inputs(train_df, validation_df, config, split_summary)
    folds = make_shared_fold_assignments(train_df, config)
    oof_predictions = generate_oof_predictions(train_df, config, folds)
    threshold_analysis = build_threshold_analysis(oof_predictions)
    oof_metrics = build_oof_model_metrics(oof_predictions)
    threshold_candidates = build_threshold_candidates(threshold_analysis)
    selected_candidate = select_development_candidate(oof_metrics, threshold_candidates)
    validation_metrics, validation_predictions, validation_probabilities = evaluate_validation(
        train_df,
        validation_df,
        config,
        str(selected_candidate["selected_model"]),
        float(selected_candidate["selected_threshold"]),
    )
    metrics = build_metrics_summary(
        train_df,
        validation_df,
        config,
        folds,
        oof_metrics,
        threshold_candidates,
        selected_candidate,
        validation_metrics,
    )
    artifacts = write_artifacts(
        oof_predictions,
        threshold_analysis,
        metrics,
        validation_predictions,
        validation_df,
        config,
        validation_probabilities,
        root,
    )
    return ModelComparisonResult(
        metrics=metrics,
        oof_predictions=oof_predictions,
        threshold_analysis=threshold_analysis,
        validation_predictions=validation_predictions,
        artifacts=artifacts,
    )

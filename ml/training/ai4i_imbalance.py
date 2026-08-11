"""Train-only AI4I class-imbalance and threshold strategy utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
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
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
IMBALANCE_ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "imbalance"
IMBALANCE_DOC_RELATIVE_PATH = Path("docs") / "ml" / "ai4i_imbalance_strategy.md"
OOF_PREDICTIONS_FILENAME = "imbalance_train_oof_predictions.csv"
THRESHOLD_ANALYSIS_FILENAME = "logistic_threshold_analysis.csv"
IMBALANCE_METRICS_FILENAME = "imbalance_strategy_metrics.json"
VALIDATION_PREDICTIONS_FILENAME = "imbalance_validation_predictions.csv"
MODEL_NAMES = ("standard_logistic", "balanced_logistic")
MODEL_CLASS_WEIGHTS = {"standard_logistic": None, "balanced_logistic": "balanced"}
CV_SPLITS = 5
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = ai4i_baseline.LOGISTIC_MAX_ITER
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
class ImbalanceArtifacts:
    """Paths produced by an imbalance strategy run."""

    oof_predictions_csv: Path
    threshold_analysis_csv: Path
    metrics_json: Path
    validation_predictions_csv: Path
    markdown_report: Path
    plot_paths: list[Path]


@dataclass(frozen=True)
class ImbalanceResult:
    """Full result returned by an imbalance strategy experiment."""

    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    threshold_analysis: pd.DataFrame
    validation_predictions: pd.DataFrame
    artifacts: ImbalanceArtifacts


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def imbalance_assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / IMBALANCE_ASSETS_RELATIVE_DIR


def imbalance_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / IMBALANCE_DOC_RELATIVE_PATH


def oof_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / OOF_PREDICTIONS_FILENAME


def threshold_analysis_path(root: Path | None = None) -> Path:
    return reports_directory(root) / THRESHOLD_ANALYSIS_FILENAME


def imbalance_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / IMBALANCE_METRICS_FILENAME


def imbalance_validation_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / VALIDATION_PREDICTIONS_FILENAME


def imbalance_plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets = imbalance_assets_directory(root)
    return {
        "train_oof_precision_recall_curves": assets / "train_oof_precision_recall_curves.png",
        "train_oof_threshold_precision_recall": assets / "train_oof_threshold_precision_recall.png",
        "train_oof_threshold_f_scores": assets / "train_oof_threshold_f_scores.png",
        "validation_threshold_comparison_confusion_matrix": assets
        / "validation_threshold_comparison_confusion_matrix.png",
        "validation_precision_recall_curve": assets / "validation_precision_recall_curve.png",
    }


def _rounded_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def make_stratified_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)


def build_logistic_pipeline(
    config: ai4i_modeling.ModelingConfig,
    model_name: str,
) -> Pipeline:
    if model_name not in MODEL_CLASS_WEIGHTS:
        raise ValueError(f"Unknown Logistic Regression variant: {model_name}")
    ai4i_baseline.validate_feature_policy(config)
    return Pipeline(
        steps=[
            ("preprocessor", ai4i_baseline.build_preprocessor(config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=RANDOM_SEED,
                    class_weight=MODEL_CLASS_WEIGHTS[model_name],
                ),
            ),
        ]
    )


def build_model_pipelines(config: ai4i_modeling.ModelingConfig) -> dict[str, Pipeline]:
    return {model_name: build_logistic_pipeline(config, model_name) for model_name in MODEL_NAMES}


def validate_training_frame(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
) -> None:
    ai4i_baseline.validate_feature_policy(config)
    ai4i_baseline.validate_split_frame(
        train_df,
        config,
        "train",
        ai4i_baseline.expected_split_rows(split_summary, "train"),
    )


def validate_validation_frame(
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
) -> None:
    ai4i_baseline.validate_split_frame(
        validation_df,
        config,
        "validation",
        ai4i_baseline.expected_split_rows(split_summary, "validation"),
    )


def generate_oof_probabilities_for_model(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    model_name: str,
    cv: StratifiedKFold | None = None,
) -> np.ndarray:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    probabilities = np.full(len(train_df), np.nan, dtype=float)
    base_pipeline = build_logistic_pipeline(config, model_name)
    splitter = cv or make_stratified_kfold()

    for train_index, holdout_index in splitter.split(features, target):
        fold_pipeline = clone(base_pipeline)
        fold_pipeline.fit(features.iloc[train_index], target.iloc[train_index])
        probabilities[holdout_index] = fold_pipeline.predict_proba(features.iloc[holdout_index])[
            :, ai4i_baseline.POSITIVE_CLASS
        ]

    if np.isnan(probabilities).any():
        raise ValueError(f"OOF probabilities were not generated for every row: {model_name}")
    return probabilities


def generate_oof_predictions(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
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
            cv,
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
        elif isinstance(value, np.floating):
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


def build_oof_model_metrics(
    oof_predictions: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
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
    standard_ap = float(oof_metrics["standard_logistic"]["average_precision"])
    balanced_ap = float(oof_metrics["balanced_logistic"]["average_precision"])
    ap_difference = abs(standard_ap - balanced_ap)
    if ap_difference < AP_TIE_TOLERANCE:
        selected_model = "standard_logistic"
        reason = (
            "OOF AP difference is less than 0.01, so the simpler standard Logistic "
            "Regression variant is preferred."
        )
    elif balanced_ap > standard_ap:
        selected_model = "balanced_logistic"
        reason = "Balanced Logistic Regression has higher train OOF Average Precision."
    else:
        selected_model = "standard_logistic"
        reason = "Standard Logistic Regression has higher train OOF Average Precision."

    selected_threshold = float(threshold_candidates[selected_model]["max_f2"]["threshold"])
    return {
        "policy": [
            "Compare standard_logistic and balanced_logistic by TRAIN OOF Average Precision.",
            "If their AP difference is less than 0.01, prefer standard_logistic.",
            "Otherwise prefer the model with higher TRAIN OOF Average Precision.",
            "For the selected model, use its TRAIN OOF max-F2 threshold.",
        ],
        "selected_model": selected_model,
        "selected_threshold": _rounded_float(selected_threshold),
        "ap_difference": _rounded_float(ap_difference),
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
    pipeline = build_logistic_pipeline(config, model_name)
    pipeline.fit(features, target)
    return pipeline


def predict_probabilities(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> np.ndarray:
    features, _ = ai4i_baseline.extract_features_and_target(frame, config)
    return pipeline.predict_proba(features)[:, ai4i_baseline.POSITIVE_CLASS]


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
    predictions = pd.DataFrame(
        {
            "source_udi": validation_df[config.derived_traceability_field].astype(int),
            "target": target.astype(int),
            "probability": probabilities.astype(float),
            "prediction_threshold_0_5": threshold_predictions(probabilities, 0.5).astype(int),
            "prediction_selected_threshold": threshold_predictions(
                probabilities, selected_threshold
            ).astype(int),
        }
    ).sort_values("source_udi", kind="mergesort")
    return metrics, predictions.reset_index(drop=True), probabilities


def build_metrics_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    oof_metrics: Mapping[str, Mapping[str, Any]],
    threshold_candidates: Mapping[str, Mapping[str, Any]],
    selected_candidate: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    selected_threshold_metrics = validation_metrics["selected_threshold"]
    threshold_0_5_metrics = validation_metrics["threshold_0_5"]
    return {
        "experiment": {
            "name": "AI4I imbalance and threshold strategy",
            "objective": "Binary classification of Machine failure",
            "phase": "train-oof model development with validation-only evaluation",
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
                "Preprocessing is fitted inside each fold pipeline on fold-training rows only."
            ),
            "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
            "numerical": "Five process variables -> StandardScaler()",
        },
        "model_variants": {
            "standard_logistic": {"class_weight": None, "max_iter": LOGISTIC_MAX_ITER},
            "balanced_logistic": {"class_weight": "balanced", "max_iter": LOGISTIC_MAX_ITER},
        },
        "train_oof_results": {
            "standard_logistic": dict(oof_metrics["standard_logistic"]),
            "balanced_logistic": dict(oof_metrics["balanced_logistic"]),
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
            "threshold_0_5": dict(threshold_0_5_metrics),
            "selected_threshold": dict(selected_threshold_metrics),
            "selection_uses_validation": False,
        },
        "f2_note": (
            "F2 gives recall more weight than precision, but it is exploratory and not "
            "a confirmed business cost function."
        ),
    }


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


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_train_oof_precision_recall_curves(oof_predictions: pd.DataFrame, path: Path) -> None:
    _, axis = plt.subplots(figsize=(7, 5))
    target = oof_predictions["target"]
    for model_name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(
            target, oof_predictions[f"{model_name}_probability"]
        )
        axis.plot(recall, precision, label=model_name)
    axis.axhline(float(target.mean()), linestyle="--", label="Positive prevalence")
    axis.set_title("AI4I Train OOF Precision-Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def plot_threshold_precision_recall(
    threshold_analysis: pd.DataFrame,
    selected_model: str,
    selected_threshold: float,
    path: Path,
) -> None:
    subset = threshold_analysis[threshold_analysis["model"] == selected_model]
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(subset["threshold"], subset["precision"], label="Precision")
    axis.plot(subset["threshold"], subset["recall"], label="Recall")
    axis.axvline(selected_threshold, linestyle="--", label="Selected threshold")
    axis.set_title("AI4I Train OOF Precision and Recall by Threshold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Metric value")
    axis.legend()
    _save_figure(path)


def plot_threshold_f_scores(
    threshold_analysis: pd.DataFrame,
    selected_model: str,
    selected_threshold: float,
    path: Path,
) -> None:
    subset = threshold_analysis[threshold_analysis["model"] == selected_model]
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(subset["threshold"], subset["f1"], label="F1")
    axis.plot(subset["threshold"], subset["f2"], label="F2")
    axis.axvline(selected_threshold, linestyle="--", label="Selected threshold")
    axis.set_title("AI4I Train OOF F-Scores by Threshold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.legend()
    _save_figure(path)


def plot_validation_confusion_comparison(
    threshold_0_5_matrix: list[list[int]],
    selected_matrix: list[list[int]],
    selected_threshold: float,
    path: Path,
) -> None:
    matrices = [
        ("Threshold 0.50", np.array(threshold_0_5_matrix, dtype=int)),
        (f"Selected threshold {selected_threshold:.2f}", np.array(selected_matrix, dtype=int)),
    ]
    _, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for axis, (title, values) in zip(axes, matrices, strict=True):
        image = axis.imshow(values)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks([0, 1], labels=["No failure", "Failure"])
        axis.set_yticks([0, 1], labels=["No failure", "Failure"])
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{labels[row, column]}\n{values[row, column]}",
                    ha="center",
                    va="center",
                )
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save_figure(path)


def plot_validation_precision_recall_curve(
    validation_df: pd.DataFrame,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(validation_df["Machine failure"], probabilities)
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="Selected model")
    axis.axhline(float(validation_df["Machine failure"].mean()), linestyle="--", label="Prevalence")
    axis.set_title("AI4I Validation Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def create_plots(
    oof_predictions: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_probabilities: np.ndarray,
    metrics: Mapping[str, Any],
    root: Path | None = None,
) -> list[Path]:
    paths = imbalance_plot_paths(root)
    selected_model = metrics["selected_model"]
    selected_threshold = float(metrics["selected_threshold"])
    validation_results = metrics["validation_results"]
    plot_train_oof_precision_recall_curves(
        oof_predictions, paths["train_oof_precision_recall_curves"]
    )
    plot_threshold_precision_recall(
        threshold_analysis,
        selected_model,
        selected_threshold,
        paths["train_oof_threshold_precision_recall"],
    )
    plot_threshold_f_scores(
        threshold_analysis,
        selected_model,
        selected_threshold,
        paths["train_oof_threshold_f_scores"],
    )
    plot_validation_confusion_comparison(
        validation_results["threshold_0_5"]["confusion_matrix"],
        validation_results["selected_threshold"]["confusion_matrix"],
        selected_threshold,
        paths["validation_threshold_comparison_confusion_matrix"],
    )
    plot_validation_precision_recall_curve(
        validation_df,
        validation_probabilities,
        paths["validation_precision_recall_curve"],
    )
    return [paths[name] for name in sorted(paths)]


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def render_markdown(metrics: Mapping[str, Any]) -> str:
    standard = metrics["train_oof_results"]["standard_logistic"]
    balanced = metrics["train_oof_results"]["balanced_logistic"]
    selected = metrics["candidate_selection_policy"]
    validation = metrics["validation_results"]
    standard_05 = standard["threshold_0_5"]
    balanced_05 = balanced["threshold_0_5"]
    validation_05 = validation["threshold_0_5"]
    validation_selected = validation["selected_threshold"]
    return "\n".join(
        [
            "# AI4I Imbalance and Threshold Strategy",
            "",
            "## Motivation",
            "The previous Logistic Regression baseline had useful ranking metrics but low "
            "default-threshold recall. This phase investigates whether class weighting and "
            "a lower decision threshold improve minority-class detection.",
            "",
            "## Experimental Protocol",
            "The training split is used for 5-fold out-of-fold model development, threshold "
            "analysis, and candidate selection. Validation is evaluated once after the model "
            "variant and threshold are selected from training OOF results.",
            "",
            "## Locked Test Set",
            "The test split remains locked and was not read, summarized, predicted, or evaluated.",
            "",
            "## Cross-Validation",
            "`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` creates OOF "
            "probabilities. Each fold fits its own preprocessing pipeline on fold-training "
            "rows only, so held-out rows are transformed by a pipeline not fitted on them.",
            "",
            "## Standard Logistic Regression",
            f"Train OOF AP: {standard['average_precision']}. Train OOF ROC-AUC: "
            f"{standard['roc_auc']}. At threshold 0.5, precision "
            f"{standard_05['precision']}, recall {standard_05['recall']}, F1 "
            f"{standard_05['f1']}, F2 {standard_05['f2']}.",
            "",
            "## Class-Weighted Logistic Regression",
            f"Train OOF AP: {balanced['average_precision']}. Train OOF ROC-AUC: "
            f"{balanced['roc_auc']}. At threshold 0.5, precision "
            f"{balanced_05['precision']}, recall {balanced_05['recall']}, F1 "
            f"{balanced_05['f1']}, F2 {balanced_05['f2']}.",
            "",
            "## Threshold Analysis",
            "Thresholds from 0.01 to 0.99 are evaluated on train OOF probabilities. "
            "Accuracy is not used as a threshold-selection objective because failures are rare.",
            "",
            "## Candidate Selection Policy",
            "The documented policy compares train OOF Average Precision. If AP differs by "
            "less than 0.01, the simpler standard model is selected; otherwise, the higher-AP "
            "model is selected. The selected model then uses its train OOF max-F2 threshold.",
            f"Selected model: `{selected['selected_model']}`. Selected threshold: "
            f"{selected['selected_threshold']}. Reason: {selected['reason']}",
            "",
            "## Validation Evaluation",
            f"Validation AP: {validation['threshold_independent']['average_precision']}. "
            f"Validation ROC-AUC: {validation['threshold_independent']['roc_auc']}.",
            f"At threshold 0.5, precision {validation_05['precision']}, recall "
            f"{validation_05['recall']}, F1 {validation_05['f1']}, F2 {validation_05['f2']}.",
            f"At the selected threshold, precision {validation_selected['precision']}, "
            f"recall {validation_selected['recall']}, F1 {validation_selected['f1']}, "
            f"F2 {validation_selected['f2']}.",
            "",
            "## Precision vs Recall Trade-off",
            "Lower thresholds can reduce false negatives by predicting more positives, but this "
            "usually increases false positives. False positives and false negatives require "
            "domain and business cost information before any final maintenance policy "
            "can be chosen.",
            "",
            "## Limitations",
            "F2 gives recall more weight than precision, but it is exploratory here and is not an "
            "official business objective. The selected threshold is a model-development candidate, "
            "not a production-optimal operating point.",
            "",
            "## Next Steps",
            "Future phases may compare additional model families, perform formal model selection, "
            "track experiments, add explainability, and eventually evaluate the locked test split.",
            "",
        ]
    )


def write_artifacts(
    oof_predictions: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    metrics: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_probabilities: np.ndarray,
    root: Path | None = None,
) -> ImbalanceArtifacts:
    oof_path = oof_predictions_path(root)
    threshold_path = threshold_analysis_path(root)
    metrics_path = imbalance_metrics_path(root)
    validation_predictions_path = imbalance_validation_predictions_path(root)
    markdown_path = imbalance_doc_path(root)

    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_csv(oof_path, index=False, float_format="%.10f")
    threshold_analysis.to_csv(threshold_path, index=False, float_format="%.6f")
    write_json(metrics, metrics_path)
    validation_predictions.to_csv(validation_predictions_path, index=False, float_format="%.10f")
    plot_paths = create_plots(
        oof_predictions,
        threshold_analysis,
        validation_df,
        validation_probabilities,
        metrics,
        root,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(metrics), encoding="utf-8")
    return ImbalanceArtifacts(
        oof_predictions_csv=oof_path,
        threshold_analysis_csv=threshold_path,
        metrics_json=metrics_path,
        validation_predictions_csv=validation_predictions_path,
        markdown_report=markdown_path,
        plot_paths=plot_paths,
    )


def run_imbalance_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
    root: Path | None = None,
) -> ImbalanceResult:
    validate_training_frame(train_df, config, split_summary)
    validate_validation_frame(validation_df, config, split_summary)
    train_udis = set(train_df[config.derived_traceability_field].tolist())
    validation_udis = set(validation_df[config.derived_traceability_field].tolist())
    if train_udis & validation_udis:
        raise ValueError("Train and validation source_udi values must not overlap.")

    oof_predictions = generate_oof_predictions(train_df, config)
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
        validation_probabilities,
        root,
    )
    return ImbalanceResult(
        metrics=metrics,
        oof_predictions=oof_predictions,
        threshold_analysis=threshold_analysis,
        validation_predictions=validation_predictions,
        artifacts=artifacts,
    )

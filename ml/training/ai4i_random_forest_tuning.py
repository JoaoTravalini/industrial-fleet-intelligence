"""Train-only AI4I Random Forest targeted tuning utilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
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
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
TUNING_ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "random_forest_tuning"
TUNING_DOC_RELATIVE_PATH = Path("docs") / "ml" / "ai4i_random_forest_tuning.md"
COMPARISON_METRICS_FILENAME = "model_comparison_metrics.json"
NESTED_OOF_PREDICTIONS_FILENAME = "random_forest_tuning_nested_oof_predictions.csv"
OUTER_FOLDS_FILENAME = "random_forest_tuning_outer_folds.csv"
GRID_RESULTS_FILENAME = "random_forest_tuning_grid_results.csv"
THRESHOLD_ANALYSIS_FILENAME = "random_forest_tuning_threshold_analysis.csv"
TUNING_METRICS_FILENAME = "random_forest_tuning_metrics.json"
VALIDATION_PREDICTIONS_FILENAME = "random_forest_tuning_validation_predictions.csv"
RANDOM_SEED = 42
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
FULL_TRAIN_CV_SPLITS = 5
PARAMETER_GRID = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 12],
    "classifier__min_samples_leaf": [1, 3],
}
FIXED_MAX_FEATURES = "sqrt"
FIXED_CLASS_WEIGHT = "balanced_subsample"
FIXED_RANDOM_FOREST_CONFIGURATION = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": FIXED_MAX_FEATURES,
    "class_weight": FIXED_CLASS_WEIGHT,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
THRESHOLDS = tuple(round(value / 100, 2) for value in range(1, 100))
RECALL_TARGET = 0.70
PROMOTION_AP_DELTA = 0.005
TEST_SET_STATUS = "LOCKED / NOT USED"
THRESHOLD_COLUMNS = [
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
THRESHOLD_COUNT_COLUMNS = {
    "predicted_positive_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
}
NESTED_OOF_COLUMNS = ["source_udi", "target", "probability", "outer_fold"]
OUTER_FOLD_COLUMNS = [
    "outer_fold",
    "training_rows",
    "holdout_rows",
    "training_positive_count",
    "holdout_positive_count",
    "best_inner_average_precision",
    "selected_n_estimators",
    "selected_max_depth",
    "selected_min_samples_leaf",
]
GRID_RESULT_COLUMNS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "mean_test_average_precision",
    "std_test_average_precision",
    "rank_test_average_precision",
]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class NestedOofResult:
    """Nested OOF predictions and per-outer-fold selection records."""

    oof_predictions: pd.DataFrame
    outer_folds: pd.DataFrame


@dataclass(frozen=True)
class RandomForestTuningArtifacts:
    """Paths produced by a Random Forest tuning run."""

    nested_oof_predictions_csv: Path
    outer_folds_csv: Path
    grid_results_csv: Path
    threshold_analysis_csv: Path
    metrics_json: Path
    validation_predictions_csv: Path
    markdown_report: Path
    plot_paths: list[Path]


@dataclass(frozen=True)
class RandomForestTuningResult:
    """Full result returned by a Random Forest tuning experiment."""

    metrics: dict[str, Any]
    nested_oof_predictions: pd.DataFrame
    outer_folds: pd.DataFrame
    grid_results: pd.DataFrame
    threshold_analysis: pd.DataFrame
    validation_predictions: pd.DataFrame
    artifacts: RandomForestTuningArtifacts


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def tuning_assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / TUNING_ASSETS_RELATIVE_DIR


def tuning_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / TUNING_DOC_RELATIVE_PATH


def comparison_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / COMPARISON_METRICS_FILENAME


def nested_oof_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / NESTED_OOF_PREDICTIONS_FILENAME


def outer_folds_path(root: Path | None = None) -> Path:
    return reports_directory(root) / OUTER_FOLDS_FILENAME


def grid_results_path(root: Path | None = None) -> Path:
    return reports_directory(root) / GRID_RESULTS_FILENAME


def threshold_analysis_path(root: Path | None = None) -> Path:
    return reports_directory(root) / THRESHOLD_ANALYSIS_FILENAME


def tuning_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / TUNING_METRICS_FILENAME


def validation_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / VALIDATION_PREDICTIONS_FILENAME


def tuning_plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets = tuning_assets_directory(root)
    return {
        "nested_oof_precision_recall_curve": assets / "nested_oof_precision_recall_curve.png",
        "nested_oof_threshold_precision_recall": assets
        / "nested_oof_threshold_precision_recall.png",
        "nested_oof_threshold_f_scores": assets / "nested_oof_threshold_f_scores.png",
        "grid_search_average_precision": assets / "grid_search_average_precision.png",
        "validation_tuned_confusion_matrix": assets / "validation_tuned_confusion_matrix.png",
        "fixed_vs_tuned_average_precision": assets / "fixed_vs_tuned_average_precision.png",
    }


def _rounded_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def format_max_depth(value: int | float | None | str) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and np.isnan(value):
        return "None"
    if isinstance(value, str) and value.lower() == "none":
        return "None"
    return str(int(value))


def normalize_max_depth(value: int | float | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return int(value)


def parameter_label(params: Mapping[str, Any]) -> str:
    return (
        f"n={int(params['n_estimators'])}, depth={format_max_depth(params['max_depth'])}, "
        f"leaf={int(params['min_samples_leaf'])}"
    )


def allowed_parameter_configurations() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for n_estimators in PARAMETER_GRID["classifier__n_estimators"]:
        for max_depth in PARAMETER_GRID["classifier__max_depth"]:
            for min_samples_leaf in PARAMETER_GRID["classifier__min_samples_leaf"]:
                configs.append(
                    {
                        "n_estimators": int(n_estimators),
                        "max_depth": max_depth,
                        "min_samples_leaf": int(min_samples_leaf),
                        "max_features": FIXED_MAX_FEATURES,
                        "class_weight": FIXED_CLASS_WEIGHT,
                        "random_state": RANDOM_SEED,
                        "n_jobs": 1,
                    }
                )
    return configs


def allowed_parameter_keys() -> set[tuple[int, str, int]]:
    return {
        (item["n_estimators"], format_max_depth(item["max_depth"]), item["min_samples_leaf"])
        for item in allowed_parameter_configurations()
    }


def extract_random_forest_params(best_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": int(best_params["classifier__n_estimators"]),
        "max_depth": normalize_max_depth(best_params["classifier__max_depth"]),
        "min_samples_leaf": int(best_params["classifier__min_samples_leaf"]),
        "max_features": FIXED_MAX_FEATURES,
        "class_weight": FIXED_CLASS_WEIGHT,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }


def serialized_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": format_max_depth(params["max_depth"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": str(params.get("max_features", FIXED_MAX_FEATURES)),
        "class_weight": str(params.get("class_weight", FIXED_CLASS_WEIGHT)),
        "random_state": int(params.get("random_state", RANDOM_SEED)),
        "n_jobs": int(params.get("n_jobs", 1)),
    }


def make_outer_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)


def make_inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)


def make_full_train_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=FULL_TRAIN_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)


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


def build_random_forest_pipeline(
    config: ai4i_modeling.ModelingConfig,
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor(config)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=FIXED_MAX_FEATURES,
                    class_weight=FIXED_CLASS_WEIGHT,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_fixed_random_forest_pipeline(config: ai4i_modeling.ModelingConfig) -> Pipeline:
    return build_random_forest_pipeline(
        config,
        n_estimators=FIXED_RANDOM_FOREST_CONFIGURATION["n_estimators"],
        max_depth=FIXED_RANDOM_FOREST_CONFIGURATION["max_depth"],
        min_samples_leaf=FIXED_RANDOM_FOREST_CONFIGURATION["min_samples_leaf"],
    )


def build_tuned_random_forest_pipeline(
    config: ai4i_modeling.ModelingConfig,
    params: Mapping[str, Any],
) -> Pipeline:
    return build_random_forest_pipeline(
        config,
        n_estimators=int(params["n_estimators"]),
        max_depth=normalize_max_depth(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
    )


def run_inner_grid_search(
    features: pd.DataFrame,
    target: pd.Series,
    config: ai4i_modeling.ModelingConfig,
    *,
    grid_n_jobs: int = -1,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_random_forest_pipeline(config, n_estimators=200),
        param_grid=PARAMETER_GRID,
        scoring="average_precision",
        refit=True,
        cv=make_inner_cv(),
        n_jobs=grid_n_jobs,
    )
    search.fit(features, target)
    return search


def run_full_train_grid_search(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    *,
    grid_n_jobs: int = -1,
) -> GridSearchCV:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    search = GridSearchCV(
        estimator=build_random_forest_pipeline(config, n_estimators=200),
        param_grid=PARAMETER_GRID,
        scoring="average_precision",
        refit=True,
        cv=make_full_train_cv(),
        n_jobs=grid_n_jobs,
    )
    search.fit(features, target)
    return search


def generate_nested_oof_predictions(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    *,
    grid_n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
) -> NestedOofResult:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    probabilities = np.full(len(train_df), np.nan, dtype=float)
    outer_fold_values = np.full(len(train_df), -1, dtype=int)
    outer_records: list[dict[str, Any]] = []
    outer_cv = make_outer_cv()

    for outer_fold, (outer_train_index, outer_holdout_index) in enumerate(
        outer_cv.split(features, target), start=1
    ):
        if progress_callback is not None:
            progress_callback(outer_fold, OUTER_CV_SPLITS)
        outer_train_features = features.iloc[outer_train_index]
        outer_train_target = target.iloc[outer_train_index]
        outer_holdout_features = features.iloc[outer_holdout_index]
        outer_holdout_target = target.iloc[outer_holdout_index]

        search = run_inner_grid_search(
            outer_train_features,
            outer_train_target,
            config,
            grid_n_jobs=grid_n_jobs,
        )
        best_estimator = search.best_estimator_
        fold_probability = best_estimator.predict_proba(outer_holdout_features)[
            :, ai4i_baseline.POSITIVE_CLASS
        ]
        probabilities[outer_holdout_index] = fold_probability
        outer_fold_values[outer_holdout_index] = outer_fold
        selected_params = extract_random_forest_params(search.best_params_)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "training_rows": int(len(outer_train_index)),
                "holdout_rows": int(len(outer_holdout_index)),
                "training_positive_count": int(outer_train_target.sum()),
                "holdout_positive_count": int(outer_holdout_target.sum()),
                "best_inner_average_precision": _rounded_float(search.best_score_),
                "selected_n_estimators": selected_params["n_estimators"],
                "selected_max_depth": format_max_depth(selected_params["max_depth"]),
                "selected_min_samples_leaf": selected_params["min_samples_leaf"],
            }
        )

    if np.isnan(probabilities).any():
        raise ValueError("Nested OOF probabilities were not generated for every training row.")
    if np.any(outer_fold_values < 1):
        raise ValueError("Nested OOF outer fold values were not assigned for every training row.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Nested OOF probabilities contain values outside [0, 1].")

    predictions = pd.DataFrame(
        {
            "source_udi": train_df[config.derived_traceability_field].astype(int),
            "target": train_df[config.target_column].astype(int),
            "probability": probabilities.astype(float),
            "outer_fold": outer_fold_values.astype(int),
        },
        columns=NESTED_OOF_COLUMNS,
    ).sort_values("source_udi", kind="mergesort")
    outer_folds = pd.DataFrame(outer_records, columns=OUTER_FOLD_COLUMNS).sort_values(
        "outer_fold", kind="mergesort"
    )
    return NestedOofResult(
        oof_predictions=predictions.reset_index(drop=True),
        outer_folds=outer_folds.reset_index(drop=True),
    )


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
        "average_precision": _rounded_float(average_precision_score(y_true, y_probability)),
        "roc_auc": _rounded_float(roc_auc_score(y_true, y_probability)),
    }


def build_nested_oof_metrics(oof_predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        **ranking_metrics(oof_predictions["target"], oof_predictions["probability"]),
        "threshold_0_5": classification_metrics_at_threshold(
            oof_predictions["target"], oof_predictions["probability"], 0.5
        ),
    }


def build_threshold_analysis(
    oof_predictions: pd.DataFrame,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = classification_metrics_at_threshold(
            oof_predictions["target"], oof_predictions["probability"], threshold
        )
        rows.append({column: metrics[column] for column in THRESHOLD_COLUMNS})
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS).sort_values("threshold", kind="mergesort")


def threshold_row_to_dict(row: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if key in THRESHOLD_COUNT_COLUMNS:
            output[key] = int(value)
        elif isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            output[key] = _rounded_float(value)
        else:
            output[key] = value
    return output


def select_threshold_by_metric(threshold_analysis: pd.DataFrame, metric: str) -> dict[str, Any]:
    best = threshold_analysis.sort_values(
        [metric, "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return threshold_row_to_dict(best)


def select_recall_candidate(
    threshold_analysis: pd.DataFrame,
    minimum_recall: float = RECALL_TARGET,
) -> dict[str, Any] | None:
    subset = threshold_analysis[threshold_analysis["recall"] >= minimum_recall]
    if subset.empty:
        return None
    best = subset.sort_values(
        ["precision", "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return threshold_row_to_dict(best)


def build_threshold_candidates(threshold_analysis: pd.DataFrame) -> dict[str, Any]:
    recall_candidate = select_recall_candidate(threshold_analysis)
    return {
        "max_f1": select_threshold_by_metric(threshold_analysis, "f1"),
        "max_f2": select_threshold_by_metric(threshold_analysis, "f2"),
        "recall_70": recall_candidate
        if recall_candidate is not None
        else {
            "available": False,
            "reason": "No threshold in the deterministic grid achieved recall >= 0.70.",
        },
    }


def load_fixed_random_forest_reference(root: Path | None = None) -> dict[str, Any]:
    path = comparison_metrics_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Model comparison metrics artifact was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    fixed_oof = metrics["train_oof_results"]["random_forest"]
    fixed_threshold = metrics["threshold_candidates"]["random_forest"]["max_f2"]
    return {
        "source_artifact": COMPARISON_METRICS_FILENAME,
        "configuration": serialized_params(FIXED_RANDOM_FOREST_CONFIGURATION),
        "average_precision": fixed_oof["average_precision"],
        "roc_auc": fixed_oof["roc_auc"],
        "threshold_0_5": fixed_oof["threshold_0_5"],
        "max_f2_threshold": fixed_threshold,
        "methodology": (
            "Previous fixed Random Forest ordinary train OOF from model comparison phase."
        ),
    }


def grid_results_dataframe(search: GridSearchCV) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cv_results = search.cv_results_
    for index in range(len(cv_results["params"])):
        params = extract_random_forest_params(cv_results["params"][index])
        rows.append(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": format_max_depth(params["max_depth"]),
                "min_samples_leaf": params["min_samples_leaf"],
                "mean_test_average_precision": _rounded_float(cv_results["mean_test_score"][index]),
                "std_test_average_precision": _rounded_float(cv_results["std_test_score"][index]),
                "rank_test_average_precision": int(cv_results["rank_test_score"][index]),
                "_max_depth_sort": 0 if params["max_depth"] is None else int(params["max_depth"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "rank_test_average_precision",
                "n_estimators",
                "_max_depth_sort",
                "min_samples_leaf",
            ],
            kind="mergesort",
        )
        .drop(columns=["_max_depth_sort"])
        .reset_index(drop=True)
    )


def full_train_search_summary(search: GridSearchCV, grid_results: pd.DataFrame) -> dict[str, Any]:
    best_params = extract_random_forest_params(search.best_params_)
    best_serialized = serialized_params(best_params)
    best_row = grid_results[grid_results["rank_test_average_precision"] == 1].iloc[0]
    return {
        "cv_configuration": {
            "method": "StratifiedKFold",
            "n_splits": FULL_TRAIN_CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_SEED,
        },
        "best_hyperparameters": best_serialized,
        "best_mean_average_precision": _rounded_float(search.best_score_),
        "best_std_average_precision": _rounded_float(best_row["std_test_average_precision"]),
        "scoring": "average_precision",
    }


def select_promotion_candidate(
    fixed_reference: Mapping[str, Any],
    tuned_nested_oof_metrics: Mapping[str, Any],
    tuned_threshold_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_ap = float(fixed_reference["average_precision"])
    tuned_ap = float(tuned_nested_oof_metrics["average_precision"])
    ap_delta = tuned_ap - fixed_ap
    if ap_delta >= PROMOTION_AP_DELTA:
        selected_candidate = "tuned_random_forest"
        selected_threshold = float(tuned_threshold_candidates["max_f2"]["threshold"])
        threshold_source = "tuned_nested_oof_max_f2"
        reason = (
            "Tuned nested-OOF Average Precision exceeds the previous fixed Random Forest "
            "OOF AP by at least 0.005."
        )
    else:
        selected_candidate = "fixed_random_forest"
        selected_threshold = float(fixed_reference["max_f2_threshold"]["threshold"])
        threshold_source = "previous_fixed_random_forest_oof_max_f2"
        reason = (
            "Tuned nested-OOF Average Precision does not exceed the previous fixed Random "
            "Forest OOF AP by at least 0.005, so added complexity is not promoted."
        )
    return {
        "policy": [
            "Compare tuned nested-OOF Average Precision to the previous fixed Random "
            "Forest train OOF Average Precision.",
            "Promote tuned_random_forest only if tuned nested-OOF AP is at least 0.005 higher.",
            "Otherwise retain fixed_random_forest because added complexity is not justified.",
            "Validation metrics cannot change the promotion decision or selected threshold.",
        ],
        "methodology_caution": (
            "The fixed reference uses ordinary train OOF development estimates, while tuned "
            "Random Forest uses nested train OOF estimates. Small numerical differences should "
            "be interpreted cautiously."
        ),
        "fixed_average_precision": _rounded_float(fixed_ap),
        "tuned_nested_average_precision": _rounded_float(tuned_ap),
        "average_precision_delta": _rounded_float(ap_delta),
        "promotion_delta_required": PROMOTION_AP_DELTA,
        "selected_candidate": selected_candidate,
        "selected_threshold": _rounded_float(selected_threshold),
        "threshold_source": threshold_source,
        "reason": reason,
        "validation_may_change_selection": False,
    }


def selected_candidate_params(
    selected_candidate: str,
    full_train_best_params: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_candidate == "tuned_random_forest":
        return dict(full_train_best_params)
    if selected_candidate == "fixed_random_forest":
        return dict(FIXED_RANDOM_FOREST_CONFIGURATION)
    raise ValueError(f"Unknown Random Forest candidate: {selected_candidate}")


def fit_selected_pipeline(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    selected_candidate: str,
    full_train_best_params: Mapping[str, Any],
) -> Pipeline:
    features, target = ai4i_baseline.extract_features_and_target(train_df, config)
    params = selected_candidate_params(selected_candidate, full_train_best_params)
    pipeline = build_tuned_random_forest_pipeline(config, params)
    pipeline.fit(features, target)
    return pipeline


def predict_probabilities(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> np.ndarray:
    features, _ = ai4i_baseline.extract_features_and_target(frame, config)
    return pipeline.predict_proba(features)[:, ai4i_baseline.POSITIVE_CLASS]


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
    selected_candidate: str,
    full_train_best_params: Mapping[str, Any],
    selected_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    pipeline = fit_selected_pipeline(
        train_df,
        config,
        selected_candidate,
        full_train_best_params,
    )
    probabilities = predict_probabilities(pipeline, validation_df, config)
    target = validation_df[config.target_column].astype(int)
    metrics = {
        "threshold_independent": ranking_metrics(target, probabilities),
        "threshold_0_5": classification_metrics_at_threshold(target, probabilities, 0.5),
        "selected_threshold": classification_metrics_at_threshold(
            target, probabilities, selected_threshold
        ),
    }
    predictions = create_validation_predictions(
        validation_df,
        config,
        probabilities,
        selected_threshold,
    )
    return metrics, predictions, probabilities


def build_metrics_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    fixed_reference: Mapping[str, Any],
    nested_oof_metrics: Mapping[str, Any],
    threshold_candidates: Mapping[str, Any],
    full_train_summary: Mapping[str, Any],
    promotion_decision: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "AI4I targeted Random Forest tuning",
            "objective": "Binary classification of Machine failure",
            "phase": "train-only nested CV tuning with validation-only evaluation",
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
        "nested_cv_configuration": {
            "outer": {
                "method": "StratifiedKFold",
                "n_splits": OUTER_CV_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_SEED,
            },
            "inner": {
                "method": "StratifiedKFold",
                "n_splits": INNER_CV_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_SEED,
            },
            "scoring": "average_precision",
            "refit": True,
            "outer_probability_policy": (
                "Each training observation receives exactly one nested out-of-fold probability "
                "from a pipeline whose preprocessing, hyperparameter selection, and model fit "
                "exclude that observation."
            ),
        },
        "parameter_grid": {
            "n_estimators": [200, 400],
            "max_depth": ["None", "12"],
            "min_samples_leaf": [1, 3],
            "fixed": {
                "max_features": FIXED_MAX_FEATURES,
                "class_weight": FIXED_CLASS_WEIGHT,
                "random_state": RANDOM_SEED,
                "n_jobs": 1,
            },
            "candidate_count": len(allowed_parameter_configurations()),
            "candidate_configurations": [
                serialized_params(params) for params in allowed_parameter_configurations()
            ],
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
                "Preprocessing remains inside the scikit-learn Pipeline and is fitted only on "
                "the corresponding training rows for each inner CV, outer refit, or final fit."
            ),
            "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
            "numerical": "Five process variables -> passthrough",
        },
        "fixed_random_forest_reference": dict(fixed_reference),
        "tuned_nested_oof_results": dict(nested_oof_metrics),
        "threshold_grid": {
            "minimum": min(THRESHOLDS),
            "maximum": max(THRESHOLDS),
            "step": 0.01,
            "selection_uses_accuracy": False,
        },
        "threshold_candidates": dict(threshold_candidates),
        "full_train_grid_search": dict(full_train_summary),
        "promotion_policy": dict(promotion_decision),
        "selected_candidate": promotion_decision["selected_candidate"],
        "selected_threshold": promotion_decision["selected_threshold"],
        "selected_candidate_configuration": serialized_params(
            selected_candidate_params(
                str(promotion_decision["selected_candidate"]),
                full_train_summary["best_hyperparameters"],
            )
        ),
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


def plot_nested_oof_precision_recall_curve(oof_predictions: pd.DataFrame, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(
        oof_predictions["target"], oof_predictions["probability"]
    )
    prevalence = float(oof_predictions["target"].mean())
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="Tuned RF nested OOF")
    axis.axhline(prevalence, linestyle="--", label="Training positive prevalence")
    axis.set_title("AI4I Random Forest Nested OOF Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def plot_threshold_precision_recall(
    threshold_analysis: pd.DataFrame,
    selected_threshold: float,
    path: Path,
) -> None:
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(threshold_analysis["threshold"], threshold_analysis["precision"], label="Precision")
    axis.plot(threshold_analysis["threshold"], threshold_analysis["recall"], label="Recall")
    axis.axvline(selected_threshold, linestyle="--", label="Selected threshold")
    axis.set_title("AI4I Tuned RF Nested OOF Precision and Recall by Threshold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Metric value")
    axis.legend()
    _save_figure(path)


def plot_threshold_f_scores(
    threshold_analysis: pd.DataFrame,
    selected_threshold: float,
    path: Path,
) -> None:
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(threshold_analysis["threshold"], threshold_analysis["f1"], label="F1")
    axis.plot(threshold_analysis["threshold"], threshold_analysis["f2"], label="F2")
    axis.axvline(selected_threshold, linestyle="--", label="Selected threshold")
    axis.set_title("AI4I Tuned RF Nested OOF F-Scores by Threshold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.legend()
    _save_figure(path)


def plot_grid_search_average_precision(grid_results: pd.DataFrame, path: Path) -> None:
    ordered = grid_results.sort_values(
        ["rank_test_average_precision", "n_estimators", "max_depth", "min_samples_leaf"],
        kind="mergesort",
    )
    labels = [
        parameter_label(
            {
                "n_estimators": row.n_estimators,
                "max_depth": row.max_depth,
                "min_samples_leaf": row.min_samples_leaf,
            }
        )
        for row in ordered.itertuples(index=False)
    ]
    values = ordered["mean_test_average_precision"].astype(float).tolist()
    errors = ordered["std_test_average_precision"].astype(float).tolist()
    _, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(range(len(values)), values, yerr=errors)
    axis.set_title("AI4I Full-Train RF Grid Search Average Precision")
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Mean CV Average Precision")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_ylim(0, max(values) * 1.15 if values else 1)
    _save_figure(path)


def plot_validation_confusion_matrix(
    matrix: list[list[int]],
    selected_candidate: str,
    selected_threshold: float,
    path: Path,
) -> None:
    values = np.array(matrix, dtype=int)
    _, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(values)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(
        f"AI4I {selected_candidate} Validation Confusion Matrix ({selected_threshold:.2f})"
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


def plot_fixed_vs_tuned_average_precision(metrics: Mapping[str, Any], path: Path) -> None:
    fixed_ap = float(metrics["fixed_random_forest_reference"]["average_precision"])
    tuned_ap = float(metrics["tuned_nested_oof_results"]["average_precision"])
    _, axis = plt.subplots(figsize=(7, 5))
    axis.bar(["Fixed RF\nordinary OOF", "Tuned RF\nnested OOF"], [fixed_ap, tuned_ap])
    axis.set_title("Fixed RF OOF vs Tuned RF Nested OOF AP - Compare Cautiously")
    axis.set_xlabel("Development estimate protocol")
    axis.set_ylabel("Average Precision")
    axis.set_ylim(0, max(fixed_ap, tuned_ap) * 1.15)
    _save_figure(path)


def create_plots(
    oof_predictions: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    grid_results: pd.DataFrame,
    metrics: Mapping[str, Any],
    root: Path | None = None,
) -> list[Path]:
    paths = tuning_plot_paths(root)
    selected_threshold = float(metrics["selected_threshold"])
    selected_candidate = str(metrics["selected_candidate"])
    plot_nested_oof_precision_recall_curve(
        oof_predictions, paths["nested_oof_precision_recall_curve"]
    )
    plot_threshold_precision_recall(
        threshold_analysis,
        selected_threshold,
        paths["nested_oof_threshold_precision_recall"],
    )
    plot_threshold_f_scores(
        threshold_analysis,
        selected_threshold,
        paths["nested_oof_threshold_f_scores"],
    )
    plot_grid_search_average_precision(grid_results, paths["grid_search_average_precision"])
    plot_validation_confusion_matrix(
        metrics["validation_results"]["selected_threshold"]["confusion_matrix"],
        selected_candidate,
        selected_threshold,
        paths["validation_tuned_confusion_matrix"],
    )
    plot_fixed_vs_tuned_average_precision(metrics, paths["fixed_vs_tuned_average_precision"])
    return [paths[name] for name in sorted(paths)]


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def render_markdown(metrics: Mapping[str, Any], outer_folds: pd.DataFrame) -> str:
    fixed = metrics["fixed_random_forest_reference"]
    tuned = metrics["tuned_nested_oof_results"]
    promotion = metrics["promotion_policy"]
    validation = metrics["validation_results"]
    threshold = metrics["threshold_candidates"]["max_f2"]
    full_train = metrics["full_train_grid_search"]
    fold_lines = [
        "| Outer fold | Best inner AP | n_estimators | max_depth | min_samples_leaf |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in outer_folds.itertuples(index=False):
        fold_lines.append(
            f"| {row.outer_fold} | {row.best_inner_average_precision} | "
            f"{row.selected_n_estimators} | {row.selected_max_depth} | "
            f"{row.selected_min_samples_leaf} |"
        )
    validation_05 = validation["threshold_0_5"]
    validation_selected = validation["selected_threshold"]
    return "\n".join(
        [
            "# AI4I Random Forest Tuning",
            "",
            "## Motivation",
            "The fixed-configuration model-family comparison selected Random Forest through the "
            "predefined train-only simplicity rule. This phase performs a deliberately small "
            "Random Forest tuning pass to see whether modest complexity is justified.",
            "",
            "## Why Random Forest Was Selected",
            "Random Forest was the selected development candidate because its train OOF Average "
            "Precision was within 0.01 of XGBoost while being simpler under the predefined order.",
            "",
            "## Targeted Search Space",
            "The search space is intentionally small: `n_estimators` in `{200, 400}`, `max_depth` "
            'in `{None, 12}`, and `min_samples_leaf` in `{1, 3}`. `max_features="sqrt"`, '
            '`class_weight="balanced_subsample"`, `random_state=42`, and estimator `n_jobs=1` '
            "are fixed. This creates exactly eight configurations.",
            "",
            "## Nested Cross-Validation",
            "Nested CV is used so each development probability is generated by a model whose "
            "preprocessing fit, inner hyperparameter selection, and final outer-fold fit did not "
            "see the held-out row. The outer CV has five stratified folds; each outer-training "
            "slice runs a three-fold inner `GridSearchCV` scored by Average Precision.",
            "",
            "## Leakage Prevention",
            "Only the training split is used for tuning, nested OOF prediction, "
            "threshold analysis, "
            "and promotion. Validation is evaluated once after all train-derived choices "
            "are fixed. "
            "The locked test split remains untouched.",
            "",
            "## Hyperparameter Search",
            *fold_lines,
            "",
            "## Nested OOF Results",
            f"Tuned nested-OOF AP: {tuned['average_precision']}. Tuned nested-OOF ROC-AUC: "
            f"{tuned['roc_auc']}. At threshold 0.5, precision "
            f"{tuned['threshold_0_5']['precision']}, recall {tuned['threshold_0_5']['recall']}, "
            f"F1 {tuned['threshold_0_5']['f1']}, F2 {tuned['threshold_0_5']['f2']}.",
            "",
            "## Threshold Strategy",
            "Thresholds from 0.01 through 0.99 are evaluated only on nested train OOF "
            "probabilities. The tuned development threshold is the nested-OOF max-F2 threshold. "
            "F2 is recall-weighted, but it is not a real business cost function.",
            f"Tuned max-F2 threshold: {threshold['threshold']}. Precision "
            f"{threshold['precision']}, "
            f"recall {threshold['recall']}, F2 {threshold['f2']}.",
            "",
            "## Fixed vs Tuned Comparison",
            f"Fixed RF ordinary OOF AP: {fixed['average_precision']}. Tuned RF nested-OOF AP: "
            f"{tuned['average_precision']}. These estimates use different development protocols, "
            "so small differences should be interpreted cautiously rather than treated as a "
            "confirmed performance improvement.",
            "",
            "## Promotion Policy",
            "The tuned Random Forest is promoted only if tuned nested-OOF AP exceeds the previous "
            "fixed RF OOF AP by at least 0.005. Otherwise the fixed RF is retained because the "
            "additional complexity is not justified.",
            f"Selected candidate: `{promotion['selected_candidate']}`. Selected threshold: "
            f"{promotion['selected_threshold']}. Reason: {promotion['reason']}",
            "",
            "## Validation Evaluation",
            f"Validation AP: {validation['threshold_independent']['average_precision']}. "
            f"Validation ROC-AUC: {validation['threshold_independent']['roc_auc']}.",
            f"At threshold 0.5, precision {validation_05['precision']}, recall "
            f"{validation_05['recall']}, F1 {validation_05['f1']}, F2 {validation_05['f2']}.",
            "At the selected train-derived threshold, precision "
            f"{validation_selected['precision']}, "
            f"recall {validation_selected['recall']}, F1 {validation_selected['f1']}, "
            f"F2 {validation_selected['f2']}.",
            "",
            "## Limitations",
            "This is not production-ready, does not evaluate the locked test split, and does not "
            "claim performance on real industrial equipment. The search space is intentionally "
            "small and does not constitute broad optimization.",
            "",
            "## Next Steps",
            "The full-train grid search selected "
            f"`{parameter_label(full_train['best_hyperparameters'])}` "
            "for a possible tuned candidate. Later phases may make a final model decision, persist "
            "a selected model, add MLflow tracking, add SHAP explainability, and run the locked "
            "test evaluation.",
            "",
        ]
    )


def write_artifacts(
    nested_oof_predictions: pd.DataFrame,
    outer_folds: pd.DataFrame,
    grid_results: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    metrics: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
    root: Path | None = None,
) -> RandomForestTuningArtifacts:
    nested_oof_path = nested_oof_predictions_path(root)
    outer_path = outer_folds_path(root)
    grid_path = grid_results_path(root)
    threshold_path = threshold_analysis_path(root)
    metrics_path = tuning_metrics_path(root)
    validation_path = validation_predictions_path(root)
    markdown_path = tuning_doc_path(root)

    nested_oof_path.parent.mkdir(parents=True, exist_ok=True)
    nested_oof_predictions.to_csv(nested_oof_path, index=False, float_format="%.10f")
    outer_folds.to_csv(outer_path, index=False, float_format="%.6f")
    grid_results.to_csv(grid_path, index=False, float_format="%.6f")
    threshold_analysis.to_csv(threshold_path, index=False, float_format="%.6f")
    write_json(metrics, metrics_path)
    validation_predictions.to_csv(validation_path, index=False, float_format="%.10f")
    plot_paths = create_plots(
        nested_oof_predictions, threshold_analysis, grid_results, metrics, root
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(metrics, outer_folds), encoding="utf-8")
    return RandomForestTuningArtifacts(
        nested_oof_predictions_csv=nested_oof_path,
        outer_folds_csv=outer_path,
        grid_results_csv=grid_path,
        threshold_analysis_csv=threshold_path,
        metrics_json=metrics_path,
        validation_predictions_csv=validation_path,
        markdown_report=markdown_path,
        plot_paths=plot_paths,
    )


def run_random_forest_tuning_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
    *,
    root: Path | None = None,
    grid_n_jobs: int = -1,
    progress_callback: ProgressCallback | None = None,
) -> RandomForestTuningResult:
    ai4i_baseline.validate_training_inputs(train_df, validation_df, config, split_summary)
    fixed_reference = load_fixed_random_forest_reference(root)
    nested = generate_nested_oof_predictions(
        train_df,
        config,
        grid_n_jobs=grid_n_jobs,
        progress_callback=progress_callback,
    )
    nested_oof_metrics = build_nested_oof_metrics(nested.oof_predictions)
    threshold_analysis = build_threshold_analysis(nested.oof_predictions)
    threshold_candidates = build_threshold_candidates(threshold_analysis)
    full_search = run_full_train_grid_search(train_df, config, grid_n_jobs=grid_n_jobs)
    grid_results = grid_results_dataframe(full_search)
    full_train_summary = full_train_search_summary(full_search, grid_results)
    promotion_decision = select_promotion_candidate(
        fixed_reference,
        nested_oof_metrics,
        threshold_candidates,
    )
    validation_metrics, validation_predictions, _ = evaluate_validation(
        train_df,
        validation_df,
        config,
        str(promotion_decision["selected_candidate"]),
        full_train_summary["best_hyperparameters"],
        float(promotion_decision["selected_threshold"]),
    )
    metrics = build_metrics_summary(
        train_df,
        validation_df,
        config,
        fixed_reference,
        nested_oof_metrics,
        threshold_candidates,
        full_train_summary,
        promotion_decision,
        validation_metrics,
    )
    artifacts = write_artifacts(
        nested.oof_predictions,
        nested.outer_folds,
        grid_results,
        threshold_analysis,
        metrics,
        validation_predictions,
        root,
    )
    return RandomForestTuningResult(
        metrics=metrics,
        nested_oof_predictions=nested.oof_predictions,
        outer_folds=nested.outer_folds,
        grid_results=grid_results,
        threshold_analysis=threshold_analysis,
        validation_predictions=validation_predictions,
        artifacts=artifacts,
    )

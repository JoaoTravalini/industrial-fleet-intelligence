"""Validation-only AI4I baseline classification utilities."""

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocessing import ai4i_modeling

TRAIN_RELATIVE_PATH = Path("data") / "processed" / "ai4i" / "train.csv"
VALIDATION_RELATIVE_PATH = Path("data") / "processed" / "ai4i" / "validation.csv"
REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
BASELINE_ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "baseline"
BASELINE_DOC_RELATIVE_PATH = Path("docs") / "ml" / "ai4i_baseline.md"
BASELINE_METRICS_FILENAME = "baseline_metrics.json"
BASELINE_PREDICTIONS_FILENAME = "baseline_validation_predictions.csv"
LOGISTIC_COEFFICIENTS_FILENAME = "logistic_regression_coefficients.csv"
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = 1000
POSITIVE_CLASS = 1
DECISION_THRESHOLD_POLICY = "Default estimator decision behavior; no threshold tuning."
TEST_SET_STATUS = "LOCKED / NOT USED"
REQUIRED_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
)


@dataclass(frozen=True)
class BaselinePipelines:
    """Fitted baseline model pipelines."""

    dummy_classifier: Pipeline
    logistic_regression: Pipeline


@dataclass(frozen=True)
class BaselineArtifacts:
    """Paths produced by a baseline training run."""

    metrics_json: Path
    validation_predictions_csv: Path
    logistic_coefficients_csv: Path
    plot_paths: list[Path]
    markdown_report: Path


@dataclass(frozen=True)
class BaselineResult:
    """Full result returned by the baseline training runner."""

    metrics: dict[str, Any]
    validation_predictions: pd.DataFrame
    logistic_coefficients: pd.DataFrame
    artifacts: BaselineArtifacts
    pipelines: BaselinePipelines


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def train_path(root: Path | None = None) -> Path:
    return (root or project_root()) / TRAIN_RELATIVE_PATH


def validation_path(root: Path | None = None) -> Path:
    return (root or project_root()) / VALIDATION_RELATIVE_PATH


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def baseline_assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / BASELINE_ASSETS_RELATIVE_DIR


def baseline_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / BASELINE_DOC_RELATIVE_PATH


def baseline_metrics_path(root: Path | None = None) -> Path:
    return reports_directory(root) / BASELINE_METRICS_FILENAME


def baseline_predictions_path(root: Path | None = None) -> Path:
    return reports_directory(root) / BASELINE_PREDICTIONS_FILENAME


def logistic_coefficients_path(root: Path | None = None) -> Path:
    return reports_directory(root) / LOGISTIC_COEFFICIENTS_FILENAME


def baseline_plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets_dir = baseline_assets_directory(root)
    return {
        "logistic_confusion_matrix": assets_dir / "logistic_confusion_matrix.png",
        "precision_recall_curve": assets_dir / "precision_recall_curve.png",
        "roc_curve": assets_dir / "roc_curve.png",
        "logistic_top_coefficients": assets_dir / "logistic_top_coefficients.png",
    }


def baseline_input_paths(root: Path | None = None) -> dict[str, Path]:
    return {"train": train_path(root), "validation": validation_path(root)}


def predictive_feature_columns(config: ai4i_modeling.ModelingConfig) -> list[str]:
    return ai4i_modeling.predictive_feature_columns(config)


def forbidden_model_feature_columns(config: ai4i_modeling.ModelingConfig) -> set[str]:
    return {
        "UDI",
        config.derived_traceability_field,
        *config.traceability_fields,
        *config.excluded_identifiers,
        *config.excluded_leakage_sensitive_columns,
        *config.forbidden_feature_sources,
    }


def validate_feature_policy(config: ai4i_modeling.ModelingConfig) -> None:
    ai4i_modeling.validate_modeling_config(config)
    ai4i_modeling.validate_forbidden_feature_columns(config)
    feature_columns = predictive_feature_columns(config)
    violations = sorted(set(feature_columns) & forbidden_model_feature_columns(config))
    if violations:
        raise ValueError("Forbidden model feature(s): " + ", ".join(violations))


def expected_split_rows(summary: Mapping[str, Any], split_name: str) -> int:
    split_rows = summary.get("split_rows")
    if not isinstance(split_rows, Mapping) or split_name not in split_rows:
        raise ValueError(f"Missing `{split_name}` row count in modeling split summary.")
    return int(split_rows[split_name])


def load_split_frame(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {split_name} split artifact was not found: {path}")
    frame = pd.read_csv(path)
    if ai4i_modeling.SOURCE_UDI_COLUMN not in frame.columns:
        raise ValueError(f"{split_name} split is missing source_udi.")
    return frame.sort_values(ai4i_modeling.SOURCE_UDI_COLUMN, kind="mergesort").reset_index(
        drop=True
    )


def load_training_and_validation_frames(
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = baseline_input_paths(root)
    return load_split_frame(paths["train"], "train"), load_split_frame(
        paths["validation"], "validation"
    )


def validate_split_frame(
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_name: str,
    expected_rows: int,
) -> None:
    expected_columns = ai4i_modeling.modeling_frame_columns(config)
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{split_name} split has unexpected columns. Expected: " + ", ".join(expected_columns)
        )
    forbidden_present = sorted(
        column
        for column in [
            "UDI",
            *config.excluded_identifiers,
            *config.excluded_leakage_sensitive_columns,
        ]
        if column in frame.columns
    )
    if forbidden_present:
        raise ValueError(
            f"{split_name} split contains forbidden column(s): " + ", ".join(forbidden_present)
        )
    if len(frame) != expected_rows:
        raise ValueError(f"{split_name} split expected {expected_rows} rows, found {len(frame)}.")
    if frame[config.derived_traceability_field].duplicated().any():
        raise ValueError(f"{split_name} split contains duplicated source_udi values.")
    if frame.isna().any().any():
        raise ValueError(f"{split_name} split contains missing values.")
    target_values = set(frame[config.target_column].unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"{split_name} target must contain binary values 0 and 1.")


def validate_training_inputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
) -> None:
    validate_feature_policy(config)
    validate_split_frame(train_df, config, "train", expected_split_rows(split_summary, "train"))
    validate_split_frame(
        validation_df,
        config,
        "validation",
        expected_split_rows(split_summary, "validation"),
    )
    train_udis = set(train_df[config.derived_traceability_field].tolist())
    validation_udis = set(validation_df[config.derived_traceability_field].tolist())
    if train_udis & validation_udis:
        raise ValueError("Train and validation source_udi values must not overlap.")


def extract_features_and_target(
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_feature_policy(config)
    feature_columns = predictive_feature_columns(config)
    forbidden_in_features = sorted(set(feature_columns) & forbidden_model_feature_columns(config))
    if forbidden_in_features:
        raise ValueError("Forbidden feature column(s): " + ", ".join(forbidden_in_features))
    return frame.loc[:, feature_columns].copy(), frame.loc[:, config.target_column].copy()


def build_preprocessor(config: ai4i_modeling.ModelingConfig) -> ColumnTransformer:
    validate_feature_policy(config)
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


def build_dummy_pipeline(config: ai4i_modeling.ModelingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic_regression_pipeline(config: ai4i_modeling.ModelingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            (
                "classifier",
                LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=RANDOM_SEED),
            ),
        ]
    )


def fit_baseline_pipelines(
    train_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> BaselinePipelines:
    x_train, y_train = extract_features_and_target(train_df, config)
    dummy_pipeline = build_dummy_pipeline(config)
    logistic_pipeline = build_logistic_regression_pipeline(config)
    dummy_pipeline.fit(x_train, y_train)
    logistic_pipeline.fit(x_train, y_train)
    return BaselinePipelines(
        dummy_classifier=dummy_pipeline,
        logistic_regression=logistic_pipeline,
    )


def predict_with_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_frame, _ = extract_features_and_target(frame, config)
    predictions = pipeline.predict(x_frame).astype(int)
    probabilities = pipeline.predict_proba(x_frame)[:, POSITIVE_CLASS]
    return predictions, probabilities


def _rounded_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_prediction: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_prediction, labels=[0, 1])
    metrics = {
        "accuracy": _rounded_float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": _rounded_float(balanced_accuracy_score(y_true, y_prediction)),
        "precision": _rounded_float(
            precision_score(y_true, y_prediction, pos_label=1, zero_division=0)
        ),
        "recall": _rounded_float(recall_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "f1": _rounded_float(f1_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "roc_auc": _rounded_float(roc_auc_score(y_true, y_probability)),
        "average_precision": _rounded_float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
    }
    return metrics


def create_validation_predictions(
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    dummy_prediction: np.ndarray,
    dummy_probability: np.ndarray,
    logistic_prediction: np.ndarray,
    logistic_probability: np.ndarray,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "source_udi": validation_df[config.derived_traceability_field].astype(int),
            "target": validation_df[config.target_column].astype(int),
            "dummy_probability": dummy_probability.astype(float),
            "dummy_prediction": dummy_prediction.astype(int),
            "logistic_probability": logistic_probability.astype(float),
            "logistic_prediction": logistic_prediction.astype(int),
        }
    )
    return predictions.sort_values("source_udi", kind="mergesort").reset_index(drop=True)


def extract_logistic_coefficients(
    logistic_pipeline: Pipeline,
    config: ai4i_modeling.ModelingConfig,
) -> pd.DataFrame:
    preprocessor = logistic_pipeline.named_steps["preprocessor"]
    classifier = logistic_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_names.astype(str),
            "coefficient": coefficients.astype(float),
        }
    )
    coefficient_frame["absolute_coefficient"] = coefficient_frame["coefficient"].abs()
    return coefficient_frame.sort_values(
        ["absolute_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def build_metrics_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    dummy_metrics: Mapping[str, Any],
    logistic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    feature_list = predictive_feature_columns(config)
    train_positive_count = int(train_df[config.target_column].sum())
    validation_positive_count = int(validation_df[config.target_column].sum())
    return {
        "experiment": {
            "name": "AI4I baseline classification",
            "objective": "Binary classification of Machine failure",
            "phase": "validation-only baseline",
            "models": ['DummyClassifier(strategy="prior")', "LogisticRegression"],
            "test_set_status": TEST_SET_STATUS,
        },
        "data": {
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "training_positive_count": train_positive_count,
            "validation_positive_count": validation_positive_count,
            "validation_positive_percentage": _rounded_float(
                validation_positive_count / len(validation_df) * 100
            ),
            "target_column": config.target_column,
            "predictive_feature_list": feature_list,
            "categorical_features": list(config.categorical_features),
            "numerical_features": list(config.numerical_features),
            "traceability_field": config.derived_traceability_field,
            "excluded_identifiers": list(config.excluded_identifiers),
            "excluded_leakage_sensitive_fields": list(config.excluded_leakage_sensitive_columns),
            "splits_used": ["train", "validation"],
            "test_data_used": False,
        },
        "preprocessing": {
            "fit_split": "train",
            "validation_policy": (
                "Validation data transformed only by the fitted training pipeline."
            ),
            "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
            "numerical": "Five process variables -> StandardScaler()",
        },
        "decision_threshold_policy": DECISION_THRESHOLD_POLICY,
        "random_seed": RANDOM_SEED,
        "dummy_classifier": dict(dummy_metrics),
        "logistic_regression": {
            "configuration": {
                "max_iter": LOGISTIC_MAX_ITER,
                "class_weight": None,
                "solver": "lbfgs",
            },
            **dict(logistic_metrics),
        },
    }


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_logistic_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    values = np.array(matrix, dtype=int)
    _, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(values)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("AI4I Logistic Regression Confusion Matrix")
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


def plot_precision_recall_curve(
    y_true: pd.Series | np.ndarray,
    dummy_probability: np.ndarray,
    logistic_probability: np.ndarray,
    path: Path,
) -> None:
    dummy_precision, dummy_recall, _ = precision_recall_curve(y_true, dummy_probability)
    logistic_precision, logistic_recall, _ = precision_recall_curve(y_true, logistic_probability)
    prevalence = float(np.mean(y_true))
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(dummy_recall, dummy_precision, label="Dummy")
    axis.plot(logistic_recall, logistic_precision, label="Logistic Regression")
    axis.axhline(prevalence, linestyle="--", label=f"Positive prevalence ({prevalence:.3f})")
    axis.set_title("AI4I Validation Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _save_figure(path)


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    dummy_probability: np.ndarray,
    logistic_probability: np.ndarray,
    path: Path,
) -> None:
    dummy_fpr, dummy_tpr, _ = roc_curve(y_true, dummy_probability)
    logistic_fpr, logistic_tpr, _ = roc_curve(y_true, logistic_probability)
    _, axis = plt.subplots(figsize=(7, 5))
    axis.plot(dummy_fpr, dummy_tpr, label="Dummy")
    axis.plot(logistic_fpr, logistic_tpr, label="Logistic Regression")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier reference")
    axis.set_title("AI4I Validation ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    _save_figure(path)


def plot_top_coefficients(coefficients: pd.DataFrame, path: Path, limit: int = 12) -> None:
    top = coefficients.head(limit).sort_values("absolute_coefficient", ascending=True)
    _, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top["feature"], top["coefficient"])
    axis.set_title("AI4I Logistic Regression Strongest Coefficients")
    axis.set_xlabel("Coefficient")
    axis.set_ylabel("Transformed feature")
    _save_figure(path)


def create_plots(
    y_true: pd.Series,
    dummy_probability: np.ndarray,
    logistic_probability: np.ndarray,
    logistic_metrics: Mapping[str, Any],
    coefficients: pd.DataFrame,
    root: Path | None = None,
) -> list[Path]:
    paths = baseline_plot_paths(root)
    plot_logistic_confusion_matrix(
        logistic_metrics["confusion_matrix"], paths["logistic_confusion_matrix"]
    )
    plot_precision_recall_curve(
        y_true, dummy_probability, logistic_probability, paths["precision_recall_curve"]
    )
    plot_roc_curve(y_true, dummy_probability, logistic_probability, paths["roc_curve"])
    plot_top_coefficients(coefficients, paths["logistic_top_coefficients"])
    return [paths[name] for name in sorted(paths)]


def render_baseline_markdown(metrics: Mapping[str, Any], coefficients: pd.DataFrame) -> str:
    dummy = metrics["dummy_classifier"]
    logistic = metrics["logistic_regression"]
    data = metrics["data"]
    top_coefficients = coefficients.head(8)
    coefficient_lines = [
        f"- `{row.feature}`: coefficient {row.coefficient:.6f} "
        f"(absolute {row.absolute_coefficient:.6f})."
        for row in top_coefficients.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# AI4I Baseline Classification",
            "",
            "## Objective",
            "This phase establishes the first validation-only baseline for binary "
            "classification of `Machine failure` using the leakage-safe AI4I modeling dataset.",
            "",
            "## Experimental Design",
            "Only `train.csv` and `validation.csv` are used. The training split fits preprocessing "
            "and model parameters; the validation split is used only for baseline evaluation.",
            f"Training rows: {data['training_rows']}. Validation rows: {data['validation_rows']}.",
            "",
            "## Leakage Prevention",
            "The predictive feature list is restricted to `Type`, `Air temperature [K]`, "
            "`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and "
            "`Tool wear [min]`. `source_udi` is traceability-only and is never passed to a model. "
            "`Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded.",
            "",
            "## Locked Test Set",
            "The test set remains locked and was not loaded, evaluated, summarized, or used for "
            "prediction in this phase. It is reserved for future final evaluation after model "
            "selection is complete.",
            "",
            "## Preprocessing",
            "A scikit-learn `ColumnTransformer` is fitted inside each pipeline on "
            "training data only. "
            '`Type` is encoded with `OneHotEncoder(handle_unknown="ignore")`; the five numerical '
            "process variables are standardized with `StandardScaler()`.",
            "",
            "## Dummy Baseline",
            '`DummyClassifier(strategy="prior")` provides a trivial benchmark representing the '
            "target class distribution.",
            "",
            "## Logistic Regression Baseline",
            "`LogisticRegression` is the first real predictive baseline. It uses a conservative "
            "configuration with no class weighting, no resampling, no threshold tuning, and no "
            "hyperparameter search.",
            "",
            "## Validation Metrics",
            "Accuracy is not the primary comparison metric because the target is highly "
            "imbalanced. "
            "Average Precision (AP), recall, precision, F1, balanced accuracy, "
            "and ROC-AUC are more "
            "useful for this baseline review.",
            "",
            "| Model | AP | ROC-AUC | Balanced accuracy | Precision | Recall | F1 | Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Dummy | {dummy['average_precision']} | {dummy['roc_auc']} | "
            f"{dummy['balanced_accuracy']} | {dummy['precision']} | {dummy['recall']} | "
            f"{dummy['f1']} | {dummy['accuracy']} |",
            f"| Logistic Regression | {logistic['average_precision']} | {logistic['roc_auc']} | "
            f"{logistic['balanced_accuracy']} | {logistic['precision']} | "
            f"{logistic['recall']} | {logistic['f1']} | {logistic['accuracy']} |",
            "",
            f"Validation positives: {data['validation_positive_count']} "
            f"({data['validation_positive_percentage']}%). Average Precision (AP) is especially "
            "useful here because only about 3.4% of validation observations are positive.",
            "",
            "## Class Imbalance",
            "No class balancing has been applied. This baseline intentionally measures "
            "natural model "
            "behavior before any class weighting, resampling, or threshold optimization.",
            "",
            "## Logistic Regression Coefficients",
            "The strongest fitted Logistic Regression coefficients by absolute value are:",
            *coefficient_lines,
            "",
            "Coefficient magnitude is model-specific. Numerical variables are standardized, `Type` "
            "is one-hot encoded, and coefficients describe associations within this "
            "fitted baseline. "
            "They do not establish causality and are not used for automatic feature elimination.",
            "",
            "## Key Observations",
            "Logistic Regression improves Average Precision over the trivial Dummy baseline. "
            "Default-threshold recall and precision should be interpreted cautiously because no "
            "threshold tuning or imbalance strategy has been applied.",
            "",
            "## Limitations",
            "This is not production-ready and does not claim generalization to real industrial "
            "machines. AI4I is a public synthetic dataset and is not proprietary or official "
            "third-party equipment data.",
            "",
            "## Next Steps",
            "Future phases may evaluate class imbalance strategies, threshold policies, advanced "
            "classifiers, model selection, MLflow tracking, explainability, and final locked test "
            "evaluation. None of those steps are implemented in this phase.",
            "",
        ]
    )


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def write_artifacts(
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    y_true: pd.Series,
    dummy_probability: np.ndarray,
    logistic_probability: np.ndarray,
    root: Path | None = None,
) -> BaselineArtifacts:
    metrics_json = baseline_metrics_path(root)
    validation_predictions_csv = baseline_predictions_path(root)
    logistic_coefficients_csv = logistic_coefficients_path(root)
    markdown_report = baseline_doc_path(root)

    write_json(metrics, metrics_json)
    validation_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(validation_predictions_csv, index=False, float_format="%.10f")
    logistic_coefficients_csv.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(logistic_coefficients_csv, index=False, float_format="%.10f")
    plot_paths = create_plots(
        y_true,
        dummy_probability,
        logistic_probability,
        metrics["logistic_regression"],
        coefficients,
        root,
    )
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    markdown_report.write_text(render_baseline_markdown(metrics, coefficients), encoding="utf-8")
    return BaselineArtifacts(
        metrics_json=metrics_json,
        validation_predictions_csv=validation_predictions_csv,
        logistic_coefficients_csv=logistic_coefficients_csv,
        plot_paths=plot_paths,
        markdown_report=markdown_report,
    )


def run_baseline_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: ai4i_modeling.ModelingConfig,
    split_summary: Mapping[str, Any],
    root: Path | None = None,
) -> BaselineResult:
    validate_training_inputs(train_df, validation_df, config, split_summary)
    pipelines = fit_baseline_pipelines(train_df, config)
    _, y_validation = extract_features_and_target(validation_df, config)

    dummy_prediction, dummy_probability = predict_with_pipeline(
        pipelines.dummy_classifier, validation_df, config
    )
    logistic_prediction, logistic_probability = predict_with_pipeline(
        pipelines.logistic_regression, validation_df, config
    )

    dummy_metrics = calculate_metrics(y_validation, dummy_prediction, dummy_probability)
    logistic_metrics = calculate_metrics(y_validation, logistic_prediction, logistic_probability)
    predictions = create_validation_predictions(
        validation_df,
        config,
        dummy_prediction,
        dummy_probability,
        logistic_prediction,
        logistic_probability,
    )
    coefficients = extract_logistic_coefficients(pipelines.logistic_regression, config)
    metrics = build_metrics_summary(
        train_df,
        validation_df,
        config,
        dummy_metrics,
        logistic_metrics,
    )
    artifacts = write_artifacts(
        metrics,
        predictions,
        coefficients,
        y_validation,
        dummy_probability,
        logistic_probability,
        root,
    )
    return BaselineResult(
        metrics=metrics,
        validation_predictions=predictions,
        logistic_coefficients=coefficients,
        artifacts=artifacts,
        pipelines=pipelines,
    )

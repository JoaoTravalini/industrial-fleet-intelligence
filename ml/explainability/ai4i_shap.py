"""SHAP explainability for the packaged AI4I Random Forest model."""

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
import shap
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.inference import ai4i_predictor
from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

EXPECTED_SHAP_VERSION = "0.52.0"
GLOBAL_SAMPLE_SIZE = 1000
RANDOM_STATE = 42
POSITIVE_CLASS = ai4i_baseline.POSITIVE_CLASS
ADDITIVITY_TOLERANCE = 1e-6
EXPLAINED_OUTPUT = "Machine failure positive-class model output"
TEST_SET_STATUS_MESSAGE = "TEST SET STATUS: FINAL EVALUATION COMPLETE / NOT USED FOR EXPLAINABILITY"
REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
ASSETS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i" / "explainability"
TRANSFORMED_IMPORTANCE_FILENAME = "shap_transformed_feature_importance.csv"
GROUPED_IMPORTANCE_FILENAME = "shap_grouped_feature_importance.csv"
LOCAL_EXPLANATIONS_FILENAME = "shap_local_explanations.json"
SAMPLE_EXPLANATIONS_FILENAME = "shap_sample_inference_explanations.json"
SUMMARY_FILENAME = "shap_explainability_summary.json"
GROUPED_TYPE_FEATURE = "Type"
REPRESENTATIVE_CASE_NAMES = ("low_risk", "threshold_near", "high_risk")
WATERFALL_FILENAMES = {
    "low_risk": "low_risk_waterfall.png",
    "threshold_near": "threshold_near_waterfall.png",
    "high_risk": "high_risk_waterfall.png",
}


@dataclass(frozen=True)
class ModelComponents:
    """Fitted model components used for SHAP analysis."""

    preprocessor: ColumnTransformer
    classifier: RandomForestClassifier
    transformed_feature_names: list[str]


@dataclass(frozen=True)
class PositiveClassShapResult:
    """Positive-class SHAP values and additivity diagnostics."""

    values: np.ndarray
    base_value: float
    model_outputs: np.ndarray
    additivity_errors: np.ndarray
    explanation: shap.Explanation


@dataclass(frozen=True)
class RepresentativeCase:
    """A deterministic development observation selected for local explanation."""

    case_name: str
    row_index: int
    source_udi: int
    failure_probability: float
    failure_prediction: int


@dataclass(frozen=True)
class ExplainabilityResult:
    """Artifacts and metadata produced by the explainability runner."""

    transformed_feature_count: int
    grouped_feature_count: int
    global_sample_size: int
    grouped_importance: list[dict[str, Any]]
    local_explanations: dict[str, Any]
    sample_explanations: dict[str, Any]
    summary: dict[str, Any]
    plot_paths: list[Path]
    max_additivity_error: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def assets_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / ASSETS_RELATIVE_DIR


def transformed_importance_path(root: Path | None = None) -> Path:
    return reports_directory(root) / TRANSFORMED_IMPORTANCE_FILENAME


def grouped_importance_path(root: Path | None = None) -> Path:
    return reports_directory(root) / GROUPED_IMPORTANCE_FILENAME


def local_explanations_path(root: Path | None = None) -> Path:
    return reports_directory(root) / LOCAL_EXPLANATIONS_FILENAME


def sample_explanations_path(root: Path | None = None) -> Path:
    return reports_directory(root) / SAMPLE_EXPLANATIONS_FILENAME


def summary_path(root: Path | None = None) -> Path:
    return reports_directory(root) / SUMMARY_FILENAME


def plot_paths(root: Path | None = None) -> dict[str, Path]:
    assets = assets_directory(root)
    return {
        "global_importance": assets / "shap_global_importance.png",
        "beeswarm": assets / "shap_beeswarm.png",
        **{case_name: assets / filename for case_name, filename in WATERFALL_FILENAMES.items()},
    }


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return data


def rounded_float(value: float | np.floating[Any], digits: int = 12) -> float:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError("Expected a finite numeric value.")
    return round(numeric_value, digits)


def expected_grouped_features(final_config: Mapping[str, Any]) -> list[str]:
    return [GROUPED_TYPE_FEATURE, *list(final_config["numerical_features"])]


def validate_packaged_model_identity(
    predictor: ai4i_predictor.AI4IPredictor,
    root: Path,
) -> dict[str, Any]:
    packaging_summary = load_json(ai4i_predictor.packaging_summary_path(root))
    expected_threshold = float(predictor.final_config["decision_threshold"])
    checks = {
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_version": ai4i_predictor.MODEL_VERSION,
        "final_config_hash": predictor.final_config_hash,
        "frozen_threshold": expected_threshold,
    }
    mismatches = [key for key, expected in checks.items() if packaging_summary.get(key) != expected]
    if mismatches:
        raise ValueError(
            "Packaged model summary does not match trusted predictor identity: "
            + ", ".join(mismatches)
        )
    if expected_threshold != 0.14:
        raise ValueError("Frozen AI4I decision threshold must remain 0.14.")
    return packaging_summary


def load_trusted_predictor(root: Path | None = None) -> ai4i_predictor.AI4IPredictor:
    root_path = root or project_root()
    model_path = ai4i_predictor.artifact_path(root_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Packaged AI4I model artifact is missing. Run "
            ".\\.venv\\Scripts\\python.exe scripts/package_ai4i_final_model.py"
        )
    predictor = ai4i_predictor.load_predictor(root_path)
    validate_packaged_model_identity(predictor, root_path)
    return predictor


def extract_model_components(
    pipeline: Pipeline,
    final_config: Mapping[str, Any],
) -> ModelComponents:
    validated_pipeline = ai4i_predictor.validate_pipeline_structure(pipeline, final_config)
    preprocessor = validated_pipeline.named_steps["preprocessor"]
    classifier = validated_pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError("Packaged pipeline preprocessor must be a fitted ColumnTransformer.")
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Packaged pipeline classifier must be a RandomForestClassifier.")
    return ModelComponents(
        preprocessor=preprocessor,
        classifier=classifier,
        transformed_feature_names=transformed_feature_names(preprocessor),
    )


def presentation_feature_name(raw_name: str) -> str:
    name = str(raw_name)
    for prefix in ("categorical__", "numerical__", "remainder__"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
    if "__" in name:
        name = name.split("__")[-1]
    return name


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = [presentation_feature_name(name) for name in preprocessor.get_feature_names_out()]
    if not names:
        raise ValueError("No transformed feature names were produced by the fitted preprocessor.")
    if len(names) != len(set(names)):
        raise ValueError("Transformed feature names must be unique after presentation cleanup.")
    return names


def transform_model_inputs(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
) -> np.ndarray:
    transformed = preprocessor.transform(features)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    values = np.asarray(transformed, dtype=float)
    if values.ndim != 2:
        raise ValueError("Transformed model input must be a two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Transformed model input contains non-finite values.")
    return values


def positive_class_base_value(expected_value: Any, positive_class: int = POSITIVE_CLASS) -> float:
    values = np.asarray(expected_value, dtype=float)
    if values.ndim == 0:
        raise ValueError("Binary SHAP expected value must expose a class dimension.")
    if values.shape[0] <= positive_class:
        raise ValueError("Binary SHAP expected value does not include the positive class.")
    return float(values[positive_class])


def select_positive_class_shap_values(
    shap_values: Any,
    expected_value: Any,
    *,
    sample_count: int,
    feature_count: int,
    positive_class: int = POSITIVE_CLASS,
) -> tuple[np.ndarray, float]:
    if isinstance(shap_values, list | tuple):
        if len(shap_values) <= positive_class:
            raise ValueError("SHAP output does not include the positive class.")
        values = np.asarray(shap_values[positive_class], dtype=float)
        if values.shape != (sample_count, feature_count):
            raise ValueError("Positive-class SHAP array has an unexpected shape.")
        return values, positive_class_base_value(expected_value, positive_class)

    values_array = np.asarray(shap_values, dtype=float)
    if values_array.ndim != 3:
        raise ValueError(
            "Unsupported SHAP output shape. Expected an explicit binary-class output dimension."
        )
    if (
        values_array.shape[0] == sample_count
        and values_array.shape[1] == feature_count
        and values_array.shape[2] > positive_class
    ):
        return values_array[:, :, positive_class], positive_class_base_value(
            expected_value,
            positive_class,
        )
    if (
        values_array.shape[0] > positive_class
        and values_array.shape[1] == sample_count
        and values_array.shape[2] == feature_count
    ):
        return values_array[positive_class, :, :], positive_class_base_value(
            expected_value,
            positive_class,
        )
    raise ValueError("Unsupported SHAP output shape for positive-class extraction.")


def positive_class_probabilities(
    classifier: RandomForestClassifier,
    transformed_features: np.ndarray,
) -> np.ndarray:
    classes = list(classifier.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError("Random Forest classes do not include positive class 1.")
    positive_index = classes.index(POSITIVE_CLASS)
    probabilities = classifier.predict_proba(transformed_features)[:, positive_index]
    values = np.asarray(probabilities, dtype=float)
    if not np.all((0 <= values) & (values <= 1)):
        raise ValueError("Positive-class probabilities must be within [0, 1].")
    return values


def additivity_errors(
    shap_values: np.ndarray,
    base_value: float,
    model_outputs: np.ndarray,
) -> np.ndarray:
    reconstructed = float(base_value) + np.asarray(shap_values, dtype=float).sum(axis=1)
    return np.abs(reconstructed - np.asarray(model_outputs, dtype=float))


def explain_positive_class(
    classifier: RandomForestClassifier,
    transformed_features: np.ndarray,
    feature_names: Sequence[str],
) -> PositiveClassShapResult:
    if transformed_features.shape[1] != len(feature_names):
        raise ValueError("Transformed feature matrix width does not match feature names.")
    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed_features)
    values, base_value = select_positive_class_shap_values(
        raw_values,
        explainer.expected_value,
        sample_count=transformed_features.shape[0],
        feature_count=transformed_features.shape[1],
    )
    model_outputs = positive_class_probabilities(classifier, transformed_features)
    errors = additivity_errors(values, base_value, model_outputs)
    max_error = float(np.max(errors)) if len(errors) else 0.0
    if max_error > ADDITIVITY_TOLERANCE:
        raise ValueError(
            f"Positive-class SHAP additivity error {max_error:.12g} exceeds tolerance "
            f"{ADDITIVITY_TOLERANCE:.12g}."
        )
    explanation = shap.Explanation(
        values=values,
        base_values=np.full(transformed_features.shape[0], base_value),
        data=transformed_features,
        feature_names=list(feature_names),
    )
    return PositiveClassShapResult(
        values=values,
        base_value=float(base_value),
        model_outputs=model_outputs,
        additivity_errors=errors,
        explanation=explanation,
    )


def load_development_data(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, ai4i_modeling.ModelingConfig]:
    modeling_config = ai4i_modeling.load_modeling_config(ai4i_modeling.config_path(root))
    development_df = ai4i_predictor.load_development_training_data(root, modeling_config)
    feature_columns = list(ai4i_modeling.predictive_feature_columns(modeling_config))
    features = development_df.loc[:, feature_columns].copy()
    return development_df, features, modeling_config


def deterministic_global_sample(
    development_df: pd.DataFrame,
    final_config: Mapping[str, Any],
    *,
    sample_size: int = GLOBAL_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    traceability_field = str(final_config["traceability_field"])
    row_count = min(sample_size, len(development_df))
    sampled = development_df.sample(n=row_count, random_state=random_state)
    return sampled.sort_values(traceability_field, kind="mergesort").reset_index(drop=True)


def group_name_for_transformed_feature(
    feature_name: str,
    final_config: Mapping[str, Any],
) -> str:
    if feature_name in final_config["numerical_features"]:
        return feature_name
    categorical_prefixes = tuple(f"{feature}_" for feature in final_config["categorical_features"])
    if feature_name == GROUPED_TYPE_FEATURE or feature_name.startswith(categorical_prefixes):
        return GROUPED_TYPE_FEATURE
    raise ValueError(f"Unsupported transformed feature for grouping: {feature_name}")


def grouped_contribution_matrix(
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    final_config: Mapping[str, Any],
) -> tuple[list[str], np.ndarray]:
    grouped_features = expected_grouped_features(final_config)
    grouped_index = {feature: index for index, feature in enumerate(grouped_features)}
    grouped_values = np.zeros((shap_values.shape[0], len(grouped_features)), dtype=float)
    for column_index, feature_name in enumerate(feature_names):
        group_name = group_name_for_transformed_feature(feature_name, final_config)
        grouped_values[:, grouped_index[group_name]] += shap_values[:, column_index]
    return grouped_features, grouped_values


def ranked_mean_absolute_importance(
    feature_names: Sequence[str],
    shap_values: np.ndarray,
) -> list[dict[str, Any]]:
    mean_values = np.abs(np.asarray(shap_values, dtype=float)).mean(axis=0)
    rows = [
        {"feature": feature, "mean_absolute_shap": float(value)}
        for feature, value in zip(feature_names, mean_values, strict=True)
    ]
    rows.sort(key=lambda item: (-item["mean_absolute_shap"], item["feature"]))
    for rank, row in enumerate(rows, start=1):
        row["mean_absolute_shap"] = rounded_float(row["mean_absolute_shap"])
        row["rank"] = rank
    return rows


def write_importance_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["feature", "mean_absolute_shap", "rank"])
    frame.to_csv(path, index=False, float_format="%.12g")


def probability_to_prediction(probability: float, threshold: float) -> int:
    return int(float(probability) >= float(threshold))


def select_representative_cases(
    development_df: pd.DataFrame,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    traceability_field: str,
) -> list[RepresentativeCase]:
    selection_frame = pd.DataFrame(
        {
            "row_index": development_df.index.to_numpy(dtype=int),
            "source_udi": development_df[traceability_field].to_numpy(dtype=int),
            "failure_probability": np.asarray(probabilities, dtype=float),
        }
    )
    if not np.all(
        (0 <= selection_frame["failure_probability"])
        & (selection_frame["failure_probability"] <= 1)
    ):
        raise ValueError("Representative selection probabilities must be within [0, 1].")

    low = selection_frame.sort_values(
        ["failure_probability", "source_udi"],
        ascending=[True, True],
        kind="mergesort",
    ).iloc[0]
    near_frame = selection_frame.assign(
        threshold_distance=(selection_frame["failure_probability"] - float(threshold)).abs()
    )
    near = near_frame.sort_values(
        ["threshold_distance", "source_udi"],
        ascending=[True, True],
        kind="mergesort",
    ).iloc[0]
    high = selection_frame.sort_values(
        ["failure_probability", "source_udi"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]

    return [
        representative_case_from_row("low_risk", low, threshold),
        representative_case_from_row("threshold_near", near, threshold),
        representative_case_from_row("high_risk", high, threshold),
    ]


def representative_case_from_row(
    case_name: str,
    row: pd.Series,
    threshold: float,
) -> RepresentativeCase:
    probability = float(row["failure_probability"])
    return RepresentativeCase(
        case_name=case_name,
        row_index=int(row["row_index"]),
        source_udi=int(row["source_udi"]),
        failure_probability=probability,
        failure_prediction=probability_to_prediction(probability, threshold),
    )


def contribution_entries(
    feature_names: Sequence[str],
    contributions: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            "feature": feature,
            "shap_value": rounded_float(value),
            "absolute_shap": rounded_float(abs(float(value))),
        }
        for feature, value in zip(feature_names, contributions, strict=True)
    ]
    rows.sort(key=lambda item: (-item["absolute_shap"], item["feature"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_local_explanation_payload(
    cases: Sequence[RepresentativeCase],
    shap_result: PositiveClassShapResult,
    transformed_feature_names_: Sequence[str],
    grouped_feature_names: Sequence[str],
    grouped_values: np.ndarray,
    predictor: ai4i_predictor.AI4IPredictor,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    threshold = float(predictor.decision_threshold)
    for position, case in enumerate(cases):
        records.append(
            {
                "additivity_check_error": rounded_float(shap_result.additivity_errors[position]),
                "case_name": case.case_name,
                "failure_prediction": int(case.failure_prediction),
                "failure_probability": rounded_float(case.failure_probability, digits=6),
                "final_config_hash": predictor.final_config_hash,
                "frozen_threshold": threshold,
                "grouped_feature_contributions": contribution_entries(
                    grouped_feature_names,
                    grouped_values[position],
                ),
                "model_name": predictor.model_name,
                "model_version": predictor.model_version,
                "positive_class_base_value": rounded_float(shap_result.base_value),
                "source_udi": int(case.source_udi),
                "transformed_feature_contributions": contribution_entries(
                    transformed_feature_names_,
                    shap_result.values[position],
                ),
            }
        )
    return {"cases": records}


def build_sample_explanation_payload(
    predictions: Sequence[Mapping[str, Any]],
    shap_result: PositiveClassShapResult,
    grouped_feature_names: Sequence[str],
    grouped_values: np.ndarray,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        records.append(
            {
                **dict(prediction),
                "additivity_check_error": rounded_float(shap_result.additivity_errors[index]),
                "grouped_feature_contributions": contribution_entries(
                    grouped_feature_names,
                    grouped_values[index],
                ),
                "positive_class_base_value": rounded_float(shap_result.base_value),
                "sample_index": index,
            }
        )
    return {"sample_explanations": records}


def build_summary(
    predictor: ai4i_predictor.AI4IPredictor,
    global_sample_size: int,
    transformed_feature_count: int,
    grouped_feature_count: int,
    grouped_importance: Sequence[Mapping[str, Any]],
    max_additivity_error: float,
) -> dict[str, Any]:
    return {
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "explained_output": EXPLAINED_OUTPUT,
        "explainer_type": "shap.TreeExplainer",
        "final_config_hash": predictor.final_config_hash,
        "global_sample_policy": (
            "Deterministic train plus validation development subset selected with random_state=42; "
            "target labels are not used for SHAP computation."
        ),
        "global_sample_size": int(global_sample_size),
        "grouped_feature_count": int(grouped_feature_count),
        "max_observed_additivity_error": rounded_float(max_additivity_error),
        "model_name": predictor.model_name,
        "model_retrained": False,
        "model_version": predictor.model_version,
        "representative_case_names": list(REPRESENTATIVE_CASE_NAMES),
        "shap_version": shap.__version__,
        "test_data_used": False,
        "top_grouped_features": [dict(item) for item in grouped_importance[:3]],
        "transformed_feature_count": int(transformed_feature_count),
    }


def plot_global_importance(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    ordered = list(reversed(list(rows)))
    labels = [str(row["feature"]) for row in ordered]
    values = [float(row["mean_absolute_shap"]) for row in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(labels, values, color="#4e79a7")
    ax.set_title("Grouped SHAP Attribution Magnitude, Not Causality")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Original conceptual feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_beeswarm(explanation: shap.Explanation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_random_state = np.random.get_state()
    np.random.seed(RANDOM_STATE)
    try:
        plt.figure(figsize=(8.2, 5.2))
        shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names), show=False)
        fig = plt.gcf()
        fig.suptitle("Positive-Class SHAP Beeswarm for Transformed Features", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    finally:
        np.random.set_state(previous_random_state)


def plot_waterfall(
    explanation: shap.Explanation,
    path: Path,
    *,
    case_name: str,
    probability: float,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(explanation, max_display=len(explanation.feature_names), show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.4, 5.2)
    fig.suptitle(
        f"{case_name} positive-class model output | probability={probability:.6f} | "
        f"threshold={threshold:.2f}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_global_reports_and_plots(
    root: Path,
    predictor: ai4i_predictor.AI4IPredictor,
    components: ModelComponents,
    global_features: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PositiveClassShapResult, list[Path]]:
    transformed = transform_model_inputs(components.preprocessor, global_features)
    shap_result = explain_positive_class(
        components.classifier,
        transformed,
        components.transformed_feature_names,
    )
    transformed_importance = ranked_mean_absolute_importance(
        components.transformed_feature_names,
        shap_result.values,
    )
    grouped_features, grouped_values = grouped_contribution_matrix(
        components.transformed_feature_names,
        shap_result.values,
        predictor.final_config,
    )
    grouped_importance = ranked_mean_absolute_importance(grouped_features, grouped_values)
    write_importance_csv(transformed_importance, transformed_importance_path(root))
    write_importance_csv(grouped_importance, grouped_importance_path(root))

    paths = plot_paths(root)
    plot_global_importance(grouped_importance, paths["global_importance"])
    plot_beeswarm(shap_result.explanation, paths["beeswarm"])
    return (
        transformed_importance,
        grouped_importance,
        shap_result,
        [
            paths["global_importance"],
            paths["beeswarm"],
        ],
    )


def write_local_reports_and_plots(
    root: Path,
    predictor: ai4i_predictor.AI4IPredictor,
    components: ModelComponents,
    development_df: pd.DataFrame,
    development_features: pd.DataFrame,
) -> tuple[dict[str, Any], list[Path], float]:
    development_probabilities = predictor.pipeline.predict_proba(development_features)[
        :, POSITIVE_CLASS
    ]
    cases = select_representative_cases(
        development_df,
        development_probabilities,
        threshold=float(predictor.decision_threshold),
        traceability_field=str(predictor.final_config["traceability_field"]),
    )
    selected_features = development_features.iloc[[case.row_index for case in cases]].copy()
    transformed = transform_model_inputs(components.preprocessor, selected_features)
    shap_result = explain_positive_class(
        components.classifier,
        transformed,
        components.transformed_feature_names,
    )
    grouped_features, grouped_values = grouped_contribution_matrix(
        components.transformed_feature_names,
        shap_result.values,
        predictor.final_config,
    )
    payload = build_local_explanation_payload(
        cases,
        shap_result,
        components.transformed_feature_names,
        grouped_features,
        grouped_values,
        predictor,
    )
    write_json(payload, local_explanations_path(root))

    paths = plot_paths(root)
    waterfall_paths: list[Path] = []
    for index, case in enumerate(cases):
        path = paths[case.case_name]
        plot_waterfall(
            shap_result.explanation[index],
            path,
            case_name=case.case_name,
            probability=case.failure_probability,
            threshold=float(predictor.decision_threshold),
        )
        waterfall_paths.append(path)
    return payload, waterfall_paths, float(np.max(shap_result.additivity_errors))


def write_sample_explanations(
    root: Path,
    predictor: ai4i_predictor.AI4IPredictor,
    components: ModelComponents,
) -> tuple[dict[str, Any], float]:
    payload = ai4i_predictor.load_inference_payload(ai4i_predictor.sample_input_path(root))
    records = payload if isinstance(payload, list) else [payload]
    feature_frame = ai4i_predictor.validate_inference_records(records, predictor.final_config)
    predictions = predictor.predict_batch(records)
    transformed = transform_model_inputs(components.preprocessor, feature_frame)
    shap_result = explain_positive_class(
        components.classifier,
        transformed,
        components.transformed_feature_names,
    )
    grouped_features, grouped_values = grouped_contribution_matrix(
        components.transformed_feature_names,
        shap_result.values,
        predictor.final_config,
    )
    sample_payload = build_sample_explanation_payload(
        predictions,
        shap_result,
        grouped_features,
        grouped_values,
    )
    write_json(sample_payload, sample_explanations_path(root))
    return sample_payload, float(np.max(shap_result.additivity_errors))


def run_explainability(root: Path | None = None) -> ExplainabilityResult:
    root_path = root or project_root()
    predictor = load_trusted_predictor(root_path)
    components = extract_model_components(predictor.pipeline, predictor.final_config)
    development_df, development_features, _modeling_config = load_development_data(root_path)
    global_sample = deterministic_global_sample(development_df, predictor.final_config)
    global_features = global_sample.loc[
        :, list(predictor.final_config["predictive_features"])
    ].copy()

    _transformed_importance, grouped_importance, global_shap, global_plot_paths = (
        write_global_reports_and_plots(root_path, predictor, components, global_features)
    )
    local_payload, waterfall_paths, local_max_error = write_local_reports_and_plots(
        root_path,
        predictor,
        components,
        development_df,
        development_features,
    )
    sample_payload, sample_max_error = write_sample_explanations(root_path, predictor, components)
    max_additivity_error = max(
        float(np.max(global_shap.additivity_errors)),
        local_max_error,
        sample_max_error,
    )
    summary = build_summary(
        predictor,
        len(global_sample),
        len(components.transformed_feature_names),
        len(expected_grouped_features(predictor.final_config)),
        grouped_importance,
        max_additivity_error,
    )
    write_json(summary, summary_path(root_path))
    return ExplainabilityResult(
        transformed_feature_count=len(components.transformed_feature_names),
        grouped_feature_count=len(expected_grouped_features(predictor.final_config)),
        global_sample_size=len(global_sample),
        grouped_importance=grouped_importance,
        local_explanations=local_payload,
        sample_explanations=sample_payload,
        summary=summary,
        plot_paths=[*global_plot_paths, *waterfall_paths],
        max_additivity_error=max_additivity_error,
    )

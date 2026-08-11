from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.explainability import ai4i_shap
from scripts import check_ai4i_shap

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
PREDICTIVE_FEATURES = ["Type", *NUMERICAL_FEATURES]


def final_config() -> dict[str, Any]:
    return {
        "categorical_features": ["Type"],
        "decision_threshold": 0.14,
        "excluded_identifier_fields": ["Product ID"],
        "excluded_leakage_sensitive_fields": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "hyperparameters": {
            "class_weight": "balanced_subsample",
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_estimators": 17,
            "n_jobs": 1,
            "random_state": 42,
        },
        "model_family": "RandomForestClassifier",
        "numerical_features": NUMERICAL_FEATURES,
        "predictive_features": PREDICTIVE_FEATURES,
        "target": "Machine failure",
        "traceability_field": "source_udi",
    }


def synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["L", 298.0, 308.1, 1450, 42.0, 20],
            ["M", 300.1, 309.9, 1320, 54.0, 155],
            ["H", 296.7, 306.8, 1700, 31.0, 35],
            ["L", 303.2, 312.8, 1210, 61.0, 220],
            ["M", 299.5, 309.1, 1510, 39.0, 70],
            ["H", 301.1, 311.4, 1180, 66.0, 240],
            ["L", 297.4, 307.6, 1605, 35.5, 45],
            ["M", 302.0, 312.0, 1275, 58.2, 205],
        ],
        columns=PREDICTIVE_FEATURES,
    )


def fitted_pipeline() -> Pipeline:
    config = final_config()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "one_hot_encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        )
                    ]
                ),
                ["Type"],
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(**config["hyperparameters"])
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pipeline.fit(synthetic_features(), y)
    return pipeline


def test_extracts_fitted_steps_and_transformed_feature_names() -> None:
    pipeline = fitted_pipeline()
    components = ai4i_shap.extract_model_components(pipeline, final_config())

    assert components.preprocessor is pipeline.named_steps["preprocessor"]
    assert components.classifier is pipeline.named_steps["classifier"]
    assert components.transformed_feature_names == [
        "Type_H",
        "Type_L",
        "Type_M",
        *NUMERICAL_FEATURES,
    ]


def test_transform_uses_existing_preprocessor_without_model_fitting_calls() -> None:
    pipeline = fitted_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]

    def forbidden_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("explainability must not call model-fitting methods")

    pipeline.fit = forbidden_call  # type: ignore[method-assign]
    preprocessor.fit = forbidden_call  # type: ignore[method-assign]
    preprocessor.fit_transform = forbidden_call  # type: ignore[method-assign]

    transformed = ai4i_shap.transform_model_inputs(preprocessor, synthetic_features().head(2))

    assert transformed.shape == (2, 8)


def test_selects_positive_class_from_binary_shap_output() -> None:
    values = np.zeros((2, 3, 2), dtype=float)
    values[:, :, 0] = -7.0
    values[:, :, 1] = 3.0

    selected, base_value = ai4i_shap.select_positive_class_shap_values(
        values,
        [0.8, 0.2],
        sample_count=2,
        feature_count=3,
    )

    assert np.all(selected == 3.0)
    assert base_value == 0.2


def test_rejects_unsupported_shap_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported SHAP output shape"):
        ai4i_shap.select_positive_class_shap_values(
            np.zeros((2, 3)),
            [0.8, 0.2],
            sample_count=2,
            feature_count=3,
        )


def test_actual_tree_explainer_values_match_positive_class_probability() -> None:
    pipeline = fitted_pipeline()
    components = ai4i_shap.extract_model_components(pipeline, final_config())
    transformed = ai4i_shap.transform_model_inputs(components.preprocessor, synthetic_features())

    shap_result = ai4i_shap.explain_positive_class(
        components.classifier,
        transformed,
        components.transformed_feature_names,
    )

    positive_probability = components.classifier.predict_proba(transformed)[:, 1]
    negative_probability = components.classifier.predict_proba(transformed)[:, 0]
    reconstructed = shap_result.base_value + shap_result.values.sum(axis=1)
    assert np.allclose(reconstructed, positive_probability, atol=ai4i_shap.ADDITIVITY_TOLERANCE)
    assert not np.allclose(reconstructed, negative_probability, atol=ai4i_shap.ADDITIVITY_TOLERANCE)


def test_grouped_type_contributions_are_additive() -> None:
    names = ["Type_H", "Type_L", "Air temperature [K]", "Torque [Nm]"]
    values = np.array([[0.1, -0.2, 0.3, -0.4], [0.5, 0.25, -0.1, 0.2]])

    grouped_names, grouped_values = ai4i_shap.grouped_contribution_matrix(
        names,
        values,
        final_config(),
    )

    assert grouped_names == ["Type", *NUMERICAL_FEATURES]
    assert np.allclose(grouped_values[:, 0], [-0.1, 0.75])
    assert np.allclose(grouped_values[:, 1], [0.3, -0.1])
    assert np.allclose(grouped_values[:, 4], [-0.4, 0.2])


def test_global_mean_absolute_importance_ranking_is_deterministic() -> None:
    rows = ai4i_shap.ranked_mean_absolute_importance(
        ["b_feature", "a_feature", "c_feature"],
        np.array([[1.0, -1.0, 0.2], [-1.0, 1.0, -0.2]]),
    )

    assert [row["feature"] for row in rows] == ["a_feature", "b_feature", "c_feature"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_representative_case_selection_and_tie_breaking() -> None:
    frame = pd.DataFrame({"source_udi": [30, 10, 20, 5]})
    probabilities = np.array([0.03, 0.16, 0.12, 0.16])

    cases = ai4i_shap.select_representative_cases(
        frame,
        probabilities,
        threshold=0.14,
        traceability_field="source_udi",
    )
    by_name = {case.case_name: case for case in cases}

    assert by_name["low_risk"].source_udi == 30
    assert by_name["threshold_near"].source_udi == 5
    assert by_name["high_risk"].source_udi == 5
    assert by_name["threshold_near"].failure_prediction == 1


def test_probability_to_threshold_prediction_uses_frozen_threshold() -> None:
    assert ai4i_shap.probability_to_prediction(0.14, 0.14) == 1
    assert ai4i_shap.probability_to_prediction(0.139999, 0.14) == 0


def test_additivity_consistency_calculation() -> None:
    values = np.array([[0.1, 0.2], [-0.05, 0.25]])
    errors = ai4i_shap.additivity_errors(values, 0.1, np.array([0.4, 0.3]))

    assert np.allclose(errors, [0.0, 0.0])


def test_local_explanation_output_excludes_target_label() -> None:
    explanation = shap.Explanation(
        values=np.array([[0.1, -0.1]]),
        base_values=np.array([0.5]),
        data=np.array([[1.0, 2.0]]),
        feature_names=["Type_H", "Torque [Nm]"],
    )
    shap_result = ai4i_shap.PositiveClassShapResult(
        values=np.array([[0.1, -0.1]]),
        base_value=0.5,
        model_outputs=np.array([0.5]),
        additivity_errors=np.array([0.0]),
        explanation=explanation,
    )
    predictor = ai4i_predictor_stub()
    case = ai4i_shap.RepresentativeCase("low_risk", 0, 101, 0.03, 0)

    payload = ai4i_shap.build_local_explanation_payload(
        [case],
        shap_result,
        ["Type_H", "Torque [Nm]"],
        ["Type", "Torque [Nm]"],
        np.array([[0.1, -0.1]]),
        predictor,
    )

    assert "Machine failure" not in json.dumps(payload)
    assert payload["cases"][0]["source_udi"] == 101


def ai4i_predictor_stub() -> Any:
    class StubPredictor:
        final_config = final_config()
        final_config_hash = "config-hash"
        model_name = "ai4i-failure-risk-random-forest"
        model_version = "1.0.0"

        @property
        def decision_threshold(self) -> float:
            return 0.14

    return StubPredictor()


def test_sample_explanation_output_structure() -> None:
    explanation = shap.Explanation(
        values=np.array([[0.2, -0.1]]),
        base_values=np.array([0.5]),
        data=np.array([[1.0, 2.0]]),
        feature_names=["Type_H", "Torque [Nm]"],
    )
    shap_result = ai4i_shap.PositiveClassShapResult(
        values=np.array([[0.2, -0.1]]),
        base_value=0.5,
        model_outputs=np.array([0.6]),
        additivity_errors=np.array([0.0]),
        explanation=explanation,
    )
    payload = ai4i_shap.build_sample_explanation_payload(
        [
            {
                "failure_probability": 0.6,
                "failure_prediction": 1,
                "decision_threshold": 0.14,
                "model_name": "ai4i-failure-risk-random-forest",
                "model_version": "1.0.0",
                "final_config_hash": "config-hash",
            }
        ],
        shap_result,
        ["Type", "Torque [Nm]"],
        np.array([[0.2, -0.1]]),
    )

    record = payload["sample_explanations"][0]
    assert record["sample_index"] == 0
    assert record["failure_prediction"] == 1
    assert len(record["grouped_feature_contributions"]) == 2


def test_source_guard_detects_restricted_sources(tmp_path: Path) -> None:
    safe = tmp_path / "safe.py"
    unsafe = tmp_path / "unsafe.py"
    safe.write_text("print('safe')\n", encoding="utf-8")
    unsafe.write_text("path = 'test.csv'\nmodel.fit(data)\n", encoding="utf-8")

    assert check_ai4i_shap.source_guard_violations([safe]) == []
    violations = check_ai4i_shap.source_guard_violations([unsafe])
    assert any("test.csv" in item for item in violations)
    assert any(".fit(" in item for item in violations)


def test_real_explainability_sources_pass_static_guard() -> None:
    paths = [
        Path("ml/explainability/ai4i_shap.py"),
        Path("scripts/explain_ai4i_model.py"),
    ]

    assert check_ai4i_shap.source_guard_violations(paths) == []

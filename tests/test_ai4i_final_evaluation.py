from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from ml.evaluation import ai4i_final_evaluation as final_eval
from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline

NUMERIC_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
LEAKAGE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")


def config() -> ai4i_modeling.ModelingConfig:
    return ai4i_modeling.ModelingConfig(
        dataset_name="AI4I 2020 Predictive Maintenance Dataset",
        modeling_objective="Binary classification of Machine failure",
        target_column="Machine failure",
        categorical_features=("Type",),
        numerical_features=NUMERIC_COLUMNS,
        traceability_fields=("UDI",),
        derived_traceability_field="source_udi",
        excluded_identifiers=("Product ID",),
        excluded_leakage_sensitive_columns=LEAKAGE_COLUMNS,
        forbidden_feature_sources=("Machine failure", *LEAKAGE_COLUMNS),
        random_seed=42,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        stratify_on="Machine failure",
        future_preprocessing_design={},
    )


def modeling_frame(start_udi: int = 1, row_count: int = 18) -> pd.DataFrame:
    rows = []
    for offset in range(row_count):
        rows.append(
            {
                "source_udi": start_udi + offset,
                "Type": ["L", "M", "H"][offset % 3],
                "Air temperature [K]": 295.0 + offset * 0.1,
                "Process temperature [K]": 305.0 + offset * 0.2,
                "Rotational speed [rpm]": 1200 + offset * 5,
                "Torque [Nm]": 30.0 + offset * 0.5,
                "Tool wear [min]": offset,
                "Machine failure": 1 if offset % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def comparison_metrics() -> dict[str, object]:
    return {
        "candidate_selection_policy": {
            "ap_tie_tolerance": 0.01,
            "model_average_precision": {
                "random_forest": 0.749231,
                "standard_logistic": 0.465735,
                "xgboost": 0.752493,
            },
            "primary_metric": "train_oof_average_precision",
            "reason": (
                "Random Forest is within 0.01 train OOF AP of the best candidate "
                "and is simpler under the predefined order."
            ),
            "selected_model": "random_forest",
            "selected_threshold": 0.14,
            "validation_may_change_selection": False,
        }
    }


def tuning_metrics() -> dict[str, object]:
    return {
        "fixed_random_forest_reference": {"average_precision": 0.749231},
        "promotion_policy": {
            "average_precision_delta": 0.003413,
            "fixed_average_precision": 0.749231,
            "promotion_delta_required": 0.005,
            "reason": (
                "Tuned nested-OOF Average Precision does not exceed the previous "
                "fixed Random Forest OOF AP by at least 0.005, so added complexity "
                "is not promoted."
            ),
            "selected_candidate": "fixed_random_forest",
            "selected_threshold": 0.14,
            "tuned_nested_average_precision": 0.752644,
            "validation_may_change_selection": False,
        },
        "tuned_nested_oof_results": {"average_precision": 0.752644},
        "validation_results": {
            "threshold_independent": {"average_precision": 0.680945, "roc_auc": 0.974743},
            "threshold_0_5": {
                "precision": 0.807692,
                "recall": 0.411765,
                "f1": 0.545455,
                "f2": 0.456522,
                "confusion_matrix": [[1444, 5], [30, 21]],
            },
            "selected_threshold": {
                "threshold": 0.14,
                "precision": 0.5,
                "recall": 0.764706,
                "f1": 0.604651,
                "f2": 0.691489,
                "confusion_matrix": [[1410, 39], [12, 39]],
            },
            "selection_uses_validation": False,
        },
    }


def final_config() -> dict[str, object]:
    return final_eval.expected_final_model_config(config(), comparison_metrics(), tuning_metrics())


def test_frozen_configuration_validation_accepts_expected_policy():
    final_eval.validate_final_model_config(
        final_config(),
        config(),
        comparison_metrics(),
        tuning_metrics(),
    )


def test_frozen_configuration_validation_rejects_threshold_change():
    bad_config = final_config()
    bad_config["decision_threshold"] = 0.5

    with pytest.raises(ValueError, match="threshold"):
        final_eval.validate_final_model_config(
            bad_config,
            config(),
            comparison_metrics(),
            tuning_metrics(),
        )


def test_configuration_hashing_is_deterministic_for_key_order():
    first = {"model": {"b": 2, "a": [1, 2]}, "threshold": 0.14}
    second = {"threshold": 0.14, "model": {"a": [1, 2], "b": 2}}

    assert final_eval.final_config_hash(first) == final_eval.final_config_hash(second)


def test_fixed_random_forest_construction_and_hyperparameter_policy():
    pipeline = final_eval.build_frozen_random_forest_pipeline(config(), final_config())
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical = preprocessor.transformers[0][1].named_steps["one_hot_encoder"]

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 300
    assert classifier.max_depth is None
    assert classifier.min_samples_leaf == 1
    assert classifier.max_features == "sqrt"
    assert classifier.class_weight == "balanced_subsample"
    assert classifier.random_state == 42
    assert classifier.n_jobs == 1
    assert isinstance(categorical, OneHotEncoder)
    assert categorical.handle_unknown == "ignore"
    assert preprocessor.transformers[1][1] == "passthrough"


def test_frozen_threshold_is_exactly_0_14():
    assert final_config()["decision_threshold"] == 0.14


def test_feature_extraction_uses_only_predictive_columns():
    features, target = ai4i_baseline.extract_features_and_target(modeling_frame(), config())

    assert list(features.columns) == ["Type", *NUMERIC_COLUMNS]
    assert "source_udi" not in features.columns
    assert set(LEAKAGE_COLUMNS).isdisjoint(features.columns)
    assert target.name == "Machine failure"


def test_train_validation_combination_is_deterministic_and_keeps_rows():
    cfg = config()
    train = modeling_frame(start_udi=20, row_count=5)
    validation = modeling_frame(start_udi=1, row_count=4)

    combined = final_eval.combine_development_training_data(train, validation, cfg)

    assert len(combined) == 9
    assert combined["source_udi"].tolist() == sorted(combined["source_udi"].tolist())
    assert int(combined["Machine failure"].sum()) == int(train["Machine failure"].sum()) + int(
        validation["Machine failure"].sum()
    )


def test_test_separation_rejects_overlap():
    cfg = config()
    train = modeling_frame(start_udi=1, row_count=5)
    validation = modeling_frame(start_udi=100, row_count=5)
    test = modeling_frame(start_udi=5, row_count=5)

    with pytest.raises(ValueError, match="Train and test"):
        final_eval.validate_split_separation(train, validation, test, cfg)


def test_preprocessing_fit_excludes_test_features():
    cfg = config()
    development = modeling_frame(start_udi=1, row_count=24)
    test = modeling_frame(start_udi=100, row_count=6)
    test.loc[:, "Type"] = "Z"
    pipeline = final_eval.fit_final_pipeline(development, cfg, final_config())
    test_features, test_target = ai4i_baseline.extract_features_and_target(test, cfg)

    final_eval.evaluate_frozen_pipeline(pipeline, 0.14, test_features, test_target)

    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["one_hot_encoder"]
    )
    assert "Z" not in set(encoder.categories_[0])


def test_final_metric_calculation_for_reference_and_frozen_thresholds():
    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    ranking = final_eval.ranking_metrics(target, probabilities)
    threshold_05 = final_eval.classification_metrics_at_threshold(target, probabilities, 0.5)
    threshold_014 = final_eval.classification_metrics_at_threshold(target, probabilities, 0.14)

    assert ranking == {"average_precision": 0.833333, "roc_auc": 0.75}
    assert threshold_05["confusion_matrix"] == [[1, 1], [1, 1]]
    assert threshold_05["precision"] == 0.5
    assert threshold_05["recall"] == 0.5
    assert threshold_014["confusion_matrix"] == [[1, 1], [0, 2]]
    assert threshold_014["precision"] == 0.666667
    assert threshold_014["recall"] == 1.0
    assert threshold_014["f1"] == 0.8
    assert threshold_014["f2"] == 0.909091


def test_development_test_delta_calculation():
    validation = {
        "threshold_independent": {"average_precision": 0.7, "roc_auc": 0.9},
        "threshold_0_14": {"precision": 0.5, "recall": 0.8, "f1": 0.6, "f2": 0.7},
    }
    test = {
        "threshold_independent": {"average_precision": 0.75, "roc_auc": 0.88},
        "threshold_0_14": {"precision": 0.55, "recall": 0.7, "f1": 0.62, "f2": 0.68},
    }

    assert final_eval.calculate_test_minus_validation_deltas(validation, test) == {
        "average_precision": 0.05,
        "roc_auc": -0.02,
        "precision_at_0_14": 0.05,
        "recall_at_0_14": -0.1,
        "f1_at_0_14": 0.02,
        "f2_at_0_14": -0.02,
    }


def test_prediction_report_construction():
    cfg = config()
    test = modeling_frame(start_udi=10, row_count=4).sample(frac=1.0, random_state=42)
    probabilities = np.array([0.1, 0.5, 0.13, 0.2])

    report = final_eval.build_prediction_report(test, cfg, probabilities, 0.14)

    assert list(report.columns) == final_eval.EXPECTED_PREDICTION_COLUMNS
    assert report["source_udi"].tolist() == sorted(test["source_udi"].tolist())
    assert report["probability"].between(0, 1, inclusive="both").all()
    assert set(report["prediction_threshold_0_5"]).issubset({0, 1})
    assert set(report["prediction_threshold_0_14"]).issubset({0, 1})


def test_forbidden_features_remain_excluded_from_final_config():
    predictive_features = set(final_config()["predictive_features"])

    assert "source_udi" not in predictive_features
    assert "UDI" not in predictive_features
    assert "Product ID" not in predictive_features
    assert set(LEAKAGE_COLUMNS).isdisjoint(predictive_features)


def test_validation_rejects_forbidden_feature_policy():
    bad_modeling_config = replace(config(), categorical_features=("Type", "TWF"))
    bad_config = final_eval.expected_final_model_config(
        bad_modeling_config,
        comparison_metrics(),
        tuning_metrics(),
    )

    with pytest.raises(ValueError, match="Forbidden"):
        final_eval.validate_final_model_config(
            bad_config,
            bad_modeling_config,
            comparison_metrics(),
            tuning_metrics(),
        )


def test_changing_test_data_cannot_mutate_frozen_specification():
    cfg = config()
    frozen_config = final_config()
    development = modeling_frame(start_udi=1, row_count=30)
    first_test = modeling_frame(start_udi=100, row_count=6)
    second_test = first_test.copy()
    second_test.loc[:, "Machine failure"] = 1 - second_test["Machine failure"]
    second_test.loc[:, "Type"] = "Z"
    second_test.loc[:, NUMERIC_COLUMNS] = second_test.loc[:, NUMERIC_COLUMNS] + 9999

    pipeline = final_eval.fit_final_pipeline(development, cfg, frozen_config)
    initial_hash = final_eval.final_config_hash(frozen_config)
    classifier_params = {
        key: pipeline.named_steps["classifier"].get_params()[key]
        for key in final_eval.FROZEN_HYPERPARAMETERS
    }
    frozen_threshold = frozen_config["decision_threshold"]
    preprocessor_config = pipeline.named_steps["preprocessor"].get_params(deep=False)

    first_features, first_target = ai4i_baseline.extract_features_and_target(first_test, cfg)
    second_features, second_target = ai4i_baseline.extract_features_and_target(second_test, cfg)
    final_eval.evaluate_frozen_pipeline(pipeline, frozen_threshold, first_features, first_target)
    final_eval.evaluate_frozen_pipeline(pipeline, frozen_threshold, second_features, second_target)

    assert final_eval.final_config_hash(frozen_config) == initial_hash
    assert {
        key: pipeline.named_steps["classifier"].get_params()[key]
        for key in final_eval.FROZEN_HYPERPARAMETERS
    } == classifier_params
    assert pipeline.named_steps["preprocessor"].get_params(deep=False) == preprocessor_config
    assert frozen_config["decision_threshold"] == 0.14
    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["one_hot_encoder"]
    )
    assert "Z" not in set(encoder.categories_[0])

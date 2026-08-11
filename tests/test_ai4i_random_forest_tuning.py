from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline
from ml.training import ai4i_random_forest_tuning as tuning

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


def modeling_frame(start_udi: int = 1, row_count: int = 60) -> pd.DataFrame:
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
                "Machine failure": 1 if offset % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def threshold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "threshold": 0.2,
                "precision": 0.4,
                "recall": 0.8,
                "f1": 0.5,
                "f2": 0.7,
                "balanced_accuracy": 0.7,
                "predicted_positive_count": 5,
                "true_positive": 4,
                "false_positive": 1,
                "true_negative": 4,
                "false_negative": 1,
            },
            {
                "threshold": 0.3,
                "precision": 0.4,
                "recall": 0.8,
                "f1": 0.5,
                "f2": 0.7,
                "balanced_accuracy": 0.7,
                "predicted_positive_count": 5,
                "true_positive": 4,
                "false_positive": 1,
                "true_negative": 4,
                "false_negative": 1,
            },
            {
                "threshold": 0.4,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.64,
                "f2": 0.62,
                "balanced_accuracy": 0.75,
                "predicted_positive_count": 4,
                "true_positive": 3,
                "false_positive": 1,
                "true_negative": 4,
                "false_negative": 2,
            },
        ],
        columns=tuning.THRESHOLD_COLUMNS,
    )


def fixed_reference(ap: float = 0.749) -> dict[str, object]:
    return {
        "average_precision": ap,
        "roc_auc": 0.95,
        "max_f2_threshold": {"threshold": 0.14},
        "threshold_0_5": {"confusion_matrix": [[1, 0], [1, 1]]},
    }


def nested_metrics(ap: float) -> dict[str, object]:
    return {
        "average_precision": ap,
        "roc_auc": 0.96,
        "threshold_0_5": {"confusion_matrix": [[1, 0], [1, 1]]},
    }


def validation_metrics(ap: float) -> dict[str, object]:
    return {
        "threshold_independent": {"average_precision": ap, "roc_auc": 0.5},
        "threshold_0_5": {"confusion_matrix": [[4, 1], [2, 3]]},
        "selected_threshold": {"confusion_matrix": [[3, 2], [1, 4]]},
    }


def test_exact_eight_configuration_parameter_grid():
    configs = tuning.allowed_parameter_configurations()

    assert len(configs) == 8
    assert tuning.allowed_parameter_keys() == {
        (200, "None", 1),
        (200, "None", 3),
        (200, "12", 1),
        (200, "12", 3),
        (400, "None", 1),
        (400, "None", 3),
        (400, "12", 1),
        (400, "12", 3),
    }
    assert {item["class_weight"] for item in configs} == {"balanced_subsample"}
    assert {item["max_features"] for item in configs} == {"sqrt"}


def test_nested_cv_fold_configurations():
    outer = tuning.make_outer_cv()
    inner = tuning.make_inner_cv()

    assert outer.n_splits == 5
    assert outer.shuffle is True
    assert outer.random_state == 42
    assert inner.n_splits == 3
    assert inner.shuffle is True
    assert inner.random_state == 42


class RecordingEstimator:
    fit_predict_pairs: list[tuple[set[int], set[int]]] = []

    def __init__(self, fit_indices: set[int], probability: float = 0.25) -> None:
        self.fit_indices = fit_indices
        self.probability = probability

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.__class__.fit_predict_pairs.append((self.fit_indices, set(x.index.tolist())))
        probability = np.full(len(x), self.probability)
        return np.column_stack([1 - probability, probability])


class StubSearch:
    def __init__(self, fit_indices: set[int]) -> None:
        self.best_score_ = 0.5
        self.best_params_ = {
            "classifier__n_estimators": 200,
            "classifier__max_depth": None,
            "classifier__min_samples_leaf": 1,
        }
        self.best_estimator_ = RecordingEstimator(fit_indices)


def test_outer_holdout_cannot_influence_inner_grid_search(monkeypatch):
    cfg = config()
    train = modeling_frame(row_count=45)
    observed_inner_indices: list[set[int]] = []
    RecordingEstimator.fit_predict_pairs = []

    def run_inner_stub(
        features: pd.DataFrame,
        target: pd.Series,
        _config: ai4i_modeling.ModelingConfig,
        *,
        grid_n_jobs: int = -1,
    ) -> StubSearch:
        assert grid_n_jobs == 1
        assert set(features.index) == set(target.index)
        observed_inner_indices.append(set(features.index.tolist()))
        return StubSearch(set(features.index.tolist()))

    monkeypatch.setattr(tuning, "run_inner_grid_search", run_inner_stub)

    result = tuning.generate_nested_oof_predictions(train, cfg, grid_n_jobs=1)

    assert len(observed_inner_indices) == 5
    assert len(result.oof_predictions) == len(train)
    for inner_indices, (_, predicted_indices) in zip(
        observed_inner_indices,
        RecordingEstimator.fit_predict_pairs,
        strict=True,
    ):
        assert inner_indices.isdisjoint(predicted_indices)
    predicted_once = [
        index
        for _, predicted_indices in RecordingEstimator.fit_predict_pairs
        for index in predicted_indices
    ]
    assert sorted(predicted_once) == list(range(len(train)))


def test_preprocessing_fits_only_on_outer_training_data():
    cfg = config()
    train = modeling_frame(row_count=45)
    features, target = ai4i_baseline.extract_features_and_target(train, cfg)
    first_train_index, first_holdout_index = next(tuning.make_outer_cv().split(features, target))
    features.loc[first_holdout_index, "Type"] = "Z"
    pipeline = tuning.build_random_forest_pipeline(cfg, n_estimators=5)

    pipeline.fit(features.iloc[first_train_index], target.iloc[first_train_index])

    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["one_hot_encoder"]
    )
    assert "Z" not in set(encoder.categories_[0])


def test_nested_oof_predictions_cover_every_observation_once(monkeypatch):
    cfg = config()
    train = modeling_frame(row_count=45)
    RecordingEstimator.fit_predict_pairs = []

    def run_inner_stub(
        features: pd.DataFrame,
        target: pd.Series,
        _config: ai4i_modeling.ModelingConfig,
        *,
        grid_n_jobs: int = -1,
    ) -> StubSearch:
        return StubSearch(set(features.index.tolist()))

    monkeypatch.setattr(tuning, "run_inner_grid_search", run_inner_stub)

    result = tuning.generate_nested_oof_predictions(train, cfg, grid_n_jobs=1)

    assert len(result.oof_predictions) == len(train)
    assert result.oof_predictions["source_udi"].is_unique
    assert set(result.oof_predictions["source_udi"]) == set(train["source_udi"])
    assert result.oof_predictions["probability"].between(0, 1).all()
    assert set(result.oof_predictions["outer_fold"]) == {1, 2, 3, 4, 5}
    assert len(result.outer_folds) == 5


def test_hyperparameter_extraction_and_deterministic_representation():
    extracted = tuning.extract_random_forest_params(
        {
            "classifier__n_estimators": 400,
            "classifier__max_depth": None,
            "classifier__min_samples_leaf": 3,
        }
    )

    assert extracted["n_estimators"] == 400
    assert extracted["max_depth"] is None
    assert extracted["min_samples_leaf"] == 3
    assert tuning.format_max_depth(None) == "None"
    assert tuning.format_max_depth(float("nan")) == "None"
    assert tuning.format_max_depth(12) == "12"
    assert tuning.parameter_label(extracted) == "n=400, depth=None, leaf=3"
    assert tuning.serialized_params(extracted)["max_depth"] == "None"


def test_max_f2_threshold_selection_prefers_higher_threshold_on_tie():
    candidates = tuning.build_threshold_candidates(threshold_frame())

    assert candidates["max_f2"]["threshold"] == 0.3
    assert candidates["max_f2"]["f2"] == 0.7


def test_promotion_policy_promotes_tuned_when_delta_is_large_enough():
    candidates = {"max_f2": {"threshold": 0.22}}

    selected = tuning.select_promotion_candidate(
        fixed_reference(0.749),
        nested_metrics(0.754),
        candidates,
    )

    assert selected["selected_candidate"] == "tuned_random_forest"
    assert selected["selected_threshold"] == 0.22


def test_promotion_policy_retains_fixed_when_delta_is_too_small():
    candidates = {"max_f2": {"threshold": 0.22}}

    selected = tuning.select_promotion_candidate(
        fixed_reference(0.749),
        nested_metrics(0.7539),
        candidates,
    )

    assert selected["selected_candidate"] == "fixed_random_forest"
    assert selected["selected_threshold"] == 0.14


def test_validation_metrics_cannot_alter_promotion_decision():
    cfg = config()
    train = modeling_frame(1, 45)
    validation = modeling_frame(101, 15)
    fixed = fixed_reference(0.749)
    tuned = nested_metrics(0.754)
    candidates = {"max_f2": {"threshold": 0.22}}
    promotion = tuning.select_promotion_candidate(fixed, tuned, candidates)
    full_train_summary = {
        "best_hyperparameters": {
            "n_estimators": 400,
            "max_depth": "None",
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": 1,
        },
        "best_mean_average_precision": 0.75,
        "best_std_average_precision": 0.01,
    }

    summary_a = tuning.build_metrics_summary(
        train,
        validation,
        cfg,
        fixed,
        tuned,
        candidates,
        full_train_summary,
        promotion,
        validation_metrics(0.1),
    )
    summary_b = tuning.build_metrics_summary(
        train,
        validation,
        cfg,
        fixed,
        tuned,
        candidates,
        full_train_summary,
        promotion,
        validation_metrics(0.9),
    )

    assert summary_a["selected_candidate"] == summary_b["selected_candidate"]
    assert summary_a["selected_threshold"] == summary_b["selected_threshold"] == 0.22
    assert summary_a["validation_results"]["selection_uses_validation"] is False
    assert summary_b["validation_results"]["selection_uses_validation"] is False


def test_fixed_random_forest_reconstruction():
    pipeline = tuning.build_fixed_random_forest_pipeline(config())
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 300
    assert classifier.max_depth is None
    assert classifier.min_samples_leaf == 1
    assert classifier.max_features == "sqrt"
    assert classifier.class_weight == "balanced_subsample"
    assert classifier.random_state == 42
    assert classifier.n_jobs == 1


def test_tuned_random_forest_construction():
    pipeline = tuning.build_tuned_random_forest_pipeline(
        config(),
        {
            "n_estimators": 400,
            "max_depth": "12",
            "min_samples_leaf": 3,
        },
    )
    classifier = pipeline.named_steps["classifier"]

    assert classifier.n_estimators == 400
    assert classifier.max_depth == 12
    assert classifier.min_samples_leaf == 3
    assert classifier.max_features == "sqrt"
    assert classifier.class_weight == "balanced_subsample"


def test_forbidden_leakage_features_remain_excluded():
    bad_config = replace(config(), numerical_features=(*NUMERIC_COLUMNS, "TWF"))

    with pytest.raises(ValueError, match="Forbidden column"):
        tuning.build_random_forest_pipeline(bad_config)

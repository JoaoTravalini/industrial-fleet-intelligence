from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline
from ml.training import ai4i_model_comparison as comparison

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
    rows = []
    for model_name, base_threshold in [
        ("standard_logistic", 0.2),
        ("random_forest", 0.3),
        ("xgboost", 0.4),
    ]:
        rows.extend(
            [
                {
                    "model": model_name,
                    "threshold": base_threshold,
                    "precision": 0.4,
                    "recall": 0.8,
                    "f1": 0.52,
                    "f2": 0.64,
                    "balanced_accuracy": 0.7,
                    "predicted_positive_count": 10,
                    "true_positive": 4,
                    "false_positive": 6,
                    "true_negative": 8,
                    "false_negative": 1,
                },
                {
                    "model": model_name,
                    "threshold": base_threshold + 0.1,
                    "precision": 0.5,
                    "recall": 0.7,
                    "f1": 0.58,
                    "f2": 0.65,
                    "balanced_accuracy": 0.75,
                    "predicted_positive_count": 8,
                    "true_positive": 4,
                    "false_positive": 4,
                    "true_negative": 10,
                    "false_negative": 1,
                },
            ]
        )
    return pd.DataFrame(rows, columns=comparison.REQUIRED_THRESHOLD_COLUMNS)


def validation_metrics(ap: float) -> dict[str, object]:
    return {
        "threshold_independent": {"roc_auc": 0.5, "average_precision": ap},
        "threshold_0_5": {"confusion_matrix": [[4, 1], [2, 3]]},
        "selected_threshold": {"confusion_matrix": [[3, 2], [1, 4]]},
    }


def test_shared_fold_assignments_are_deterministic_and_reusable():
    cfg = config()
    train = modeling_frame()

    first = comparison.make_shared_fold_assignments(train, cfg)
    second = comparison.make_shared_fold_assignments(train, cfg)

    assert len(first) == 5
    assert len(second) == 5
    for (first_train, first_holdout), (second_train, second_holdout) in zip(
        first,
        second,
        strict=True,
    ):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_holdout, second_holdout)
        assert set(first_train).isdisjoint(set(first_holdout))


def test_model_preprocessing_policies_are_expected():
    cfg = config()
    y = modeling_frame()["Machine failure"]
    pipelines = comparison.build_model_pipelines(cfg, y)

    logistic_preprocessor = pipelines["standard_logistic"].named_steps["preprocessor"]
    logistic_numerical = logistic_preprocessor.transformers[1][1]
    assert isinstance(logistic_numerical.named_steps["standard_scaler"], StandardScaler)

    for model_name in ["random_forest", "xgboost"]:
        preprocessor = pipelines[model_name].named_steps["preprocessor"]
        assert preprocessor.transformers[1][1] == "passthrough"
        categorical = preprocessor.transformers[0][1].named_steps["one_hot_encoder"]
        assert isinstance(categorical, OneHotEncoder)
        assert categorical.handle_unknown == "ignore"


def test_random_forest_fixed_configuration():
    pipeline = comparison.build_random_forest_pipeline(config())
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 300
    assert classifier.class_weight == "balanced_subsample"
    assert classifier.random_state == 42
    assert classifier.n_jobs == 1


def test_xgboost_fixed_configuration_and_cpu_execution():
    pipeline = comparison.build_xgboost_pipeline(config(), scale_pos_weight=3.5)
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()

    assert isinstance(classifier, XGBClassifier)
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.9
    assert params["colsample_bytree"] == 0.9
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == "logloss"
    assert params["random_state"] == 42
    assert params["n_jobs"] == 1
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"
    assert params["scale_pos_weight"] == 3.5


class RecordingEstimator:
    fit_predict_pairs_by_model: dict[str, list[tuple[set[int], set[int]]]] = {}

    def __init__(self, model_name: str, probability: float = 0.25) -> None:
        self.model_name = model_name
        self.probability = probability

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RecordingEstimator:
        self.fit_indices_ = set(x.index.tolist())
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.__class__.fit_predict_pairs_by_model.setdefault(self.model_name, []).append(
            (self.fit_indices_, set(x.index.tolist()))
        )
        probability = np.full(len(x), self.probability)
        return np.column_stack([1 - probability, probability])


def test_oof_predictions_cover_each_training_observation_once_and_use_shared_folds(monkeypatch):
    cfg = config()
    train = modeling_frame(row_count=45)
    probabilities = {
        "standard_logistic": 0.2,
        "random_forest": 0.4,
        "xgboost": 0.6,
    }
    RecordingEstimator.fit_predict_pairs_by_model = {}

    def build_stub(
        _config: ai4i_modeling.ModelingConfig,
        model_name: str,
        _training_target: pd.Series | np.ndarray | None = None,
    ) -> RecordingEstimator:
        return RecordingEstimator(model_name, probabilities[model_name])

    monkeypatch.setattr(comparison, "build_model_pipeline", build_stub)

    oof = comparison.generate_oof_predictions(train, cfg)

    assert len(oof) == len(train)
    assert oof["source_udi"].is_unique
    assert set(oof["source_udi"]) == set(train["source_udi"])
    for model_name in comparison.MODEL_NAMES:
        assert oof[f"{model_name}_probability"].between(0, 1).all()
        assert len(RecordingEstimator.fit_predict_pairs_by_model[model_name]) == 5

    standard_pairs = RecordingEstimator.fit_predict_pairs_by_model["standard_logistic"]
    for model_name in ["random_forest", "xgboost"]:
        assert RecordingEstimator.fit_predict_pairs_by_model[model_name] == standard_pairs
    for fit_indices, predict_indices in standard_pairs:
        assert fit_indices.isdisjoint(predict_indices)


def test_logistic_preprocessing_for_oof_fold_does_not_use_held_out_features():
    cfg = config()
    train = modeling_frame(row_count=45)
    folds = comparison.make_shared_fold_assignments(train, cfg)
    fit_index, holdout_index = folds[0]
    train.loc[holdout_index, list(NUMERIC_COLUMNS)] = 999999.0
    features, target = ai4i_baseline.extract_features_and_target(train, cfg)
    pipeline = comparison.build_standard_logistic_pipeline(cfg)

    pipeline.fit(features.iloc[fit_index], target.iloc[fit_index])

    scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numerical"]
        .named_steps["standard_scaler"]
    )
    expected_fit_means = features.iloc[fit_index].loc[:, NUMERIC_COLUMNS].mean().to_numpy()
    all_row_means = features.loc[:, NUMERIC_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_fit_means)
    assert not np.allclose(scaler.mean_, all_row_means)


def test_scale_pos_weight_calculation_uses_training_labels_only():
    target = pd.Series([0, 0, 0, 1, 1])

    assert comparison.calculate_scale_pos_weight(target) == 1.5


class ScaleRecordingEstimator:
    scales: list[float] = []
    fit_predict_pairs: list[tuple[set[int], set[int]]] = []

    def __init__(self, scale_pos_weight: float) -> None:
        self.scale_pos_weight = scale_pos_weight
        self.__class__.scales.append(scale_pos_weight)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ScaleRecordingEstimator:
        self.fit_indices_ = set(x.index.tolist())
        self.fit_target_sum_ = int(y.sum())
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.__class__.fit_predict_pairs.append((self.fit_indices_, set(x.index.tolist())))
        probability = np.full(len(x), 0.5)
        return np.column_stack([1 - probability, probability])


def test_xgboost_fold_scale_pos_weight_uses_fold_training_labels_only(monkeypatch):
    cfg = config()
    train = modeling_frame(row_count=45)
    folds = comparison.make_shared_fold_assignments(train, cfg)
    target = train["Machine failure"]
    expected_weights = [
        comparison.calculate_scale_pos_weight(target.iloc[fit_index]) for fit_index, _ in folds
    ]
    ScaleRecordingEstimator.scales = []
    ScaleRecordingEstimator.fit_predict_pairs = []

    def build_xgboost_stub(
        _config: ai4i_modeling.ModelingConfig,
        scale_pos_weight: float,
    ) -> ScaleRecordingEstimator:
        return ScaleRecordingEstimator(scale_pos_weight)

    monkeypatch.setattr(comparison, "build_xgboost_pipeline", build_xgboost_stub)

    comparison.generate_oof_probabilities_for_model(train, cfg, "xgboost", folds)

    assert ScaleRecordingEstimator.scales == pytest.approx(expected_weights)
    assert len(ScaleRecordingEstimator.fit_predict_pairs) == 5
    for fit_indices, predict_indices in ScaleRecordingEstimator.fit_predict_pairs:
        assert fit_indices.isdisjoint(predict_indices)


def test_threshold_candidates_include_selected_max_f2_thresholds():
    candidates = comparison.build_threshold_candidates(threshold_frame())

    assert candidates["standard_logistic"]["max_f2"]["threshold"] == 0.3
    assert candidates["random_forest"]["max_f2"]["threshold"] == 0.4
    assert candidates["xgboost"]["max_f2"]["threshold"] == 0.5


def test_ap_model_selection_prefers_highest_ap_when_margin_is_large():
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.40},
        "random_forest": {"average_precision": 0.43},
        "xgboost": {"average_precision": 0.47},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.2}},
        "random_forest": {"max_f2": {"threshold": 0.3}},
        "xgboost": {"max_f2": {"threshold": 0.4}},
    }

    selected = comparison.select_development_candidate(oof_metrics, candidates)

    assert selected["selected_model"] == "xgboost"
    assert selected["selected_threshold"] == 0.4


def test_simplicity_tie_break_policy_prefers_simpler_model_within_ap_tolerance():
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.461},
        "random_forest": {"average_precision": 0.466},
        "xgboost": {"average_precision": 0.470},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.2}},
        "random_forest": {"max_f2": {"threshold": 0.3}},
        "xgboost": {"max_f2": {"threshold": 0.4}},
    }

    selected = comparison.select_development_candidate(oof_metrics, candidates)

    assert selected["selected_model"] == "standard_logistic"
    assert selected["selected_threshold"] == 0.2
    assert selected["models_within_tie_tolerance"] == [
        "standard_logistic",
        "random_forest",
        "xgboost",
    ]


def test_validation_metrics_cannot_alter_candidate_selection():
    cfg = config()
    train = modeling_frame(1, 45)
    validation = modeling_frame(101, 15)
    folds = comparison.make_shared_fold_assignments(train, cfg)
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.40, "threshold_0_5": {}},
        "random_forest": {"average_precision": 0.45, "threshold_0_5": {}},
        "xgboost": {"average_precision": 0.50, "threshold_0_5": {}},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.2}},
        "random_forest": {"max_f2": {"threshold": 0.3}},
        "xgboost": {"max_f2": {"threshold": 0.4}},
    }
    selected = comparison.select_development_candidate(oof_metrics, candidates)

    summary_a = comparison.build_metrics_summary(
        train,
        validation,
        cfg,
        folds,
        oof_metrics,
        candidates,
        selected,
        validation_metrics(0.1),
    )
    summary_b = comparison.build_metrics_summary(
        train,
        validation,
        cfg,
        folds,
        oof_metrics,
        candidates,
        selected,
        validation_metrics(0.9),
    )

    assert summary_a["selected_model"] == summary_b["selected_model"] == "xgboost"
    assert summary_a["selected_threshold"] == summary_b["selected_threshold"] == 0.4
    assert summary_a["validation_results"]["selection_uses_validation"] is False
    assert summary_b["validation_results"]["selection_uses_validation"] is False


def test_forbidden_features_cannot_enter_any_model_family():
    bad_config = replace(config(), numerical_features=(*NUMERIC_COLUMNS, "TWF"))
    y = pd.Series([0, 1, 0, 1])

    for model_name in comparison.MODEL_NAMES:
        with pytest.raises(ValueError, match="Forbidden column"):
            comparison.build_model_pipeline(bad_config, model_name, y)

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_imbalance as imbalance

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


def modeling_frame(start_udi: int = 1, row_count: int = 50) -> pd.DataFrame:
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
                "Machine failure": 1 if offset % 5 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def threshold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "standard_logistic",
                "threshold": 0.2,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.6,
                "f2": 0.7,
                "balanced_accuracy": 0.6,
                "predicted_positive_count": 4,
                "true_positive": 2,
                "false_positive": 2,
                "true_negative": 4,
                "false_negative": 1,
            },
            {
                "model": "standard_logistic",
                "threshold": 0.3,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.6,
                "f2": 0.7,
                "balanced_accuracy": 0.6,
                "predicted_positive_count": 4,
                "true_positive": 2,
                "false_positive": 2,
                "true_negative": 4,
                "false_negative": 1,
            },
            {
                "model": "balanced_logistic",
                "threshold": 0.4,
                "precision": 0.4,
                "recall": 0.6,
                "f1": 0.48,
                "f2": 0.55,
                "balanced_accuracy": 0.58,
                "predicted_positive_count": 5,
                "true_positive": 2,
                "false_positive": 3,
                "true_negative": 3,
                "false_negative": 1,
            },
        ]
    )


def test_stratified_kfold_configuration():
    cv = imbalance.make_stratified_kfold()

    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_model_variants_use_identical_features_and_different_class_weights():
    cfg = config()
    pipelines = imbalance.build_model_pipelines(cfg)

    standard_preprocessor = pipelines["standard_logistic"].named_steps["preprocessor"]
    balanced_preprocessor = pipelines["balanced_logistic"].named_steps["preprocessor"]
    assert [item[0] for item in standard_preprocessor.transformers] == [
        item[0] for item in balanced_preprocessor.transformers
    ]
    assert [item[2] for item in standard_preprocessor.transformers] == [
        item[2] for item in balanced_preprocessor.transformers
    ]
    assert pipelines["standard_logistic"].named_steps["classifier"].class_weight is None
    assert pipelines["balanced_logistic"].named_steps["classifier"].class_weight == "balanced"


def test_oof_predictions_cover_each_training_observation_once():
    cfg = config()
    train = modeling_frame()

    oof = imbalance.generate_oof_predictions(train, cfg)

    assert len(oof) == len(train)
    assert oof["source_udi"].is_unique
    assert set(oof["source_udi"]) == set(train["source_udi"])
    assert oof["standard_logistic_probability"].between(0, 1).all()
    assert oof["balanced_logistic_probability"].between(0, 1).all()


class RecordingEstimator(BaseEstimator):
    fit_predict_pairs: list[tuple[set[int], set[int]]] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RecordingEstimator:
        self.fit_indices_ = set(x.index.tolist())
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.__class__.fit_predict_pairs.append((self.fit_indices_, set(x.index.tolist())))
        probability = np.full(len(x), 0.25)
        return np.column_stack([1 - probability, probability])


def test_oof_fold_predictions_are_not_fitted_on_held_out_rows(monkeypatch):
    cfg = config()
    train = modeling_frame()
    RecordingEstimator.fit_predict_pairs = []

    monkeypatch.setattr(
        imbalance,
        "build_logistic_pipeline",
        lambda _config, _model_name: RecordingEstimator(),
    )

    probabilities = imbalance.generate_oof_probabilities_for_model(
        train,
        cfg,
        "standard_logistic",
    )

    assert len(probabilities) == len(train)
    assert len(RecordingEstimator.fit_predict_pairs) == 5
    for fit_indices, predict_indices in RecordingEstimator.fit_predict_pairs:
        assert fit_indices.isdisjoint(predict_indices)


def test_threshold_metric_calculation_includes_f2():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.9])

    metrics = imbalance.classification_metrics_at_threshold(y_true, probability, 0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["f2"] == 0.5


def test_threshold_analysis_has_expected_grid_and_models():
    oof = pd.DataFrame(
        {
            "source_udi": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "standard_logistic_probability": [0.1, 0.2, 0.8, 0.9],
            "balanced_logistic_probability": [0.3, 0.4, 0.6, 0.7],
        }
    )

    analysis = imbalance.build_threshold_analysis(oof, thresholds=(0.25, 0.5))

    assert list(analysis.columns) == imbalance.REQUIRED_THRESHOLD_COLUMNS
    assert set(analysis["model"]) == set(imbalance.MODEL_NAMES)
    assert analysis.groupby("model")["threshold"].apply(list).to_dict() == {
        "balanced_logistic": [0.25, 0.5],
        "standard_logistic": [0.25, 0.5],
    }


def test_max_f1_and_max_f2_threshold_selection_prefers_higher_threshold_on_tie():
    analysis = threshold_frame()

    max_f1 = imbalance.select_threshold_by_metric(analysis, "standard_logistic", "f1")
    max_f2 = imbalance.select_threshold_by_metric(analysis, "standard_logistic", "f2")

    assert max_f1["threshold"] == 0.3
    assert max_f2["threshold"] == 0.3


def test_recall_70_candidate_logic_and_tie_breaking():
    analysis = threshold_frame()

    candidate = imbalance.select_recall_candidate(analysis, "standard_logistic")
    missing = imbalance.select_recall_candidate(analysis, "balanced_logistic")

    assert candidate is not None
    assert candidate["threshold"] == 0.3
    assert missing is None


def test_candidate_model_selection_policy_prefers_standard_when_ap_difference_is_small():
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.40},
        "balanced_logistic": {"average_precision": 0.405},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.21}},
        "balanced_logistic": {"max_f2": {"threshold": 0.19}},
    }

    selected = imbalance.select_development_candidate(oof_metrics, candidates)

    assert selected["selected_model"] == "standard_logistic"
    assert selected["selected_threshold"] == 0.21


def test_candidate_model_selection_policy_prefers_higher_ap_when_difference_is_large():
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.40},
        "balanced_logistic": {"average_precision": 0.43},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.21}},
        "balanced_logistic": {"max_f2": {"threshold": 0.19}},
    }

    selected = imbalance.select_development_candidate(oof_metrics, candidates)

    assert selected["selected_model"] == "balanced_logistic"
    assert selected["selected_threshold"] == 0.19


def test_validation_metrics_cannot_alter_selected_threshold():
    cfg = config()
    train = modeling_frame(1, 20)
    validation = modeling_frame(101, 8)
    oof_metrics = {
        "standard_logistic": {"average_precision": 0.40, "threshold_0_5": {}},
        "balanced_logistic": {"average_precision": 0.43, "threshold_0_5": {}},
    }
    candidates = {
        "standard_logistic": {"max_f2": {"threshold": 0.21}},
        "balanced_logistic": {"max_f2": {"threshold": 0.19}},
    }
    selected = imbalance.select_development_candidate(oof_metrics, candidates)
    validation_metrics_a = {
        "threshold_independent": {"roc_auc": 0.5, "average_precision": 0.1},
        "threshold_0_5": {"confusion_matrix": [[1, 0], [1, 0]]},
        "selected_threshold": {"confusion_matrix": [[1, 0], [0, 1]]},
    }
    validation_metrics_b = {
        "threshold_independent": {"roc_auc": 1.0, "average_precision": 1.0},
        "threshold_0_5": {"confusion_matrix": [[0, 1], [0, 1]]},
        "selected_threshold": {"confusion_matrix": [[0, 1], [1, 0]]},
    }

    summary_a = imbalance.build_metrics_summary(
        train, validation, cfg, oof_metrics, candidates, selected, validation_metrics_a
    )
    summary_b = imbalance.build_metrics_summary(
        train, validation, cfg, oof_metrics, candidates, selected, validation_metrics_b
    )

    assert summary_a["selected_threshold"] == summary_b["selected_threshold"] == 0.19
    assert summary_a["selected_model"] == summary_b["selected_model"] == "balanced_logistic"


def test_forbidden_leakage_fields_cannot_become_model_features():
    bad_config = replace(config(), numerical_features=(*NUMERIC_COLUMNS, "TWF"))

    with pytest.raises(ValueError, match="Forbidden column"):
        imbalance.build_logistic_pipeline(bad_config, "standard_logistic")

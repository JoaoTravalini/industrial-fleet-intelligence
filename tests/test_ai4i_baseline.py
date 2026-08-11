from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocessing import ai4i_modeling
from ml.training import ai4i_baseline as baseline

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


def modeling_frame(
    start_udi: int, row_count: int, include_unknown_type: bool = False
) -> pd.DataFrame:
    rows = []
    for offset in range(row_count):
        source_udi = start_udi + offset
        target = 1 if offset % 4 == 0 else 0
        machine_type = "X" if include_unknown_type and offset == 0 else ["L", "M", "H"][offset % 3]
        rows.append(
            {
                "source_udi": source_udi,
                "Type": machine_type,
                "Air temperature [K]": 295.0 + offset,
                "Process temperature [K]": 305.0 + offset * 0.5,
                "Rotational speed [rpm]": 1200 + offset * 20,
                "Torque [Nm]": 30.0 + offset * 1.5,
                "Tool wear [min]": offset * 3,
                "Machine failure": target,
            }
        )
    return pd.DataFrame(rows)


def split_summary(train_rows: int, validation_rows: int) -> dict[str, object]:
    return {"split_rows": {"train": train_rows, "validation": validation_rows}}


def test_baseline_input_paths_include_only_train_and_validation(tmp_path):
    paths = baseline.baseline_input_paths(tmp_path)

    assert set(paths) == {"train", "validation"}
    assert all("test" not in path.name for path in paths.values())


def test_feature_and_target_extraction_excludes_traceability():
    cfg = config()
    frame = modeling_frame(1, 12)

    features, target = baseline.extract_features_and_target(frame, cfg)

    assert list(features.columns) == ["Type", *NUMERIC_COLUMNS]
    assert "source_udi" not in features.columns
    assert target.name == "Machine failure"


def test_forbidden_leakage_features_are_rejected_by_policy():
    bad_config = replace(config(), numerical_features=(*NUMERIC_COLUMNS, "TWF"))

    with pytest.raises(ValueError, match="Forbidden column"):
        baseline.validate_feature_policy(bad_config)


def test_identifier_columns_cannot_become_model_features():
    bad_config = replace(config(), categorical_features=("Type", "source_udi", "Product ID"))

    with pytest.raises(ValueError, match="Forbidden column"):
        baseline.validate_feature_policy(bad_config)


def test_split_frame_rejects_extra_forbidden_columns():
    cfg = config()
    frame = modeling_frame(1, 12)
    frame["TWF"] = 0

    with pytest.raises(ValueError, match="unexpected columns"):
        baseline.validate_split_frame(frame, cfg, "train", 12)


def test_preprocessor_uses_expected_transformers():
    cfg = config()

    preprocessor = baseline.build_preprocessor(cfg)

    assert [name for name, _, _ in preprocessor.transformers] == ["categorical", "numerical"]
    categorical_pipeline = preprocessor.transformers[0][1]
    numerical_pipeline = preprocessor.transformers[1][1]
    assert isinstance(categorical_pipeline.named_steps["one_hot_encoder"], OneHotEncoder)
    assert isinstance(numerical_pipeline.named_steps["standard_scaler"], StandardScaler)


def test_categorical_unknown_values_are_ignored_at_validation_time():
    cfg = config()
    train = modeling_frame(1, 24)
    validation = modeling_frame(101, 8, include_unknown_type=True)
    pipeline = baseline.build_logistic_regression_pipeline(cfg)

    x_train, y_train = baseline.extract_features_and_target(train, cfg)
    x_validation, _ = baseline.extract_features_and_target(validation, cfg)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_validation)[:, 1]

    assert len(probabilities) == len(validation)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_validation_values_do_not_affect_fitted_numerical_scaler():
    cfg = config()
    train = modeling_frame(1, 20)
    validation = modeling_frame(101, 5)
    validation.loc[:, NUMERIC_COLUMNS] = 999999.0
    pipeline = baseline.build_logistic_regression_pipeline(cfg)

    x_train, y_train = baseline.extract_features_and_target(train, cfg)
    x_validation, _ = baseline.extract_features_and_target(validation, cfg)
    pipeline.fit(x_train, y_train)
    pipeline.predict_proba(x_validation)

    scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numerical"]
        .named_steps["standard_scaler"]
    )
    expected_means = train.loc[:, NUMERIC_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_means)


def test_metric_calculation_and_confusion_matrix_structure():
    y_true = np.array([0, 0, 1, 1])
    y_prediction = np.array([0, 1, 0, 1])
    y_probability = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = baseline.calculate_metrics(y_true, y_prediction, y_probability)

    assert set(baseline.REQUIRED_METRIC_KEYS).issubset(metrics)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_validation_prediction_construction_is_deterministic_and_bounded():
    cfg = config()
    validation = modeling_frame(101, 4)
    dummy_probability = np.array([0.25, 0.25, 0.25, 0.25])
    logistic_probability = np.array([0.1, 0.2, 0.8, 0.9])
    dummy_prediction = np.array([0, 0, 0, 0])
    logistic_prediction = np.array([0, 0, 1, 1])

    predictions = baseline.create_validation_predictions(
        validation,
        cfg,
        dummy_prediction,
        dummy_probability,
        logistic_prediction,
        logistic_probability,
    )

    assert list(predictions.columns) == [
        "source_udi",
        "target",
        "dummy_probability",
        "dummy_prediction",
        "logistic_probability",
        "logistic_prediction",
    ]
    assert predictions["source_udi"].tolist() == sorted(validation["source_udi"].tolist())
    assert predictions["dummy_probability"].between(0, 1).all()
    assert predictions["logistic_probability"].between(0, 1).all()


def test_coefficient_extraction_excludes_traceability_and_leakage_columns():
    cfg = config()
    train = modeling_frame(1, 24)
    pipeline = baseline.build_logistic_regression_pipeline(cfg)
    x_train, y_train = baseline.extract_features_and_target(train, cfg)
    pipeline.fit(x_train, y_train)

    coefficients = baseline.extract_logistic_coefficients(pipeline, cfg)

    assert list(coefficients.columns) == ["feature", "coefficient", "absolute_coefficient"]
    forbidden = {"source_udi", "UDI", "Product ID", *LEAKAGE_COLUMNS}
    assert forbidden.isdisjoint(set(coefficients["feature"]))
    assert coefficients["absolute_coefficient"].is_monotonic_decreasing


def test_metrics_summary_is_deterministic_and_records_no_test_usage():
    cfg = config()
    train = modeling_frame(1, 20)
    validation = modeling_frame(101, 8)
    metric_block = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "average_precision": 0.25,
        "confusion_matrix": [[6, 0], [2, 0]],
    }

    first = baseline.build_metrics_summary(train, validation, cfg, metric_block, metric_block)
    second = baseline.build_metrics_summary(train, validation, cfg, metric_block, metric_block)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["data"]["test_data_used"] is False
    assert first["data"]["predictive_feature_list"] == ["Type", *NUMERIC_COLUMNS]


def test_run_baseline_experiment_writes_artifacts_without_test_input(tmp_path):
    cfg = config()
    train = modeling_frame(1, 28)
    validation = modeling_frame(101, 12)

    result = baseline.run_baseline_experiment(
        train,
        validation,
        cfg,
        split_summary(len(train), len(validation)),
        root=tmp_path,
    )

    assert result.artifacts.metrics_json.exists()
    assert result.artifacts.validation_predictions_csv.exists()
    assert result.artifacts.logistic_coefficients_csv.exists()
    assert result.artifacts.markdown_report.exists()
    assert all(path.exists() and path.stat().st_size > 0 for path in result.artifacts.plot_paths)
    assert len(result.validation_predictions) == len(validation)
    assert result.metrics["data"]["splits_used"] == ["train", "validation"]
    assert result.metrics["experiment"]["test_set_status"] == baseline.TEST_SET_STATUS

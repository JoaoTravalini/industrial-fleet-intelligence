from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from ml.tracking import ai4i_mlflow


def tracking_config() -> dict[str, object]:
    return {
        "artifact_root_relative_path": ".mlflow/artifacts",
        "backend_relative_path": ".mlflow/mlflow.db",
        "experiment_name": "industrial-fleet-ai4i",
        "historical_import_policy": "retrospective_import",
        "project": "industrial-fleet-intelligence",
        "tracking_mode": "local_sqlite",
    }


def manifest() -> dict[str, object]:
    return {
        "experiment_name": "industrial-fleet-ai4i",
        "historical_import_policy": "retrospective_import",
        "runs": [
            {
                "development_stage": "baseline_modeling",
                "run_key": "baseline_logistic_regression",
                "run_name": "AI4I baseline Logistic Regression",
                "source_artifacts": [
                    "reports/ai4i/baseline_metrics.json",
                    "docs/ml/ai4i_baseline.md",
                ],
                "source_report": "reports/ai4i/baseline_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "imbalance_threshold_strategy",
                "run_key": "logistic_imbalance_strategy",
                "run_name": "AI4I Logistic Regression imbalance strategy",
                "source_artifacts": [
                    "reports/ai4i/imbalance_strategy_metrics.json",
                    "docs/ml/ai4i_imbalance_strategy.md",
                ],
                "source_report": "reports/ai4i/imbalance_strategy_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "model_family_comparison",
                "run_key": "model_comparison_logistic",
                "run_name": "AI4I model comparison Logistic Regression",
                "source_artifacts": [
                    "reports/ai4i/model_comparison_metrics.json",
                    "docs/ml/ai4i_model_comparison.md",
                ],
                "source_report": "reports/ai4i/model_comparison_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "model_family_comparison",
                "run_key": "model_comparison_random_forest",
                "run_name": "AI4I model comparison Random Forest",
                "source_artifacts": [
                    "reports/ai4i/model_comparison_metrics.json",
                    "docs/ml/ai4i_model_comparison.md",
                ],
                "source_report": "reports/ai4i/model_comparison_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "model_family_comparison",
                "run_key": "model_comparison_xgboost",
                "run_name": "AI4I model comparison XGBoost",
                "source_artifacts": [
                    "reports/ai4i/model_comparison_metrics.json",
                    "docs/ml/ai4i_model_comparison.md",
                ],
                "source_report": "reports/ai4i/model_comparison_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "random_forest_tuning",
                "run_key": "random_forest_targeted_tuning",
                "run_name": "AI4I Random Forest targeted tuning",
                "source_artifacts": [
                    "reports/ai4i/random_forest_tuning_metrics.json",
                    "docs/ml/ai4i_random_forest_tuning.md",
                ],
                "source_report": "reports/ai4i/random_forest_tuning_metrics.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "final_holdout_evaluation",
                "run_key": "final_holdout_evaluation",
                "run_name": "AI4I final holdout evaluation",
                "source_artifacts": [
                    "reports/ai4i/final_test_metrics.json",
                    "reports/ai4i/final_model_decision.json",
                    "docs/ml/ai4i_final_evaluation.md",
                ],
                "source_report": "reports/ai4i/final_test_metrics.json",
                "test_data_used": True,
                "tracking_provenance": "retrospective_import",
            },
            {
                "development_stage": "model_packaging",
                "run_key": "final_model_packaging",
                "run_name": "AI4I final model packaging",
                "source_artifacts": [
                    "reports/ai4i/model_packaging_summary.json",
                    "docs/ml/ai4i_model_serving.md",
                ],
                "source_report": "reports/ai4i/model_packaging_summary.json",
                "test_data_used": False,
                "tracking_provenance": "retrospective_import",
            },
        ],
    }


def threshold_metrics() -> dict[str, object]:
    return {
        "accuracy": 0.9,
        "balanced_accuracy": 0.8,
        "precision": 0.5,
        "recall": 0.7,
        "f1": 0.58,
        "f2": 0.65,
        "threshold": 0.14,
    }


def baseline_report() -> dict[str, object]:
    return {
        "random_seed": 42,
        "logistic_regression": {
            "accuracy": 0.91,
            "average_precision": 0.4,
            "balanced_accuracy": 0.7,
            "configuration": {"class_weight": None, "max_iter": 1000, "solver": "lbfgs"},
            "f1": 0.3,
            "precision": 0.6,
            "recall": 0.2,
            "roc_auc": 0.86,
        },
    }


def imbalance_report() -> dict[str, object]:
    return {
        "cv_configuration": {"n_splits": 5},
        "selected_model": "standard_logistic",
        "selected_threshold": 0.14,
        "threshold_candidates": {"standard_logistic": {"max_f2": threshold_metrics()}},
        "train_oof_results": {"standard_logistic": {"average_precision": 0.46, "roc_auc": 0.89}},
        "validation_results": {"selected_threshold": threshold_metrics()},
    }


def model_comparison_report() -> dict[str, object]:
    return {
        "candidate_selection_policy": {"primary_metric": "train_oof_average_precision"},
        "cv_configuration": {"n_splits": 5},
        "model_configurations": {
            "standard_logistic": {
                "family": "sklearn.linear_model.LogisticRegression",
                "max_iter": 1000,
            },
            "random_forest": {
                "family": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": 300,
            },
            "xgboost": {"family": "xgboost.XGBClassifier", "max_depth": 4},
        },
        "selected_model": "random_forest",
        "selected_threshold": 0.14,
        "threshold_candidates": {
            "standard_logistic": {"max_f2": threshold_metrics()},
            "random_forest": {"max_f2": threshold_metrics()},
            "xgboost": {"max_f2": threshold_metrics()},
        },
        "train_oof_results": {
            "standard_logistic": {"average_precision": 0.46, "roc_auc": 0.89},
            "random_forest": {"average_precision": 0.74, "roc_auc": 0.95},
            "xgboost": {"average_precision": 0.75, "roc_auc": 0.97},
        },
    }


def tuning_report() -> dict[str, object]:
    return {
        "nested_cv_configuration": {
            "inner": {"n_splits": 3},
            "outer": {"n_splits": 5},
        },
        "parameter_grid": {"candidate_count": 8},
        "promotion_policy": {
            "average_precision_delta": 0.003,
            "fixed_average_precision": 0.749,
            "promotion_delta_required": 0.005,
            "selected_candidate": "fixed_random_forest",
            "selected_threshold": 0.14,
            "threshold_source": "previous_fixed_random_forest_oof_max_f2",
        },
        "threshold_candidates": {"max_f2": threshold_metrics()},
        "tuned_nested_oof_results": {"average_precision": 0.752, "roc_auc": 0.959},
        "validation_results": {"selected_threshold": threshold_metrics()},
    }


def final_metrics_report() -> dict[str, object]:
    return {
        "development_positive_count": 288,
        "development_training_row_count": 8500,
        "final_model_configuration_hash": "abc123",
        "frozen_threshold": 0.14,
        "hyperparameters": {
            "class_weight": "balanced_subsample",
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_estimators": 300,
            "n_jobs": 1,
            "random_state": 42,
        },
        "model_family": "RandomForestClassifier",
        "test_metrics": {
            "threshold_0_14": threshold_metrics(),
            "threshold_0_5": threshold_metrics(),
            "threshold_independent": {"average_precision": 0.77, "roc_auc": 0.96},
        },
        "test_positive_count": 51,
        "test_row_count": 1500,
        "training_data_policy": "train + validation",
    }


def packaging_report() -> dict[str, object]:
    return {
        "final_config_hash": "abc123",
        "frozen_threshold": 0.14,
        "joblib_version": "1.5.3",
        "model_name": "ai4i-failure-risk-random-forest",
        "model_version": "1.0.0",
        "python_version": "3.12.10",
        "relative_local_artifact_path": "ml/artifacts/ai4i/final_model.joblib",
        "scikit_learn_version": "1.9.0",
        "serialization_format": "joblib",
        "test_data_used_for_packaging": False,
        "training_positive_count": 288,
        "training_row_count": 8500,
    }


def write_text(path: Path, text: str = "synthetic report") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict[str, object]) -> None:
    ai4i_mlflow.write_json(data, path)


def write_temp_project(root: Path) -> None:
    write_json(root / "ml/config/ai4i_mlflow_tracking.json", tracking_config())
    write_json(root / "ml/config/ai4i_mlflow_manifest.json", manifest())
    write_json(root / "reports/ai4i/baseline_metrics.json", baseline_report())
    write_json(root / "reports/ai4i/imbalance_strategy_metrics.json", imbalance_report())
    write_json(root / "reports/ai4i/model_comparison_metrics.json", model_comparison_report())
    write_json(root / "reports/ai4i/random_forest_tuning_metrics.json", tuning_report())
    write_json(root / "reports/ai4i/final_test_metrics.json", final_metrics_report())
    write_json(
        root / "reports/ai4i/final_model_decision.json",
        {"final_decision": {"configuration_hash": "abc123"}},
    )
    write_json(root / "reports/ai4i/model_packaging_summary.json", packaging_report())
    for relative_path in [
        "docs/ml/ai4i_baseline.md",
        "docs/ml/ai4i_imbalance_strategy.md",
        "docs/ml/ai4i_model_comparison.md",
        "docs/ml/ai4i_random_forest_tuning.md",
        "docs/ml/ai4i_final_evaluation.md",
        "docs/ml/ai4i_model_serving.md",
    ]:
        write_text(root / relative_path)


def test_tracking_uri_construction(tmp_path: Path):
    write_json(tmp_path / "ml/config/ai4i_mlflow_tracking.json", tracking_config())

    assert (
        ai4i_mlflow.tracking_uri(tmp_path) == f"sqlite:///{tmp_path.as_posix()}/.mlflow/mlflow.db"
    )


def test_manifest_validation_and_stable_run_keys():
    specs = ai4i_mlflow.validate_manifest(manifest())

    assert tuple(spec.run_key for spec in specs) == ai4i_mlflow.EXPECTED_RUN_KEYS
    assert len({spec.run_key for spec in specs}) == len(specs)


def test_manifest_rejects_manual_metrics():
    bad_manifest = manifest()
    bad_manifest["runs"][0]["metrics"] = {"average_precision": 0.5}

    with pytest.raises(ValueError, match="metrics"):
        ai4i_mlflow.validate_manifest(bad_manifest)


def test_metric_and_parameter_extraction_from_reports():
    params, metrics = ai4i_mlflow.baseline_run_payload(baseline_report())

    assert params["model_family"] == "LogisticRegression"
    assert params["max_iter"] == "1000"
    assert metrics["validation_average_precision"] == 0.4
    assert metrics["validation_roc_auc"] == 0.86


def test_model_comparison_extraction_uses_contextual_metric_names():
    params, metrics = ai4i_mlflow.model_comparison_payload(
        "model_comparison_random_forest",
        model_comparison_report(),
    )

    assert params["model_key"] == "random_forest"
    assert params["n_estimators"] == "300"
    assert metrics["oof_average_precision"] == 0.74
    assert metrics["oof_max_f2_threshold_precision"] == 0.5


def test_historical_provenance_and_test_data_tags(tmp_path: Path):
    write_temp_project(tmp_path)
    prepared = ai4i_mlflow.load_prepared_runs(tmp_path)
    tags_by_key = {item.spec.run_key: item.tags for item in prepared}

    assert {tags["tracking_provenance"] for tags in tags_by_key.values()} == {
        "retrospective_import"
    }
    assert tags_by_key["final_holdout_evaluation"]["test_data_used"] == "true"
    assert tags_by_key["final_model_packaging"]["test_data_used"] == "false"
    assert tags_by_key["final_holdout_evaluation"]["model_frozen_before_evaluation"] == "true"


def test_idempotent_import_creates_no_duplicate_semantic_runs(tmp_path: Path):
    write_temp_project(tmp_path)

    first = ai4i_mlflow.import_historical_runs(tmp_path)
    second = ai4i_mlflow.import_historical_runs(tmp_path)

    assert len(first.imported_run_keys) == len(ai4i_mlflow.EXPECTED_RUN_KEYS)
    assert first.existing_run_keys == ()
    assert second.imported_run_keys == ()
    assert len(second.existing_run_keys) == len(ai4i_mlflow.EXPECTED_RUN_KEYS)
    client, _uri, config = ai4i_mlflow.configure_tracking(tmp_path)
    experiment_id = ai4i_mlflow.get_or_create_experiment(client, config, tmp_path)
    for run_key in ai4i_mlflow.EXPECTED_RUN_KEYS:
        assert len(ai4i_mlflow.search_runs_by_key(client, experiment_id, run_key)) == 1


def test_duplicate_run_detection(tmp_path: Path):
    write_temp_project(tmp_path)
    ai4i_mlflow.import_historical_runs(tmp_path)
    client, _uri, config = ai4i_mlflow.configure_tracking(tmp_path)
    experiment_id = ai4i_mlflow.get_or_create_experiment(client, config, tmp_path)
    with mlflow.start_run(experiment_id=experiment_id, run_name="duplicate"):
        mlflow.set_tag("run_key", "baseline_logistic_regression")

    with pytest.raises(ValueError, match="Duplicate"):
        ai4i_mlflow.import_historical_runs(tmp_path)


def test_source_conflict_detection(tmp_path: Path):
    write_temp_project(tmp_path)
    ai4i_mlflow.import_historical_runs(tmp_path)
    changed = baseline_report()
    changed["logistic_regression"]["average_precision"] = 0.99
    write_json(tmp_path / "reports/ai4i/baseline_metrics.json", changed)

    with pytest.raises(ValueError, match="conflicts"):
        ai4i_mlflow.import_historical_runs(tmp_path)


def test_runtime_ids_excluded_from_deterministic_summary():
    summary = ai4i_mlflow.deterministic_tracking_summary(tracking_config(), manifest())

    assert not any("run_id" in str(item) for item in summary.values())
    assert not any("experiment_id" in str(item) for item in summary.values())
    assert "tracking_uri" not in str(summary)


def test_raw_or_processed_dataset_paths_cannot_be_logged(tmp_path: Path):
    spec = ai4i_mlflow.RunSpec(
        run_key="bad",
        run_name="bad",
        development_stage="bad",
        source_report="data/processed/ai4i/train.csv",
        source_artifacts=("data/processed/ai4i/train.csv",),
        test_data_used=False,
        tracking_provenance="retrospective_import",
    )

    with pytest.raises(ValueError, match="Disallowed"):
        ai4i_mlflow.validate_source_artifacts(tmp_path, spec)

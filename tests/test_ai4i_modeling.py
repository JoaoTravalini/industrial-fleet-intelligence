from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from ml.preprocessing import ai4i_modeling as modeling

NUMERIC_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
LEAKAGE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")


def valid_config() -> modeling.ModelingConfig:
    return modeling.ModelingConfig(
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


def synthetic_source_df(row_count: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(1, row_count + 1):
        machine_failure = 1 if index <= row_count // 5 else 0
        rows.append(
            {
                "UDI": index,
                "Product ID": f"P{index:05d}",
                "Type": ["L", "M", "H"][index % 3],
                "Air temperature [K]": 298.0 + (index % 7) * 0.1,
                "Process temperature [K]": 308.0 + (index % 5) * 0.1,
                "Rotational speed [rpm]": 1200 + index,
                "Torque [Nm]": 35.0 + (index % 11),
                "Tool wear [min]": index % 250,
                "Machine failure": machine_failure,
                "TWF": 1 if machine_failure and index % 5 == 0 else 0,
                "HDF": 1 if machine_failure and index % 5 == 1 else 0,
                "PWF": 1 if machine_failure and index % 5 == 2 else 0,
                "OSF": 1 if machine_failure and index % 5 == 3 else 0,
                "RNF": 1 if machine_failure and index % 5 == 4 else 0,
            }
        )
    return pd.DataFrame(rows)


def split_udi_sets(splits: dict[str, pd.DataFrame]) -> dict[str, set[int]]:
    return {name: set(frame["source_udi"].tolist()) for name, frame in splits.items()}


def test_modeling_configuration_validation_accepts_policy():
    config = valid_config()

    modeling.validate_modeling_config(config)

    assert modeling.predictive_feature_columns(config) == ["Type", *NUMERIC_COLUMNS]


def test_modeling_configuration_validation_rejects_bad_split_sum():
    config = replace(valid_config(), test_fraction=0.20)

    with pytest.raises(ValueError, match="sum to 1.0"):
        modeling.validate_modeling_config(config)


def test_safe_modeling_frame_contains_traceability_features_and_target_only():
    config = valid_config()
    source_df = synthetic_source_df()

    frame = modeling.construct_modeling_frame(source_df, config)

    assert list(frame.columns) == ["source_udi", "Type", *NUMERIC_COLUMNS, "Machine failure"]
    assert frame["source_udi"].tolist() == source_df["UDI"].tolist()
    assert frame["source_udi"].is_unique


def test_identifier_and_leakage_columns_are_excluded_from_modeling_frame():
    config = valid_config()
    frame = modeling.construct_modeling_frame(synthetic_source_df(), config)

    forbidden_columns = {"UDI", "Product ID", *LEAKAGE_COLUMNS}

    assert forbidden_columns.isdisjoint(frame.columns)


def test_leakage_columns_cannot_become_predictive_features():
    config = replace(valid_config(), numerical_features=(*NUMERIC_COLUMNS, "TWF"))

    with pytest.raises(ValueError, match="Forbidden column"):
        modeling.validate_forbidden_feature_columns(config)


def test_source_identifier_cannot_become_predictive_feature():
    config = replace(valid_config(), categorical_features=("Type", "UDI"))

    with pytest.raises(ValueError, match="Forbidden column"):
        modeling.validate_forbidden_feature_columns(config)


def test_deterministic_stratified_splitting_is_reproducible():
    config = valid_config()
    frame = modeling.construct_modeling_frame(synthetic_source_df(), config)

    first = modeling.create_stratified_splits(frame, config)
    second = modeling.create_stratified_splits(frame, config)

    assert {name: len(split) for name, split in first.items()} == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    for split_name in modeling.SPLIT_NAMES:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])


def test_stratification_preserves_target_rates_closely():
    config = valid_config()
    frame = modeling.construct_modeling_frame(synthetic_source_df(), config)
    splits = modeling.create_stratified_splits(frame, config)

    assert modeling.split_target_counts(splits["train"], "Machine failure") == {
        "0": 56,
        "1": 14,
    }
    assert modeling.split_target_counts(splits["validation"], "Machine failure") == {
        "0": 12,
        "1": 3,
    }
    assert modeling.split_target_counts(splits["test"], "Machine failure") == {
        "0": 12,
        "1": 3,
    }


def test_split_integrity_reports_no_overlap_and_complete_coverage():
    config = valid_config()
    source_df = synthetic_source_df()
    frame = modeling.construct_modeling_frame(source_df, config)
    splits = modeling.create_stratified_splits(frame, config)

    report = modeling.validate_split_integrity(
        splits,
        config,
        source_udis=source_df["UDI"].tolist(),
        expected_total_rows=len(source_df),
    )
    udi_sets = split_udi_sets(splits)

    assert report.is_valid
    assert not (udi_sets["train"] & udi_sets["validation"])
    assert not (udi_sets["train"] & udi_sets["test"])
    assert not (udi_sets["validation"] & udi_sets["test"])
    assert set().union(*udi_sets.values()) == set(source_df["UDI"].tolist())


def test_split_assignments_are_minimal_and_consistent():
    config = valid_config()
    frame = modeling.construct_modeling_frame(synthetic_source_df(), config)
    splits = modeling.create_stratified_splits(frame, config)

    assignments = modeling.build_split_assignments(splits)
    results = modeling.validate_split_assignments(assignments, splits, config)

    assert list(assignments.columns) == ["source_udi", "split"]
    assert len(assignments) == len(frame)
    assert assignments["source_udi"].is_unique
    assert all(result.status is modeling.Status.PASS for result in results)


def test_split_summary_contains_feature_policy_and_counts():
    config = valid_config()
    frame = modeling.construct_modeling_frame(synthetic_source_df(), config)
    splits = modeling.create_stratified_splits(frame, config)

    summary = modeling.build_split_summary(splits, config)

    assert summary["random_seed"] == 42
    assert summary["total_rows"] == 100
    assert summary["feature_list"] == ["Type", *NUMERIC_COLUMNS]
    assert summary["excluded_identifiers"] == ["Product ID"]
    assert summary["excluded_leakage_sensitive_fields"] == list(LEAKAGE_COLUMNS)
    assert summary["target_counts_per_split"]["train"] == {"0": 56, "1": 14}


def test_generated_artifact_validation_checks_summary_consistency(tmp_path):
    config = valid_config()
    source_df = synthetic_source_df()
    frame = modeling.construct_modeling_frame(source_df, config)
    splits = modeling.create_stratified_splits(frame, config)
    summary = modeling.build_split_summary(splits, config)
    summary_path = tmp_path / "modeling_split_summary.json"

    modeling.write_modeling_artifacts(splits, summary, tmp_path, summary_path)
    report = modeling.validate_generated_artifacts(source_df, config, tmp_path, summary_path)

    assert report.is_valid
    assert any(result.name == "split_summary File" for result in report.results)
    assert any(result.name == "Split Summary Consistency" for result in report.results)

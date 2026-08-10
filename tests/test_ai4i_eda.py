import json

import pandas as pd

from ml.analysis import ai4i_eda


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "UDI": 1,
                "Product ID": "L00001",
                "Type": "L",
                "Air temperature [K]": 298.0,
                "Process temperature [K]": 308.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 10,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
            {
                "UDI": 2,
                "Product ID": "M00002",
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1300,
                "Torque [Nm]": 55.0,
                "Tool wear [min]": 100,
                "Machine failure": 1,
                "TWF": 1,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
            {
                "UDI": 3,
                "Product ID": "H00003",
                "Type": "H",
                "Air temperature [K]": 299.0,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1600,
                "Torque [Nm]": 35.0,
                "Tool wear [min]": 20,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            },
            {
                "UDI": 4,
                "Product ID": "L00004",
                "Type": "L",
                "Air temperature [K]": 305.0,
                "Process temperature [K]": 315.0,
                "Rotational speed [rpm]": 1200,
                "Torque [Nm]": 65.0,
                "Tool wear [min]": 200,
                "Machine failure": 1,
                "TWF": 0,
                "HDF": 1,
                "PWF": 1,
                "OSF": 0,
                "RNF": 0,
            },
        ]
    )


def test_machine_failure_summary_calculates_distribution_and_percentage():
    result = ai4i_eda.machine_failure_summary(sample_df())

    assert result["positive_count"] == 2
    assert result["negative_count"] == 2
    assert result["failure_percentage"] == 50.0
    assert result["positive_to_negative_ratio"] == 1.0


def test_type_failure_summary_calculates_failure_rates():
    summary = ai4i_eda.type_failure_summary(sample_df()).set_index("type")

    assert summary.loc["L", "row_count"] == 2
    assert summary.loc["L", "failure_count"] == 1
    assert summary.loc["L", "failure_rate"] == 50.0
    assert summary.loc["M", "failure_rate"] == 100.0
    assert summary.loc["H", "failure_rate"] == 0.0


def test_failure_mode_counts_and_overlap_logic():
    df = sample_df()
    positive_summary = ai4i_eda.failure_mode_positive_summary(df).set_index("label")
    overlap = ai4i_eda.failure_mode_overlap_counts(df)
    observed = ai4i_eda.observed_target_flag_relationship(df)

    assert positive_summary.loc["TWF", "count"] == 1
    assert positive_summary.loc["HDF", "count"] == 1
    assert positive_summary.loc["PWF", "count"] == 1
    assert overlap == {
        "zero_active_failure_mode_flags": 2,
        "exactly_one_active_failure_mode_flag": 1,
        "more_than_one_active_failure_mode_flag": 1,
    }
    assert observed["machine_failure_1_with_any_failure_mode_flag"] == 2
    assert observed["machine_failure_0_with_any_failure_mode_flag"] == 0


def test_descriptive_statistics_has_expected_structure():
    statistics = ai4i_eda.descriptive_statistics(sample_df())

    assert set(statistics["variable"]) == set(ai4i_eda.PRIMARY_NUMERIC_COLUMNS)
    assert {"count", "mean", "std", "min", "25%", "median", "75%", "max"}.issubset(
        statistics.columns
    )


def test_numeric_comparison_by_failure_has_each_variable_and_class():
    comparison = ai4i_eda.numeric_comparison_by_failure(sample_df())

    assert len(comparison) == len(ai4i_eda.PRIMARY_NUMERIC_COLUMNS) * 2
    assert set(comparison["machine_failure"]) == {0, 1}
    assert {"mean", "median", "std"}.issubset(comparison.columns)


def test_correlation_matrix_has_expected_shape():
    matrix = ai4i_eda.correlation_matrix(sample_df())

    assert matrix.shape == (6, 6)
    assert list(matrix.columns) == ai4i_eda.CORRELATION_COLUMNS


def test_build_summary_is_json_serializable_and_deterministic():
    df = sample_df()
    descriptive = ai4i_eda.descriptive_statistics(df)
    type_summary = ai4i_eda.type_failure_summary(df)
    failure_modes = ai4i_eda.failure_mode_summary(df)
    numeric_by_failure = ai4i_eda.numeric_comparison_by_failure(df)
    correlations = ai4i_eda.correlation_matrix(df)

    summary = ai4i_eda.build_summary(
        df,
        descriptive,
        type_summary,
        failure_modes,
        numeric_by_failure,
        correlations,
    )

    first = json.dumps(summary, sort_keys=True)
    second = json.dumps(summary, sort_keys=True)
    assert first == second
    assert summary["dataset"]["name"] == ai4i_eda.DATASET_NAME


def test_create_plots_writes_expected_png_files(tmp_path):
    df = sample_df()
    type_summary = ai4i_eda.type_failure_summary(df)

    plot_paths = ai4i_eda.create_plots(df, type_summary, tmp_path)

    assert len(plot_paths) == 10
    assert all(path.exists() for path in plot_paths)
    assert all(path.suffix == ".png" for path in plot_paths)
    assert all(path.stat().st_size > 0 for path in plot_paths)

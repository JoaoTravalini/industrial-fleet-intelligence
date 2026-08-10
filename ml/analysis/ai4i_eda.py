"""Reusable exploratory analysis for the AI4I 2020 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
DATASET_SOURCE = "UCI Machine Learning Repository"
DATASET_ID = 601
DATASET_DOI = "10.24432/C5HS5C"
RAW_DATASET_RELATIVE_PATH = Path("data") / "raw" / "ai4i" / "ai4i2020.csv"
REPORTS_RELATIVE_DIR = Path("reports") / "ai4i"
PLOTS_RELATIVE_DIR = Path("docs") / "assets" / "ai4i"
EDA_DOC_RELATIVE_PATH = Path("docs") / "data" / "ai4i_eda.md"
TARGET_COLUMN = "Machine failure"
TYPE_COLUMN = "Type"
UDI_COLUMN = "UDI"
PRODUCT_ID_COLUMN = "Product ID"
TYPE_ORDER = ["L", "M", "H"]
TARGET_ORDER = [0, 1]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
PRIMARY_NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CORRELATION_COLUMNS = [*PRIMARY_NUMERIC_COLUMNS, TARGET_COLUMN]
EXPECTED_COLUMNS = [
    UDI_COLUMN,
    PRODUCT_ID_COLUMN,
    TYPE_COLUMN,
    *PRIMARY_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
]
NUMERIC_PLOT_FILENAMES = {
    "Air temperature [K]": "air_temperature_distribution.png",
    "Process temperature [K]": "process_temperature_distribution.png",
    "Rotational speed [rpm]": "rotational_speed_distribution.png",
    "Torque [Nm]": "torque_distribution.png",
    "Tool wear [min]": "tool_wear_distribution.png",
}


@dataclass(frozen=True)
class EdaArtifacts:
    """Paths produced by a full AI4I EDA run."""

    summary_json: Path
    descriptive_statistics_csv: Path
    type_failure_summary_csv: Path
    failure_mode_summary_csv: Path
    numeric_by_failure_summary_csv: Path
    correlation_matrix_csv: Path
    markdown_report: Path
    plot_paths: list[Path]


@dataclass(frozen=True)
class EdaResult:
    """Full result object returned by the EDA runner."""

    summary: dict[str, Any]
    artifacts: EdaArtifacts


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_dataset_path(root: Path | None = None) -> Path:
    return (root or project_root()) / RAW_DATASET_RELATIVE_PATH


def reports_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / REPORTS_RELATIVE_DIR


def plots_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / PLOTS_RELATIVE_DIR


def eda_doc_path(root: Path | None = None) -> Path:
    return (root or project_root()) / EDA_DOC_RELATIVE_PATH


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_file = path or raw_dataset_path()
    if not dataset_file.exists():
        raise FileNotFoundError(f"AI4I raw dataset was not found: {dataset_file}")
    return pd.read_csv(dataset_file)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing expected AI4I column(s): " + ", ".join(missing))


def to_builtin(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def round_float(value: Any, digits: int = 6) -> float | None:
    converted = to_builtin(value)
    if converted is None:
        return None
    return round(float(converted), digits)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({key: to_builtin(value) for key, value in row.items()})
    return records


def dataframe_dtypes(df: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def missing_value_counts(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(count) for column, count in df.isna().sum().items()}


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "data_types": dataframe_dtypes(df),
        "missing_value_counts": missing_value_counts(df),
        "duplicated_full_rows": int(df.duplicated().sum()),
        "unique_udi_count": int(df[UDI_COLUMN].nunique(dropna=True)),
        "unique_product_id_count": int(df[PRODUCT_ID_COLUMN].nunique(dropna=True)),
    }


def value_distribution(
    series: pd.Series, order: list[Any] | None = None
) -> dict[str, dict[str, float | int]]:
    total = len(series)
    counts = series.value_counts(dropna=False)
    keys = order if order is not None else sorted(counts.index.tolist())
    distribution: dict[str, dict[str, float | int]] = {}
    for key in keys:
        count = int(counts.get(key, 0))
        distribution[str(key)] = {
            "count": count,
            "percentage": round_float((count / total) * 100) if total else 0.0,
        }
    return distribution


def machine_failure_summary(df: pd.DataFrame) -> dict[str, Any]:
    distribution = value_distribution(df[TARGET_COLUMN], TARGET_ORDER)
    positive_count = int(distribution["1"]["count"])
    negative_count = int(distribution["0"]["count"])
    return {
        "distribution": distribution,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "failure_percentage": round_float((positive_count / len(df)) * 100) if len(df) else 0.0,
        "positive_to_negative_ratio": round_float(positive_count / negative_count)
        if negative_count
        else None,
        "is_class_imbalanced": positive_count < negative_count,
    }


def type_failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    for machine_type in TYPE_ORDER:
        subset = df[df[TYPE_COLUMN] == machine_type]
        row_count = int(len(subset))
        failure_count = int(subset[TARGET_COLUMN].sum())
        non_failure_count = row_count - failure_count
        rows.append(
            {
                "type": machine_type,
                "row_count": row_count,
                "percentage_of_dataset": round_float((row_count / total_rows) * 100)
                if total_rows
                else 0.0,
                "failure_count": failure_count,
                "non_failure_count": non_failure_count,
                "failure_rate": round_float((failure_count / row_count) * 100)
                if row_count
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def failure_mode_positive_summary(df: pd.DataFrame) -> pd.DataFrame:
    machine_failure_positives = int(df[TARGET_COLUMN].sum())
    total_rows = len(df)
    rows: list[dict[str, Any]] = []
    for column in FAILURE_MODE_COLUMNS:
        positive_count = int(df[column].sum())
        rows.append(
            {
                "section": "failure_mode",
                "label": column,
                "count": positive_count,
                "percentage_of_rows": round_float((positive_count / total_rows) * 100)
                if total_rows
                else 0.0,
                "percentage_of_machine_failure_positives": round_float(
                    (positive_count / machine_failure_positives) * 100
                )
                if machine_failure_positives
                else None,
            }
        )
    return pd.DataFrame(rows)


def failure_mode_overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    active_flags = df[FAILURE_MODE_COLUMNS].sum(axis=1)
    return {
        "zero_active_failure_mode_flags": int((active_flags == 0).sum()),
        "exactly_one_active_failure_mode_flag": int((active_flags == 1).sum()),
        "more_than_one_active_failure_mode_flag": int((active_flags > 1).sum()),
    }


def observed_target_flag_relationship(df: pd.DataFrame) -> dict[str, int]:
    active_flags = df[FAILURE_MODE_COLUMNS].sum(axis=1)
    has_any_failure_mode = active_flags > 0
    machine_failure = df[TARGET_COLUMN] == 1
    return {
        "machine_failure_1_with_any_failure_mode_flag": int(
            (machine_failure & has_any_failure_mode).sum()
        ),
        "machine_failure_1_with_no_failure_mode_flags": int(
            (machine_failure & ~has_any_failure_mode).sum()
        ),
        "machine_failure_0_with_any_failure_mode_flag": int(
            (~machine_failure & has_any_failure_mode).sum()
        ),
        "machine_failure_0_with_no_failure_mode_flags": int(
            (~machine_failure & ~has_any_failure_mode).sum()
        ),
    }


def failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = dataframe_records(failure_mode_positive_summary(df))
    for label, count in failure_mode_overlap_counts(df).items():
        rows.append(
            {
                "section": "failure_mode_overlap",
                "label": label,
                "count": count,
                "percentage_of_rows": round_float((count / total_rows) * 100)
                if total_rows
                else 0.0,
                "percentage_of_machine_failure_positives": None,
            }
        )
    for label, count in observed_target_flag_relationship(df).items():
        rows.append(
            {
                "section": "target_flag_relationship",
                "label": label,
                "count": count,
                "percentage_of_rows": round_float((count / total_rows) * 100)
                if total_rows
                else 0.0,
                "percentage_of_machine_failure_positives": None,
            }
        )
    return pd.DataFrame(rows)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    statistics = df[PRIMARY_NUMERIC_COLUMNS].describe(percentiles=[0.25, 0.5, 0.75]).T
    statistics = statistics[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    statistics = statistics.rename(columns={"50%": "median"})
    statistics.index.name = "variable"
    return statistics.reset_index().round(6)


def numeric_comparison_by_failure(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in PRIMARY_NUMERIC_COLUMNS:
        for target_value in TARGET_ORDER:
            subset = df[df[TARGET_COLUMN] == target_value][variable]
            rows.append(
                {
                    "variable": variable,
                    "machine_failure": target_value,
                    "count": int(subset.count()),
                    "mean": round_float(subset.mean()),
                    "median": round_float(subset.median()),
                    "std": round_float(subset.std()),
                }
            )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[CORRELATION_COLUMNS].corr(method="pearson").round(6)


def numeric_ranges(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        column: {
            "minimum": round_float(df[column].min()),
            "maximum": round_float(df[column].max()),
        }
        for column in PRIMARY_NUMERIC_COLUMNS
    }


def strongest_target_correlations(
    correlations: pd.DataFrame, limit: int = 3
) -> list[dict[str, Any]]:
    target_correlations = correlations[TARGET_COLUMN].drop(index=TARGET_COLUMN)
    ordered = target_correlations.reindex(
        target_correlations.abs().sort_values(ascending=False).index
    ).head(limit)
    return [
        {"variable": str(variable), "correlation": round_float(value)}
        for variable, value in ordered.items()
    ]


def build_summary(
    df: pd.DataFrame,
    descriptive_stats: pd.DataFrame,
    type_summary: pd.DataFrame,
    failure_modes: pd.DataFrame,
    numeric_by_failure: pd.DataFrame,
    correlations: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "uci_dataset_id": DATASET_ID,
            "doi": DATASET_DOI,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "quality": quality_summary(df),
        "machine_failure": machine_failure_summary(df),
        "types": {row["type"]: row for row in dataframe_records(type_summary)},
        "failure_modes": {
            "positive_counts": {
                row["label"]: int(row["count"])
                for row in dataframe_records(failure_modes)
                if row["section"] == "failure_mode"
            },
            "overlap": failure_mode_overlap_counts(df),
            "observed_target_flag_relationship": observed_target_flag_relationship(df),
        },
        "numeric_ranges": numeric_ranges(df),
        "descriptive_statistics": dataframe_records(descriptive_stats),
        "numeric_by_machine_failure": dataframe_records(numeric_by_failure),
        "correlations": {
            "matrix": {
                row_label: {
                    column: round_float(correlations.loc[row_label, column])
                    for column in correlations.columns
                }
                for row_label in correlations.index
            },
            "strongest_machine_failure_associations": strongest_target_correlations(correlations),
        },
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_bar_chart(values: pd.Series, title: str, x_label: str, y_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    values.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.tick_params(axis="x", rotation=0)
    for index, value in enumerate(values.tolist()):
        axis.text(index, value, str(int(value)), ha="center", va="bottom")
    save_figure(path)
    plt.close(figure)


def create_histogram(series: pd.Series, title: str, x_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(series, bins=30)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Row count")
    save_figure(path)
    plt.close(figure)


def create_correlation_plot(correlations: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlations.values, vmin=-1, vmax=1)
    axis.set_title("AI4I Pearson Correlation Matrix")
    axis.set_xticks(range(len(correlations.columns)))
    axis.set_xticklabels(correlations.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlations.index)))
    axis.set_yticklabels(correlations.index)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for row_index, row_label in enumerate(correlations.index):
        for column_index, column_label in enumerate(correlations.columns):
            value = correlations.loc[row_label, column_label]
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
    save_figure(path)
    plt.close(figure)


def create_plots(df: pd.DataFrame, type_summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    machine_failure_counts = df[TARGET_COLUMN].value_counts().reindex(TARGET_ORDER, fill_value=0)
    path = output_dir / "machine_failure_distribution.png"
    create_bar_chart(
        machine_failure_counts,
        "AI4I Machine Failure Distribution",
        "Machine failure class",
        "Row count",
        path,
    )
    plot_paths.append(path)

    type_counts = df[TYPE_COLUMN].value_counts().reindex(TYPE_ORDER, fill_value=0)
    path = output_dir / "product_type_distribution.png"
    create_bar_chart(type_counts, "AI4I Product Type Distribution", "Type", "Row count", path)
    plot_paths.append(path)

    failure_mode_counts = df[FAILURE_MODE_COLUMNS].sum().reindex(FAILURE_MODE_COLUMNS)
    path = output_dir / "failure_mode_counts.png"
    create_bar_chart(
        failure_mode_counts,
        "AI4I Failure Mode Positive Counts",
        "Failure mode flag",
        "Positive row count",
        path,
    )
    plot_paths.append(path)

    for column, filename in NUMERIC_PLOT_FILENAMES.items():
        path = output_dir / filename
        create_histogram(df[column], f"AI4I {column} Distribution", column, path)
        plot_paths.append(path)

    failure_rates = type_summary_df.set_index("type")["failure_rate"].reindex(TYPE_ORDER)
    path = output_dir / "failure_rate_by_type.png"
    create_bar_chart(
        failure_rates,
        "AI4I Machine Failure Rate by Product Type",
        "Type",
        "Failure rate (%)",
        path,
    )
    plot_paths.append(path)

    path = output_dir / "correlation_matrix.png"
    create_correlation_plot(correlation_matrix(df), path)
    plot_paths.append(path)

    return plot_paths


def render_markdown(summary: dict[str, Any]) -> str:
    quality = summary["quality"]
    machine_failure = summary["machine_failure"]
    positive_count = machine_failure["positive_count"]
    rows = summary["dataset"]["rows"]
    failure_pct = machine_failure["failure_percentage"]
    strongest = summary["correlations"]["strongest_machine_failure_associations"]
    type_lines = [
        f"- Type `{machine_type}`: {values['row_count']} rows, "
        f"{values['failure_count']} failures, {values['failure_rate']}% failure rate."
        for machine_type, values in summary["types"].items()
    ]
    failure_mode_lines = [
        f"- `{mode}`: {count} positive rows."
        for mode, count in summary["failure_modes"]["positive_counts"].items()
    ]
    correlation_lines = [
        f"- `{item['variable']}`: Pearson correlation with Machine failure = {item['correlation']}."
        for item in strongest
    ]
    overlap = summary["failure_modes"]["overlap"]

    return "\n".join(
        [
            "# AI4I Exploratory Data Analysis",
            "",
            "## Dataset Overview",
            f"FACT: The dataset contains {rows} rows and {summary['dataset']['columns']} columns.",
            f"FACT: The source is the {summary['dataset']['source']} dataset ID "
            f"{summary['dataset']['uci_dataset_id']} with DOI {summary['dataset']['doi']}.",
            "OBSERVATION: This is an external public synthetic dataset and is separate from "
            "the fictional `MCH-XXXX` PostgreSQL fleet and future streaming telemetry simulator.",
            "",
            "## Data Quality",
            f"FACT: Missing values total {sum(quality['missing_value_counts'].values())}; "
            f"duplicated full rows total {quality['duplicated_full_rows']}.",
            f"FACT: Unique UDI count is {quality['unique_udi_count']} and unique Product ID count "
            f"is {quality['unique_product_id_count']}.",
            "OBSERVATION: The raw dataset is left unchanged; this EDA produces only "
            "derived reports and static charts.",
            "",
            "## Target Distribution",
            f"FACT: Machine failure has {positive_count} positive observations out of {rows} "
            f"({failure_pct}%).",
            "OBSERVATION: The target is class imbalanced because non-failure observations dominate "
            "the dataset.",
            "",
            "## Product Types",
            *type_lines,
            "OBSERVATION: Type-level summaries are descriptive only and do not establish causal "
            "business meaning.",
            "",
            "## Failure Modes",
            *failure_mode_lines,
            f"FACT: Rows with zero active failure-mode flags: "
            f"{overlap['zero_active_failure_mode_flags']}.",
            f"FACT: Rows with exactly one active failure-mode flag: "
            f"{overlap['exactly_one_active_failure_mode_flag']}.",
            f"FACT: Rows with more than one active failure-mode flag: "
            f"{overlap['more_than_one_active_failure_mode_flag']}.",
            "OBSERVATION: Failure-mode flags are reported factually; this phase does not assume "
            "that `Machine failure` is exactly the logical OR of the individual flags.",
            "",
            "## Numerical Variables",
            "FACT: Descriptive statistics and target-group comparisons are written to "
            "`reports/ai4i/descriptive_statistics.csv` and "
            "`reports/ai4i/numeric_by_failure_summary.csv`.",
            "OBSERVATION: Differences between failure and non-failure groups are descriptive only; "
            "no statistical tests or causal claims are made in this phase.",
            "",
            "## Correlation Overview",
            *correlation_lines,
            "OBSERVATION: Correlation describes linear association only. It is not evidence of "
            "causality and is not used here for feature selection.",
            "",
            "## Key Observations",
            "OBSERVATION: The dataset is structurally complete for the expected AI4I columns.",
            "OBSERVATION: Machine failure positives are rare relative to non-failures.",
            "OBSERVATION: Failure-mode flags are sparse and target-adjacent by design.",
            "",
            "## Modeling Considerations",
            "FUTURE MODELING CONSIDERATION: The class imbalance should be handled deliberately "
            "during a later modeling phase, without changing the raw dataset here.",
            "FUTURE MODELING CONSIDERATION: Failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, "
            "and `RNF` may have a very direct relationship with `Machine failure` and could "
            "cause target leakage if blindly included as model input features. This phase does "
            "not make the feature-selection decision.",
            "",
            "## Limitations",
            "FACT: This EDA does not perform preprocessing, feature engineering, train/test "
            "splitting, model training, or database import.",
            "FACT: AI4I must not be described as real industrial data, proprietary data, or "
            "data generated by the local fictional fleet.",
            "",
        ]
    )


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary), encoding="utf-8")


def run_eda(
    dataset_file: Path | None = None,
    reports_dir: Path | None = None,
    plots_dir: Path | None = None,
    markdown_path: Path | None = None,
) -> EdaResult:
    df = load_dataset(dataset_file)
    validate_required_columns(df)

    reports = reports_dir or reports_directory()
    plots = plots_dir or plots_directory()
    markdown = markdown_path or eda_doc_path()

    descriptive_stats = descriptive_statistics(df)
    type_summary = type_failure_summary(df)
    failure_modes = failure_mode_summary(df)
    numeric_by_failure = numeric_comparison_by_failure(df)
    correlations = correlation_matrix(df)
    summary = build_summary(
        df,
        descriptive_stats,
        type_summary,
        failure_modes,
        numeric_by_failure,
        correlations,
    )

    summary_path = reports / "summary.json"
    descriptive_path = reports / "descriptive_statistics.csv"
    type_path = reports / "type_failure_summary.csv"
    failure_mode_path = reports / "failure_mode_summary.csv"
    numeric_by_failure_path = reports / "numeric_by_failure_summary.csv"
    correlation_path = reports / "correlation_matrix.csv"

    write_summary_json(summary, summary_path)
    save_dataframe(descriptive_stats, descriptive_path)
    save_dataframe(type_summary, type_path)
    save_dataframe(failure_modes, failure_mode_path)
    save_dataframe(numeric_by_failure, numeric_by_failure_path)
    correlations.to_csv(correlation_path, float_format="%.6f")
    plot_paths = create_plots(df, type_summary, plots)
    write_markdown_report(summary, markdown)

    artifacts = EdaArtifacts(
        summary_json=summary_path,
        descriptive_statistics_csv=descriptive_path,
        type_failure_summary_csv=type_path,
        failure_mode_summary_csv=failure_mode_path,
        numeric_by_failure_summary_csv=numeric_by_failure_path,
        correlation_matrix_csv=correlation_path,
        markdown_report=markdown,
        plot_paths=plot_paths,
    )
    return EdaResult(summary=summary, artifacts=artifacts)

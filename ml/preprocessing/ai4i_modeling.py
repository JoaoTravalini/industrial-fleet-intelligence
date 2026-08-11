"""Leakage-safe AI4I modeling dataset preparation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import ceil, floor, isclose
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG_RELATIVE_PATH = Path("ml") / "config" / "ai4i_modeling.json"
RAW_DATASET_RELATIVE_PATH = Path("data") / "raw" / "ai4i" / "ai4i2020.csv"
PROCESSED_RELATIVE_DIR = Path("data") / "processed" / "ai4i"
SPLIT_SUMMARY_RELATIVE_PATH = Path("reports") / "ai4i" / "modeling_split_summary.json"
SOURCE_UDI_COLUMN = "source_udi"
SPLIT_COLUMN = "split"
TARGET_VALUES = (0, 1)
SPLIT_NAMES = ("train", "validation", "test")


class Status(StrEnum):
    """Validation status values printed by modeling-data checkers."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ValidationResult:
    """A single modeling-data validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation output for generated AI4I modeling data."""

    results: list[ValidationResult]
    summary: dict[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        return not any(result.status is Status.FAIL and result.mandatory for result in self.results)


@dataclass(frozen=True)
class ModelingConfig:
    """Machine-readable AI4I modeling feature and split policy."""

    dataset_name: str
    modeling_objective: str
    target_column: str
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    traceability_fields: tuple[str, ...]
    derived_traceability_field: str
    excluded_identifiers: tuple[str, ...]
    excluded_leakage_sensitive_columns: tuple[str, ...]
    forbidden_feature_sources: tuple[str, ...]
    random_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    stratify_on: str
    future_preprocessing_design: Mapping[str, Any]


@dataclass(frozen=True)
class ModelingArtifacts:
    """Paths produced by the modeling-data preparation runner."""

    train_csv: Path
    validation_csv: Path
    test_csv: Path
    split_assignments_csv: Path
    split_summary_json: Path


@dataclass(frozen=True)
class ModelingPreparationResult:
    """Full result returned by a modeling-data preparation run."""

    config: ModelingConfig
    splits: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    artifacts: ModelingArtifacts
    validation_report: ValidationReport


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / CONFIG_RELATIVE_PATH


def raw_dataset_path(root: Path | None = None) -> Path:
    return (root or project_root()) / RAW_DATASET_RELATIVE_PATH


def processed_data_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / PROCESSED_RELATIVE_DIR


def split_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / SPLIT_SUMMARY_RELATIVE_PATH


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field `{key}` must be a non-empty string.")
    return value


def _require_string_sequence(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Configuration field `{key}` must be a list of strings.")
    return tuple(value)


def _duplicate_values(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def modeling_config_from_mapping(mapping: Mapping[str, Any]) -> ModelingConfig:
    split_fractions = mapping.get("split_fractions")
    if not isinstance(split_fractions, Mapping):
        raise ValueError("Configuration field `split_fractions` must be an object.")

    future_preprocessing_design = mapping.get("future_preprocessing_design")
    if not isinstance(future_preprocessing_design, Mapping):
        raise ValueError("Configuration field `future_preprocessing_design` must be an object.")

    config = ModelingConfig(
        dataset_name=_require_string(mapping, "dataset_name"),
        modeling_objective=_require_string(mapping, "modeling_objective"),
        target_column=_require_string(mapping, "target_column"),
        categorical_features=_require_string_sequence(mapping, "categorical_features"),
        numerical_features=_require_string_sequence(mapping, "numerical_features"),
        traceability_fields=_require_string_sequence(mapping, "traceability_fields"),
        derived_traceability_field=_require_string(mapping, "derived_traceability_field"),
        excluded_identifiers=_require_string_sequence(mapping, "excluded_identifiers"),
        excluded_leakage_sensitive_columns=_require_string_sequence(
            mapping, "excluded_leakage_sensitive_columns"
        ),
        forbidden_feature_sources=_require_string_sequence(mapping, "forbidden_feature_sources"),
        random_seed=int(mapping.get("random_seed")),
        train_fraction=float(split_fractions.get("train")),
        validation_fraction=float(split_fractions.get("validation")),
        test_fraction=float(split_fractions.get("test")),
        stratify_on=_require_string(mapping, "stratify_on"),
        future_preprocessing_design=future_preprocessing_design,
    )
    validate_modeling_config(config)
    return config


def load_modeling_config(path: Path | None = None) -> ModelingConfig:
    config_file = path or config_path()
    with config_file.open("r", encoding="utf-8") as file:
        return modeling_config_from_mapping(json.load(file))


def predictive_feature_columns(config: ModelingConfig) -> list[str]:
    return [*config.categorical_features, *config.numerical_features]


def modeling_frame_columns(config: ModelingConfig) -> list[str]:
    return [
        config.derived_traceability_field,
        *predictive_feature_columns(config),
        config.target_column,
    ]


def required_source_columns(config: ModelingConfig) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *config.traceability_fields,
                *predictive_feature_columns(config),
                config.target_column,
                *config.excluded_identifiers,
                *config.excluded_leakage_sensitive_columns,
            ]
        )
    )


def validate_forbidden_feature_columns(config: ModelingConfig) -> None:
    forbidden = {
        config.target_column,
        config.derived_traceability_field,
        *config.traceability_fields,
        *config.excluded_identifiers,
        *config.excluded_leakage_sensitive_columns,
        *config.forbidden_feature_sources,
    }
    violations = [column for column in predictive_feature_columns(config) if column in forbidden]
    if violations:
        raise ValueError(
            "Forbidden column(s) configured as predictive features: " + ", ".join(violations)
        )


def validate_modeling_config(config: ModelingConfig) -> None:
    errors: list[str] = []
    fractions = [config.train_fraction, config.validation_fraction, config.test_fraction]
    if any(fraction <= 0 for fraction in fractions):
        errors.append("Split fractions must all be positive.")
    if not isclose(sum(fractions), 1.0, abs_tol=1e-9):
        errors.append("Train, validation, and test fractions must sum to 1.0.")
    if not isclose(config.train_fraction, 0.70, abs_tol=1e-9):
        errors.append("Train fraction must be 0.70.")
    if not isclose(config.validation_fraction, 0.15, abs_tol=1e-9):
        errors.append("Validation fraction must be 0.15.")
    if not isclose(config.test_fraction, 0.15, abs_tol=1e-9):
        errors.append("Test fraction must be 0.15.")
    if config.random_seed != 42:
        errors.append("Random seed must be 42.")
    if config.target_column != "Machine failure":
        errors.append("Target column must be `Machine failure`.")
    if config.modeling_objective != "Binary classification of Machine failure":
        errors.append("Modeling objective must be binary classification of Machine failure.")
    if config.traceability_fields != ("UDI",):
        errors.append("Traceability fields must contain only `UDI`.")
    if config.derived_traceability_field != SOURCE_UDI_COLUMN:
        errors.append("Derived traceability field must be `source_udi`.")
    if config.stratify_on != config.target_column:
        errors.append("Stratification must use the target column only.")

    duplicates = _duplicate_values(predictive_feature_columns(config))
    if duplicates:
        errors.append("Predictive features must be unique: " + ", ".join(duplicates))

    try:
        validate_forbidden_feature_columns(config)
    except ValueError as exc:
        errors.append(str(exc))

    required_forbidden_sources = {
        config.target_column,
        *config.excluded_leakage_sensitive_columns,
    }
    missing_sources = required_forbidden_sources - set(config.forbidden_feature_sources)
    if missing_sources:
        errors.append(
            "Forbidden feature sources must include: " + ", ".join(sorted(missing_sources))
        )

    if errors:
        raise ValueError("Invalid AI4I modeling configuration: " + " ".join(errors))


def load_source_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_file = path or raw_dataset_path()
    if not dataset_file.exists():
        raise FileNotFoundError(f"AI4I raw dataset was not found: {dataset_file}")
    return pd.read_csv(dataset_file)


def validate_required_source_columns(df: pd.DataFrame, config: ModelingConfig) -> None:
    missing = [column for column in required_source_columns(config) if column not in df.columns]
    if missing:
        raise ValueError("Missing required AI4I source column(s): " + ", ".join(missing))


def _target_values(series: pd.Series) -> set[int] | None:
    if series.isna().any():
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return None
    unique_values = set(int(value) for value in numeric.unique())
    if not numeric.isin(TARGET_VALUES).all():
        return None
    return unique_values


def _has_binary_target(df: pd.DataFrame, target_column: str) -> bool:
    return _target_values(df[target_column]) == set(TARGET_VALUES)


def validate_modeling_frame(frame: pd.DataFrame, config: ModelingConfig) -> None:
    expected_columns = modeling_frame_columns(config)
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "Modeling frame has an unexpected schema. Expected columns: "
            + ", ".join(expected_columns)
        )

    forbidden_in_frame = [
        column
        for column in [*config.excluded_identifiers, *config.excluded_leakage_sensitive_columns]
        if column in frame.columns
    ]
    if forbidden_in_frame:
        raise ValueError(
            "Forbidden column(s) found in modeling frame: " + ", ".join(forbidden_in_frame)
        )

    if config.derived_traceability_field in predictive_feature_columns(config):
        raise ValueError("source_udi must not be a predictive feature.")
    if frame[config.derived_traceability_field].duplicated().any():
        raise ValueError("source_udi values must be unique.")
    if frame.isna().any().any():
        raise ValueError("Modeling frame must not contain missing values.")
    if not _has_binary_target(frame, config.target_column):
        raise ValueError("Modeling target must be binary with values 0 and 1.")


def construct_modeling_frame(source_df: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    validate_modeling_config(config)
    validate_required_source_columns(source_df, config)
    validate_forbidden_feature_columns(config)

    traceability_field = config.traceability_fields[0]
    source_columns = [traceability_field, *predictive_feature_columns(config), config.target_column]
    frame = source_df.loc[:, source_columns].copy()
    frame = frame.rename(columns={traceability_field: config.derived_traceability_field})
    frame = frame.loc[:, modeling_frame_columns(config)]
    validate_modeling_frame(frame, config)
    return frame


def _sort_split(df: pd.DataFrame, traceability_column: str) -> pd.DataFrame:
    return df.sort_values(traceability_column, kind="mergesort").reset_index(drop=True)


def create_stratified_splits(
    modeling_frame: pd.DataFrame,
    config: ModelingConfig,
) -> dict[str, pd.DataFrame]:
    validate_modeling_frame(modeling_frame, config)

    train_df, temporary_df = train_test_split(
        modeling_frame,
        train_size=config.train_fraction,
        random_state=config.random_seed,
        stratify=modeling_frame[config.target_column],
    )
    validation_fraction_of_temporary = config.validation_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        train_size=validation_fraction_of_temporary,
        random_state=config.random_seed,
        stratify=temporary_df[config.target_column],
    )

    return {
        "train": _sort_split(train_df, config.derived_traceability_field),
        "validation": _sort_split(validation_df, config.derived_traceability_field),
        "test": _sort_split(test_df, config.derived_traceability_field),
    }


def _percentage(count: int, total: int) -> float:
    return round((count / total) * 100, 6) if total else 0.0


def split_target_counts(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    counts = df[target_column].value_counts().reindex(TARGET_VALUES, fill_value=0)
    return {str(target_value): int(counts[target_value]) for target_value in TARGET_VALUES}


def split_target_percentages(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    counts = split_target_counts(df, target_column)
    return {target_value: _percentage(count, len(df)) for target_value, count in counts.items()}


def build_split_assignments(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    assignment_frames: list[pd.DataFrame] = []
    for split_name in SPLIT_NAMES:
        frame = splits[split_name][[SOURCE_UDI_COLUMN]].copy()
        frame[SPLIT_COLUMN] = split_name
        assignment_frames.append(frame)
    return (
        pd.concat(assignment_frames, ignore_index=True)
        .sort_values(SOURCE_UDI_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )[[SOURCE_UDI_COLUMN, SPLIT_COLUMN]]


def build_split_summary(
    splits: Mapping[str, pd.DataFrame],
    config: ModelingConfig,
) -> dict[str, Any]:
    split_rows = {split_name: int(len(splits[split_name])) for split_name in SPLIT_NAMES}
    total_rows = sum(split_rows.values())
    return {
        "modeling_objective": config.modeling_objective,
        "random_seed": config.random_seed,
        "total_rows": total_rows,
        "split_rows": split_rows,
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "stratify_on": config.stratify_on,
        "target_column": config.target_column,
        "target_counts_per_split": {
            split_name: split_target_counts(splits[split_name], config.target_column)
            for split_name in SPLIT_NAMES
        },
        "target_percentages_per_split": {
            split_name: split_target_percentages(splits[split_name], config.target_column)
            for split_name in SPLIT_NAMES
        },
        "categorical_features": list(config.categorical_features),
        "numerical_features": list(config.numerical_features),
        "feature_list": predictive_feature_columns(config),
        "traceability_field": config.derived_traceability_field,
        "source_traceability_field": config.traceability_fields[0],
        "excluded_identifiers": list(config.excluded_identifiers),
        "excluded_leakage_sensitive_fields": list(config.excluded_leakage_sensitive_columns),
        "modeling_frame_columns": modeling_frame_columns(config),
        "future_preprocessing": {
            "categorical": 'Type -> OneHotEncoder(handle_unknown="ignore")',
            "numerical": "Five process variables -> preprocessing determined during modeling",
            "fit_policy": (
                "Fit preprocessing transformations on training data only inside a pipeline."
            ),
        },
    }


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def write_modeling_artifacts(
    splits: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    output_dir: Path | None = None,
    summary_file: Path | None = None,
) -> ModelingArtifacts:
    processed_dir = output_dir or processed_data_directory()
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    test_csv = processed_dir / "test.csv"
    split_assignments_csv = processed_dir / "split_assignments.csv"
    split_summary_json = summary_file or split_summary_path()

    splits["train"].to_csv(train_csv, index=False)
    splits["validation"].to_csv(validation_csv, index=False)
    splits["test"].to_csv(test_csv, index=False)
    build_split_assignments(splits).to_csv(split_assignments_csv, index=False)
    write_json(summary, split_summary_json)

    return ModelingArtifacts(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
        split_assignments_csv=split_assignments_csv,
        split_summary_json=split_summary_json,
    )


def _expected_count_range(total_rows: int, fraction: float) -> tuple[int, int]:
    expected = total_rows * fraction
    return floor(expected), ceil(expected)


def validate_split_integrity(
    splits: Mapping[str, pd.DataFrame],
    config: ModelingConfig,
    source_udis: Sequence[Any] | None = None,
    expected_total_rows: int | None = None,
    stratification_tolerance_percentage_points: float = 1.0,
) -> ValidationReport:
    results: list[ValidationResult] = []

    try:
        validate_modeling_config(config)
        results.append(ValidationResult("Modeling Configuration", Status.PASS, "Policy is valid."))
    except ValueError as exc:
        results.append(ValidationResult("Modeling Configuration", Status.FAIL, str(exc)))
        return ValidationReport(results)

    missing_splits = [split_name for split_name in SPLIT_NAMES if split_name not in splits]
    unexpected_splits = [split_name for split_name in splits if split_name not in SPLIT_NAMES]
    if not missing_splits and not unexpected_splits:
        results.append(ValidationResult("Split Names", Status.PASS, "All expected splits exist."))
    else:
        details = []
        if missing_splits:
            details.append("missing " + ", ".join(missing_splits))
        if unexpected_splits:
            details.append("unexpected " + ", ".join(unexpected_splits))
        results.append(ValidationResult("Split Names", Status.FAIL, "; ".join(details)))
        return ValidationReport(results)

    expected_columns = modeling_frame_columns(config)
    for split_name in SPLIT_NAMES:
        split_df = splits[split_name]
        if list(split_df.columns) == expected_columns:
            results.append(
                ValidationResult(f"{split_name} Schema", Status.PASS, "Expected columns present.")
            )
        else:
            results.append(
                ValidationResult(
                    f"{split_name} Schema",
                    Status.FAIL,
                    "Expected columns: " + ", ".join(expected_columns),
                )
            )

        forbidden_columns = [
            column
            for column in [*config.excluded_identifiers, *config.excluded_leakage_sensitive_columns]
            if column in split_df.columns
        ]
        if forbidden_columns:
            results.append(
                ValidationResult(
                    f"{split_name} Forbidden Columns",
                    Status.FAIL,
                    "Found forbidden column(s): " + ", ".join(forbidden_columns),
                )
            )
        else:
            results.append(
                ValidationResult(
                    f"{split_name} Forbidden Columns",
                    Status.PASS,
                    "No forbidden columns found.",
                )
            )

        missing_values = int(split_df.isna().sum().sum())
        results.append(
            ValidationResult(
                f"{split_name} Missing Values",
                Status.PASS if missing_values == 0 else Status.FAIL,
                "No missing values found."
                if missing_values == 0
                else f"Found {missing_values} missing value(s).",
            )
        )

        duplicate_count = int(split_df[config.derived_traceability_field].duplicated().sum())
        results.append(
            ValidationResult(
                f"{split_name} source_udi Uniqueness",
                Status.PASS if duplicate_count == 0 else Status.FAIL,
                "source_udi values are unique."
                if duplicate_count == 0
                else f"Found {duplicate_count} duplicated source_udi value(s).",
            )
        )

        binary_target = _has_binary_target(split_df, config.target_column)
        results.append(
            ValidationResult(
                f"{split_name} Target Binary",
                Status.PASS if binary_target else Status.FAIL,
                "Target contains binary values 0 and 1."
                if binary_target
                else "Target must contain only binary values 0 and 1.",
            )
        )

    combined = pd.concat([splits[split_name] for split_name in SPLIT_NAMES], ignore_index=True)
    expected_total = expected_total_rows or (
        len(source_udis) if source_udis is not None else len(combined)
    )
    total_rows = int(len(combined))
    results.append(
        ValidationResult(
            "Total Row Count",
            Status.PASS if total_rows == expected_total else Status.FAIL,
            f"Total rows equal {expected_total}."
            if total_rows == expected_total
            else f"Expected {expected_total} rows, found {total_rows}.",
        )
    )

    global_duplicates = int(combined[config.derived_traceability_field].duplicated().sum())
    results.append(
        ValidationResult(
            "Global source_udi Uniqueness",
            Status.PASS if global_duplicates == 0 else Status.FAIL,
            "Every source_udi appears once globally."
            if global_duplicates == 0
            else f"Found {global_duplicates} duplicated source_udi value(s) globally.",
        )
    )

    overlaps: list[str] = []
    split_sets = {
        split_name: set(splits[split_name][config.derived_traceability_field].tolist())
        for split_name in SPLIT_NAMES
    }
    for index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[index + 1 :]:
            overlap_count = len(split_sets[left_name] & split_sets[right_name])
            if overlap_count:
                overlaps.append(f"{left_name}/{right_name}: {overlap_count}")
    results.append(
        ValidationResult(
            "Split Overlap",
            Status.PASS if not overlaps else Status.FAIL,
            "No source_udi overlap between splits."
            if not overlaps
            else "Overlap found: " + ", ".join(overlaps),
        )
    )

    if source_udis is not None:
        expected_udis = set(source_udis)
        actual_udis = set(combined[config.derived_traceability_field].tolist())
        results.append(
            ValidationResult(
                "Source UDI Coverage",
                Status.PASS if actual_udis == expected_udis else Status.FAIL,
                f"All {len(expected_udis)} source UDI values are covered exactly once."
                if actual_udis == expected_udis
                else "Split UDI union does not match source UDI values.",
            )
        )

    proportion_failures: list[str] = []
    expected_fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    for split_name, fraction in expected_fractions.items():
        low, high = _expected_count_range(expected_total, fraction)
        count = len(splits[split_name])
        if count < low or count > high:
            proportion_failures.append(f"{split_name}: expected {low}-{high}, found {count}")
    results.append(
        ValidationResult(
            "Split Proportions",
            Status.PASS if not proportion_failures else Status.FAIL,
            "Split row counts match requested fractions within integer rounding."
            if not proportion_failures
            else "; ".join(proportion_failures),
        )
    )

    class_rate_failures: list[str] = []
    source_positive_rate = float(combined[config.target_column].mean() * 100)
    for split_name in SPLIT_NAMES:
        split_rate = float(splits[split_name][config.target_column].mean() * 100)
        difference = abs(split_rate - source_positive_rate)
        if difference > stratification_tolerance_percentage_points:
            class_rate_failures.append(
                f"{split_name}: {split_rate:.6f}% vs source {source_positive_rate:.6f}%"
            )
    results.append(
        ValidationResult(
            "Target Stratification",
            Status.PASS if not class_rate_failures else Status.FAIL,
            "Target class rates remain approximately stratified."
            if not class_rate_failures
            else "; ".join(class_rate_failures),
        )
    )

    summary = build_split_summary(splits, config)
    return ValidationReport(results, summary)


def generated_artifact_paths(output_dir: Path | None = None) -> dict[str, Path]:
    processed_dir = output_dir or processed_data_directory()
    return {
        "train": processed_dir / "train.csv",
        "validation": processed_dir / "validation.csv",
        "test": processed_dir / "test.csv",
        "split_assignments": processed_dir / "split_assignments.csv",
    }


def load_generated_splits(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    paths = generated_artifact_paths(output_dir)
    return {
        "train": pd.read_csv(paths["train"]),
        "validation": pd.read_csv(paths["validation"]),
        "test": pd.read_csv(paths["test"]),
    }


def load_split_assignments(output_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(generated_artifact_paths(output_dir)["split_assignments"])


def validate_split_assignments(
    assignments: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    config: ModelingConfig,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    expected_columns = [config.derived_traceability_field, SPLIT_COLUMN]
    results.append(
        ValidationResult(
            "Split Assignments Schema",
            Status.PASS if list(assignments.columns) == expected_columns else Status.FAIL,
            "split_assignments.csv contains source_udi and split only."
            if list(assignments.columns) == expected_columns
            else "split_assignments.csv must contain only source_udi and split.",
        )
    )

    expected_total = sum(len(splits[split_name]) for split_name in SPLIT_NAMES)
    results.append(
        ValidationResult(
            "Split Assignments Row Count",
            Status.PASS if len(assignments) == expected_total else Status.FAIL,
            f"split_assignments.csv contains {expected_total} rows."
            if len(assignments) == expected_total
            else f"Expected {expected_total} rows, found {len(assignments)}.",
        )
    )

    duplicate_count = int(assignments[config.derived_traceability_field].duplicated().sum())
    results.append(
        ValidationResult(
            "Split Assignments Uniqueness",
            Status.PASS if duplicate_count == 0 else Status.FAIL,
            "Each source_udi appears once in split_assignments.csv."
            if duplicate_count == 0
            else f"Found {duplicate_count} duplicated source_udi value(s).",
        )
    )

    invalid_splits = sorted(set(assignments[SPLIT_COLUMN].tolist()) - set(SPLIT_NAMES))
    results.append(
        ValidationResult(
            "Split Assignments Values",
            Status.PASS if not invalid_splits else Status.FAIL,
            "Split values are train, validation, or test."
            if not invalid_splits
            else "Unexpected split value(s): " + ", ".join(invalid_splits),
        )
    )

    consistency_failures: list[str] = []
    for split_name in SPLIT_NAMES:
        expected_udis = set(splits[split_name][config.derived_traceability_field].tolist())
        actual_udis = set(
            assignments.loc[
                assignments[SPLIT_COLUMN] == split_name, config.derived_traceability_field
            ].tolist()
        )
        if actual_udis != expected_udis:
            consistency_failures.append(split_name)
    results.append(
        ValidationResult(
            "Split Assignments Consistency",
            Status.PASS if not consistency_failures else Status.FAIL,
            "Assignments match generated split files."
            if not consistency_failures
            else "Assignments do not match split file(s): " + ", ".join(consistency_failures),
        )
    )
    return results


def load_split_summary(path: Path | None = None) -> dict[str, Any]:
    summary_file = path or split_summary_path()
    with summary_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_generated_artifacts(
    source_df: pd.DataFrame,
    config: ModelingConfig,
    output_dir: Path | None = None,
    summary_file: Path | None = None,
) -> ValidationReport:
    paths = generated_artifact_paths(output_dir)
    summary_path = summary_file or split_summary_path()
    all_paths = {**paths, "split_summary": summary_path}
    results: list[ValidationResult] = []
    missing_files: list[str] = []
    for name, path in all_paths.items():
        if path.exists():
            results.append(ValidationResult(f"{name} File", Status.PASS, "Expected file exists."))
        else:
            missing_files.append(name)
            results.append(
                ValidationResult(
                    f"{name} File", Status.FAIL, f"Expected file is missing: {path.name}"
                )
            )
    if missing_files:
        return ValidationReport(results)

    try:
        splits = load_generated_splits(output_dir)
        assignments = load_split_assignments(output_dir)
        persisted_summary = load_split_summary(summary_path)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        results.append(ValidationResult("Readable Artifacts", Status.FAIL, str(exc)))
        return ValidationReport(results)

    split_report = validate_split_integrity(
        splits,
        config,
        source_udis=source_df[config.traceability_fields[0]].tolist(),
        expected_total_rows=len(source_df),
    )
    results.extend(split_report.results)
    results.extend(validate_split_assignments(assignments, splits, config))
    results.append(
        ValidationResult(
            "Split Summary Consistency",
            Status.PASS if persisted_summary == split_report.summary else Status.FAIL,
            "modeling_split_summary.json matches generated split artifacts."
            if persisted_summary == split_report.summary
            else "modeling_split_summary.json does not match generated split artifacts.",
        )
    )
    return ValidationReport(results, split_report.summary)


def prepare_modeling_data(
    source_df: pd.DataFrame,
    config: ModelingConfig,
    output_dir: Path | None = None,
    summary_file: Path | None = None,
) -> ModelingPreparationResult:
    modeling_frame = construct_modeling_frame(source_df, config)
    splits = create_stratified_splits(modeling_frame, config)
    pre_write_report = validate_split_integrity(
        splits,
        config,
        source_udis=source_df[config.traceability_fields[0]].tolist(),
        expected_total_rows=len(source_df),
    )
    summary = pre_write_report.summary or build_split_summary(splits, config)
    empty_artifacts = ModelingArtifacts(Path(), Path(), Path(), Path(), Path())
    if not pre_write_report.is_valid:
        return ModelingPreparationResult(
            config=config,
            splits=splits,
            summary=summary,
            artifacts=empty_artifacts,
            validation_report=pre_write_report,
        )

    artifacts = write_modeling_artifacts(splits, summary, output_dir, summary_file)
    validation_report = validate_generated_artifacts(source_df, config, output_dir, summary_file)
    return ModelingPreparationResult(
        config=config,
        splits=splits,
        summary=summary,
        artifacts=artifacts,
        validation_report=validation_report,
    )

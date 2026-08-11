"""Prepare leakage-safe AI4I modeling datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from scripts import check_ai4i  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ai4i_modeling.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def print_validation_results(report: ai4i_modeling.ValidationReport) -> None:
    name_width = max(len(result.name) for result in report.results)
    for result in report.results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")


def print_split_summary(summary: dict[str, object]) -> None:
    print()
    print("Split summary:")
    split_rows = summary["split_rows"]
    target_counts = summary["target_counts_per_split"]
    target_percentages = summary["target_percentages_per_split"]
    for split_name in ai4i_modeling.SPLIT_NAMES:
        rows = split_rows[split_name]
        counts = target_counts[split_name]
        percentages = target_percentages[split_name]
        print(
            f"  {split_name}: {rows} rows; "
            f"Machine failure 0={counts['0']} ({percentages['0']}%), "
            f"1={counts['1']} ({percentages['1']}%)"
        )


def print_raw_failures(report: check_ai4i.ValidationReport) -> None:
    print("Industrial Fleet Intelligence Platform AI4I modeling data preparation")
    print()
    print("FAIL Raw AI4I dataset is missing or structurally invalid.")
    for result in report.results:
        if result.status is check_ai4i.Status.FAIL:
            print(f"FAIL {result.name} {result.message}")


def main() -> int:
    dataset_file = check_ai4i.dataset_path()
    raw_report = check_ai4i.validate_dataset_file(dataset_file)
    if not raw_report.is_valid:
        print_raw_failures(raw_report)
        return 1

    print("Industrial Fleet Intelligence Platform AI4I modeling data preparation")
    print()
    print("PASS Raw AI4I dataset validated before preparation.")

    try:
        config = ai4i_modeling.load_modeling_config()
        source_df = ai4i_modeling.load_source_dataset(dataset_file)
        result = ai4i_modeling.prepare_modeling_data(source_df, config)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Preparation failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"FAIL Preparation encountered an unexpected error: {exc}")
        return 2

    print("PASS Modeling configuration validated.")
    print("PASS Leakage-safe modeling frame constructed.")
    print("PASS Deterministic stratified split artifacts written.")
    print_validation_results(result.validation_report)
    if result.summary:
        print_split_summary(result.summary)

    print()
    print("Generated artifacts:")
    for path in [
        result.artifacts.train_csv,
        result.artifacts.validation_csv,
        result.artifacts.test_csv,
        result.artifacts.split_assignments_csv,
        result.artifacts.split_summary_json,
    ]:
        print(f"  {relative_path(path)}")

    pass_count = sum(
        1 for item in result.validation_report.results if item.status is ai4i_modeling.Status.PASS
    )
    warn_count = sum(
        1 for item in result.validation_report.results if item.status is ai4i_modeling.Status.WARN
    )
    fail_count = sum(
        1 for item in result.validation_report.results if item.status is ai4i_modeling.Status.FAIL
    )
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    return 0 if result.validation_report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only validator for generated AI4I modeling datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from scripts import check_ai4i  # noqa: E402


def print_results(report: ai4i_modeling.ValidationReport) -> None:
    name_width = max(len(result.name) for result in report.results)
    for result in report.results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")


def print_split_summary(summary: dict[str, object] | None) -> None:
    if summary is None:
        return
    print()
    print("Class distribution:")
    target_counts = summary["target_counts_per_split"]
    target_percentages = summary["target_percentages_per_split"]
    for split_name in ai4i_modeling.SPLIT_NAMES:
        counts = target_counts[split_name]
        percentages = target_percentages[split_name]
        print(
            f"  {split_name}: "
            f"Machine failure 0={counts['0']} ({percentages['0']}%), "
            f"1={counts['1']} ({percentages['1']}%)"
        )


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I modeling data validation")
    print()

    try:
        config = ai4i_modeling.load_modeling_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Modeling configuration could not be loaded: {exc}")
        return 1

    dataset_file = check_ai4i.dataset_path()
    raw_report = check_ai4i.validate_dataset_file(dataset_file)
    if not raw_report.is_valid:
        print("FAIL Raw AI4I dataset is missing or structurally invalid.")
        for result in raw_report.results:
            if result.status is check_ai4i.Status.FAIL:
                print(f"FAIL {result.name} {result.message}")
        return 1

    try:
        source_df = ai4i_modeling.load_source_dataset(dataset_file)
        report = ai4i_modeling.validate_generated_artifacts(source_df, config)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Modeling artifact validation failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_results(report)
    print_split_summary(report.summary)

    pass_count = sum(1 for item in report.results if item.status is ai4i_modeling.Status.PASS)
    warn_count = sum(1 for item in report.results if item.status is ai4i_modeling.Status.WARN)
    fail_count = sum(1 for item in report.results if item.status is ai4i_modeling.Status.FAIL)
    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run reproducible AI4I exploratory data analysis."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analysis import ai4i_eda  # noqa: E402
from scripts import check_ai4i  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ai4i_eda.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def print_structural_failures(report: check_ai4i.ValidationReport) -> None:
    print("Industrial Fleet Intelligence Platform AI4I EDA")
    print()
    print("FAIL Raw AI4I dataset is missing or structurally invalid.")
    for result in report.results:
        if result.status is check_ai4i.Status.FAIL:
            print(f"FAIL {result.name} {result.message}")


def print_success(result: ai4i_eda.EdaResult) -> None:
    summary = result.summary
    artifacts = result.artifacts
    print("Industrial Fleet Intelligence Platform AI4I EDA")
    print()
    print("PASS Raw AI4I dataset validated before analysis.")
    rows = summary["dataset"]["rows"]
    columns = summary["dataset"]["columns"]
    print(f"PASS Dataset dimensions {rows} rows x {columns} columns.")
    print(
        "PASS Machine failure positives "
        f"{summary['machine_failure']['positive_count']} "
        f"({summary['machine_failure']['failure_percentage']}%)."
    )
    print("PASS Generated deterministic report artifacts:")
    for path in [
        artifacts.summary_json,
        artifacts.descriptive_statistics_csv,
        artifacts.type_failure_summary_csv,
        artifacts.failure_mode_summary_csv,
        artifacts.numeric_by_failure_summary_csv,
        artifacts.correlation_matrix_csv,
        artifacts.markdown_report,
        *artifacts.plot_paths,
    ]:
        print(f"  {relative_path(path)}")


def main() -> int:
    dataset_file = check_ai4i.dataset_path()
    validation_report = check_ai4i.validate_dataset_file(dataset_file)
    if not validation_report.is_valid:
        print_structural_failures(validation_report)
        return 1

    try:
        result = ai4i_eda.run_eda(dataset_file=dataset_file)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform AI4I EDA")
        print()
        print(f"FAIL EDA failed: {exc}")
        return 2

    print_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

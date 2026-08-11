"""Train validation-only AI4I baseline classifiers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from ml.training import ai4i_baseline  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ai4i_baseline.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def print_metrics(label: str, metrics: dict[str, object]) -> None:
    print(
        f"{label}: AP={metrics['average_precision']}, ROC-AUC={metrics['roc_auc']}, "
        f"balanced_accuracy={metrics['balanced_accuracy']}, precision={metrics['precision']}, "
        f"recall={metrics['recall']}, F1={metrics['f1']}, accuracy={metrics['accuracy']}"
    )


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I baseline classification")
    print("TEST SET STATUS: LOCKED / NOT USED")
    print()

    try:
        config = ai4i_modeling.load_modeling_config()
        split_summary = ai4i_modeling.load_split_summary()
        train_df, validation_df = ai4i_baseline.load_training_and_validation_frames()
        result = ai4i_baseline.run_baseline_experiment(
            train_df=train_df,
            validation_df=validation_df,
            config=config,
            split_summary=split_summary,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Baseline training failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"FAIL Baseline training encountered an unexpected error: {exc}")
        return 2

    metrics = result.metrics
    print("PASS Required train and validation artifacts validated.")
    print("PASS Modeling policy validated; forbidden leakage fields are absent from model input.")
    print("PASS Dummy baseline fitted on train and evaluated on validation only.")
    print("PASS Logistic Regression baseline fitted on train and evaluated on validation only.")
    print()
    print("Validation comparison:")
    print_metrics("Dummy", metrics["dummy_classifier"])
    print_metrics("Logistic Regression", metrics["logistic_regression"])
    print()
    print("Generated artifacts:")
    for path in [
        result.artifacts.metrics_json,
        result.artifacts.validation_predictions_csv,
        result.artifacts.logistic_coefficients_csv,
        result.artifacts.markdown_report,
        *result.artifacts.plot_paths,
    ]:
        print(f"  {relative_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

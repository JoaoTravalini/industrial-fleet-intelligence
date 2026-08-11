"""Run AI4I targeted Random Forest tuning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from ml.training import ai4i_baseline, ai4i_random_forest_tuning  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ai4i_random_forest_tuning.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def print_progress(outer_fold: int, total_folds: int) -> None:
    print(f"Outer fold {outer_fold}/{total_folds}")


def print_nested_summary(metrics: dict[str, Any]) -> None:
    tuned = metrics["tuned_nested_oof_results"]
    threshold_0_5 = tuned["threshold_0_5"]
    max_f2 = metrics["threshold_candidates"]["max_f2"]
    print(
        f"Tuned nested OOF: AP={tuned['average_precision']}, ROC-AUC={tuned['roc_auc']}, "
        f"threshold 0.5 precision={threshold_0_5['precision']}, "
        f"recall={threshold_0_5['recall']}, F1={threshold_0_5['f1']}, "
        f"F2={threshold_0_5['f2']}"
    )
    print(
        f"Tuned nested OOF max-F2 threshold={max_f2['threshold']}, "
        f"precision={max_f2['precision']}, recall={max_f2['recall']}, F2={max_f2['f2']}"
    )


def print_validation_summary(metrics: dict[str, Any]) -> None:
    validation = metrics["validation_results"]
    threshold_0_5 = validation["threshold_0_5"]
    selected = validation["selected_threshold"]
    print(
        f"Validation AP={validation['threshold_independent']['average_precision']}, "
        f"ROC-AUC={validation['threshold_independent']['roc_auc']}"
    )
    print(
        f"Validation threshold 0.5: precision={threshold_0_5['precision']}, "
        f"recall={threshold_0_5['recall']}, F1={threshold_0_5['f1']}, "
        f"F2={threshold_0_5['f2']}, confusion_matrix={threshold_0_5['confusion_matrix']}"
    )
    print(
        f"Validation selected threshold: precision={selected['precision']}, "
        f"recall={selected['recall']}, F1={selected['f1']}, F2={selected['f2']}, "
        f"confusion_matrix={selected['confusion_matrix']}"
    )


def main() -> int:
    print("Industrial Fleet Intelligence Platform AI4I Random Forest tuning")
    print("TEST SET STATUS: LOCKED / NOT USED")
    print()

    try:
        config = ai4i_modeling.load_modeling_config()
        split_summary = ai4i_modeling.load_split_summary()
        train_df, validation_df = ai4i_baseline.load_training_and_validation_frames()
        result = ai4i_random_forest_tuning.run_random_forest_tuning_experiment(
            train_df=train_df,
            validation_df=validation_df,
            config=config,
            split_summary=split_summary,
            progress_callback=print_progress,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Random Forest tuning failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"FAIL Random Forest tuning encountered an unexpected error: {exc}")
        return 2

    metrics = result.metrics
    promotion = metrics["promotion_policy"]
    full_train = metrics["full_train_grid_search"]
    print()
    print("PASS Modeling configuration validated.")
    print("PASS Train-only nested CV completed for Random Forest tuning.")
    print("PASS Threshold analysis completed using nested train OOF predictions only.")
    print("PASS Full-train GridSearchCV completed using train only.")
    print(
        "PASS Selected Random Forest candidate: "
        f"{promotion['selected_candidate']} at threshold {promotion['selected_threshold']}"
    )
    print("PASS Validation evaluated once after train-derived selection.")
    print()
    print("Nested OOF summary:")
    print_nested_summary(metrics)
    print()
    print("Full-train GridSearchCV best configuration:")
    print(
        f"  {ai4i_random_forest_tuning.parameter_label(full_train['best_hyperparameters'])}; "
        f"mean AP={full_train['best_mean_average_precision']}, "
        f"std={full_train['best_std_average_precision']}"
    )
    print()
    print("Promotion decision:")
    print(
        f"  fixed AP={promotion['fixed_average_precision']}, "
        f"tuned nested AP={promotion['tuned_nested_average_precision']}, "
        f"delta={promotion['average_precision_delta']}"
    )
    print(f"  {promotion['reason']}")
    print()
    print("Validation comparison:")
    print_validation_summary(metrics)
    print()
    print("Generated artifacts:")
    for path in [
        result.artifacts.metrics_json,
        result.artifacts.nested_oof_predictions_csv,
        result.artifacts.outer_folds_csv,
        result.artifacts.grid_results_csv,
        result.artifacts.threshold_analysis_csv,
        result.artifacts.validation_predictions_csv,
        result.artifacts.markdown_report,
        *result.artifacts.plot_paths,
    ]:
        print(f"  {relative_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

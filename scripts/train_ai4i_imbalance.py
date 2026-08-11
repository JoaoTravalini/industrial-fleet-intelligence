"""Run AI4I class-imbalance and threshold strategy experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import ai4i_modeling  # noqa: E402
from ml.training import ai4i_baseline, ai4i_imbalance  # noqa: E402


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ai4i_imbalance.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def print_oof_summary(metrics: dict[str, object]) -> None:
    for model_name in ai4i_imbalance.MODEL_NAMES:
        model_metrics = metrics["train_oof_results"][model_name]
        threshold_metrics = model_metrics["threshold_0_5"]
        print(
            f"{model_name}: OOF AP={model_metrics['average_precision']}, "
            f"ROC-AUC={model_metrics['roc_auc']}, threshold 0.5 precision="
            f"{threshold_metrics['precision']}, recall={threshold_metrics['recall']}, "
            f"F1={threshold_metrics['f1']}, F2={threshold_metrics['f2']}"
        )


def print_validation_summary(metrics: dict[str, object]) -> None:
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
    print("Industrial Fleet Intelligence Platform AI4I imbalance strategy")
    print("TEST SET STATUS: LOCKED / NOT USED")
    print()

    try:
        config = ai4i_modeling.load_modeling_config()
        split_summary = ai4i_modeling.load_split_summary()
        train_df, validation_df = ai4i_baseline.load_training_and_validation_frames()
        result = ai4i_imbalance.run_imbalance_experiment(
            train_df=train_df,
            validation_df=validation_df,
            config=config,
            split_summary=split_summary,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL Imbalance strategy training failed: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print(f"FAIL Imbalance strategy encountered an unexpected error: {exc}")
        return 2

    metrics = result.metrics
    selected = metrics["candidate_selection_policy"]
    print("PASS Modeling configuration validated.")
    print("PASS Train-only 5-fold OOF probabilities generated for both Logistic variants.")
    print("PASS Threshold analysis completed using train OOF predictions only.")
    print(
        "PASS Selected development candidate: "
        f"{selected['selected_model']} at threshold {selected['selected_threshold']}"
    )
    print("PASS Validation evaluated once after train-derived selection.")
    print()
    print("Train OOF comparison:")
    print_oof_summary(metrics)
    print()
    print("Validation comparison:")
    print_validation_summary(metrics)
    print()
    print("Generated artifacts:")
    for path in [
        result.artifacts.metrics_json,
        result.artifacts.oof_predictions_csv,
        result.artifacts.threshold_analysis_csv,
        result.artifacts.validation_predictions_csv,
        result.artifacts.markdown_report,
        *result.artifacts.plot_paths,
    ]:
        print(f"  {relative_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the final AI4I holdout evaluation for the frozen classifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation import ai4i_final_evaluation  # noqa: E402
from ml.preprocessing import ai4i_modeling  # noqa: E402


def main() -> int:
    try:
        modeling_config = ai4i_modeling.load_modeling_config(
            ai4i_modeling.config_path(PROJECT_ROOT)
        )
        comparison_metrics = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.model_comparison_metrics_path(PROJECT_ROOT)
        )
        tuning_metrics = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.random_forest_tuning_metrics_path(PROJECT_ROOT)
        )
        final_config = ai4i_final_evaluation.load_final_model_config(
            ai4i_final_evaluation.final_config_path(PROJECT_ROOT)
        )
        ai4i_final_evaluation.validate_final_model_config(
            final_config,
            modeling_config,
            comparison_metrics,
            tuning_metrics,
        )
        config_hash = ai4i_final_evaluation.final_config_hash(final_config)

        print(f"Final model configuration SHA-256: {config_hash}")
        print(ai4i_final_evaluation.FROZEN_SPEC_MESSAGE)
        print("Fitting frozen Random Forest on train + validation development data.")
        print(ai4i_final_evaluation.TEST_UNLOCK_MESSAGE)

        result = ai4i_final_evaluation.run_final_evaluation(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL Final AI4I holdout evaluation failed: {exc}", file=sys.stderr)
        return 1

    metrics = result.metrics
    test_metrics = metrics["test_metrics"]
    threshold_014 = test_metrics["threshold_0_14"]
    print("PASS Final AI4I holdout evaluation artifacts generated.")
    print(
        "Development training rows: "
        f"{metrics['development_training_row_count']} "
        f"(positives: {metrics['development_positive_count']})"
    )
    print(f"Test rows: {metrics['test_row_count']} (positives: {metrics['test_positive_count']})")
    print(
        "Test AP: "
        f"{test_metrics['threshold_independent']['average_precision']} | "
        f"Test ROC-AUC: {test_metrics['threshold_independent']['roc_auc']}"
    )
    print(
        "Primary threshold 0.14: "
        f"precision={threshold_014['precision']}, "
        f"recall={threshold_014['recall']}, "
        f"f1={threshold_014['f1']}, "
        f"f2={threshold_014['f2']}"
    )
    print(f"Predictions: {result.artifacts.predictions_csv}")
    print(f"Metrics: {result.artifacts.metrics_json}")
    print(f"Decision artifact: {result.artifacts.decision_json}")
    print(f"Report: {result.artifacts.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

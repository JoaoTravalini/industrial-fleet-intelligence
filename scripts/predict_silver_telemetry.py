"""Run trusted AI4I batch inference over adapted canonical Silver telemetry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import ai4i_telemetry  # noqa: E402


def main() -> int:
    try:
        summary = ai4i_telemetry.run_prediction_pipeline(root=PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL AI4I telemetry inference failed: {exc}", file=sys.stderr)
        return 1

    print("PASS AI4I telemetry inference completed.")
    print(f"Adapter records: {summary.adapter_record_count}")
    print(f"Predicted records: {summary.prediction_record_count}")
    print(f"Unique event IDs: {summary.unique_event_id_count}")
    print(f"Positive predictions: {summary.positive_prediction_count}")
    print(f"Negative predictions: {summary.negative_prediction_count}")
    print(f"Minimum failure probability: {summary.min_failure_probability:.6f}")
    print(f"Maximum failure probability: {summary.max_failure_probability:.6f}")
    print(f"Mean failure probability: {summary.mean_failure_probability:.6f}")
    print(f"Model name: {summary.model_name}")
    print(f"Model version: {summary.model_version}")
    print(f"Threshold: {summary.frozen_threshold:.2f}")
    print(f"Prediction output: {summary.output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

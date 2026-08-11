"""Package the frozen AI4I final model as a local joblib artifact."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import ai4i_predictor  # noqa: E402

PACKAGING_TEST_STATUS = "TEST SET STATUS: FINAL EVALUATION COMPLETE / NOT USED FOR PACKAGING"


def main() -> int:
    try:
        print(PACKAGING_TEST_STATUS)
        result = ai4i_predictor.package_model(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL AI4I final model packaging failed: {exc}", file=sys.stderr)
        return 1

    print("PASS Packaged frozen AI4I final model.")
    print(f"Model name: {result.model_name}")
    print(f"Model version: {result.model_version}")
    print(f"Final config SHA-256: {result.final_config_hash}")
    print(f"Artifact path: {result.artifact_path}")
    print(f"Artifact SHA-256: {result.model_artifact_sha256}")
    print(f"Metadata path: {result.metadata_path}")
    print(f"Tracked packaging summary: {result.packaging_summary_path}")
    print(
        f"Training rows: {result.training_row_count} (positives: {result.training_positive_count})"
    )
    print(f"joblib version: {result.joblib_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

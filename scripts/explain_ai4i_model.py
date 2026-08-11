"""Generate local SHAP explainability artifacts for the packaged AI4I model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability import ai4i_shap  # noqa: E402


def main() -> int:
    try:
        print(ai4i_shap.TEST_SET_STATUS_MESSAGE)
        result = ai4i_shap.run_explainability(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL AI4I SHAP explainability failed: {exc}", file=sys.stderr)
        return 1

    print("PASS AI4I SHAP explainability completed.")
    print(f"Model: {result.summary['model_name']} {result.summary['model_version']}")
    print(f"SHAP version: {result.summary['shap_version']}")
    print(f"Global development sample size: {result.global_sample_size}")
    print(f"Transformed feature count: {result.transformed_feature_count}")
    print(f"Grouped feature count: {result.grouped_feature_count}")
    top_features = ", ".join(
        f"{item['feature']}={item['mean_absolute_shap']}" for item in result.grouped_importance[:3]
    )
    print(f"Top grouped attributions: {top_features}")
    print(f"Maximum additivity error: {result.summary['max_observed_additivity_error']}")
    for path in result.plot_paths:
        print(f"Plot: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

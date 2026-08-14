"""Generate operational SHAP explanations for persisted AI4I telemetry predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability import ai4i_telemetry_shap  # noqa: E402


def main() -> int:
    print("Industrial Fleet Intelligence Platform operational AI4I explainability")
    print()
    try:
        result = ai4i_telemetry_shap.run_operational_explainability(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

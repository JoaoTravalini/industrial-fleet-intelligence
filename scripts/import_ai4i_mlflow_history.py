"""Import historical AI4I development reports into local MLflow."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.tracking import ai4i_mlflow  # noqa: E402


def main() -> int:
    try:
        result = ai4i_mlflow.import_historical_runs(PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"FAIL AI4I MLflow history import failed: {exc}", file=sys.stderr)
        return 1

    print("PASS AI4I MLflow historical import completed.")
    print(f"Experiment: {result.experiment_name}")
    print(f"Expected semantic runs: {result.expected_run_count}")
    print(
        "Imported run keys: "
        + (", ".join(result.imported_run_keys) if result.imported_run_keys else "none")
    )
    print(
        "Existing run keys: "
        + (", ".join(result.existing_run_keys) if result.existing_run_keys else "none")
    )
    print("Tracking provenance: retrospective_import")
    print("Runtime state: .mlflow/ local SQLite store and local artifact directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

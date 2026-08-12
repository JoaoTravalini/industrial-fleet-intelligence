"""Run the Spark Gold descriptive analytics snapshot rebuild inside the Spark container."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.batch import gold_transformation  # noqa: E402


def main() -> int:
    spark = None
    try:
        config = gold_transformation.load_gold_config(
            PROJECT_ROOT / gold_transformation.CONFIG_RELATIVE_PATH
        )
        silver_path = Path(gold_transformation.container_path(config.silver_input_path))
        if not silver_path.exists():
            print(
                f"FAIL Canonical Silver does not exist: {config.silver_input_path}", file=sys.stderr
            )
            return 1

        spark = gold_transformation.create_spark_session(config)
        counts = gold_transformation.rebuild_gold_snapshot(spark, config)

        print(f"Application: {config.application_name}")
        print(f"Spark master: {config.master}")
        print(f"Silver input rows: {counts.silver_row_count}")
        print(f"Distinct machines: {counts.silver_machine_count}")
        print(f"Machine summary rows: {counts.machine_summary_row_count}")
        print(f"Machine window rows: {counts.machine_window_row_count}")
        print(f"Fleet summary rows: {counts.fleet_summary_row_count}")
        print(f"Machine summary event-count sum: {counts.machine_summary_event_count_sum}")
        print(f"Window event-count sum: {counts.machine_windows_event_count_sum}")
        print("PASS Spark Gold snapshot rebuild completed.")
        return 0
    except (
        gold_transformation.SparkGoldConfigError,
        gold_transformation.SparkGoldValidationError,
    ) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())

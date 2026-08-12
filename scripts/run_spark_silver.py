"""Run the Spark Silver telemetry snapshot rebuild inside the Spark container."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.streaming import silver_transformation  # noqa: E402


def main() -> int:
    spark = None
    try:
        config = silver_transformation.load_silver_config(
            PROJECT_ROOT / silver_transformation.CONFIG_RELATIVE_PATH
        )
        bronze_path = Path(silver_transformation.container_path(config.bronze_input_path))
        if not bronze_path.exists():
            print(f"FAIL Bronze input does not exist: {config.bronze_input_path}", file=sys.stderr)
            return 1

        spark = silver_transformation.create_spark_session(config)
        counts = silver_transformation.rebuild_silver_snapshot(spark, config)

        print(f"Application: {config.application_name}")
        print(f"Spark master: {config.master}")
        print(f"Bronze input rows: {counts.bronze_row_count}")
        print(f"Valid pre-dedup rows: {counts.valid_pre_dedup_row_count}")
        print(f"Canonical Silver rows: {counts.canonical_silver_row_count}")
        print(f"Duplicate audit rows: {counts.duplicate_audit_row_count}")
        print(f"Quarantine rows: {counts.quarantine_row_count}")
        print("PASS Spark Silver snapshot rebuild completed.")
        return 0
    except (
        silver_transformation.SparkSilverConfigError,
        silver_transformation.SparkSilverValidationError,
    ) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())

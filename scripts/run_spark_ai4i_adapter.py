"""Run the Spark AI4I feature adapter inside the Spark container."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.batch import ai4i_feature_adapter  # noqa: E402


def main() -> int:
    spark = None
    try:
        config = ai4i_feature_adapter.load_adapter_config(
            PROJECT_ROOT / ai4i_feature_adapter.CONFIG_RELATIVE_PATH
        )
        source_path = Path(ai4i_feature_adapter.container_path(config.source))
        if not source_path.exists():
            print(f"FAIL Canonical Silver does not exist: {config.source}", file=sys.stderr)
            return 1

        spark = ai4i_feature_adapter.create_spark_session(config)
        counts = ai4i_feature_adapter.rebuild_adapter_output(spark, config)

        print(f"Application: {config.application_name}")
        print(f"Spark master: {config.master}")
        print(f"Adapter version: {config.adapter_version}")
        print(f"Silver input rows: {counts.silver_row_count}")
        print(f"Silver unique event IDs: {counts.silver_distinct_event_id_count}")
        print(f"Adapter output rows: {counts.adapter_row_count}")
        print(f"Adapter unique event IDs: {counts.adapter_distinct_event_id_count}")
        print(f"Adapter output path: {config.output}")
        print("PASS Spark AI4I feature adapter completed.")
        return 0
    except (
        ai4i_feature_adapter.AI4IFeatureAdapterConfigError,
        ai4i_feature_adapter.AI4IFeatureAdapterValidationError,
    ) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())

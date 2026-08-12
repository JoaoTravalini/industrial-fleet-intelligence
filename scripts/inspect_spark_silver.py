"""Inspect Spark Silver telemetry outputs from inside the Spark container."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.streaming import silver_transformation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-rules-check",
        action="store_true",
        help="Run an in-memory Silver rules check instead of inspecting persisted outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spark = None
    try:
        config = silver_transformation.load_silver_config(
            PROJECT_ROOT / silver_transformation.CONFIG_RELATIVE_PATH
        )
        spark = silver_transformation.create_spark_session(config)
        if args.synthetic_rules_check:
            summary = silver_transformation.run_synthetic_rules_check(spark)
        else:
            summary = silver_transformation.inspect_silver_outputs(spark, config)
        print(json.dumps(summary, sort_keys=True))
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

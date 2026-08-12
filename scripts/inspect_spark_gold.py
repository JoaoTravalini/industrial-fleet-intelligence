"""Inspect Spark Gold descriptive analytics outputs from inside the Spark container."""

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

from pipelines.batch import gold_transformation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-analytics-check",
        action="store_true",
        help="Run an in-memory Gold analytics check instead of inspecting persisted outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spark = None
    try:
        config = gold_transformation.load_gold_config(
            PROJECT_ROOT / gold_transformation.CONFIG_RELATIVE_PATH
        )
        spark = gold_transformation.create_spark_session(config)
        if args.synthetic_analytics_check:
            summary = gold_transformation.run_synthetic_analytics_check(spark, config)
        else:
            summary = gold_transformation.inspect_gold_outputs(spark, config)
        print(json.dumps(summary, sort_keys=True))
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

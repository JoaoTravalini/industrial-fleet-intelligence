"""Inspect the local Bronze telemetry Parquet dataset inside Spark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.streaming import bronze_ingestion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Bronze telemetry Parquet data.")
    parser.add_argument("--expected-payloads-json", default=None)
    parser.add_argument("--expected-records-json", default=None)
    return parser.parse_args()


def parse_json_argument(value: str | None, name: str) -> list[Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON array.")
    return parsed


def main() -> int:
    args = parse_args()
    spark = None
    try:
        expected_payloads = parse_json_argument(
            args.expected_payloads_json,
            "--expected-payloads-json",
        )
        expected_records = parse_json_argument(
            args.expected_records_json,
            "--expected-records-json",
        )
        if expected_payloads is not None and not all(
            isinstance(payload, str) for payload in expected_payloads
        ):
            raise ValueError("--expected-payloads-json must contain only strings.")
        if expected_records is not None and not all(
            isinstance(record, dict) for record in expected_records
        ):
            raise ValueError("--expected-records-json must contain only objects.")

        config = bronze_ingestion.load_spark_config(
            PROJECT_ROOT / bronze_ingestion.CONFIG_RELATIVE_PATH
        )
        spark = bronze_ingestion.create_spark_session(config)
        summary = bronze_ingestion.inspect_bronze_dataset(
            spark,
            config,
            expected_payloads=expected_payloads,
            expected_records=expected_records,
        )
        print(json.dumps(summary.to_dict(), sort_keys=True))
    except Exception as exc:
        print(f"FAIL Bronze inspection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

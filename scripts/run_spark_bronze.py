"""Run available-now Kafka to Bronze Structured Streaming ingestion inside Spark."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.streaming import bronze_ingestion  # noqa: E402


def main() -> int:
    spark = None
    try:
        config = bronze_ingestion.load_spark_config(
            PROJECT_ROOT / bronze_ingestion.CONFIG_RELATIVE_PATH
        )
        spark = bronze_ingestion.create_spark_session(config)
        query = bronze_ingestion.start_bronze_query(spark, config, available_now=True)
        print(f"Application: {config.application_name}")
        print(f"Spark master: {config.master}")
        print(f"Kafka bootstrap: {config.kafka_bootstrap_servers}")
        print(f"Kafka topic: {config.kafka_topic}")
        print(f"Bronze output: {config.bronze_output_path}")
        print(f"Checkpoint: {config.checkpoint_path}")
        query.awaitTermination()
        progress = query.lastProgress or {}
        print(f"Query id: {query.id}")
        print(f"Query name: {query.name}")
        print(f"Last batch id: {progress.get('batchId', 'unavailable')}")
        print(f"Input rows in last batch: {progress.get('numInputRows', 'unavailable')}")
        print("PASS Spark Bronze available-now ingestion completed.")
    except Exception as exc:
        print(f"FAIL Spark Bronze ingestion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

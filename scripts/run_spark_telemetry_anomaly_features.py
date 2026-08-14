"""Export operational anomaly features from Silver inside the Spark container."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.batch import telemetry_anomaly_features  # noqa: E402


def main() -> int:
    spark = None
    try:
        config = telemetry_anomaly_features.default_feature_config()
        spark = telemetry_anomaly_features.create_spark_session(config)
        counts = telemetry_anomaly_features.rebuild_feature_output(spark, config)
        print(json.dumps(counts.to_dict(), sort_keys=True))
        return 0
    except telemetry_anomaly_features.TelemetryAnomalyFeatureExtractionError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())

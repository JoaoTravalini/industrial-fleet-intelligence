"""Run local AI4I failure-risk inference from a JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import ai4i_predictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local AI4I failure-risk inference.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ai4i_predictor.sample_input_path(PROJECT_ROOT),
        help="JSON object or array of objects using the AI4I inference feature contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    try:
        predictor = ai4i_predictor.load_predictor(PROJECT_ROOT)
        payload = ai4i_predictor.load_inference_payload(input_path)
        predictions = predictor.predict_batch(payload if isinstance(payload, list) else [payload])
    except (OSError, ValueError) as exc:
        print(f"FAIL AI4I inference failed: {exc}", file=sys.stderr)
        return 1

    response = {
        "model_name": ai4i_predictor.MODEL_NAME,
        "model_version": ai4i_predictor.MODEL_VERSION,
        "input_record_count": len(predictions),
        "predictions": predictions,
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

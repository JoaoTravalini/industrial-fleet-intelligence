"""Warm the configured local Ollama copilot model without using project tools."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.copilot.config import get_copilot_settings  # noqa: E402
from services.copilot.ollama_client import OllamaClient, OllamaClientError  # noqa: E402


def main() -> int:
    settings = get_copilot_settings()
    client = OllamaClient(settings)
    print("Industrial Fleet Intelligence Platform copilot warmup\n", flush=True)
    print("Provider: ollama", flush=True)
    print(f"Model: {settings.ollama_model}", flush=True)
    print(f"Keep alive: {settings.ollama_keep_alive}", flush=True)
    print(f"num_ctx: {settings.num_ctx}", flush=True)
    print(f"num_predict: {settings.num_predict}", flush=True)
    print("Checking local Ollama...", flush=True)

    try:
        models = client.list_models()
    except OllamaClientError as exc:
        print(f"FAIL Ollama unavailable: {exc}", flush=True)
        return 1
    if settings.ollama_model not in models:
        print(f"FAIL Configured model is not installed: {settings.ollama_model}", flush=True)
        return 1

    print("RUN  Minimal warmup request...", flush=True)
    started_at = time.perf_counter()
    try:
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Reply only with OK.",
                }
            ],
            tools=None,
            timeout_seconds=settings.request_timeout_seconds,
        )
    except OllamaClientError as exc:
        print(f"FAIL Warmup request failed: {exc}", flush=True)
        return 1

    elapsed = time.perf_counter() - started_at
    print(f"PASS Model ready in {elapsed:.1f}s.", flush=True)
    if response.metrics:
        print(f"Telemetry: {response.metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

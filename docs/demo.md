# Demo Guide

This 5-10 minute walkthrough is designed for recruiters, hiring managers, and technical reviewers.

## Before The Demo

1. Start Docker Desktop with Linux containers.
2. Ensure PostgreSQL contains materialized platform state.
3. Start the API and dashboard:

```powershell
.\scripts\start_local_platform.ps1
```

4. Optional Copilot mode: start Ollama and ensure `qwen3:4b-instruct` is installed.

## Talk Track

1. **Open the overview page.** Explain that the project is local-first, zero paid services, and uses fictional operational assets plus the public synthetic AI4I dataset.
2. **Open Machines.** Show server-side pagination and explain that PostgreSQL stores materialized operational state, not raw telemetry history.
3. **Open `MCH-0001`.** Walk through prediction probability, anomaly score, vibration, pressure, and persisted SHAP attribution. Emphasize that predictions are model estimates, anomaly scores are detector scores, and SHAP values are attributions.
4. **Open Alerts.** Explain deterministic alert materialization from persisted prediction/anomaly state and note that alert lifecycle mutations are intentionally out of scope.
5. **Open Drift.** Show AI4I model-input drift and anomaly-input drift separately. Explain PSI as an input-distribution diagnostic, not model accuracy.
6. **Open Copilot.** Explain that it uses local Ollama only, source-grounded retrieval, and read-only validated tools.

## Suggested Copilot Prompts

- `Give me a fleet overview.`
- `Explain MCH-0001 using the latest available evidence.`
- `What does anomaly score mean in this project?`
- `Explain the latest drift status.`
- `Why did the latest MCH-0001 prediction receive that model output?`

## Closing Points

- The final AI4I model and threshold were selected before the holdout test was opened.
- The dashboard does not run model inference, Spark, Kafka, SHAP, anomaly scoring, or drift calculation inside API requests.
- Runtime data and model binaries are intentionally ignored and reproducible from documented scripts.
- The project avoids paid APIs, cloud billing, proprietary branding, and confidential data.

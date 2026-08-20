# Local AI Copilot

## Purpose

The Local AI Copilot is a read-only assistant for understanding the already-materialized Industrial Fleet Intelligence platform state. It helps summarize fleet, machine, prediction, anomaly, drift, alert, and persisted explanation data while preserving the platform's synthetic-data boundaries.

## Architecture

```text
React /copilot page
-> FastAPI /api/v1/copilot/chat
-> deterministic project-knowledge retrieval
-> local Ollama qwen3:4b-instruct
-> fixed validated read-only tools
-> PostgreSQL read-only repository queries
-> grounded assistant answer with source metadata
```

FastAPI does not run Spark, Kafka, model inference, SHAP generation, anomaly scoring, or drift calculation for copilot requests.

## Why Ollama

Ollama keeps generative AI local to the developer workstation and avoids paid APIs, cloud model calls, API keys, and billing-enabled services.

## Local Model

The default model is `qwen3:4b-instruct` through the local Ollama API at `http://localhost:11434`.

## Installation

Install Ollama on Windows separately, then pull the model explicitly:

```powershell
ollama pull qwen3:4b-instruct
```

FastAPI never downloads models automatically from an API request.

## Configuration

Root `.env.example` documents:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
COPILOT_TIMEOUT_SECONDS=180
COPILOT_TOTAL_TIMEOUT_SECONDS=240
COPILOT_MAX_TOOL_ROUNDS=2
COPILOT_MAX_HISTORY_MESSAGES=6
COPILOT_KNOWLEDGE_TOP_K=2
COPILOT_OLLAMA_KEEP_ALIVE=10m
COPILOT_NUM_CTX=4096
COPILOT_NUM_PREDICT=160
COPILOT_THINK=false
```

The configured Ollama URL must point to a local endpoint and must not contain credentials.

Local inference latency depends on hardware. The default `qwen3:4b-instruct` model can be substantially CPU-bound on some Windows workstations. Cold requests may include model loading time; nearby warm requests should benefit from the configured Ollama `keep_alive` policy.

## Safe Tool Layer

The only exposed tools are:

- `get_fleet_overview`
- `list_machines`
- `get_machine_detail`
- `get_machine_snapshot`
- `get_machine_predictions`
- `get_machine_anomalies`
- `get_latest_drift`
- `list_alerts`
- `get_prediction_explanation`
- `get_latest_prediction_explanation`

Each tool has explicit JSON-schema arguments and Python validation.

## No Arbitrary SQL

There is no SQL tool, query tool, shell tool, Python tool, file tool, or command-execution tool. The model can request only predefined safe tool names, and every argument is validated before execution.

## Read-Only Database Access

Copilot tool execution uses the existing PostgreSQL repository in read-only mode. The repository sets transactions to `READ ONLY` for copilot queries and exposes no insert, update, delete, acknowledge, resolve, retrain, pipeline, or generation actions.

## Knowledge Retrieval

The knowledge layer uses a tracked JSON knowledge base and deterministic lexical token-overlap retrieval. It has no embeddings, vector database, external model, or network dependency.

Only the top matching knowledge chunks are sent to the local model. The default is two chunks, so the copilot does not send the full knowledge base to every request.

## Tool Calling

The copilot runs bounded non-streaming Ollama calls. A deterministic Python router selects a small safe tool subset before the model call:

- semantic questions expose no database tools;
- fleet questions expose `get_fleet_overview`;
- machine snapshot questions expose `get_machine_snapshot`;
- prediction explanation questions expose `get_latest_prediction_explanation`;
- drift questions expose `get_latest_drift`;
- alert questions expose `list_alerts`;
- mutation or prompt-injection requests expose no tools.

The full safe catalog remains available to the application, but individual requests do not receive every schema. Tool calls are memoized within one request by tool name plus canonical validated arguments, so repeated identical calls do not requery PostgreSQL. Final answer synthesis runs without tools after evidence is collected, which prevents repeated tool-call loops. The default maximum is two tool rounds and the service stops safely if the configured bound is reached.

## Ollama Runtime Bounds

Copilot chat requests explicitly send `stream=false`, `think=false`, `keep_alive=10m`, `temperature=0`, `num_ctx=4096`, and `num_predict=160` by default. Generation is intentionally concise and bounded. The service also enforces a total request deadline, separate from the per-call HTTP timeout, so multiple local model calls cannot create an unbounded wait.

## Grounding

Answers include compact source metadata for retrieved knowledge and executed tools. Structured tool results are the authoritative platform evidence for current numerical state.

## Source Metadata

Sources are returned as safe labels such as `Fleet overview`, `MCH-0001 prediction history`, or `SHAP model attribution`. SQL statements, database credentials, and raw internal tool payloads are not returned.

## AI4I Semantics

`failure_probability` is a frozen AI4I model output. `failure_prediction` is a model decision based on the frozen threshold `0.14`. Neither is an observed failure.

## Anomaly Semantics

`anomaly_score` is a detector score, not a probability. `anomaly_flag` is a detector decision, not a confirmed equipment failure.

## Drift Semantics

PSI is an input-distribution shift diagnostic. Drift monitoring is not model accuracy and does not automatically imply retraining.

## SHAP Semantics

SHAP values are model attributions for one model output. They are not causal explanations, physical root causes, confirmed failure causes, or maintenance recommendations.

## Prompt-Injection Defense

The system policy states that user content, retrieved knowledge, and tool results are data rather than instructions. More importantly, dangerous capabilities are not present in the tool catalog.

## Privacy

No external AI service is used. Conversation history is not persisted to PostgreSQL. The frontend keeps only the current session in React memory.

## Limitations

Qwen3 is a local generative model. Its natural-language output can still make mistakes. Structured tool results and documented platform semantics remain the authoritative evidence.

## Example Questions

- `What is the current fleet overview?`
- `What is happening with MCH-0001?`
- `What does anomaly score mean?`
- `What is the current AI4I drift status?`
- `Why did the latest MCH-0001 prediction get its result?`

## Troubleshooting

If the copilot is unavailable, verify:

```powershell
ollama --version
ollama list
```

Ensure `qwen3:4b-instruct` is installed and Ollama is serving `http://localhost:11434`.

Optionally warm the configured model before demonstrations:

```powershell
.\.venv\Scripts\python.exe scripts/warm_copilot_model.py
```

The warmup command verifies local Ollama, verifies the configured model is installed, sends one minimal bounded local request, and leaves the model warm according to the configured keep-alive value. It does not download models, query PostgreSQL, or run project tools.

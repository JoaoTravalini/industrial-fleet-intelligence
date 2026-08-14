# Operational AI4I Explainability

## Purpose

Operational AI4I explainability materializes SHAP attributions for already-created telemetry predictions. It helps inspect how the frozen AI4I model inputs contributed to one persisted model output.

## Relationship to Predictions

Explanations are downstream of `data/predictions/ai4i/telemetry_predictions.jsonl`. They do not change `failure_probability`, `failure_prediction`, the frozen `0.14` threshold, model identity, anomaly scoring, drift monitoring, or alert policy.

## Materialization Architecture

The implemented path is:

```text
canonical adapted telemetry
-> existing frozen AI4I model
-> existing SHAP positive-class explainer
-> data/explanations/ai4i/telemetry_explanations.jsonl
-> PostgreSQL prediction_explanations
-> FastAPI read endpoint
-> React dashboard explanation panel
```

SHAP is calculated by `scripts/explain_ai4i_telemetry_predictions.py`, not by FastAPI request handlers.

## Model Input Integrity

The materializer loads canonical adapter records from `data/model_input/ai4i/telemetry/` and validates that each reconstructed `model_input` has the same `model_input_sha256` stored in the corresponding prediction record. This proves the explained input is the same six-feature input used for the persisted prediction.

## SHAP Semantics

The implementation reuses `ml/explainability/ai4i_shap.py` and `shap.TreeExplainer` for the trusted packaged Random Forest model. SHAP values are signed decimal model attributions for the model output.

## Positive-Class Attribution

The explained output is the positive-class failure-risk model output. Positive SHAP values push the model output toward higher predicted failure risk. Negative SHAP values push the model output toward lower predicted failure risk.

## Semantic Feature Grouping

The persisted and API-facing explanation contains exactly these six semantic features:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The internal one-hot encoded `Type` columns are grouped back into the semantic `Type` feature.

## Additivity

For every operational explanation, the materializer checks that:

```text
base_value + sum(grouped SHAP contributions)
```

matches the positive-class model output within the existing project SHAP tolerance.

## Persistence

Migration `006_prediction_explanations.sql` creates `prediction_explanations`. Each row links to `model_predictions` through `model_prediction_id` and stores explainer identity, model-input hash, base value, model output value, contribution sum, additivity error, and deterministic JSONB semantic feature contributions.

The stable persistence identity is:

```text
model_prediction_id + explainer_name + explainer_version + explanation_config_hash
```

Repeated persistence is idempotent. Conflicting immutable values fail instead of overwriting history.

## API Access

The read-only endpoint is:

```text
GET /api/v1/machines/{machine_code}/predictions/{event_id}/explanation
```

It reads persisted PostgreSQL state only. It does not load the model artifact, instantiate SHAP, predict, or calculate missing explanations on demand.

## Frontend Visualization

The machine detail page selects the latest prediction by default and fetches its persisted explanation. Selecting a different prediction row updates the explanation query. The SHAP chart shows all six semantic features with actual event feature values and signed decimal SHAP attributions.

## No Causal Claims

Operational SHAP explanations are model attributions. They are not causal explanations, physical root-cause analysis, confirmed failure causes, observed failure labels, or maintenance recommendations.

## Reproducibility

The materialized JSONL output is deterministic. Re-running explanation generation with unchanged predictions, adapter records, model artifact, SHAP version, and explanation configuration produces byte-identical JSONL.

## Limitations

AI4I is a public synthetic dataset adapted for this local portfolio platform. The explanation describes the frozen model behavior on synthetic AI4I-style inputs and does not prove production performance on real industrial equipment.

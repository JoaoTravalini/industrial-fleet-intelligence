# AI4I Model Explainability

## Purpose
This phase adds local SHAP explainability for the already packaged AI4I final Random Forest model. It explains model behavior for the positive class and generates deterministic reports and plots without changing the model, threshold, inference contract, or historical MLflow state.

## Frozen Model
The explained model is the trusted local artifact at `ml/artifacts/ai4i/final_model.joblib`. It is loaded through `ml.inference.ai4i_predictor.load_predictor`, which validates artifact metadata, model identity, the frozen configuration hash, and the pipeline structure before use.

## SHAP Method
The implementation uses `shap.TreeExplainer` because the packaged estimator is a fitted scikit-learn `RandomForestClassifier`. The fitted pipeline preprocessor is reused only through `transform`; no model or preprocessing step is retrained during explainability.

## Positive-Class Explanation
SHAP values are selected explicitly for `Machine failure = 1`. The implementation validates the SHAP output shape and fails if the installed SHAP/scikit-learn combination does not expose a supported binary-class output dimension.

## Global Explanation Dataset
Global explanations use a deterministic 1,000-row subset from the combined train + validation development data. Target labels are not used for SHAP computation. The locked final holdout split is not read or explained.

## Transformed Features
The transformed feature report explains the fitted model feature space: one-hot `Type` components plus the five numerical AI4I inputs. Feature names are obtained from the fitted preprocessor instead of hard-coding encoder category order.

## Grouped Original Features
For readability, the one-hot `Type` SHAP contributions are summed back into the conceptual `Type` feature for each observation before global aggregation. Numerical features map one-to-one. This grouping is additive for interpretation only and does not change the fitted model representation.

## Global Feature Attributions
The generated global reports are:

- `reports/ai4i/shap_transformed_feature_importance.csv`
- `reports/ai4i/shap_grouped_feature_importance.csv`
- `docs/assets/ai4i/explainability/shap_global_importance.png`
- `docs/assets/ai4i/explainability/shap_beeswarm.png`

Mean absolute SHAP magnitude describes attribution strength for the model output. It does not establish physical causality.

## Representative Local Explanations
Three development observations are selected deterministically from train + validation model probabilities:

- `low_risk`: lowest predicted failure probability.
- `threshold_near`: closest probability to the frozen `0.14` threshold.
- `high_risk`: highest predicted failure probability.

The local explanation report is `reports/ai4i/shap_local_explanations.json`, and the corresponding waterfall plots are stored under `docs/assets/ai4i/explainability/`.

## Decision Threshold
The final decision threshold remains frozen at `0.14`. SHAP explains how model features contribute to the positive-class probability; it does not alter the probability or the threshold rule `failure_probability >= 0.14`.

## Additivity Validation
For every generated explanation group, the implementation checks that base value plus positive-class SHAP contributions reconstructs the Random Forest positive-class output within a documented floating-point tolerance. This guards against accidentally explaining the wrong class.

## Interpretation Guidelines
Use these artifacts to understand how the packaged model behaves on synthetic AI4I-style inputs. A positive contribution increases the model output relative to the base value, and a negative contribution decreases it. These are model attributions, not physical causes of failure.

## Limitations
AI4I is a public synthetic dataset. The explanations do not prove real industrial generalization and do not imply that any feature physically causes machine failure. The grouped `Type` view is for readability and is not a newly trained feature representation.

## Operational Explanation Access
Operational telemetry prediction explanations can now be materialized separately from these development reports, persisted in PostgreSQL, and served through the read-only FastAPI dashboard API. That operational path explains already-created telemetry predictions from `data/predictions/ai4i/telemetry_predictions.jsonl` and validates `model_input_sha256` alignment with the canonical adapter records.

Prediction and explanation remain separate concepts. SHAP does not change prediction outputs, binary decisions, the frozen threshold, or business alert policy, and FastAPI does not calculate SHAP during requests.

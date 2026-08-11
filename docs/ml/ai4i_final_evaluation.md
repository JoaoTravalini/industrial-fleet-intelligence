# AI4I Final Holdout Evaluation

## Development Process
The leakage-safe AI4I development process completed baseline modeling, imbalance analysis, train-only threshold development, non-linear model-family comparison, and targeted Random Forest tuning before the test split was opened.

## Frozen Model Decision
The final specification was frozen before final holdout evaluation. The selected model remains the fixed Random Forest retained by the predefined tuning promotion policy.

## Final Random Forest
The final classifier is `RandomForestClassifier` with `n_estimators=300`, `max_depth=None`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=1`.

## Frozen Decision Threshold
The frozen operating threshold is 0.14. It was selected from train-only OOF development predictions and was not selected from the test split.

## Train + Validation Refit
The frozen pipeline was fitted once on 8500 combined train + validation rows with 288 positive labels. The test split was not used for preprocessing fitting.

## Locked Test Protocol
The test split was opened only for final evaluation, containing 1500 rows with 51 positive labels. No model choice, feature policy, hyperparameter, preprocessing step, or threshold was changed after viewing test results.

## Final Test Metrics
Average Precision: 0.770679. ROC-AUC: 0.968707.

## Confusion Matrix
At threshold 0.14, the confusion matrix is `[[1399, 50], [8, 43]]` with precision 0.462366, recall 0.843137, F1 0.597222, and F2 0.723906.

## Precision / Recall Trade-off
At threshold 0.5, precision is 0.958333 and recall is 0.45098. At the frozen threshold 0.14, precision is 0.462366 and recall is 0.843137. Threshold 0.5 is reported only as a reference.

## Validation vs Test
Previous validation metrics are loaded from tracked development artifacts. The test-minus-validation deltas are AP 0.089734, ROC-AUC -0.006036, precision@0.14 -0.037634, recall@0.14 0.078431, F1@0.14 -0.007429, and F2@0.14 0.032417. These differences are descriptive holdout variation only.

## Interpretation
The final evaluation provides a reproducible holdout estimate for this public synthetic dataset. It does not establish production readiness or real-world industrial generalization.

## Limitations
AI4I is synthetic and public. The result should not be interpreted causally, and it does not represent any specific manufacturer, fleet, or industrial operating site.

## Next Steps
Later phases may add model persistence, MLflow tracking, SHAP explainability, local serving, and dashboard integration without revisiting the final holdout test split for model selection.

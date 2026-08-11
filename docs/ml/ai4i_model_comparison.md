# AI4I Non-Linear Model Comparison

## Objective
This phase compares the existing standard Logistic Regression reference with fixed Random Forest and XGBoost baselines for binary classification of `Machine failure`.

## Experimental Protocol
The comparison uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split only. All three model families use identical fold assignments. Each out-of-fold probability comes from a pipeline whose preprocessing and model parameters were fitted without that held-out row.

## Locked Test Set
The locked test split was not read, counted, predicted, evaluated, or used for candidate selection. Final test evaluation is reserved for a later phase.

## Feature Policy
All models receive the same predictive information: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. `source_udi` is traceability-only. `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded.

## Logistic Regression Reference
The reference model keeps the same standard Logistic Regression configuration used in the imbalance phase: no class weighting, `max_iter=1000`, and training-fold-only standardization for numerical variables.

## Random Forest
The Random Forest baseline uses a fixed configuration with `n_estimators=300`, `class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=1`. No hyperparameter search or validation-based tuning is performed.

## XGBoost
The XGBoost baseline uses a conservative CPU-only fixed configuration with `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `objective="binary:logistic"`, `eval_metric="logloss"`, `tree_method="hist"`, `device="cpu"`, `random_state=42`, and `n_jobs=1`. `scale_pos_weight` is calculated only from the labels available to each fitted training fold, or from full train only if XGBoost is the selected final validation candidate.

## Train OOF Results
| Model | OOF AP | OOF ROC-AUC | Max-F2 threshold | Precision | Recall | F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.465735 | 0.898862 | 0.14 | 0.365729 | 0.603376 | 0.533981 |
| Random Forest | 0.749231 | 0.959211 | 0.14 | 0.496203 | 0.827004 | 0.72971 |
| XGBoost | 0.752493 | 0.971056 | 0.67 | 0.644928 | 0.751055 | 0.727124 |

## Threshold Strategy
Thresholds from 0.01 through 0.99 are evaluated on train OOF probabilities only. Each model is associated with its own train-derived max-F2 threshold. F2 is used as an exploratory recall-weighted operating-point summary, not as a confirmed business cost function.

## Model Selection Policy
The predefined selection policy chooses the highest train OOF Average Precision. If another candidate is within 0.01 AP of the best candidate, the simpler model is preferred in this order: `standard_logistic`, `random_forest`, `xgboost`. Validation metrics cannot change the selected model or selected threshold.
Selected model: `random_forest`. Selected threshold: 0.14. Reason: Random Forest is within 0.01 train OOF AP of the best candidate and is simpler under the predefined order.

## Validation Evaluation
Validation AP: 0.680945. Validation ROC-AUC: 0.974743.
At threshold 0.5, precision 0.807692, recall 0.411765, F1 0.545455, F2 0.456522.
At the train-selected threshold, precision 0.5, recall 0.764706, F1 0.604651, F2 0.691489.

## Complexity vs Performance
Non-linear models are evaluated because they can represent interactions that a linear model may miss. Complexity is only justified when train-only development metrics show a measurable ranking improvement large enough to overcome the simplicity tie policy.

## Limitations
This is a fixed-configuration development comparison, not production readiness. It does not claim performance on real industrial equipment, does not persist a model, and does not provide feature-importance or explainability conclusions.

## Next Steps
Later phases may add hyperparameter tuning, final model selection, locked test evaluation, model persistence, MLflow tracking, and SHAP explainability.

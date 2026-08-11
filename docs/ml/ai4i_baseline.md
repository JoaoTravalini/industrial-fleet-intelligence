# AI4I Baseline Classification

## Objective
This phase establishes the first validation-only baseline for binary classification of `Machine failure` using the leakage-safe AI4I modeling dataset.

## Experimental Design
Only `train.csv` and `validation.csv` are used. The training split fits preprocessing and model parameters; the validation split is used only for baseline evaluation.
Training rows: 7000. Validation rows: 1500.

## Leakage Prevention
The predictive feature list is restricted to `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. `source_udi` is traceability-only and is never passed to a model. `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded.

## Locked Test Set
The test set remains locked and was not loaded, evaluated, summarized, or used for prediction in this phase. It is reserved for future final evaluation after model selection is complete.

## Preprocessing
A scikit-learn `ColumnTransformer` is fitted inside each pipeline on training data only. `Type` is encoded with `OneHotEncoder(handle_unknown="ignore")`; the five numerical process variables are standardized with `StandardScaler()`.

## Dummy Baseline
`DummyClassifier(strategy="prior")` provides a trivial benchmark representing the target class distribution.

## Logistic Regression Baseline
`LogisticRegression` is the first real predictive baseline. It uses a conservative configuration with no class weighting, no resampling, no threshold tuning, and no hyperparameter search.

## Validation Metrics
Accuracy is not the primary comparison metric because the target is highly imbalanced. Average Precision (AP), recall, precision, F1, balanced accuracy, and ROC-AUC are more useful for this baseline review.

| Model | AP | ROC-AUC | Balanced accuracy | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.034 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.966 |
| Logistic Regression | 0.38438 | 0.862177 | 0.567247 | 0.636364 | 0.137255 | 0.225806 | 0.968 |

Validation positives: 51 (3.4%). Average Precision (AP) is especially useful here because only about 3.4% of validation observations are positive.

## Class Imbalance
No class balancing has been applied. This baseline intentionally measures natural model behavior before any class weighting, resampling, or threshold optimization.

## Logistic Regression Coefficients
The strongest fitted Logistic Regression coefficients by absolute value are:
- `Torque [Nm]`: coefficient 2.744303 (absolute 2.744303).
- `Rotational speed [rpm]`: coefficient 2.065650 (absolute 2.065650).
- `Air temperature [K]`: coefficient 1.466065 (absolute 1.466065).
- `Process temperature [K]`: coefficient -0.989626 (absolute 0.989626).
- `Tool wear [min]`: coefficient 0.808147 (absolute 0.808147).
- `Type_L`: coefficient 0.430508 (absolute 0.430508).
- `Type_H`: coefficient -0.339300 (absolute 0.339300).
- `Type_M`: coefficient -0.129803 (absolute 0.129803).

Coefficient magnitude is model-specific. Numerical variables are standardized, `Type` is one-hot encoded, and coefficients describe associations within this fitted baseline. They do not establish causality and are not used for automatic feature elimination.

## Key Observations
Logistic Regression improves Average Precision over the trivial Dummy baseline. Default-threshold recall and precision should be interpreted cautiously because no threshold tuning or imbalance strategy has been applied.

## Limitations
This is not production-ready and does not claim generalization to real industrial machines. AI4I is a public synthetic dataset and is not proprietary or official third-party equipment data.

## Next Steps
Future phases may evaluate class imbalance strategies, threshold policies, advanced classifiers, model selection, MLflow tracking, explainability, and final locked test evaluation. None of those steps are implemented in this phase.

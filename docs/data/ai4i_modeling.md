# AI4I Modeling Dataset

## Prediction Objective

The first predictive-maintenance modeling problem is binary classification of `Machine failure` using the public synthetic AI4I 2020 Predictive Maintenance Dataset. This phase defines the modeling dataset and deterministic splits only; no model is trained.

## Feature Policy

Predictive categorical features:

- `Type`

Predictive numerical features:

- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The target is `Machine failure`. The generated modeling CSV files keep `Type` in its original categorical form and keep the numerical variables in their original source units.

## Leakage Prevention

The failure-mode flags `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from predictive features and generated modeling datasets because they are target-adjacent outcome indicators. Including them as ordinary input variables could leak information about `Machine failure` into the model.

No features are derived from `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, or `RNF` in this phase.

## Identifier Handling

`UDI` is renamed to `source_udi` only in derived modeling artifacts. It is retained for traceability and split-integrity validation, but it is not a predictive feature.

`Product ID` is excluded because it is an identifier, not a process measurement. It is not included in the modeling frame and must not be used as a predictive feature.

## Train / Validation / Test Strategy

The split policy is machine-readable in `ml/config/ai4i_modeling.json`:

- Random seed: `42`
- Train fraction: `0.70`
- Validation fraction: `0.15`
- Test fraction: `0.15`
- Stratification column: `Machine failure`

The preparation logic uses deterministic stratified splitting with scikit-learn. It first splits 70% train and 30% temporary data, then splits the temporary data equally into validation and test sets. Stratification uses only `Machine failure`; it does not stratify by `Type` or any other field.

## Class Distribution

Generated split sizes and target distributions:

| Split | Rows | Machine failure = 0 | Machine failure = 1 | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| Train | 7000 | 6763 | 237 | 3.385714% |
| Validation | 1500 | 1449 | 51 | 3.4% |
| Test | 1500 | 1449 | 51 | 3.4% |

The full dataset contains 10000 rows with 339 positive `Machine failure` observations. Stratification is used because positive failures are rare relative to non-failures.

## Reproducibility

Generated modeling files are written under `data/processed/ai4i/`:

- `train.csv`
- `validation.csv`
- `test.csv`
- `split_assignments.csv`

`split_assignments.csv` contains only `source_udi` and `split`. Generated processed datasets are ignored by Git and can be reconstructed from the raw AI4I CSV and tracked configuration.

A deterministic machine-readable summary is tracked at `reports/ai4i/modeling_split_summary.json`. It does not contain absolute paths, usernames, runtime timestamps, or machine-specific information.

## Future Preprocessing

Future preprocessing should be fitted on training data only and stored inside a scikit-learn `Pipeline` to prevent leakage.

Planned preprocessing design:

- Categorical: `Type` -> `OneHotEncoder(handle_unknown="ignore")`
- Numerical: the five process variables -> preprocessing determined during modeling

This phase does not fit `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, or any other transformer. It also does not normalize variables, standardize variables, convert `Type` to dummy columns, perform feature selection, create polynomial features, train a model, calculate class weights, or perform class balancing.

## Limitations

The generated dataset is derived from an external public synthetic dataset, not from the fictional local `MCH-XXXX` fleet and not from future streaming telemetry. No oversampling, undersampling, SMOTE, synthetic observation generation, model training, model evaluation, or hyperparameter tuning has been performed.
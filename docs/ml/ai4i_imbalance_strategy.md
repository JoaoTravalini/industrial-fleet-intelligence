# AI4I Imbalance and Threshold Strategy

## Motivation
The previous Logistic Regression baseline had useful ranking metrics but low default-threshold recall. This phase investigates whether class weighting and a lower decision threshold improve minority-class detection.

## Experimental Protocol
The training split is used for 5-fold out-of-fold model development, threshold analysis, and candidate selection. Validation is evaluated once after the model variant and threshold are selected from training OOF results.

## Locked Test Set
The test split remains locked and was not read, summarized, predicted, or evaluated.

## Cross-Validation
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` creates OOF probabilities. Each fold fits its own preprocessing pipeline on fold-training rows only, so held-out rows are transformed by a pipeline not fitted on them.

## Standard Logistic Regression
Train OOF AP: 0.465735. Train OOF ROC-AUC: 0.898862. At threshold 0.5, precision 0.742424, recall 0.206751, F1 0.323432, F2 0.241617.

## Class-Weighted Logistic Regression
Train OOF AP: 0.435379. Train OOF ROC-AUC: 0.901906. At threshold 0.5, precision 0.139636, recall 0.810127, F1 0.238213, F2 0.413259.

## Threshold Analysis
Thresholds from 0.01 to 0.99 are evaluated on train OOF probabilities. Accuracy is not used as a threshold-selection objective because failures are rare.

## Candidate Selection Policy
The documented policy compares train OOF Average Precision. If AP differs by less than 0.01, the simpler standard model is selected; otherwise, the higher-AP model is selected. The selected model then uses its train OOF max-F2 threshold.
Selected model: `standard_logistic`. Selected threshold: 0.14. Reason: Standard Logistic Regression has higher train OOF Average Precision.

## Validation Evaluation
Validation AP: 0.38438. Validation ROC-AUC: 0.862177.
At threshold 0.5, precision 0.636364, recall 0.137255, F1 0.225806, F2 0.162791.
At the selected threshold, precision 0.328947, recall 0.490196, F1 0.393701, F2 0.446429.

## Precision vs Recall Trade-off
Lower thresholds can reduce false negatives by predicting more positives, but this usually increases false positives. False positives and false negatives require domain and business cost information before any final maintenance policy can be chosen.

## Limitations
F2 gives recall more weight than precision, but it is exploratory here and is not an official business objective. The selected threshold is a model-development candidate, not a production-optimal operating point.

## Next Steps
Future phases may compare additional model families, perform formal model selection, track experiments, add explainability, and eventually evaluate the locked test split.

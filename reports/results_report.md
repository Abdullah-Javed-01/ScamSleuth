# ScamSleuth Results Report

## 1. TF-IDF + Logistic Regression Baseline

The first baseline model used TF-IDF lexical features with Logistic Regression. Only the training set was used to fit the TF-IDF vocabulary and model, while the validation set was used for evaluation.

### Baseline Metrics

- Accuracy: 0.6667
- Precision: 0.6667
- Recall: 0.6667
- F1-score: 0.6667
- ROC-AUC: 0.7096
- PR-AUC: 0.7501

### Confusion Matrix

- True Negatives: 30
- False Positives: 15
- False Negatives: 15
- True Positives: 30

### Baseline Findings

Coefficient inspection showed that the model learned several meaningful scam-associated terms such as `buy`, `wallet`, `card`, `forward`, and `gift cards`.

However, it also assigned importance to weak or dataset-specific terms such as `august`, `10`, and `am`.

This indicates that lexical information contains useful signal but is insufficient by itself. Structural and recruitment-specific behavioral features are therefore required to help the model capture patterns that cannot be represented reliably through vocabulary alone.

## 2. Feature Ablation Study

An ablation study was performed to determine the contribution of structural and behavioral engineered features.

| Feature Set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF only | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.7096 | 0.7501 |
| TF-IDF + Structural | 0.6000 | 0.5882 | 0.6667 | 0.6250 | 0.6509 | 0.6633 |
| TF-IDF + Behavioral | 0.6667 | 0.8000 | 0.4444 | 0.5714 | 0.7862 | 0.8208 |
| TF-IDF + Structural + Behavioral | 0.6444 | 0.7826 | 0.4000 | 0.5294 | 0.7195 | 0.7791 |

### Interpretation

The structural feature group reduced both ROC-AUC and PR-AUC when added to the lexical baseline, indicating that these features introduced more noise than useful generalizable signal.

Behavioral features produced the strongest ranking performance, improving ROC-AUC from 0.7096 to 0.7862 and PR-AUC from 0.7501 to 0.8208.

At the default probability threshold of 0.50, however, the TF-IDF + Behavioral model became conservative when predicting Scam. Precision increased to 0.8000 while recall decreased to 0.4444.

Therefore, threshold-dependent metrics at 0.50 are not sufficient to select the final operating configuration. Threshold selection will be performed separately after model tuning.

The behavioral feature definitions were frozen before this validation comparison and were not modified based on validation performance.

## 3. Decision Threshold Selection

The tuned Logistic Regression model uses predicted Scam probabilities, but the default classification threshold of 0.50 was not accepted automatically.

Thresholds from 0.05 to 0.95 were evaluated on the validation split.

| Operating Point | Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.8077 | 0.4667 | 0.5915 | 0.5097 | 5 | 24 |
| Best F1 | 0.31 | 0.6939 | 0.7556 | 0.7234 | 0.7424 | 15 | 11 |
| Best F2 | 0.16 | 0.5357 | 1.0000 | 0.6977 | 0.8523 | 39 | 0 |

### Selected Operating Threshold: 0.31

Threshold 0.31 was selected because ScamSleuth should place meaningful emphasis on detecting scams while avoiding an excessive number of false alarms.

Lowering the threshold from 0.50 to 0.31 reduced false negatives from 24 to 11 and increased correctly detected scams from 21 to 34.

Although threshold 0.16 achieved perfect recall and the highest F2 score on validation, it incorrectly flagged 39 of 45 Safe examples as Scam. This false-positive rate was considered operationally excessive.

Therefore, 0.31 was selected as a more practical precision-recall trade-off.

The threshold was frozen before final test-set evaluation.

## 4. Final Test Evaluation

After feature design, model selection, hyperparameter tuning, and threshold selection were completed, all decisions were frozen before final test evaluation.

The selected system used:

- TF-IDF lexical features
- 11 frozen behavioral features
- Logistic Regression
- `C = 4.0`
- TF-IDF unigrams
- `min_df = 2`
- `max_df = 0.95`
- `sublinear_tf = True`
- Decision threshold = `0.31`

The final model was trained on the combined training and validation development data (510 examples) and evaluated on the previously untouched 90-example test split.

### Final Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.9222 |
| Precision | 0.8800 |
| Recall | 0.9778 |
| F1-score | 0.9263 |
| F2-score | 0.9565 |
| ROC-AUC | 0.9753 |
| PR-AUC | 0.9838 |

### Final Confusion Matrix

| | Predicted Safe | Predicted Scam |
|---|---:|---:|
| Actual Safe | 39 | 6 |
| Actual Scam | 1 | 44 |

The final system correctly detected 44 of 45 Scam examples while incorrectly flagging 6 of 45 Safe examples.

The test result was not used for additional feature engineering, hyperparameter tuning, or threshold adjustment.

### Interpretation and Limitation

Test performance was substantially higher than validation performance. This may reflect differences in difficulty between held-out template clusters, the larger final development training set, and sampling variation from the relatively small 90-example test set.

Because the dataset is synthetic, these results should be interpreted as performance on the designed ScamSleuth benchmark rather than an estimate of real-world recruitment-scam accuracy.

## 5. Model Explainability

Global Logistic Regression coefficients were inspected to understand which features influenced the final classifier.

The strongest Scam-associated features were primarily the hand-engineered behavioral signals, including:

- `selection_bypass_flag`
- `suspicious_application_link_flag`
- `payment_request_flag`
- `paid_training_flag`
- `lookalike_domain_flag`
- `money_transfer_flag`
- `equipment_purchase_flag`
- `identity_document_flag`

This indicates that the final classifier relies substantially on recruitment-specific behavioral patterns rather than lexical vocabulary alone.

Some TF-IDF terms also received strong coefficients. However, terms such as `august`, `september`, `online`, and other ordinary recruitment vocabulary demonstrate that lexical coefficients may capture dataset-specific correlations and should not be interpreted as universal fraud indicators.

### Individual Prediction Explanation

For a test message requesting a processing fee before onboarding, the model predicted:

- Prediction: Scam
- Scam probability: 0.9284
- Decision threshold: 0.31

The strongest contribution toward Scam was `payment_request_flag`, with a positive contribution of approximately 3.53.

Interestingly, the isolated TF-IDF feature `pay` contributed slightly toward Safe, while the behavioral payment-request feature strongly pushed toward Scam. This demonstrates the value of combining lexical features with higher-level behavioral patterns.

Feature contributions describe how the trained model reached a prediction; they do not establish that an individual word or feature proves fraud.

## 6. Error Analysis

The frozen final model produced 7 errors on the 90-example held-out test set:

- 6 False Positives
- 1 False Negative

Five of the six False Positive examples were annotated as Hard examples. Their Scam probabilities ranged from approximately 0.316 to 0.410, only slightly above the selected decision threshold of 0.31.

These examples contained realistic but potentially suspicious-looking characteristics such as urgent recruitment, informal referrals, small family businesses, international hiring, temporary projects, or limited web presence. The model therefore showed a tendency to flag ambiguous legitimate recruitment when weak lexical warning signs accumulated.

The only False Negative was a visa-processing scam in which the applicant was required to send EUR 60 to a personal account before receiving an employment contract. The model assigned a Scam probability of 0.1543. This indicates that some fraudulent payment mechanisms can be missed when their wording falls outside the frozen behavioral patterns.

### Adversarial Stress-Test Errors

The 16-example adversarial stress test produced:

- 4 False Positives
- 1 False Negative

Several False Positives used explicit negation, such as statements that candidates were not required to buy equipment, pay training fees, or receive money through personal bank accounts.

This demonstrates a limitation of the current feature representation: lexical and regex-based features can identify suspicious phrases but do not fully understand whether those behaviors are being required, prohibited, or merely discussed.

The adversarial False Negative involved credential theft phrased differently from the training examples, demonstrating that pattern-based behavioral features remain sensitive to phrasing variation.

### Main Failure Modes

1. **Threshold-borderline ambiguity** — difficult legitimate recruitment examples may fall slightly above the selected Scam threshold.
2. **Negation and context limitations** — suspicious vocabulary may trigger even when the text explicitly warns against the behavior.
3. **Behavioral pattern coverage gaps** — unseen formulations of payment or credential theft may not activate the intended engineered features.
4. **Lexical dataset associations** — TF-IDF features can learn correlations that are specific to the synthetic benchmark rather than universally meaningful.

No model, feature, or threshold changes were made after inspecting the test or adversarial results.
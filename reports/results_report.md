# ScamSleuth Results Report

## 1. Project Summary

ScamSleuth is a binary text-classification project for recruitment-related content.

The target labels are:

- **Safe** — recruitment communication with no sufficient evidence of deceptive or improper behavior.
- **Scam** — recruitment communication containing meaningful evidence of deceptive behavior intended to obtain money, credentials, sensitive information, account access, identity documents, illegal financial assistance, or another improper benefit.

The final dataset contains **600 synthetic recruitment examples**:

| Label | Count |
|---|---:|
| Safe | 300 |
| Scam | 300 |

The data is also balanced across the three supported text types:

- Recruiter messages
- Recruitment emails
- Job/internship postings

Because the dataset is balanced, no SMOTE, oversampling, undersampling, or class weighting was applied. Class balance was instead preserved during splitting and monitored during evaluation.

---

## 2. Leakage-Aware Dataset Split

Synthetic examples can share template-level wording, so a normal random row split could produce overly optimistic evaluation results.

ScamSleuth therefore splits data by `template_cluster`.

| Split | Rows | Template Clusters |
|---|---:|---:|
| Train | 420 | 84 |
| Validation | 90 | 18 |
| Test | 90 | 18 |
| **Total** | **600** | **120** |

No template cluster appears in more than one split.

The split is deterministic with random seed `42`.

The source dataset fingerprint and split metadata are stored in:

```text
artifacts/split_metadata.json
```

---

## 3. TF-IDF + Logistic Regression Baseline

The first baseline model used TF-IDF lexical features with Logistic Regression.

Only the training split was used to fit the TF-IDF vocabulary and model. The validation set was used only for evaluation.

### Baseline Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.6667 |
| Precision | 0.6667 |
| Recall | 0.6667 |
| F1-score | 0.6667 |
| ROC-AUC | 0.7096 |
| PR-AUC | 0.7501 |

### Confusion Matrix

```text
[[30 15]
 [15 30]]
```

This corresponds to:

- True Negatives: 30
- False Positives: 15
- False Negatives: 15
- True Positives: 30

### Baseline Findings

Coefficient inspection showed that the lexical model learned several meaningful scam-associated terms such as `buy`, `wallet`, `card`, `forward`, and `gift cards`.

However, it also assigned importance to weak or dataset-specific terms such as `august`, `10`, and `am`.

This indicates that lexical information contains useful signal but is insufficient by itself. Recruitment-specific engineered features were therefore evaluated to capture behaviors that cannot be represented reliably through vocabulary alone.

---

## 4. Feature Engineering

Three feature families were evaluated:

1. TF-IDF lexical features
2. Structural features
3. Behavioral pattern features

### Structural Features

The structural feature set included:

- `word_count`
- `char_count`
- `sentence_count`
- `url_count`
- `question_count`
- `caps_ratio`
- `digit_ratio`
- `currency_reference_count`

### Behavioral Features

The final behavioral feature set included:

1. `payment_request_flag`
2. `credential_request_flag`
3. `urgency_flag`
4. `identity_document_flag`
5. `equipment_purchase_flag`
6. `money_transfer_flag`
7. `paid_training_flag`
8. `suspicious_application_link_flag`
9. `selection_bypass_flag`
10. `cheque_overpayment_flag`
11. `lookalike_domain_flag`

Each engineered feature and its hypothesis is documented in:

```text
features/FEATURE_JUSTIFICATIONS.md
```

Only information derived from the raw `text` field was used as predictive input.

Metadata such as `difficulty`, `scam_category`, `strong_signals`, `weak_signals`, and `label_reason` was excluded from model features.

---

## 5. Feature Ablation Study

An ablation study was performed to determine the contribution of structural and behavioral engineered features.

| Feature Set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF only | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.7096 | 0.7501 |
| TF-IDF + Structural | 0.6000 | 0.5882 | 0.6667 | 0.6250 | 0.6509 | 0.6633 |
| TF-IDF + Behavioral | 0.6667 | 0.8000 | 0.4444 | 0.5714 | **0.7862** | **0.8208** |
| TF-IDF + Structural + Behavioral | 0.6444 | 0.7826 | 0.4000 | 0.5294 | 0.7195 | 0.7791 |

### Interpretation

The structural feature group reduced both ROC-AUC and PR-AUC when added to the lexical baseline, indicating that these features introduced more noise than useful generalizable signal.

Behavioral features produced the strongest ranking performance, improving ROC-AUC from 0.7096 to 0.7862 and PR-AUC from 0.7501 to 0.8208.

At the default probability threshold of 0.50, however, the TF-IDF + Behavioral model became conservative when predicting Scam. Precision increased to 0.8000 while recall decreased to 0.4444.

Therefore, the preferred feature configuration became:

```text
TF-IDF + 11 Behavioral Features
```

Structural features were retained as a documented experiment but excluded from the final classifier.

The behavioral feature definitions were frozen before the final model evaluation.

---

## 6. Model Selection and Hyperparameter Tuning

Two model families were evaluated:

- Logistic Regression
- Linear SVM

Hyperparameter tuning used:

```text
StratifiedGroupKFold
```

with:

```text
n_splits = 5
```

and `template_cluster` as the grouping variable.

This kept related synthetic variants together during cross-validation.

### Best Grouped-CV Results

#### Logistic Regression

```text
C = 4.0
TF-IDF min_df = 2
TF-IDF ngram_range = (1, 1)
Grouped-CV PR-AUC = 0.9302
```

#### Linear SVM

```text
C = 2.0
TF-IDF min_df = 2
TF-IDF ngram_range = (1, 1)
Grouped-CV PR-AUC = 0.9474
```

### Validation Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.6778 | 0.8077 | 0.4667 | 0.5915 | **0.7728** | **0.8149** |
| Tuned Linear SVM | **0.6889** | **0.8148** | **0.4889** | **0.6111** | 0.7620 | 0.8108 |

Logistic Regression was selected before final test inspection because its validation ranking performance was comparable to Linear SVM and it provides native probability output.

That probability output enabled deliberate threshold selection and straightforward coefficient-based explanations.

---

## 7. Frozen Model Family Comparison on Held-Out Test Set

After model selection was completed using training and validation data, both frozen candidate model families were evaluated on the same held-out test set for final reporting.

Both models used the same preferred feature configuration:

```text
TF-IDF + 11 Behavioral Features
```

For this symmetric comparison, each classifier used its standard decision boundary:

- Logistic Regression: probability `>= 0.50`
- Linear SVM: decision function `>= 0`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.8889 | 1.0000 | 0.7778 | 0.8750 | 0.9753 | 0.9838 |
| Tuned Linear SVM | **0.9000** | **1.0000** | **0.8000** | **0.8889** | **0.9812** | **0.9859** |

Linear SVM achieved slightly stronger held-out test metrics under the default decision boundaries.

However, this comparison was performed only after the model-selection process had already been frozen. It was **not used to change the selected final model**.

Logistic Regression remained the preferred operational model because:

- it was selected using validation evidence before test inspection;
- its validation ranking performance was comparable to Linear SVM;
- it provides native probability estimates;
- probability output enabled deliberate threshold selection;
- its coefficients support straightforward global and individual explanations.

The comparison output is saved in:

```text
reports/model_comparison_test.csv
```

---

## 8. Decision Threshold Selection

The tuned Logistic Regression model produces predicted Scam probabilities, but the default classification threshold of 0.50 was not accepted automatically.

Thresholds from 0.05 to 0.95 were evaluated on the validation split.

| Operating Point | Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.8077 | 0.4667 | 0.5915 | 0.5097 | 5 | 24 |
| Best F1 | **0.31** | 0.6939 | 0.7556 | **0.7234** | 0.7424 | 15 | 11 |
| Best F2 | 0.16 | 0.5357 | 1.0000 | 0.6977 | **0.8523** | 39 | 0 |

### Selected Operating Threshold: 0.31

Threshold 0.31 was selected because ScamSleuth should place meaningful emphasis on detecting scams while avoiding an excessive number of false alarms.

Lowering the threshold from 0.50 to 0.31 reduced false negatives from 24 to 11 and increased correctly detected scams from 21 to 34 on the validation split.

Although threshold 0.16 achieved perfect recall and the highest F2 score, it incorrectly flagged 39 of 45 Safe validation examples as Scam.

That false-positive rate was considered operationally excessive.

Therefore, `0.31` was selected as the final operating threshold and frozen before final test evaluation.

---

## 9. Final Operational Test Evaluation

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

The final model was trained on the combined training and validation development data:

```text
510 examples
```

and evaluated on the previously untouched:

```text
90-example test split
```

### Final Test Metrics

| Metric | Score |
|---|---:|
| Accuracy | **0.9222** |
| Precision | **0.8800** |
| Recall | **0.9778** |
| F1-score | **0.9263** |
| F2-score | **0.9565** |
| ROC-AUC | **0.9753** |
| PR-AUC | **0.9838** |

### Final Confusion Matrix

| | Predicted Safe | Predicted Scam |
|---|---:|---:|
| Actual Safe | 39 | 6 |
| Actual Scam | 1 | 44 |

The final system correctly detected **44 of 45 Scam examples** while incorrectly flagging 6 of 45 Safe examples.

The test result was not used for additional feature engineering, hyperparameter tuning, or threshold adjustment.

### Confusion Matrix Figure

![Final Test Confusion Matrix](figures/confusion_matrix.png)

### ROC Curve

![Final Test ROC Curve](figures/roc_curve.png)

### Precision-Recall Curve

![Final Test Precision-Recall Curve](figures/precision_recall_curve.png)

### Interpretation

Test performance was substantially higher than validation performance.

Possible reasons include:

- differences in difficulty between held-out template clusters;
- the larger final development training set;
- sampling variation from the relatively small 90-example test set.

Because the dataset is synthetic, these results should be interpreted as performance on the designed ScamSleuth benchmark rather than an estimate of real-world recruitment-scam accuracy.

---

## 10. Model Explainability

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

Some TF-IDF terms also received strong coefficients.

However, terms such as `august`, `september`, `online`, and other ordinary recruitment vocabulary demonstrate that lexical coefficients may capture dataset-specific correlations and should not be interpreted as universal fraud indicators.

### Global Feature Importance

Positive coefficients push predictions toward **Scam**, while negative coefficients push predictions toward **Safe**.

![Global Feature Importance](figures/feature_importance.png)

Feature importance should be interpreted as model association rather than universal fraud evidence.

### Individual Prediction Explanation

For an illustrative recruitment message requesting a processing fee before onboarding, the model predicted:

- Prediction: Scam
- Scam probability: 0.9284
- Decision threshold: 0.31

The strongest contribution toward Scam was:

```text
payment_request_flag
```

with a positive contribution of approximately `3.53`.

Interestingly, the isolated TF-IDF feature `pay` contributed slightly toward Safe, while the behavioral payment-request feature strongly pushed toward Scam.

This demonstrates the value of combining lexical features with higher-level behavioral patterns.

Feature contributions describe how the trained model reached a prediction; they do not establish that an individual word or feature proves fraud.

---

## 11. Adversarial Stress Test

A separate 16-example adversarial stress-test set was created to probe:

- unusual scam wording;
- legitimate messages containing suspicious vocabulary;
- explicit negation;
- mixed legitimate and suspicious signals.

The stress-test dataset is stored in:

```text
data/stress_test/adversarial_stress_test.csv
```

### Stress-Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.6875 |
| Precision | 0.6364 |
| Recall | 0.8750 |
| F1 | 0.7368 |

Confusion matrix:

```text
[[4 4]
 [1 7]]
```

The lower adversarial performance demonstrates that the model is less robust to deliberately difficult wording than its held-out benchmark results suggest.

The stress test was used only for robustness analysis.

No model, feature, regex rule, hyperparameter, or threshold was modified after these failures were inspected.

---

## 12. Error Analysis

The frozen final model produced 7 errors on the 90-example held-out test set:

- 6 False Positives
- 1 False Negative

Five of the six False Positive examples were annotated as **Hard**.

Their Scam probabilities ranged from approximately `0.316` to `0.410`, only slightly above the selected decision threshold of `0.31`.

These examples contained realistic but potentially suspicious-looking characteristics such as:

- urgent recruitment;
- informal referrals;
- small family businesses;
- international hiring;
- temporary projects;
- limited web presence.

The model therefore showed a tendency to flag ambiguous legitimate recruitment when weak lexical warning signs accumulated.

The only False Negative was a visa-processing scam in which the applicant was required to send EUR 60 to a personal account before receiving an employment contract.

The model assigned a Scam probability of `0.1543`.

This indicates that some fraudulent payment mechanisms can be missed when their wording falls outside the frozen behavioral patterns.

### Adversarial Stress-Test Errors

The 16-example adversarial stress test produced:

- 4 False Positives
- 1 False Negative

Several False Positives used explicit negation, such as statements that candidates were not required to:

- buy equipment;
- pay training fees;
- receive money through personal bank accounts.

This demonstrates a limitation of the current feature representation: lexical and regex-based features can identify suspicious phrases but do not fully understand whether those behaviors are being required, prohibited, or merely discussed.

The adversarial False Negative involved credential theft phrased differently from the training examples.

This demonstrates that pattern-based behavioral features remain sensitive to phrasing variation.

### Main Failure Modes

1. **Threshold-borderline ambiguity** — difficult legitimate recruitment examples may fall slightly above the selected Scam threshold.
2. **Negation and context limitations** — suspicious vocabulary may trigger even when the text explicitly warns against the behavior.
3. **Behavioral pattern coverage gaps** — unseen formulations of payment or credential theft may not activate the intended engineered features.
4. **Lexical dataset associations** — TF-IDF features can learn correlations specific to the synthetic benchmark rather than universally meaningful signals.

Detailed error outputs are stored in:

```text
reports/test_error_analysis.csv
reports/stress_test_error_analysis.csv
reports/stress_test_predictions.csv
```

---

## 13. Limitations

### Synthetic Dataset

The dataset is synthetic and cannot represent the full diversity of real-world recruitment communication.

### Small Evaluation Sets

Validation and test splits each contain only 90 examples, so individual predictions can noticeably affect reported percentages.

### Negation and Context

Regex and lexical features do not provide full semantic understanding.

For example:

```text
Candidates are not required to buy equipment.
```

may contain many of the same surface signals as:

```text
Candidates must buy equipment.
```

### Behavioral Phrasing Variation

Fraudulent behavior can be expressed using unseen wording that falls outside the frozen pattern set.

### Lexical Artifacts

TF-IDF can learn dataset-specific correlations from ordinary recruitment vocabulary, dates, locations, and formatting.

### Probability Calibration

The decision threshold was selected on validation probabilities from a model fitted only on the training split.

The final model was later refit on Train + Validation, which can shift probability calibration slightly.

A stronger production workflow would select the operating threshold from out-of-fold development predictions before fitting the final model.

---

## 14. Future Hardening

A production-oriented ScamSleuth system could be strengthened using:

- transformer or semantic text representations;
- improved negation and context understanding;
- character-level features for obfuscated scam wording;
- URL and domain reputation intelligence;
- larger real-world recruitment datasets;
- out-of-fold threshold selection;
- explicit probability calibration;
- multilingual recruitment examples;
- continuous adversarial evaluation;
- human review for borderline predictions.

These improvements should be evaluated on genuinely external recruitment data before any production claims are made.

---

## 15. Stakeholder Interpretation

The final benchmark results show that ScamSleuth can identify many recruitment scams while maintaining useful precision on the designed synthetic test set.

The selected operating threshold prioritizes scam detection more strongly than the default 0.50 threshold.

However, the adversarial stress test demonstrates that benchmark accuracy alone is not enough to establish real-world reliability.

For practical use, ScamSleuth should therefore be treated as a **screening aid** rather than an automated fraud accusation system.

A high Scam score should trigger additional verification such as:

- checking the company identity;
- confirming the recruiter through official channels;
- reviewing application URLs;
- refusing applicant-paid recruitment fees;
- avoiding disclosure of passwords, OTPs, or banking credentials;
- validating written employment terms before transferring money or sensitive documents.

The most appropriate operational use is to help prioritize suspicious recruitment communication for human review.

# ScamSleuth Feature Engineering — Hypotheses and Justifications

This document explains the engineered features evaluated in ScamSleuth and the hypothesis behind each feature.

The objective is not to create hard-coded scam rules. Each engineered feature represents a potentially useful signal that can be combined with TF-IDF lexical information by the machine-learning model.

A feature being present does **not** automatically mean that a recruitment message is fraudulent.

---

## Final Feature Families

ScamSleuth evaluated three main feature families:

1. TF-IDF lexical features
2. Structural features
3. Behavioral pattern features

The final preferred classifier uses:

```text
TF-IDF + Behavioral Features
```

Structural features were implemented and evaluated but excluded from the final classifier after ablation testing showed weaker validation performance.

---

# 1. Structural Features

Structural features describe measurable properties of the text without directly encoding whether a message is fraudulent.

## `word_count`

**Definition:**
Number of words in the recruitment text.

**Hypothesis:**
Very short or unusually long recruitment messages may differ in style from normal hiring communication.

**Limitation:**
Length alone is not a reliable scam indicator. Legitimate recruiters may send both very short and very detailed messages.

---

## `char_count`

**Definition:**
Total number of characters in the text.

**Hypothesis:**
Character length provides another representation of message size and may capture differences not fully reflected by word count.

**Limitation:**
This feature is strongly correlated with word count and does not directly represent fraudulent behavior.

---

## `sentence_count`

**Definition:**
Approximate number of sentences after URLs and email addresses are protected from sentence-boundary splitting.

**Hypothesis:**
Message structure and level of detail may differ between legitimate recruitment communication and some scam templates.

**Limitation:**
Informal messages often use weak punctuation, so sentence count is only approximate.

---

## `url_count`

**Definition:**
Number of URLs found in the recruitment text.

**Hypothesis:**
Links are commonly used in legitimate job applications but can also be used for phishing or fake recruitment portals.

**Limitation:**
A URL by itself is not suspicious. Both Safe and Scam examples intentionally contain URLs.

---

## `question_count`

**Definition:**
Number of question marks in the text.

**Hypothesis:**
Recruitment messages asking many questions may have a different interaction style from static job advertisements or scam templates.

**Limitation:**
Question frequency is highly context dependent and is not inherently associated with fraud.

---

## `caps_ratio`

**Definition:**
Ratio of uppercase alphabetic characters to all alphabetic characters.

**Hypothesis:**
Excessive capitalization can sometimes accompany promotional, urgent, or manipulative communication.

**Limitation:**
Capitalization can also result from acronyms, formatting, company names, or ordinary writing style.

---

## `digit_ratio`

**Definition:**
Ratio of numeric characters to total characters.

**Hypothesis:**
Recruitment scams involving fees, salaries, phone numbers, deadlines, or payment instructions may contain relatively more numeric content.

**Limitation:**
Legitimate job advertisements frequently contain salary ranges, dates, working hours, and addresses.

---

## `currency_reference_count`

**Definition:**
Number of currency symbols or supported currency codes such as PKR, USD, GBP, EUR, AED, SAR, CAD, AUD, INR, MYR, and QAR.

**Hypothesis:**
Scam messages involving applicant-paid fees or financial transfers may contain explicit monetary references.

**Limitation:**
Salary information is a normal part of legitimate recruitment communication, so currency references are not treated as scam evidence on their own.

---

# Structural Feature Decision

Structural features were tested through feature ablation.

Validation performance:

| Feature Set | ROC-AUC | PR-AUC |
|---|---:|---:|
| TF-IDF only | 0.7096 | 0.7501 |
| TF-IDF + Structural | 0.6509 | 0.6633 |
| TF-IDF + Behavioral | 0.7862 | 0.8208 |
| TF-IDF + Structural + Behavioral | 0.7195 | 0.7791 |

Adding structural features reduced validation ranking performance.

Therefore, structural features were retained as a documented experiment but were **excluded from the preferred final classifier**.

This decision was made from validation results before final test evaluation.

---

# 2. Behavioral Pattern Features

Behavioral features are binary indicators derived from the raw recruitment text.

They encode suspicious recruitment behaviors rather than individual keywords whenever possible.

Each feature returns:

```text
0 = pattern not detected
1 = pattern detected
```

Binary indicators were chosen instead of raw match counts because overlapping regular expressions could otherwise artificially inflate feature values.

---

## `payment_request_flag`

**Definition:**
Detects language indicating that the applicant is expected to pay a recruitment-related fee or charge.

Examples may include:

- Interview fees
- Processing charges
- Visa fees
- Registration fees
- Security deposits
- Applicant-funded recruitment costs

**Hypothesis:**
Requiring applicants to send money before or during recruitment is one of the strongest indicators of recruitment fraud.

**Safeguards:**
The implementation attempts to avoid triggering on:

- Explicitly negated payment statements
- Employer-funded costs

**Limitation:**
Rule-based negation handling is incomplete and unusual phrasing may still be missed.

---

## `credential_request_flag`

**Definition:**
Detects requests for sensitive authentication information such as:

- Passwords
- OTPs
- PINs
- CVV/security codes
- Login credentials

**Hypothesis:**
Legitimate recruiters should not require authentication credentials as part of ordinary recruitment.

**Limitation:**
Unseen wording or indirect credential requests may not match the frozen patterns.

---

## `urgency_flag`

**Definition:**
Detects strong time-pressure language such as:

- Urgent
- Immediately
- Within 24 hours
- Within 48 hours

**Hypothesis:**
Artificial urgency can be used to reduce careful verification and push applicants toward fast decisions.

**Limitation:**
Legitimate recruitment can also involve genuine deadlines, so urgency is a weak supporting feature rather than decisive evidence.

---

## `identity_document_flag`

**Definition:**
Detects references to sensitive identity documents such as:

- Passport
- CNIC
- National ID
- Identity card
- Driver licence
- Bank statement
- Utility bill
- Live selfie

**Hypothesis:**
Premature or inappropriate requests for identity documentation may indicate identity-theft-oriented recruitment fraud.

**Limitation:**
Identity verification can be completely legitimate after an offer or during formal onboarding. Therefore this feature is intentionally treated as contextual rather than automatically fraudulent.

---

## `equipment_purchase_flag`

**Definition:**
Detects instructions to buy or order work-related equipment such as:

- Laptop
- Workstation
- Device
- Computer equipment

**Hypothesis:**
Fake equipment-purchase schemes are a known recruitment-scam pattern, particularly when applicants are directed to specific vendors or reimbursed through suspicious payment mechanisms.

**Limitation:**
The current regex representation does not fully understand negation. For example, a statement saying that candidates are **not** required to buy equipment may still contain overlapping suspicious vocabulary.

---

## `money_transfer_flag`

**Definition:**
Detects instructions involving receiving funds in a personal account and forwarding or transferring the money.

**Hypothesis:**
Recruitment offers that ask applicants to move money through personal accounts may be money-mule schemes.

**Limitation:**
Finance-related legitimate jobs may discuss money transfers without requiring the applicant to personally move funds.

Negation handling reduces some false positives but is not equivalent to semantic language understanding.

---

## `paid_training_flag`

**Definition:**
Detects recruitment processes that require candidates to purchase training, certification, enrolment, or related services.

**Hypothesis:**
Mandatory paid training or certificates sold through a recruitment partner can be used as the monetization mechanism in fake-job schemes.

**Limitation:**
Legitimate jobs can include training. The signal is designed around required applicant purchases rather than ordinary employer-provided training, but unusual wording may still cause false positives.

---

## `suspicious_application_link_flag`

**Definition:**
Detects URLs appearing together with requests for sensitive information such as:

- Card details
- CVV/security codes
- Passwords
- OTPs
- PINs
- Login credentials
- Account access

**Hypothesis:**
A recruitment link requesting sensitive financial or authentication information is substantially more suspicious than the mere presence of an application URL.

**Design decision:**
This feature deliberately requires both:

```text
URL + sensitive-information context
```

rather than marking every URL as suspicious.

---

## `selection_bypass_flag`

**Definition:**
Detects language suggesting that normal recruitment stages have been bypassed, such as:

- No interview required
- Without interview
- No further application stage
- Direct onboarding

**Hypothesis:**
Fraudulent recruiters may claim immediate selection to reduce scrutiny and create excitement or urgency.

**Limitation:**
Some legitimate temporary, referral-based, or simplified hiring processes may genuinely have fewer stages.

---

## `cheque_overpayment_flag`

**Definition:**
Detects cheque/check payment scenarios involving:

- Excess payment
- More money than required
- Remainder or unused balance
- Instructions to return or transfer the excess

**Hypothesis:**
Fake-cheque overpayment is a distinct fraud mechanism where the victim receives a fraudulent payment and is instructed to return part of it before the cheque is discovered to be invalid.

**Limitation:**
This feature is intentionally narrow and may miss more subtle variants of the same fraud mechanism.

---

## `lookalike_domain_flag`

**Definition:**
Detects simple digit substitutions inside URL hostnames, such as:

```text
c0mpany.example
netw0rks.example
```

**Hypothesis:**
Impersonation scams may use domains visually similar to legitimate company names.

**Limitation:**
This is a weak heuristic. Legitimate domains can contain digits, and sophisticated impersonation domains may not use obvious substitutions.

The feature is therefore treated as supporting evidence rather than proof of fraud.

---

# 3. Lexical TF-IDF Features

The final model also uses TF-IDF lexical features.

Preprocessing includes:

- Lowercasing
- URL replacement with `URLTOKEN`
- Email replacement with `EMAILTOKEN`
- Whitespace normalization

The final selected TF-IDF configuration is:

```text
ngram_range = (1, 1)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

## Hypothesis

TF-IDF allows the model to learn recurring lexical patterns that are useful beyond the manually engineered features.

## Limitation

Lexical associations are statistical rather than semantic.

For example, ordinary words such as dates, locations, recruitment terminology, or phrases such as `offer`, `online`, or `pay` may receive positive or negative coefficients because of dataset-specific correlations.

These coefficients should not be interpreted as universal scam rules.

---

# Feature Leakage Safeguards

Only information derived from the raw `text` field is supplied to the classifier.

The following metadata columns are **not** used as predictive features:

```text
label
difficulty
scam_category
template_cluster
strong_signals
weak_signals
label_reason
```

These fields are retained only for:

- Dataset auditing
- Leakage-aware splitting
- Evaluation
- Error analysis

This prevents the classifier from directly learning human annotations or scam-category labels that would reveal the target.

---

# Final Feature Configuration

After ablation testing and model selection, the frozen final classifier uses:

```text
TF-IDF lexical features
+
11 behavioral pattern features
```

Structural features remain implemented and documented because they were part of the experimental feature-engineering process, but validation evidence showed that they did not improve the preferred classifier.

No engineered feature, regex rule, model hyperparameter, or threshold was changed after final test-set inspection.

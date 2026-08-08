# ScamSleuth Feature Justifications

Hand-engineered features are designed as supporting signals rather than deterministic rules. A feature being present does not by itself prove that a recruitment message is fraudulent.

| Feature | Hypothesis | Limitation |
|---|---|---|
| `url_count` | Scam communication may direct applicants to external payment, credential, or impersonation pages. | Legitimate recruiters also use application and scheduling links. |
| `shortened_url_present` | URL shorteners can hide the true destination of a recruitment link. | Legitimate organizations may also use shortened links. |
| `exclamation_count` | Excessive punctuation may correlate with pressure-oriented communication. | Legitimate promotional or informal recruitment messages can also use exclamation marks. |
| `caps_ratio` | Excessive capitalization may indicate pressure or attention-seeking language. | Legitimate headings and acronyms may also use capitals. |
| `urgency_count` | Artificial deadlines can pressure applicants before they independently verify an opportunity. | Legitimate hiring processes may have genuine deadlines. |
| `payment_request_count` | Upfront payment requests are a strong recruitment-fraud signal. | Some legitimate third-party expenses may exist, so context matters. |
| `credential_request_count` | Requests for passwords, OTPs, or account credentials are inappropriate in legitimate recruitment. | Ordinary account creation should not be confused with requests for existing credentials. |
| `equipment_purchase_count` | Recruiter-directed applicant-funded equipment purchases are common in fake remote-job scams. | Legitimate workers may occasionally purchase equipment independently. |
| `money_transfer_count` | Requests to receive or forward third-party money through personal accounts strongly indicate money-mule activity. | Legitimate finance roles handle money through company-controlled systems. |
| `identity_document_count` | Excessive or premature requests for identity documents can indicate identity theft. | Legitimate employers may need identification during formal onboarding. |


## Features Excluded After Training-Data Inspection

Some initially proposed structural features were tested but excluded from the current model feature set after inspecting their variance on the training split.

| Feature | Decision | Reason |
|---|---|---|
| `email_count` | Excluded | All 420 training examples had a value of 0, so the feature had zero variance and could not provide predictive information. |
| `exclamation_count` | Excluded | All 420 training examples had a value of 0, so the feature had zero variance. |
| `currency_symbol_count` | Replaced | Currency amounts in the dataset are mainly expressed using codes such as PKR, USD, and AED rather than symbols. It was replaced by `currency_reference_count`. |

## Behavioral Feature Freeze

The behavioral regex features were designed and audited using only the training split.

Final behavioral feature set:

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

The patterns were frozen before evaluating the hybrid model on the validation split.

Validation and test results will not be used to rewrite individual regex rules. This reduces the risk of overfitting feature engineering to held-out examples.
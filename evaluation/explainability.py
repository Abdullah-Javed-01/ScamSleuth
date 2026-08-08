import numpy as np
import pandas as pd


def get_feature_names(model):
    """
    Recover TF-IDF and behavioral feature names
    from the fitted ScamSleuth pipeline.
    """

    feature_union = model.named_steps[
        "features"
    ]

    tfidf = dict(
        feature_union.transformer_list
    )["tfidf"]

    tfidf_names = [
        f"tfidf::{name}"
        for name in tfidf.get_feature_names_out()
    ]

    behavioral = dict(
        feature_union.transformer_list
    )["behavioral"]

    behavioral_names = [
        "behavior::payment_request_flag",
        "behavior::credential_request_flag",
        "behavior::urgency_flag",
        "behavior::identity_document_flag",
        "behavior::equipment_purchase_flag",
        "behavior::money_transfer_flag",
        "behavior::paid_training_flag",
        "behavior::suspicious_application_link_flag",
        "behavior::selection_bypass_flag",
        "behavior::cheque_overpayment_flag",
        "behavior::lookalike_domain_flag",
    ]

    return np.array(
        tfidf_names + behavioral_names
    )


def get_global_feature_importance(
    model,
    top_n=20,
):
    """
    Return the strongest global Scam and Safe
    coefficients learned by Logistic Regression.
    """

    feature_names = get_feature_names(
        model
    )

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = classifier.coef_[0]

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature-name count does not match "
            "classifier coefficient count."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    scam_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(top_n)
    )

    safe_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(top_n)
    )

    return (
        scam_features,
        safe_features,
    )


def explain_prediction(
    model,
    text,
    threshold,
    top_n=10,
):
    """
    Explain one ScamSleuth prediction using
    feature-level Logistic Regression contributions.
    """

    feature_names = get_feature_names(
        model
    )

    feature_union = model.named_steps[
        "features"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    X = feature_union.transform(
        [text]
    )

    coefficients = classifier.coef_[0]

    contribution_values = (
        X.multiply(coefficients)
        .toarray()[0]
    )

    probability = model.predict_proba(
        [text]
    )[0, 1]

    prediction = (
        "Scam"
        if probability >= threshold
        else "Safe"
    )

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contribution_values,
        }
    )

    explanation_df = explanation_df[
        explanation_df["contribution"] != 0
    ]

    scam_contributors = (
        explanation_df[
            explanation_df["contribution"] > 0
        ]
        .sort_values(
            "contribution",
            ascending=False,
        )
        .head(top_n)
    )

    safe_contributors = (
        explanation_df[
            explanation_df["contribution"] < 0
        ]
        .sort_values(
            "contribution",
            ascending=True,
        )
        .head(top_n)
    )

    return {
        "prediction": prediction,
        "scam_probability": probability,
        "threshold": threshold,
        "intercept": classifier.intercept_[0],
        "top_scam_contributors": scam_contributors,
        "top_safe_contributors": safe_contributors,
    }
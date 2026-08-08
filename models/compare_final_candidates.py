import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    RANDOM_SEED,
)

from features.text_preprocessing import preprocess_text
from features.feature_pipeline import BehavioralFeatureTransformer
from evaluation.metrics import calculate_metrics


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def build_features():
    """
    Build the frozen preferred feature configuration:
    TF-IDF + behavioral features.
    """

    return FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    ngram_range=(1, 1),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "behavioral",
                BehavioralFeatureTransformer(),
            ),
        ]
    )


def build_logistic_model():
    """
    Frozen tuned Logistic Regression candidate.
    """

    return Pipeline(
        [
            (
                "features",
                build_features(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_svm_model():
    """
    Frozen tuned Linear SVM candidate.
    """

    return Pipeline(
        [
            (
                "features",
                build_features(),
            ),
            (
                "classifier",
                LinearSVC(
                    C=2.0,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_candidate(
    name,
    model,
    X_development,
    y_development,
    X_test,
    y_test,
):
    """
    Train a frozen candidate on development data
    and evaluate it on the same held-out test set.

    Default classifier decision boundaries are used
    so the candidate comparison remains symmetric:
    Logistic Regression -> probability >= 0.50
    Linear SVM -> decision function >= 0
    """

    model.fit(
        X_development,
        y_development,
    )

    y_pred = model.predict(
        X_test
    )

    if hasattr(
        model,
        "predict_proba",
    ):
        y_score = model.predict_proba(
            X_test
        )[:, 1]

    else:
        y_score = model.decision_function(
            X_test
        )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_score,
    )

    return {
        "model": name,
        **metrics,
    }


def main():

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    X_development = development_df[
        "text"
    ]

    y_development = development_df[
        "label"
    ].map(
        LABEL_MAP
    )

    X_test = test_df[
        "text"
    ]

    y_test = test_df[
        "label"
    ].map(
        LABEL_MAP
    )

    results = []

    results.append(
        evaluate_candidate(
            "Tuned Logistic Regression",
            build_logistic_model(),
            X_development,
            y_development,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_candidate(
            "Tuned Linear SVM",
            build_svm_model(),
            X_development,
            y_development,
            X_test,
            y_test,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "FROZEN MODEL FAMILY COMPARISON — HELD-OUT TEST SET"
    )

    print(
        "=" * 95
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    output_path = (
        "reports/model_comparison_test.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        "\nImportant:"
    )

    print(
        "This post-selection test comparison is "
        "reported for completeness only."
    )

    print(
        "It was not used to change the selected "
        "model, features, hyperparameters, or threshold."
    )


if __name__ == "__main__":
    main()
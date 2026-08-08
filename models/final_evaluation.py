import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    RANDOM_SEED,
    DECISION_THRESHOLD,
)

from features.text_preprocessing import preprocess_text
from features.feature_pipeline import BehavioralFeatureTransformer


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def build_final_model():

    features = FeatureUnion(
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

    return Pipeline(
        [
            (
                "features",
                features,
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


def main():

    # --------------------------------------------------
    # 1. Load frozen splits
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )


    # --------------------------------------------------
    # 2. Combine train + validation
    #
    # All model and threshold decisions are already
    # frozen, so validation can now join training.
    # --------------------------------------------------

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

    print(
        "Final development rows:",
        len(development_df),
    )

    print(
        "Final test rows:",
        len(test_df),
    )

    print(
        "Decision threshold:",
        DECISION_THRESHOLD,
    )


    # --------------------------------------------------
    # 3. Train frozen final model
    # --------------------------------------------------

    model = build_final_model()

    model.fit(
        X_development,
        y_development,
    )


    # --------------------------------------------------
    # 4. Test probabilities
    # --------------------------------------------------

    y_score = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------
    # 5. Apply frozen threshold
    # --------------------------------------------------

    y_pred = (
        y_score >= DECISION_THRESHOLD
    ).astype(int)


    # --------------------------------------------------
    # 6. Final metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_test,
        y_pred,
        beta=2,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score,
    )

    pr_auc = average_precision_score(
        y_test,
        y_score,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )


    # --------------------------------------------------
    # 7. Print FINAL test result
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SCAMSLEUTH TEST PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"accuracy  : {accuracy:.4f}"
    )

    print(
        f"precision : {precision:.4f}"
    )

    print(
        f"recall    : {recall:.4f}"
    )

    print(
        f"f1        : {f1:.4f}"
    )

    print(
        f"f2        : {f2:.4f}"
    )

    print(
        f"roc_auc   : {roc_auc:.4f}"
    )

    print(
        f"pr_auc    : {pr_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )


if __name__ == "__main__":
    main()
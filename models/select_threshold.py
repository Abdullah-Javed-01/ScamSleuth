import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
)
from sklearn.linear_model import (
    LogisticRegression,
)

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    RANDOM_SEED,
)

from features.text_preprocessing import (
    preprocess_text,
)

from features.feature_pipeline import (
    BehavioralFeatureTransformer,
)

from evaluation.threshold_analysis import (
    analyze_thresholds,
)


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def build_tuned_logistic_model():

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


def print_threshold_result(
    title,
    row,
):
    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    print(
        f"Threshold : {row['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {row['accuracy']:.4f}"
    )

    print(
        f"Precision : {row['precision']:.4f}"
    )

    print(
        f"Recall    : {row['recall']:.4f}"
    )

    print(
        f"F1        : {row['f1']:.4f}"
    )

    print(
        f"F2        : {row['f2']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        [
            [
                int(row["tn"]),
                int(row["fp"]),
            ],
            [
                int(row["fn"]),
                int(row["tp"]),
            ],
        ]
    )


def main():

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    X_train = train_df["text"]

    y_train = train_df["label"].map(
        LABEL_MAP
    )

    X_validation = validation_df[
        "text"
    ]

    y_validation = validation_df[
        "label"
    ].map(
        LABEL_MAP
    )


    # ----------------------------------------------
    # Fit tuned model on TRAIN only
    # ----------------------------------------------

    model = build_tuned_logistic_model()

    model.fit(
        X_train,
        y_train,
    )


    # ----------------------------------------------
    # Validation probabilities
    # ----------------------------------------------

    validation_scores = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    # ----------------------------------------------
    # Threshold analysis
    # ----------------------------------------------

    results = analyze_thresholds(
        y_validation,
        validation_scores,
    )


    # Default threshold
    default_row = results.iloc[
        (
            results["threshold"] - 0.50
        ).abs().argmin()
    ]

    # Highest F1
    best_f1_row = results.loc[
        results["f1"].idxmax()
    ]

    # Highest F2
    best_f2_row = results.loc[
        results["f2"].idxmax()
    ]


    print_threshold_result(
        "DEFAULT THRESHOLD",
        default_row,
    )

    print_threshold_result(
        "BEST F1 THRESHOLD",
        best_f1_row,
    )

    print_threshold_result(
        "BEST F2 THRESHOLD",
        best_f2_row,
    )


    # ----------------------------------------------
    # Show top F2 candidates
    # ----------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TOP 10 THRESHOLDS BY F2"
    )

    print(
        "=" * 90
    )

    columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "fp",
        "fn",
        "tp",
        "tn",
    ]

    print(
        results
        .sort_values(
            "f2",
            ascending=False,
        )
        [columns]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
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
from sklearn.svm import (
    LinearSVC,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    confusion_matrix,
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

from evaluation.metrics import (
    calculate_metrics,
)


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def build_feature_union():
    """
    Combine lexical TF-IDF features with the frozen
    behavioral feature set.
    """

    return FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
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


def evaluate_on_validation(
    name,
    estimator,
    validation_text,
    y_validation,
):
    """
    Evaluate a tuned estimator on the untouched
    validation split.
    """

    y_pred = estimator.predict(
        validation_text
    )

    if hasattr(
        estimator,
        "predict_proba",
    ):
        y_score = estimator.predict_proba(
            validation_text
        )[:, 1]

    else:
        y_score = estimator.decision_function(
            validation_text
        )

    metrics = calculate_metrics(
        y_validation,
        y_pred,
        y_score,
    )

    matrix = confusion_matrix(
        y_validation,
        y_pred,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{name} — VALIDATION"
    )

    print(
        "=" * 70
    )

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )

    return {
        "model": name,
        **metrics,
    }


def main():

    # --------------------------------------------------
    # 1. Load TRAIN + VALIDATION only
    # --------------------------------------------------

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

    groups = train_df[
        "template_cluster"
    ]

    X_validation = validation_df[
        "text"
    ]

    y_validation = validation_df[
        "label"
    ].map(
        LABEL_MAP
    )

    print(
        "Training rows:",
        len(train_df),
    )

    print(
        "Training clusters:",
        groups.nunique(),
    )

    print(
        "Validation rows:",
        len(validation_df),
    )


    # --------------------------------------------------
    # 2. Leakage-aware grouped cross-validation
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


    # ==================================================
    # MODEL 1 — LOGISTIC REGRESSION
    # ==================================================

    logistic_pipeline = Pipeline(
        [
            (
                "features",
                build_feature_union(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    logistic_grid = {
        "features__tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],

        "features__tfidf__min_df": [
            1,
            2,
        ],

        "classifier__C": [
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
        ],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TUNING LOGISTIC REGRESSION"
    )

    print(
        "=" * 70
    )

    logistic_search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    print(
        "\nBest Logistic Regression parameters:"
    )

    print(
        logistic_search.best_params_
    )

    print(
        "Best grouped-CV PR-AUC:",
        round(
            logistic_search.best_score_,
            4,
        ),
    )


    # ==================================================
    # MODEL 2 — LINEAR SVM
    # ==================================================

    svm_pipeline = Pipeline(
        [
            (
                "features",
                build_feature_union(),
            ),
            (
                "classifier",
                LinearSVC(
                    random_state=RANDOM_SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )

    svm_grid = {
        "features__tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],

        "features__tfidf__min_df": [
            1,
            2,
        ],

        "classifier__C": [
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.0,
        ],
    }

    svm_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=svm_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TUNING LINEAR SVM"
    )

    print(
        "=" * 70
    )

    svm_search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    print(
        "\nBest Linear SVM parameters:"
    )

    print(
        svm_search.best_params_
    )

    print(
        "Best grouped-CV PR-AUC:",
        round(
            svm_search.best_score_,
            4,
        ),
    )


    # --------------------------------------------------
    # 3. Evaluate tuned models on VALIDATION
    # --------------------------------------------------

    results = []

    logistic_result = evaluate_on_validation(
        "Tuned Logistic Regression",
        logistic_search.best_estimator_,
        X_validation,
        y_validation,
    )

    results.append(
        logistic_result
    )

    svm_result = evaluate_on_validation(
        "Tuned Linear SVM",
        svm_search.best_estimator_,
        X_validation,
        y_validation,
    )

    results.append(
        svm_result
    )


    # --------------------------------------------------
    # 4. Final candidate comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TUNED MODEL COMPARISON"
    )

    print(
        "=" * 90
    )

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
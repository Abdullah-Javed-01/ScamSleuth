import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    RANDOM_SEED,
)

from features.text_preprocessing import preprocess_text
from features.feature_pipeline import (
    build_structural_feature_frame,
    build_behavioral_feature_frame,
)

from evaluation.metrics import calculate_metrics


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def evaluate_model(
    name,
    X_train,
    X_validation,
    y_train,
    y_validation,
):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_validation
    )

    y_score = model.predict_proba(
        X_validation
    )[:, 1]

    metrics = calculate_metrics(
        y_validation,
        y_pred,
        y_score,
    )

    matrix = confusion_matrix(
        y_validation,
        y_pred,
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Feature count:",
        X_train.shape[1],
    )

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "model": name,
        **metrics,
    }


def main():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    y_train = train_df["label"].map(
        LABEL_MAP
    )

    y_validation = validation_df["label"].map(
        LABEL_MAP
    )


    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(
        train_df["text"]
    )

    X_validation_tfidf = vectorizer.transform(
        validation_df["text"]
    )


    # --------------------------------------------------
    # Structural features
    # --------------------------------------------------

    train_struct = (
        build_structural_feature_frame(
            train_df["text"]
        )
    )

    validation_struct = (
        build_structural_feature_frame(
            validation_df["text"]
        )
    )

    scaler = StandardScaler()

    train_struct_scaled = scaler.fit_transform(
        train_struct
    )

    validation_struct_scaled = scaler.transform(
        validation_struct
    )

    train_struct_sparse = csr_matrix(
        train_struct_scaled
    )

    validation_struct_sparse = csr_matrix(
        validation_struct_scaled
    )


    # --------------------------------------------------
    # Behavioral features
    # --------------------------------------------------

    train_behavior = (
        build_behavioral_feature_frame(
            train_df["text"]
        )
    )

    validation_behavior = (
        build_behavioral_feature_frame(
            validation_df["text"]
        )
    )

    train_behavior_sparse = csr_matrix(
        train_behavior.values
    )

    validation_behavior_sparse = csr_matrix(
        validation_behavior.values
    )


    # --------------------------------------------------
    # Build feature combinations
    # --------------------------------------------------

    feature_sets = {
        "A - TF-IDF only": (
            X_train_tfidf,
            X_validation_tfidf,
        ),

        "B - TF-IDF + Structural": (
            hstack(
                [
                    X_train_tfidf,
                    train_struct_sparse,
                ],
                format="csr",
            ),
            hstack(
                [
                    X_validation_tfidf,
                    validation_struct_sparse,
                ],
                format="csr",
            ),
        ),

        "C - TF-IDF + Behavioral": (
            hstack(
                [
                    X_train_tfidf,
                    train_behavior_sparse,
                ],
                format="csr",
            ),
            hstack(
                [
                    X_validation_tfidf,
                    validation_behavior_sparse,
                ],
                format="csr",
            ),
        ),

        "D - TF-IDF + Structural + Behavioral": (
            hstack(
                [
                    X_train_tfidf,
                    train_struct_sparse,
                    train_behavior_sparse,
                ],
                format="csr",
            ),
            hstack(
                [
                    X_validation_tfidf,
                    validation_struct_sparse,
                    validation_behavior_sparse,
                ],
                format="csr",
            ),
        ),
    }


    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    results = []

    for name, (
        X_train,
        X_validation,
    ) in feature_sets.items():

        result = evaluate_model(
            name,
            X_train,
            X_validation,
            y_train,
            y_validation,
        )

        results.append(
            result
        )


    # --------------------------------------------------
    # Comparison table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 90)
    print("FEATURE SET COMPARISON")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
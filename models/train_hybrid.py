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


def main():

    # --------------------------------------------------
    # 1. Load training and validation data
    # --------------------------------------------------

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    print(
        "Train shape:",
        train_df.shape,
    )

    print(
        "Validation shape:",
        validation_df.shape,
    )

    y_train = train_df["label"].map(
        LABEL_MAP
    )

    y_validation = validation_df["label"].map(
        LABEL_MAP
    )


    # --------------------------------------------------
    # 2. TF-IDF lexical features
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
    # 3. Structural features
    # --------------------------------------------------

    X_train_struct = (
        build_structural_feature_frame(
            train_df["text"]
        )
    )

    X_validation_struct = (
        build_structural_feature_frame(
            validation_df["text"]
        )
    )

    structural_feature_names = (
        X_train_struct.columns.tolist()
    )


    # --------------------------------------------------
    # 4. Scale structural features
    #
    # Fit scaler ONLY on training data.
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_struct_scaled = scaler.fit_transform(
        X_train_struct
    )

    X_validation_struct_scaled = scaler.transform(
        X_validation_struct
    )


    # --------------------------------------------------
    # 5. Behavioral features
    # --------------------------------------------------

    X_train_behavior = (
        build_behavioral_feature_frame(
            train_df["text"]
        )
    )

    X_validation_behavior = (
        build_behavioral_feature_frame(
            validation_df["text"]
        )
    )

    behavioral_feature_names = (
        X_train_behavior.columns.tolist()
    )


    # --------------------------------------------------
    # 6. Combine feature families
    # --------------------------------------------------

    X_train_hybrid = hstack(
        [
            X_train_tfidf,
            csr_matrix(
                X_train_struct_scaled
            ),
            csr_matrix(
                X_train_behavior.values
            ),
        ],
        format="csr",
    )

    X_validation_hybrid = hstack(
        [
            X_validation_tfidf,
            csr_matrix(
                X_validation_struct_scaled
            ),
            csr_matrix(
                X_validation_behavior.values
            ),
        ],
        format="csr",
    )


    # --------------------------------------------------
    # 7. Display feature dimensions
    # --------------------------------------------------

    print(
        "\nTF-IDF features:",
        X_train_tfidf.shape[1],
    )

    print(
        "Structural features:",
        len(structural_feature_names),
    )

    print(
        "Behavioral features:",
        len(behavioral_feature_names),
    )

    print(
        "Total hybrid features:",
        X_train_hybrid.shape[1],
    )

    print(
        "\nTraining hybrid matrix:",
        X_train_hybrid.shape,
    )

    print(
        "Validation hybrid matrix:",
        X_validation_hybrid.shape,
    )


    # --------------------------------------------------
    # 8. Train Logistic Regression
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train_hybrid,
        y_train,
    )


    # --------------------------------------------------
    # 9. Validation predictions
    # --------------------------------------------------

    y_pred = model.predict(
        X_validation_hybrid
    )

    y_score = model.predict_proba(
        X_validation_hybrid
    )[:, 1]


    # --------------------------------------------------
    # 10. Metrics
    # --------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        y_pred,
        y_score,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "HYBRID TF-IDF + ENGINEERED FEATURES"
    )

    print(
        "=" * 60
    )

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )


    # --------------------------------------------------
    # 11. Confusion matrix
    # --------------------------------------------------

    conf_matrix = confusion_matrix(
        y_validation,
        y_pred,
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        conf_matrix
    )


if __name__ == "__main__":
    main()
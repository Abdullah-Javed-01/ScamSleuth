import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    RANDOM_SEED,
)

from features.text_preprocessing import preprocess_text
from evaluation.metrics import calculate_metrics


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def main():

    # -------------------------------------------------
    # Load Train and Validation only
    # -------------------------------------------------

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    print("Train shape:", train_df.shape)
    print(
        "Validation shape:",
        validation_df.shape
    )

    # -------------------------------------------------
    # Prepare target labels
    # -------------------------------------------------

    y_train = (
        train_df["label"]
        .map(LABEL_MAP)
    )

    y_validation = (
        validation_df["label"]
        .map(LABEL_MAP)
    )

    # -------------------------------------------------
    # TF-IDF
    # -------------------------------------------------

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_df["text"]
    )

    X_validation = vectorizer.transform(
        validation_df["text"]
    )

    print(
        "TF-IDF vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    print(
        "Training feature matrix:",
        X_train.shape
    )

    print(
        "Validation feature matrix:",
        X_validation.shape
    )

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Validation predictions
    # -------------------------------------------------

    y_pred = model.predict(
        X_validation
    )

    y_score = model.predict_proba(
        X_validation
    )[:, 1]

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        y_pred,
        y_score,
    )

    print("\n" + "=" * 60)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )
    
    conf_matrix = confusion_matrix(
        y_validation,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    
    top_scam_features = (
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=False
        )
        .head(20)
    )

    print("\n" + "=" * 60)
    print("TOP TF-IDF FEATURES ASSOCIATED WITH SCAM")
    print("=" * 60)

    print(
        top_scam_features.to_string(
            index=False
        )
    )
    
    top_safe_features = (
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=True
        )
        .head(20)
    )

    print("\n" + "=" * 60)
    print("TOP TF-IDF FEATURES ASSOCIATED WITH SAFE")
    print("=" * 60)

    print(
        top_safe_features.to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()
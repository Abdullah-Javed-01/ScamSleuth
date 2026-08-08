import json

import joblib
import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    FINAL_MODEL_PATH,
    FINAL_MODEL_METADATA_PATH,
    DECISION_THRESHOLD,
)

from models.final_evaluation import (
    build_final_model,
)


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def main():

    # --------------------------------------------------
    # Load development data
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )


    # --------------------------------------------------
    # Build and fit the frozen final model
    # --------------------------------------------------

    model = build_final_model()

    model.fit(
        development_df["text"],
        development_df["label"].map(
            LABEL_MAP
        ),
    )


    # --------------------------------------------------
    # Save model artifact
    # --------------------------------------------------

    FINAL_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        FINAL_MODEL_PATH,
    )


    # --------------------------------------------------
    # Save human-readable metadata
    # --------------------------------------------------

    metadata = {
        "model": "Logistic Regression",
        "feature_set": [
            "TF-IDF lexical features",
            "11 behavioral features",
        ],
        "training_rows": len(
            development_df
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "classifier_C": 4.0,
        "tfidf_ngram_range": [
            1,
            1,
        ],
        "tfidf_min_df": 2,
        "tfidf_max_df": 0.95,
        "tfidf_sublinear_tf": True,
        "labels": {
            "0": "Safe",
            "1": "Scam",
        },
        "test_data_used_for_training": False,
    }

    with open(
        FINAL_MODEL_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


    print(
        "Final ScamSleuth model trained successfully."
    )

    print(
        "Development rows:",
        len(development_df),
    )

    print(
        "Decision threshold:",
        DECISION_THRESHOLD,
    )

    print(
        "\nSaved model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        "\nSaved metadata:"
    )

    print(
        FINAL_MODEL_METADATA_PATH
    )


if __name__ == "__main__":
    main()
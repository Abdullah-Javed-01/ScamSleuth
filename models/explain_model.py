import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    DECISION_THRESHOLD,
)

from models.final_evaluation import (
    build_final_model,
)

from evaluation.explainability import (
    get_global_feature_importance,
    explain_prediction,
)


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def main():

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

    model = build_final_model()

    model.fit(
        development_df["text"],
        development_df["label"].map(
            LABEL_MAP
        ),
    )


    # --------------------------------------------------
    # Global feature importance
    # --------------------------------------------------

    scam_features, safe_features = (
        get_global_feature_importance(
            model,
            top_n=20,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TOP GLOBAL FEATURES PUSHING TOWARD SCAM"
    )

    print(
        "=" * 80
    )

    print(
        scam_features.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "TOP GLOBAL FEATURES PUSHING TOWARD SAFE"
    )

    print(
        "=" * 80
    )

    print(
        safe_features.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # Example individual prediction
    # --------------------------------------------------

    example = """
    Congratulations. You have been selected for
    the remote role. Please pay the processing fee
    before onboarding so that your appointment can
    be activated.
    """

    explanation = explain_prediction(
        model,
        example,
        DECISION_THRESHOLD,
        top_n=10,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "INDIVIDUAL PREDICTION EXPLANATION"
    )

    print(
        "=" * 80
    )

    print(
        "Prediction:",
        explanation["prediction"],
    )

    print(
        "Scam probability:",
        round(
            explanation[
                "scam_probability"
            ],
            4,
        ),
    )

    print(
        "Decision threshold:",
        explanation["threshold"],
    )


    print(
        "\nTop Scam contributors:"
    )

    print(
        explanation[
            "top_scam_contributors"
        ].to_string(
            index=False
        )
    )


    print(
        "\nTop Safe contributors:"
    )

    print(
        explanation[
            "top_safe_contributors"
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
import argparse

import joblib

from config import (
    FINAL_MODEL_PATH,
    DECISION_THRESHOLD,
)

from evaluation.explainability import (
    explain_prediction,
)


def load_model():
    """
    Load the frozen ScamSleuth model.
    """

    if not FINAL_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Final model artifact not found. "
            "Run `python -m models.train_final` first."
        )

    return joblib.load(
        FINAL_MODEL_PATH
    )


def predict_text(
    text: str,
    include_explanation: bool = True,
):
    """
    Predict whether recruitment text is Safe or Scam.
    """

    model = load_model()

    scam_probability = model.predict_proba(
        [text]
    )[0, 1]

    prediction = (
        "Scam"
        if scam_probability
        >= DECISION_THRESHOLD
        else "Safe"
    )

    result = {
        "prediction": prediction,
        "scam_probability": float(
            scam_probability
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
    }

    if include_explanation:

        explanation = explain_prediction(
            model,
            text,
            DECISION_THRESHOLD,
            top_n=5,
        )

        result[
            "top_scam_contributors"
        ] = explanation[
            "top_scam_contributors"
        ]

        result[
            "top_safe_contributors"
        ] = explanation[
            "top_safe_contributors"
        ]

    return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify recruitment text "
            "as Safe or Scam."
        )
    )

    parser.add_argument(
        "--text",
        required=True,
        help=(
            "Recruitment message, email, "
            "or job-posting text."
        ),
    )

    args = parser.parse_args()

    result = predict_text(
        args.text
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SCAMSLEUTH PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        "Prediction:",
        result["prediction"],
    )

    print(
        "Scam probability:",
        f"{result['scam_probability']:.4f}",
    )

    print(
        "Decision threshold:",
        result["decision_threshold"],
    )


    print(
        "\nTop signals toward Scam:"
    )

    scam_contributors = result[
        "top_scam_contributors"
    ]

    if scam_contributors.empty:

        print(
            "No positive feature contributions."
        )

    else:

        print(
            scam_contributors.to_string(
                index=False
            )
        )


    print(
        "\nTop signals toward Safe:"
    )

    safe_contributors = result[
        "top_safe_contributors"
    ]

    if safe_contributors.empty:

        print(
            "No negative feature contributions."
        )

    else:

        print(
            safe_contributors.to_string(
                index=False
            )
        )


    print(
        "\nNote: This prediction is a screening aid "
        "and should not be treated as proof that a "
        "recruitment opportunity is fraudulent."
    )


if __name__ == "__main__":
    main()
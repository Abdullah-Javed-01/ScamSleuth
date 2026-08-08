from pathlib import Path

import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    DECISION_THRESHOLD,
)

from models.final_evaluation import (
    build_final_model,
)


LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}

REVERSE_LABEL_MAP = {
    0: "Safe",
    1: "Scam",
}

TEST_ERROR_PATH = Path(
    "reports/test_error_analysis.csv"
)

STRESS_RESULTS_PATH = Path(
    "reports/stress_test_predictions.csv"
)

STRESS_ERROR_PATH = Path(
    "reports/stress_test_error_analysis.csv"
)


def build_frozen_model():

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

    return model


def analyze_test_errors(model):

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    probabilities = model.predict_proba(
        test_df["text"]
    )[:, 1]

    predictions = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    analysis_df = test_df.copy()

    analysis_df[
        "scam_probability"
    ] = probabilities.round(4)

    analysis_df[
        "predicted_label"
    ] = [
        REVERSE_LABEL_MAP[value]
        for value in predictions
    ]

    analysis_df[
        "error_type"
    ] = "Correct"

    analysis_df.loc[
        (
            (analysis_df["label"] == "Safe")
            & (
                analysis_df["predicted_label"]
                == "Scam"
            )
        ),
        "error_type",
    ] = "False Positive"

    analysis_df.loc[
        (
            (analysis_df["label"] == "Scam")
            & (
                analysis_df["predicted_label"]
                == "Safe"
            )
        ),
        "error_type",
    ] = "False Negative"

    errors = analysis_df[
        analysis_df["error_type"]
        != "Correct"
    ].copy()

    TEST_ERROR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        TEST_ERROR_PATH,
        index=False,
        encoding="utf-8",
    )

    return errors


def analyze_stress_errors():

    stress_df = pd.read_csv(
        STRESS_RESULTS_PATH
    )

    errors = stress_df[
        stress_df["correct"] == False
    ].copy()

    errors["error_type"] = errors.apply(
        lambda row:
            "False Positive"
            if (
                row["label"] == "Safe"
                and row["predicted_label"] == "Scam"
            )
            else "False Negative",
        axis=1,
    )

    errors.to_csv(
        STRESS_ERROR_PATH,
        index=False,
        encoding="utf-8",
    )

    return errors


def print_test_errors(errors):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL TEST-SET ERRORS"
    )

    print(
        "=" * 100
    )

    print(
        "Total errors:",
        len(errors),
    )

    print(
        "\nError types:"
    )

    print(
        errors[
            "error_type"
        ].value_counts()
    )

    columns = [
        "id",
        "label",
        "predicted_label",
        "scam_probability",
        "text_type",
        "difficulty",
        "scam_category",
        "error_type",
    ]

    print(
        "\n"
        + errors[
            columns
        ].to_string(
            index=False
        )
    )


def print_test_error_text(errors):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TEST ERROR DETAILS"
    )

    print(
        "=" * 100
    )

    for _, row in errors.iterrows():

        print(
            "\nID:",
            row["id"],
        )

        print(
            "Actual:",
            row["label"],
        )

        print(
            "Predicted:",
            row["predicted_label"],
        )

        print(
            "Probability:",
            row["scam_probability"],
        )

        print(
            "Text type:",
            row["text_type"],
        )

        print(
            "Difficulty:",
            row["difficulty"],
        )

        print(
            "Scam category:",
            row["scam_category"],
        )

        print(
            "\nTEXT:"
        )

        print(
            row["text"]
        )

        print(
            "\nLABEL REASON:"
        )

        print(
            row["label_reason"]
        )

        print(
            "-" * 100
        )


def print_stress_errors(errors):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ADVERSARIAL STRESS-TEST ERRORS"
    )

    print(
        "=" * 100
    )

    print(
        "Total errors:",
        len(errors),
    )

    columns = [
        "id",
        "scenario",
        "label",
        "predicted_label",
        "scam_probability",
        "error_type",
    ]

    print(
        errors[
            columns
        ].to_string(
            index=False
        )
    )


def main():

    model = build_frozen_model()

    test_errors = analyze_test_errors(
        model
    )

    stress_errors = analyze_stress_errors()

    print_test_errors(
        test_errors
    )

    print_test_error_text(
        test_errors
    )

    print_stress_errors(
        stress_errors
    )

    print(
        "\nSaved:"
    )

    print(
        TEST_ERROR_PATH
    )

    print(
        STRESS_ERROR_PATH
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    DECISION_THRESHOLD,
)

from models.final_evaluation import (
    build_final_model,
)


STRESS_TEST_PATH = Path(
    "data/stress_test/adversarial_stress_test.csv"
)

RESULTS_PATH = Path(
    "reports/stress_test_predictions.csv"
)

LABEL_MAP = {
    "Safe": 0,
    "Scam": 1,
}


def main():

    # --------------------------------------------------
    # Recreate the frozen final model
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

    model = build_final_model()

    model.fit(
        development_df["text"],
        development_df["label"].map(
            LABEL_MAP
        ),
    )


    # --------------------------------------------------
    # Load adversarial examples
    # --------------------------------------------------

    stress_df = pd.read_csv(
        STRESS_TEST_PATH
    )

    y_true = stress_df["label"].map(
        LABEL_MAP
    )


    # --------------------------------------------------
    # Frozen model predictions
    # --------------------------------------------------

    scam_probability = model.predict_proba(
        stress_df["text"]
    )[:, 1]

    y_pred = (
        scam_probability
        >= DECISION_THRESHOLD
    ).astype(int)

    predicted_labels = pd.Series(
        y_pred
    ).map(
        {
            0: "Safe",
            1: "Scam",
        }
    )


    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = stress_df.copy()

    results_df[
        "scam_probability"
    ] = scam_probability.round(4)

    results_df[
        "predicted_label"
    ] = predicted_labels

    results_df[
        "correct"
    ] = (
        results_df["label"]
        == results_df["predicted_label"]
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )


    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "ADVERSARIAL STRESS TEST"
    )

    print(
        "=" * 80
    )

    print(
        "Examples:",
        len(stress_df),
    )

    print(
        "Decision threshold:",
        DECISION_THRESHOLD,
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )


    # --------------------------------------------------
    # Per-example predictions
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INDIVIDUAL STRESS-TEST RESULTS"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "id",
        "scenario",
        "label",
        "predicted_label",
        "scam_probability",
        "correct",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # Explicitly print failures
    # --------------------------------------------------

    failures = results_df[
        ~results_df["correct"]
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STRESS-TEST FAILURES"
    )

    print(
        "=" * 100
    )

    if failures.empty:

        print(
            "No misclassifications."
        )

    else:

        for _, row in failures.iterrows():

            print(
                "\nID:",
                row["id"],
            )

            print(
                "Scenario:",
                row["scenario"],
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
                "Scam probability:",
                row["scam_probability"],
            )

            print(
                "Text:"
            )

            print(
                row["text"]
            )

            print(
                "-" * 100
            )


    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nResults saved to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
    
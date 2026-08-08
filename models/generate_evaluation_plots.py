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

from evaluation.plots import (
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
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

    test_df = pd.read_csv(
        TEST_DATA_PATH
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

    y_test = test_df["label"].map(
        LABEL_MAP
    )

    y_score = model.predict_proba(
        test_df["text"]
    )[:, 1]

    y_pred = (
        y_score >= DECISION_THRESHOLD
    ).astype(int)

    save_confusion_matrix(
        y_test,
        y_pred,
        "reports/figures/confusion_matrix.png",
    )

    save_roc_curve(
        y_test,
        y_score,
        "reports/figures/roc_curve.png",
    )

    save_pr_curve(
        y_test,
        y_score,
        "reports/figures/precision_recall_curve.png",
    )

    print(
        "Saved evaluation figures:"
    )

    print(
        "reports/figures/confusion_matrix.png"
    )

    print(
        "reports/figures/roc_curve.png"
    )

    print(
        "reports/figures/precision_recall_curve.png"
    )


if __name__ == "__main__":
    main()
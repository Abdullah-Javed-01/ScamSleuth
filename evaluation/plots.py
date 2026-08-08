from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "Safe",
            "Scam",
        ],
    )

    plt.title(
        "ScamSleuth Final Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def save_roc_curve(
    y_true,
    y_score,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
    )

    plt.title(
        "ScamSleuth Final Test ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def save_pr_curve(
    y_true,
    y_score,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
    )

    plt.title(
        "ScamSleuth Final Test Precision-Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()
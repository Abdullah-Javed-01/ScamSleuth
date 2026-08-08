from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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


def save_feature_importance(
    scam_features,
    safe_features,
    output_path,
    top_n=10,
):
    """
    Save a signed Logistic Regression coefficient chart.

    Positive coefficients push predictions toward Scam.
    Negative coefficients push predictions toward Safe.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = pd.concat(
        [
            safe_features.head(top_n),
            scam_features.head(top_n),
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "coefficient",
        ascending=True,
    )

    plt.figure(
        figsize=(10, 9)
    )

    plt.barh(
        combined["feature"],
        combined["coefficient"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Logistic Regression Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "ScamSleuth Global Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()
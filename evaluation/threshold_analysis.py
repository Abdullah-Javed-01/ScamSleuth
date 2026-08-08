import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)


def analyze_thresholds(
    y_true,
    y_score,
    thresholds=None,
):
    """
    Evaluate classification performance across
    multiple probability thresholds.
    """

    if thresholds is None:
        thresholds = np.arange(
            0.05,
            0.96,
            0.01,
        )

    rows = []

    for threshold in thresholds:

        y_pred = (
            y_score >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f2": fbeta_score(
                    y_true,
                    y_pred,
                    beta=2,
                    zero_division=0,
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    return pd.DataFrame(rows)
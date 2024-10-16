"""
This is code is used to calculate the metrics for the given model predictions.

Author: Aravindh P
"""

from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(y_true, y_pred) -> dict:
    """
    This function is used to calculate the classification model metrics.

    Args:
        y_true: This is the actual values.
        y_pred: This is the predicted values.

    Returns:
        It finally calculates all the metrics and returns in a dictionary.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    metrics = {
        "accuracy": round(100 * accuracy_score(y_true, y_pred), 2),
        "precision": round(precision_score(y_true, y_pred), 2),
        "recall": round(recall_score(y_true, y_pred), 2),
        "f1_score": round(f1_score(y_true, y_pred), 2),
        "auc": round(auc(fpr, tpr), 2),
    }

    return metrics

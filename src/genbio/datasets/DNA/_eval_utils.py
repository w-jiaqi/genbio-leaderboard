import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def evaluate_binary(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate binary classification predictions.

    Primary metric is MCC (Matthews Correlation Coefficient).

    Args:
        preds: DataFrame with integer predicted labels in a 'labels' column.
        targets: DataFrame with integer true labels in a 'labels' column.

    Returns:
        dict with keys: primary_metric, mcc, accuracy, f1, precision, recall.
    """
    y_pred = preds["labels"].values
    y_true = targets["labels"].values
    assert len(y_pred) == len(y_true), (
        f"Predictions and targets must have the same length. "
        f"Got {len(y_pred)} and {len(y_true)}"
    )

    return {
        "primary_metric": "mcc",
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
    }


def evaluate_multiclass(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate multiclass classification predictions.

    Primary metric is MCC (Matthews Correlation Coefficient).

    Args:
        preds: DataFrame with integer predicted labels in a 'labels' column.
        targets: DataFrame with integer true labels in a 'labels' column.

    Returns:
        dict with keys: primary_metric, mcc, accuracy, f1_macro, f1_weighted,
        precision_macro, recall_macro.
    """
    y_pred = preds["labels"].values
    y_true = targets["labels"].values
    assert len(y_pred) == len(y_true), (
        f"Predictions and targets must have the same length. "
        f"Got {len(y_pred)} and {len(y_true)}"
    )

    return {
        "primary_metric": "mcc",
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

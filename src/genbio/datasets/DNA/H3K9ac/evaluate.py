import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def evaluate(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate binary classification predictions.

    Note:
        Primary metric is Matthews Correlation Coefficient (MCC).

    Args:
        preds (pd.DataFrame): DataFrame with a 'labels' column containing
            predicted integer labels (0 or 1).
        targets (pd.DataFrame): DataFrame with a 'labels' column containing
            true integer labels (0 or 1).

    Returns:
        dict[str, float]: Evaluation metrics including:
            - primary_metric: 'mcc'
            - mcc: Matthews Correlation Coefficient
            - accuracy: Classification accuracy
            - f1: F1 score
            - precision: Precision
            - recall: Recall
    """
    y_pred = preds['labels'].values
    y_true = targets['labels'].values
    assert len(y_pred) == len(y_true), (
        f"Predictions and targets must have the same length, "
        f"got {len(y_pred)} and {len(y_true)}"
    )

    return {
        'primary_metric': 'mcc',
        'mcc': matthews_corrcoef(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

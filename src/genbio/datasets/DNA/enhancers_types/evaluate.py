import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def evaluate(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate multiclass classification predictions (3 classes).

    Note:
        Primary metric is Matthews Correlation Coefficient (MCC).

    Args:
        preds (pd.DataFrame): DataFrame with a 'labels' column containing
            predicted integer labels (0 to 2).
        targets (pd.DataFrame): DataFrame with a 'labels' column containing
            true integer labels (0 to 2).

    Returns:
        dict[str, float]: Evaluation metrics including:
            - primary_metric: 'mcc'
            - mcc: Matthews Correlation Coefficient
            - accuracy: Classification accuracy
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
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
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

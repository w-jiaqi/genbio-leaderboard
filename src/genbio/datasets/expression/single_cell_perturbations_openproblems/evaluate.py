import numpy as np
import pandas as pd


_METADATA_COLS = {
    "id",
    "cell_type",
    "sm_name",
    "sm_lincs_id",
    "SMILES",
    "control",
    "dose_uM",
    "timepoint_hr",
    "sm_cell_type",
    "split",
}


def _gene_columns(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if col not in _METADATA_COLS])


def _mean_rowwise_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    squared_diff = (y_true - y_pred) ** 2
    mse_per_row = np.mean(squared_diff, axis=1)
    rmse_per_row = np.sqrt(mse_per_row)
    return float(np.mean(rmse_per_row))


def evaluate(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate single-cell perturbation predictions using MRRMSE.

    Note:
        Leaderboard rankings maximize `primary_metric`, so `primary_metric`
        is set to `neg_mrrmse` (higher is better) while `mrrmse` is also
        reported directly (lower is better).

    Args:
        preds (pd.DataFrame): Predictions with `id` and all required gene columns.
        targets (pd.DataFrame): Ground-truth targets with `id` and gene columns.

    Returns:
        dict[str, float]:
            - primary_metric: 'neg_mrrmse'
            - neg_mrrmse: Negative MRRMSE (higher is better)
            - mrrmse: Mean rowwise RMSE (lower is better)
    """
    if "id" not in preds.columns:
        raise ValueError("Predictions must contain an 'id' column")
    if "id" not in targets.columns:
        raise ValueError("Targets must contain an 'id' column")

    expected_gene_cols = _gene_columns(targets)
    pred_gene_cols = _gene_columns(preds)

    missing = sorted(set(expected_gene_cols) - set(pred_gene_cols))
    if missing:
        raise ValueError(f"Predictions are missing {len(missing)} gene columns. First 10: {missing[:10]}")

    extra = sorted(set(pred_gene_cols) - set(expected_gene_cols))
    if extra:
        raise ValueError(f"Predictions contain {len(extra)} unexpected gene columns. First 10: {extra[:10]}")

    preds_sorted = preds.sort_values("id").reset_index(drop=True)
    targets_sorted = targets.sort_values("id").reset_index(drop=True)

    if len(preds_sorted) != len(targets_sorted):
        raise ValueError(f"Prediction length {len(preds_sorted)} does not match target length {len(targets_sorted)}")

    if not (preds_sorted["id"].to_numpy() == targets_sorted["id"].to_numpy()).all():
        raise ValueError("Prediction IDs do not match target IDs")

    y_pred = preds_sorted[expected_gene_cols].to_numpy()
    y_true = targets_sorted[expected_gene_cols].to_numpy()

    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain non-finite values")

    y_pred = np.clip(y_pred, -4, 4)
    mrrmse = _mean_rowwise_rmse(y_true, y_pred)

    return {
        "primary_metric": "neg_mrrmse",
        "neg_mrrmse": -mrrmse,
        "mrrmse": mrrmse,
    }

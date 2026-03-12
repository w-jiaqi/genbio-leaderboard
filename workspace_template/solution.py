"""Agent solution — edit this file and commit after each change.

Must define:
    build_and_predict(train_data, test_data) -> predictions

The data objects are whatever the dataset's load() returns:
  - Segerstolpe: AnnData (train.X is sparse, train.obs['cell_type_label'] is labels)
  - RNA translation: DataFrame (columns: 'sequence', 'labels', 'fold_id')
  - DNA tasks: DataFrame (columns: 'sequence', 'name', 'labels')

Predictions must match the format expected by evaluate():
  - Segerstolpe: AnnData with obs['cell_type_label'] as int64 predicted labels
  - RNA translation: DataFrame with 'labels' column as float predictions
  - DNA tasks: DataFrame with 'labels' column as int predicted labels
"""

import numpy as np


def build_and_predict(train_data, test_data):
    """Majority-class baseline (placeholder — replace with your approach)."""
    # Detect data type
    try:
        # AnnData (Segerstolpe)
        y_train = train_data.obs["cell_type_label"].to_numpy(dtype=np.int64)
        from collections import Counter
        majority = Counter(y_train.tolist()).most_common(1)[0][0]
        preds = test_data.copy()
        preds.obs["cell_type_label"] = np.full(len(test_data), majority, dtype=np.int64)
        return preds
    except AttributeError:
        pass

    # DataFrame with integer labels (DNA classification tasks)
    if train_data["labels"].dtype in (np.int64, np.int32, int):
        from collections import Counter
        majority = Counter(train_data["labels"].tolist()).most_common(1)[0][0]
        preds = test_data.copy()
        preds["labels"] = majority
        return preds

    # DataFrame with float labels (RNA regression tasks)
    mean_label = train_data["labels"].mean()
    preds = test_data.copy()
    preds["labels"] = mean_label
    return preds

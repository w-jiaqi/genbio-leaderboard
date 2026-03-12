import pandas as pd
from datasets import load_dataset


def load(fold_id: str) -> dict[str, pd.DataFrame]:
    """Load the enhancers dataset from the Nucleotide Transformer downstream tasks.

    This dataset has a single fixed train/test split (chromosome held-out).

    Args:
        fold_id (str): Must be "0" (single fixed split).

    Returns:
        dict[str, pd.DataFrame]: Dictionary with keys 'train' and 'test',
            each containing DataFrames with columns 'sequence', 'name', and 'labels'.
    """
    if fold_id != "0":
        raise ValueError(
            f"This dataset only supports fold '0' (fixed train/test split), got '{fold_id}'"
        )

    train_ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        data_dir="enhancers",
        split="train",
    )
    test_ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        data_dir="enhancers",
        split="test",
    )

    train_df = train_ds.to_pandas().rename(columns={"label": "labels"})
    test_df = test_ds.to_pandas().rename(columns={"label": "labels"})

    return {
        "train": train_df,
        "test": test_df,
    }

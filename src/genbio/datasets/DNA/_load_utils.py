import pandas as pd
from datasets import load_dataset


_DS_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def load_task(task_name: str, fold_id: str) -> dict[str, pd.DataFrame]:
    """Load a nucleotide transformer downstream task (revised) from HuggingFace.

    The revised dataset stores all 18 tasks in a single dataset with a 'task'
    column. We load the full split then filter by task name.

    Args:
        task_name: Value of the 'task' column (e.g. "H2AFZ", "splice_sites_all").
        fold_id: Must be "0" -- these datasets have a single fixed train/test split.

    Returns:
        dict with keys 'train' and 'test', each a DataFrame with columns
        'sequence', 'name', 'labels', and 'task'.
    """
    if fold_id != "0":
        raise ValueError(
            f"nucleotide_transformer tasks have a single fixed split; "
            f"only fold '0' is supported, got '{fold_id}'"
        )

    train_df = (
        load_dataset(_DS_NAME, split="train")
        .filter(lambda x: x["task"] == task_name)
        .to_pandas()
        .rename(columns={"label": "labels"})
    )
    test_df = (
        load_dataset(_DS_NAME, split="test")
        .filter(lambda x: x["task"] == task_name)
        .to_pandas()
        .rename(columns={"label": "labels"})
    )

    return {"train": train_df, "test": test_df}

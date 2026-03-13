import pandas as pd
from genbio.datasets.DNA._load_utils import load_task


def load(fold_id: str) -> dict[str, pd.DataFrame]:
    """Load the splice site classification (3-class: donor, acceptor, non-splice) dataset (revised).

    Args:
        fold_id: Must be "0" (single fixed train/test split).

    Returns:
        dict with keys 'train' and 'test', each a DataFrame with columns
        'sequence' (600bp DNA), 'name', and 'labels' (0, 1, or 2).
    """
    return load_task("splice_sites_all", fold_id)

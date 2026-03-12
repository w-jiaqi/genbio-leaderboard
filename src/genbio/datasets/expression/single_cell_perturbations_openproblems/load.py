from pathlib import Path
import urllib.request

import pandas as pd
import scanpy as sc


CACHE_DIR = Path.home() / ".cache" / "genbio_leaderboard" / "openproblems_single_cell_perturbations"
DATASET_BASE_URLS = [
    "https://openproblems-data.s3.amazonaws.com/resources/task_perturbation_prediction/datasets/neurips-2023-data",
    "https://openproblems-data.s3.us-east-1.amazonaws.com/resources/task_perturbation_prediction/datasets/neurips-2023-data",
]


def _download_with_fallback(filename: str, destination: Path) -> None:
    """Download a file from OpenProblems endpoints with URL fallback."""
    last_error = None
    for base_url in DATASET_BASE_URLS:
        url = f"{base_url}/{filename}"
        try:
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, destination)
            print(f"Cached {filename} at {destination}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Failed to download {filename} from all configured URLs") from last_error


def _h5ad_to_dataframe(h5ad_path: Path) -> pd.DataFrame:
    """Convert an OpenProblems h5ad differential-expression matrix to a DataFrame."""
    adata = sc.read_h5ad(h5ad_path)

    if "clipped_sign_log10_pval" not in adata.layers:
        available_layers = list(adata.layers.keys())
        raise ValueError(
            "Required layer 'clipped_sign_log10_pval' not found in "
            f"{h5ad_path}. Available layers: {available_layers}"
        )

    data_matrix = adata.layers["clipped_sign_log10_pval"]
    if hasattr(data_matrix, "toarray"):
        data_matrix = data_matrix.toarray()

    df = pd.DataFrame(data_matrix, index=adata.obs.index, columns=adata.var.index)

    for col in adata.obs.columns:
        df[col] = adata.obs[col].values

    return df


def load(fold_id: str) -> dict[str, pd.DataFrame]:
    """Load OpenProblems single-cell perturbation data.

    The task predicts differential expression profiles for unseen
    (compound, cell_type) pairs in PBMC single-cell data.

    Args:
        fold_id (str): Fold identifier. Must be "0" (single fixed split).

    Returns:
        dict[str, pd.DataFrame]:
            - train: Differential expression training data with gene and metadata columns.
            - test: Differential expression test data with `id`, metadata, and gene columns.
    """
    if fold_id != "0":
        raise ValueError(f"OpenProblems single-cell perturbations only supports fold '0', got '{fold_id}'")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    de_train_path = CACHE_DIR / "de_train.h5ad"
    de_test_path = CACHE_DIR / "de_test.h5ad"
    id_map_path = CACHE_DIR / "id_map.csv"

    if not de_train_path.exists():
        _download_with_fallback("de_train.h5ad", de_train_path)
    if not de_test_path.exists():
        _download_with_fallback("de_test.h5ad", de_test_path)
    if not id_map_path.exists():
        _download_with_fallback("id_map.csv", id_map_path)

    train_df = _h5ad_to_dataframe(de_train_path).reset_index(drop=True)
    test_df = _h5ad_to_dataframe(de_test_path).reset_index(drop=True)
    id_map = pd.read_csv(id_map_path)

    if "id" not in id_map.columns:
        id_map = id_map.reset_index().rename(columns={"index": "id"})

    if len(id_map) != len(test_df):
        raise ValueError(
            f"id_map row count ({len(id_map)}) does not match test rows ({len(test_df)})"
        )

    test_df.insert(0, "id", id_map["id"].to_numpy())

    if "cell_type" in id_map.columns and "cell_type" in test_df.columns:
        test_df["cell_type"] = id_map["cell_type"].to_numpy()
    if "sm_name" in id_map.columns and "sm_name" in test_df.columns:
        test_df["sm_name"] = id_map["sm_name"].to_numpy()

    return {
        "train": train_df,
        "test": test_df,
    }

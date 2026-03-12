# OpenProblems Single-Cell Perturbations (PBMC)

Input: differential-expression targets derived from single-cell RNA-seq perturbation data.
Output: predicted per-gene `clipped_sign_log10_pval` for unseen `(compound, cell_type)` pairs.

This task mirrors Bioml's `manual/open-problems-single-cell-perturbations` task.

## Dataset

- Source files:
  - `de_train.h5ad`
  - `de_test.h5ad`
  - `id_map.csv`
- Source bucket:
  - `s3://openproblems-data/resources/task_perturbation_prediction/datasets/neurips-2023-data/`
- Biological context:
  - PBMC perturbation experiment
  - 144 compounds
  - 24-hour treatment

## Folding

Only one split is supported: fold `0`.

## Data Interface

`load("0")` returns:

- `train` (`pd.DataFrame`): training rows with metadata and gene columns.
- `test` (`pd.DataFrame`): test rows with `id`, metadata, and gene columns.

## Evaluation

Metric follows Bioml: **Mean Rowwise RMSE (MRRMSE)** with prediction clipping to `[-4, 4]`.

Because GenBio leaderboard sorting is max-first, the primary metric is:

- `neg_mrrmse = -mrrmse` (higher is better)

The raw `mrrmse` is also reported.

## Example

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(
    name="expression/single-cell-perturbations-openproblems",
    fold="0",
    user="your_name",
)

train_df, test_df = task.setup()

# Dummy baseline: zeros for all genes
pred_df = test_df.copy()
gene_cols = [c for c in pred_df.columns if c not in {
    "id", "cell_type", "sm_name", "sm_lincs_id", "SMILES", "control", "dose_uM", "timepoint_hr", "sm_cell_type", "split"
}]
pred_df[gene_cols] = 0.0

task.evaluate(pred_df, test_df)
```

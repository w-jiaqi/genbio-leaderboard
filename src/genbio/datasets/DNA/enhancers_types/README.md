# Enhancer type classification (tissue-specific, tissue-invariant, or non-enhancer)

Name: `DNA/enhancers-types`
Category: Enhancer
Task Type: Multiclass classification (3 classes)
Input: 200bp DNA sequence
Target: Integer label (0-2)
Primary Metric: Matthews Correlation Coefficient (MCC)

## Dataset Description

Enhancer type classification (tissue-specific, tissue-invariant, or non-enhancer) from the Nucleotide Transformer downstream tasks benchmark.
Sequences are 200bp DNA sequences. The task is multiclass classification (3 classes).

**Source:** InstaDeepAI/nucleotide_transformer_downstream_tasks (HuggingFace), config `enhancers_types`

## Data Format

- **Input:** DataFrame with columns:
  - `sequence`: DNA sequence string (200bp)
  - `name`: Sequence identifier
  - `labels`: Integer classification label (0-2)

- **Output:** Predicted labels in `preds['labels']`

## Folds

| Fold ID | Description |
|---------|-------------|
| 0       | Fixed chromosome held-out train/test split |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/enhancers-types', fold='0', user='your_name')
train_df, test_df = task.setup()

# Build your model, make predictions (dummy: random)
import numpy as np
train_pred_df = train_df.copy()
train_pred_df['labels'] = np.random.randint(0, 3, size=len(train_df))
test_pred_df = test_df.copy()
test_pred_df['labels'] = np.random.randint(0, 3, size=len(test_df))

# Compute intermediate train metrics
task.evaluate(train_pred_df, train_df)

# Make a submission
task.submit(test_pred_df, name='baseline', description='Baseline submission')
```

## Citation

```
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

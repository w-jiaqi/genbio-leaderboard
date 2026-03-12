# H4 core histone modification mark

Name: `DNA/H4`
Category: Histone Mark
Task Type: Binary classification
Input: 500bp DNA sequence
Target: Integer label (0-1)
Primary Metric: Matthews Correlation Coefficient (MCC)

## Dataset Description

H4 core histone modification mark from the Nucleotide Transformer downstream tasks benchmark.
Sequences are 500bp DNA sequences. The task is binary classification.

**Source:** InstaDeepAI/nucleotide_transformer_downstream_tasks (HuggingFace), config `H4`

## Data Format

- **Input:** DataFrame with columns:
  - `sequence`: DNA sequence string (500bp)
  - `name`: Sequence identifier
  - `labels`: Integer classification label (0-1)

- **Output:** Predicted labels in `preds['labels']`

## Folds

| Fold ID | Description |
|---------|-------------|
| 0       | Fixed chromosome held-out train/test split |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/H4', fold='0', user='your_name')
train_df, test_df = task.setup()

# Build your model, make predictions (dummy: majority class)
import numpy as np
from collections import Counter
majority_label = Counter(train_df['labels'].values).most_common(1)[0][0]
train_pred_df = train_df.copy()
train_pred_df['labels'] = majority_label
test_pred_df = test_df.copy()
test_pred_df['labels'] = majority_label

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

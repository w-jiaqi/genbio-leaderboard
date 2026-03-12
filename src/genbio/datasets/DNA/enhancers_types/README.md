# Enhancer Type Classification

Name: DNA/enhancers-types
Input: 200bp DNA sequence
Target: Multiclass (0, 1, or 2)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 3

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 14,968     | 400       |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/enhancers-types', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

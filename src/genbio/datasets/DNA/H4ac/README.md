# H4ac Histone Mark Classification

Name: DNA/H4ac
Input: 500bp DNA sequence
Target: Binary (0 or 1)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 2

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 30,685     | 3,410     |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/H4ac', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

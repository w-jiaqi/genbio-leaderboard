# Acceptor Splice Site Classification

Name: DNA/splice-sites-acceptors
Input: 600bp DNA sequence
Target: Binary (0 or 1)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 2

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 19,961     | 2,218     |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/splice-sites-acceptors', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

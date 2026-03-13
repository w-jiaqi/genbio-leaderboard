# H4K20me1 Histone Mark Classification

Name: DNA/H4K20me1
Input: 1000bp DNA sequence
Target: Binary (0 or 1)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 2

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 30,000     | 2,270     |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/H4K20me1', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

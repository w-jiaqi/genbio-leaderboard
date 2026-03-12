# TATA Promoter Classification

Name: DNA/promoter-tata
Input: 300bp DNA sequence
Target: Binary (0 or 1)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 2

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 5,509      | 621       |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/promoter-tata', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

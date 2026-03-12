# Splice Site Classification (All)

Name: DNA/splice-sites-all
Input: 400bp DNA sequence
Target: Multiclass (0, 1, or 2)
Primary Metric: MCC (Matthews Correlation Coefficient)
Labels: 3

| Fold ID | Train Size | Test Size |
|---------|------------|-----------|
| 0       | 27,000     | 3,000     |

## Usage

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='DNA/splice-sites-all', fold='0', user='your_name')
train_df, test_df = task.setup()
# train_df / test_df have columns: sequence, name, labels
```

## Citation

Dalla-Torre, H., et al. "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics." bioRxiv (2023).

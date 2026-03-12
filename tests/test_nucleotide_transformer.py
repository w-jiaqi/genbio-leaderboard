"""Smoke tests for Nucleotide Transformer downstream tasks."""
import genbio.leaderboard as gl
import numpy as np
from collections import Counter


def test_binary_task():
    """Test a binary classification task (H3 histone mark)."""
    task = gl.BenchmarkTask(name='DNA/H3', fold='0', user='test_user')
    train_df, test_df = task.setup()

    assert 'sequence' in train_df.columns
    assert 'labels' in train_df.columns
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert set(train_df['labels'].unique()).issubset({0, 1})

    majority_label = Counter(train_df['labels'].values).most_common(1)[0][0]
    train_pred = train_df.copy()
    train_pred['labels'] = majority_label
    test_pred = test_df.copy()
    test_pred['labels'] = majority_label

    results = task.evaluate(train_pred, train_df)
    assert 'primary_metric' in results
    assert results['primary_metric'] == 'mcc'
    assert 'mcc' in results
    assert 'accuracy' in results
    assert 'f1' in results

    print(f"\n[PASS] Binary task DNA/H3: train={len(train_df)}, test={len(test_df)}")
    print(f"  Metrics: {results}")


def test_multiclass_task():
    """Test a multiclass classification task (splice_sites_all)."""
    task = gl.BenchmarkTask(name='DNA/splice-sites-all', fold='0', user='test_user')
    train_df, test_df = task.setup()

    assert 'sequence' in train_df.columns
    assert 'labels' in train_df.columns
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df['labels'].unique()) == 3

    train_pred = train_df.copy()
    train_pred['labels'] = np.random.randint(0, 3, size=len(train_df))
    test_pred = test_df.copy()
    test_pred['labels'] = np.random.randint(0, 3, size=len(test_df))

    results = task.evaluate(train_pred, train_df)
    assert 'primary_metric' in results
    assert results['primary_metric'] == 'mcc'
    assert 'mcc' in results
    assert 'f1_macro' in results
    assert 'f1_weighted' in results

    print(f"\n[PASS] Multiclass task DNA/splice-sites-all: train={len(train_df)}, test={len(test_df)}")
    print(f"  Metrics: {results}")


def test_fold_validation():
    """Test that invalid fold IDs are rejected."""
    task = gl.BenchmarkTask(name='DNA/H3', fold='1', user='test_user')
    try:
        task.setup()
        assert False, "Should have raised ValueError for fold '1'"
    except ValueError as e:
        print(f"\n[PASS] Fold validation: correctly rejected fold '1': {e}")


if __name__ == "__main__":
    test_fold_validation()
    test_binary_task()
    test_multiclass_task()
    print("\n\nAll smoke tests passed!")

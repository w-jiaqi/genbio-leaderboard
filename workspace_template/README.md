# Agent Workspace

You are working on the **<DATASET>** task.
Your goal is to build a performant ML model by iterating on `solution.py`.

## Quick reference

```bash
# Evaluate your solution with 3-fold CV on training data
python run_cv.py --dataset <DATASET> --user <USER> --name "describe your approach"

# Check your score history (shows commit hashes)
genbio-leaderboard history --dataset <DATASET> --fold 0 --user <USER>

# See the full leaderboard
genbio-leaderboard leaderboard --dataset <DATASET> --fold 0
```

## Your workflow

1. **Edit `solution.py`** — it must define:
   ```python
   def build_and_predict(train_data, test_data):
       # train_data, test_data: AnnData or DataFrame (depends on task)
       # Return: predictions in the format expected by evaluate()
       ...
       return preds
   ```

2. **Run CV** to evaluate on training data:
   ```bash
   python run_cv.py --dataset <DATASET> --user <USER> --name "describe your approach"
   ```

3. **Commit** your change:
   ```bash
   git add solution.py && git commit -m "describe what you changed and the CV score"
   ```

4. **Check history** to see all past scores and their commit hashes:
   ```bash
   genbio-leaderboard history --dataset <DATASET> --fold 0 --user <USER>
   ```

5. **Explore past attempts** using git:
   ```bash
   git log --oneline solution.py
   git diff <commit1> <commit2> -- solution.py
   git checkout <commit> -- solution.py
   ```

6. **Repeat** steps 1-5 until you are satisfied with the CV score.

## Rules

- **Optimize only on CV score** (training data). You never have access to the test set.
- **Commit after every change** so each attempt is tracked with a commit hash.
- **Use deterministic seeds**: `np.random.seed(0)`, `random_state=0`.
- Keep solutions self-contained with all imports in `solution.py`.
- Use lightweight dependencies (numpy, scipy, scikit-learn, pandas).

## Data format details

### Segerstolpe (classification: `expression/cell-type-classification-segerstolpe`)

```python
X = train_data.X                                    # scipy.sparse.csr_matrix (n_cells x 25453 genes)
y = train_data.obs['cell_type_label'].to_numpy()    # int64 labels (14 classes: 0-12)

# Return predictions as:
preds = test_data.copy()
preds.obs['cell_type_label'] = y_pred.astype(np.int64)
return preds
```

### RNA translation efficiency (regression: `RNA/translation-efficiency-muscle`)

```python
sequences = train_data['sequence']     # RNA sequences (strings)
y = train_data['labels'].to_numpy()    # float regression targets

# Return predictions as:
preds = test_data.copy()
preds['labels'] = y_pred
return preds
```

## Inspect the task

```python
import genbio.leaderboard as gl
task = gl.BenchmarkTask(name='<DATASET>', fold='0', user='<USER>')
task.describe()
```

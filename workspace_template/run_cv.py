"""CV evaluation harness for the agent workspace.

Loads training data, runs your solution with k-fold cross-validation,
prints the CV score, and records it to the leaderboard with the current
git commit hash.

Edit ``solution.py`` in this directory, then run::

    python run_cv.py --dataset expression/cell-type-classification-segerstolpe \\
        --user jw --name "Ridge + HVG"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

import genbio.leaderboard as gl
from genbio.datasets.utils import _load_dataset_module
from genbio.leaderboard.reporting import record_cv_score, _get_git_commit_hash


def _load_solution(script_path: str):
    """Import *script_path* and return its ``build_and_predict`` function."""
    spec = importlib.util.spec_from_file_location("_solution", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "build_and_predict"):
        raise ValueError(
            f"{script_path} must define: "
            "build_and_predict(train_data, test_data) -> preds"
        )
    return mod.build_and_predict


def _split(data, indices):
    """Subset *data* by integer indices, handling AnnData and DataFrame."""
    import anndata as ad
    import pandas as pd

    if isinstance(data, ad.AnnData):
        return data[indices]
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices].reset_index(drop=True)
    raise TypeError(f"Unsupported data type: {type(data)}")


def _get_stratify_labels(data):
    """Extract classification labels for stratified splitting, or None."""
    import anndata as ad

    if isinstance(data, ad.AnnData) and "cell_type_label" in data.obs.columns:
        return data.obs["cell_type_label"].to_numpy()
    return None


def run_cv(
    dataset: str, fold: str, solution_path: str, n_splits: int = 3,
) -> tuple[float, float, str]:
    """Run k-fold CV and return (mean_score, std_score, metric_name)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    task = gl.BenchmarkTask(name=dataset, fold=fold, user="_cv_")
    train = task.setup_train()
    evaluate_mod = _load_dataset_module(dataset, "evaluate")
    build_and_predict = _load_solution(solution_path)

    n_samples = len(train)
    labels = _get_stratify_labels(train)

    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_iter = splitter.split(np.zeros(n_samples), labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_iter = splitter.split(np.zeros(n_samples))

    scores: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
        train_fold = _split(train, tr_idx)
        val_fold = _split(train, va_idx)

        preds = build_and_predict(train_fold, val_fold)
        metrics = evaluate_mod.evaluate(preds, val_fold)
        primary = metrics["primary_metric"]
        score = float(metrics[primary])
        scores.append(score)
        print(f"  Fold {fold_idx}: {primary}={score:.4f}")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return mean_score, std_score, primary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CV on solution.py")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--fold", default="0", help="Fold identifier")
    parser.add_argument("--user", required=True, help="User identifier")
    parser.add_argument("--name", required=True, help="Short name for this run")
    parser.add_argument("--description", default="", help="Description")
    parser.add_argument("--agent", default="codex", help="Agent identifier")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--solution", default=None, help="Path to solution script")
    parser.add_argument("--no-record", action="store_true", help="Skip leaderboard recording")
    args = parser.parse_args()

    solution_path = args.solution or str(Path(__file__).resolve().parent / "solution.py")

    print(f"Dataset:  {args.dataset}")
    print(f"Solution: {solution_path}")
    print(f"CV folds: {args.cv_folds}\n")

    np.random.seed(0)
    mean_score, std_score, metric_name = run_cv(
        dataset=args.dataset, fold=args.fold,
        solution_path=solution_path, n_splits=args.cv_folds,
    )

    print(f"\nCV {metric_name}: {mean_score:.4f} ± {std_score:.4f}")

    if not args.no_record:
        filepath = record_cv_score(
            dataset=args.dataset, fold=args.fold, user=args.user,
            metric_name=metric_name, cv_score=mean_score,
            name=args.name,
            description=args.description or f"CV {metric_name}={mean_score:.4f} ± {std_score:.4f}",
            agent=args.agent,
        )
        commit = _get_git_commit_hash() or "--"
        print(f"Recorded: {args.name} | {metric_name}={mean_score:.4f} | commit={commit}")
        print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()

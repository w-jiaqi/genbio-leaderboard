"""CLI and dashboard reporting tools for the leaderboard."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Submissions stored globally so all agent workspaces share one leaderboard
SUBMISSION_DIR = Path.home() / ".genbio_leaderboard" / "submissions"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _get_git_commit_hash() -> Optional[str]:
    """Return the short git commit hash of HEAD, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# Storage functions

def save_submission(
    dataset: str,
    fold: str,
    user: str,
    metrics: Dict[str, float],
    name: str,
    description: str,
    agent: str = "unknown",
    tracking_id: Optional[str] = None,
    commit_hash: Optional[str] = None,
) -> str:
    """
    Save a submission to disk.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: User identifier
        metrics: Dictionary of metric names and values
        name: Submission name
        description: Submission description
        agent: Agent identifier, e.g. "codex", "claude"
        tracking_id: Optional external identifier for run tracking
        commit_hash: Optional git commit hash (auto-detected if None)

    Returns:
        Path to the saved submission file
    """
    if commit_hash is None:
        commit_hash = _get_git_commit_hash()

    # Create directory structure
    submission_dir = SUBMISSION_DIR / dataset / fold / user
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().isoformat()

    # Create submission data
    metrics_payload = dict(metrics)
    content_payload = {
        "user": user,
        "dataset": dataset,
        "fold": fold,
        "metrics": metrics_payload,
        "name": name,
        "description": description,
        "agent": agent,
        "tracking_id": tracking_id,
    }
    content_hash = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    submission_hash = hashlib.sha256(
        json.dumps(
            {"timestamp": timestamp, **content_payload},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()

    submission_data = {
        "timestamp": timestamp,
        "user": user,
        "dataset": dataset,
        "fold": fold,
        "metrics": metrics_payload,
        "name": name,
        "description": description,
        "agent": agent,
        "tracking_id": tracking_id,
        "commit_hash": commit_hash,
        "content_hash": content_hash,
        "submission_hash": submission_hash,
    }

    # Save to file
    filename = f"{timestamp.replace(':', '-')}.json"
    filepath = submission_dir / filename

    with open(filepath, 'w') as f:
        json.dump(submission_data, f, indent=2)

    return str(filepath)


def record_cv_score(
    dataset: str,
    fold: str,
    user: str,
    metric_name: str,
    cv_score: float,
    name: str,
    description: str = "",
    agent: str = "unknown",
    commit_hash: Optional[str] = None,
) -> str:
    """Record a CV score without evaluating on the test set."""
    metrics = {
        "primary_metric": metric_name,
        metric_name: cv_score,
        "record_type": "cv",
    }
    return save_submission(
        dataset=dataset, fold=fold, user=user, metrics=metrics,
        name=name, description=description, agent=agent,
        tracking_id=None, commit_hash=commit_hash,
    )


def load_submissions(
    dataset: str,
    fold: str,
    user: Optional[str] = None,
) -> List[Dict]:
    """
    Load submissions from disk.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: Optional user identifier. If None, load all users.

    Returns:
        List of submission dictionaries
    """
    submissions = []

    base_dir = SUBMISSION_DIR / dataset / fold

    if not base_dir.exists():
        return submissions

    if user:
        # Load submissions for specific user
        user_dir = base_dir / user
        if user_dir.exists():
            for filepath in user_dir.glob("*.json"):
                with open(filepath, 'r') as f:
                    submissions.append(json.load(f))
    else:
        # Load submissions for all users
        for user_dir in base_dir.iterdir():
            if user_dir.is_dir():
                for filepath in user_dir.glob("*.json"):
                    with open(filepath, 'r') as f:
                        submissions.append(json.load(f))

    # Sort by timestamp
    submissions.sort(key=lambda x: x['timestamp'])

    return submissions


def get_leaderboard_data(dataset: str, fold: str) -> List[Dict]:
    """
    Get leaderboard data (best submission per user).

    Args:
        dataset: Dataset name
        fold: Fold identifier

    Returns:
        List of best submissions per user, sorted by primary metric
    """
    all_submissions = load_submissions(dataset, fold)

    if not all_submissions:
        return []

    # Get primary metric name
    primary_metric = all_submissions[0]['metrics']['primary_metric']

    # Group by user and get best submission
    user_best = {}
    for submission in all_submissions:
        user = submission['user']
        metric_value = submission['metrics'][primary_metric]

        if user not in user_best or metric_value > user_best[user]['metrics'][primary_metric]:
            user_best[user] = submission

    # Convert to list and sort by primary metric (descending)
    leaderboard = list(user_best.values())
    leaderboard.sort(key=lambda x: x['metrics'][primary_metric], reverse=True)

    return leaderboard


def get_user_history(dataset: str, fold: str, user: str) -> List[Dict]:
    """
    Get submission history for a specific user.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: User identifier

    Returns:
        List of submissions sorted by timestamp
    """
    return load_submissions(dataset, fold, user)


# Display functions

def display_leaderboard(dataset: str, fold: str):
    """Display the leaderboard for the specified dataset and fold."""
    # Get leaderboard data
    dataset = dataset.replace('-', '_')
    leaderboard_data = get_leaderboard_data(dataset, fold)

    if not leaderboard_data:
        print(f"\nNo submissions found for {dataset} fold {fold}")
        return

    # Get primary metric name
    primary_metric = leaderboard_data[0]['metrics']['primary_metric']

    # Print header
    print(f"\n{'='*100}")
    print(f"Leaderboard: {dataset} (Fold {fold})")
    print(f"Primary Metric: {primary_metric}")
    print(f"{'='*110}")
    print(f"{'Rank':<6} {'User':<20} {'Name':<25} {'Score':<12} {'Commit':<10} {'Timestamp':<30}")
    print(f"{'-'*110}")

    # Print entries
    for rank, entry in enumerate(leaderboard_data, 1):
        user = entry['user']
        name = entry.get('name', 'Unnamed')  # Handle old submissions without name
        score = entry['metrics'][primary_metric]
        timestamp = entry['timestamp'][:19]  # Remove microseconds

        commit = entry.get('commit_hash', '--') or '--'
        print(f"{rank:<6} {user:<20} {name:<25} {score:<12.6f} {commit:<10} {timestamp:<30}")

    print(f"{'='*100}\n")


def display_history(dataset: str, fold: str, user: str):
    """Display submission history for a user."""
    # Get user history
    user_submissions = get_user_history(dataset, fold, user)

    if not user_submissions:
        print(f"\nNo submissions found for user '{user}' on {dataset} fold {fold}")
        return

    # Get primary metric name
    primary_metric = user_submissions[0]['metrics']['primary_metric']

    # Print header
    print(f"\n{'='*120}")
    print(f"Submission History: {user} - {dataset} (Fold {fold})")
    print(f"Primary Metric: {primary_metric}")
    print(f"{'='*130}")
    print(f"{'#':<4} {'Name':<25} {'Commit':<10} {'Timestamp':<22} {primary_metric:<12} {'Change':<10} {'Other Metrics'}")
    print(f"{'-'*130}")

    # Print entries
    prev_score = None
    for idx, submission in enumerate(user_submissions, 1):
        name = submission.get('name', 'Unnamed')  # Handle old submissions without name
        timestamp = submission['timestamp'][:19]  # Remove microseconds
        score = submission['metrics'][primary_metric]

        # Calculate change
        if prev_score is not None:
            change = score - prev_score
            change_str = f"{change:+.4f}" if change != 0 else "  --"
        else:
            change_str = "  --"

        # Format other metrics
        other_metrics = []
        for key, value in submission['metrics'].items():
            if key not in ['primary_metric', primary_metric]:
                other_metrics.append(f"{key}={value:.4f}")
        other_metrics_str = ", ".join(other_metrics)

        commit = submission.get('commit_hash', '--') or '--'
        print(f"{idx:<4} {name:<25} {commit:<10} {timestamp:<22} {score:<12.6f} {change_str:<10} {other_metrics_str}")

        prev_score = score

    print(f"{'='*130}")

    # Show improvement summary
    if len(user_submissions) > 1:
        first_score = user_submissions[0]['metrics'][primary_metric]
        best_score = max(s['metrics'][primary_metric] for s in user_submissions)
        latest_score = user_submissions[-1]['metrics'][primary_metric]

        print(f"\nSummary:")
        print(f"  First submission:  {first_score:.6f}")
        print(f"  Best submission:   {best_score:.6f}")
        print(f"  Latest submission: {latest_score:.6f}")
        print(f"  Total improvement: {latest_score - first_score:+.6f}")
        print()


# Export functions

def export_benchmark_data(
    output_file: str = "benchmark_export.csv",
) -> str:
    """
    Export all benchmark data to CSV.

    Args:
        output_file: Path to output CSV file (default: "benchmark_export.csv")

    Returns:
        Path to the exported CSV file
    """
    base_dir = SUBMISSION_DIR

    if not base_dir.exists():
        raise ValueError(f"Submissions directory not found: {base_dir}")

    # Find all submission JSON files recursively
    all_submissions = []
    for submission_file in base_dir.glob("**/*.json"):
        with open(submission_file, 'r') as f:
            submission = json.load(f)
            # Flatten metrics into top-level fields
            flat_submission = {
                'dataset': submission['dataset'],
                'fold': submission['fold'],
                'user': submission['user'],
                'timestamp': submission['timestamp'],
                'name': submission.get('name', ''),
                'description': submission.get('description', ''),
            }
            # Add all metrics as separate columns
            for key, value in submission['metrics'].items():
                flat_submission[f'metric_{key}'] = value
            all_submissions.append(flat_submission)

    if not all_submissions:
        raise ValueError("No submissions found to export")

    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(all_submissions)
    df = df.sort_values('timestamp')
    df.to_csv(output_file, index=False)
    return output_file


# Workspace init

def init_workspace(dataset: str, user: str, agent: str, output_dir: str) -> str:
    """Create a fresh agent workspace directory with template files."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Directory {out} already exists and is not empty")

    template_dir = PACKAGE_ROOT / "workspace_template"
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Workspace template not found at {template_dir}. "
            "Make sure the genbio-leaderboard package is installed correctly."
        )

    out.mkdir(parents=True, exist_ok=True)

    for src_file in template_dir.iterdir():
        if src_file.name == "README.md":
            file_content = src_file.read_text()
            file_content = file_content.replace("<DATASET>", dataset)
            file_content = file_content.replace("<USER>", user)
            file_content = file_content.replace("<AGENT>", agent)
            (out / src_file.name).write_text(file_content)
        else:
            shutil.copy2(src_file, out / src_file.name)

    subprocess.run(["git", "init"], cwd=str(out), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(out), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"Initial workspace for {dataset}"],
        cwd=str(out), capture_output=True,
    )

    return str(out)


# Test evaluation

def evaluate_on_test(
    dataset: str, fold: str, user: str, agent: str,
    workspace_dir: str, commit: Optional[str] = None,
) -> None:
    """Run the agent's solution on the held-out test set and submit."""
    from genbio.leaderboard.main import BenchmarkTask

    ws = Path(workspace_dir).resolve()
    solution_path = ws / "solution.py"

    if commit:
        result = subprocess.run(
            ["git", "show", f"{commit}:solution.py"],
            capture_output=True, text=True, cwd=str(ws),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Cannot checkout solution.py from {commit}: {result.stderr}")
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="solution_")
        tmp.write(result.stdout)
        tmp.close()
        solution_path = Path(tmp.name)

    spec = importlib.util.spec_from_file_location("_solution", str(solution_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {solution_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_and_predict"):
        raise ValueError(f"{solution_path} must define build_and_predict(train_data, test_data)")
    build_and_predict = mod.build_and_predict

    task = BenchmarkTask(name=dataset, fold=fold, user=user)
    train, test = task.setup()

    np.random.seed(0)
    preds = build_and_predict(train, test)

    commit_hash = commit or _get_git_commit_hash()
    task.submit(
        preds,
        name=f"test-eval-{commit_hash or 'HEAD'}",
        description=f"Test evaluation from commit {commit_hash or 'HEAD'}",
        agent=agent,
        tracking_id=f"test-eval-{commit_hash or 'HEAD'}",
    )

    if commit:
        solution_path.unlink(missing_ok=True)


# CLI entry point

def cli():
    """Main CLI entry point for genbio-leaderboard."""
    parser = argparse.ArgumentParser(
        description='GenBio Leaderboard CLI',
        prog='genbio-leaderboard'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser('leaderboard', help='Display leaderboard')
    leaderboard_parser.add_argument('--dataset', required=True, help='Dataset name')
    leaderboard_parser.add_argument('--fold', required=True, help='Fold identifier')
    # History command
    history_parser = subparsers.add_parser('history', help='Display submission history')
    history_parser.add_argument('--dataset', required=True, help='Dataset name')
    history_parser.add_argument('--fold', required=True, help='Fold identifier')
    history_parser.add_argument('--user', required=True, help='User identifier')
    # Export command
    export_parser = subparsers.add_parser('export', help='Export all benchmark data to CSV')
    export_parser.add_argument(
        '-o', '--output',
        default='benchmark_export.csv',
        help='Output CSV filename (default: benchmark_export.csv)'
    )

    # Init command
    init_parser = subparsers.add_parser('init', help='Create a fresh agent workspace')
    init_parser.add_argument('--dataset', required=True, help='Dataset name')
    init_parser.add_argument('--user', required=True, help='User identifier')
    init_parser.add_argument('--agent', default='codex', help='Agent identifier')
    init_parser.add_argument('--dir', required=True, help='Output directory for the workspace')
    # Record command
    record_parser = subparsers.add_parser('record', help='Record a CV score (for agent use)')
    record_parser.add_argument('--dataset', required=True, help='Dataset name')
    record_parser.add_argument('--fold', default='0', help='Fold identifier')
    record_parser.add_argument('--user', required=True, help='User identifier')
    record_parser.add_argument('--metric', required=True, help='Metric name (e.g. f1_macro)')
    record_parser.add_argument('--score', required=True, type=float, help='CV score value')
    record_parser.add_argument('--name', required=True, help='Short name for this run')
    record_parser.add_argument('--description', default='', help='Description')
    record_parser.add_argument('--agent', default='unknown', help='Agent identifier')
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Run solution on test set (post-agent)')
    evaluate_parser.add_argument('--dataset', required=True, help='Dataset name')
    evaluate_parser.add_argument('--fold', default='0', help='Fold identifier')
    evaluate_parser.add_argument('--user', required=True, help='User identifier')
    evaluate_parser.add_argument('--agent', default='codex', help='Agent identifier')
    evaluate_parser.add_argument('--workspace', required=True, help='Path to agent workspace')
    evaluate_parser.add_argument('--commit', default=None, help='Git commit hash (default: HEAD)')

    args = parser.parse_args()
    if args.command == 'init':
        ws_path = init_workspace(args.dataset, args.user, args.agent, args.dir)
        print(f"Workspace created at: {ws_path}")
        print(f"Give your agent: 'Work in {ws_path}. Read README.md.'")
    elif args.command == 'leaderboard':
        display_leaderboard(args.dataset, args.fold)
    elif args.command == 'history':
        display_history(args.dataset, args.fold, args.user)
    elif args.command == 'record':
        fp = record_cv_score(
            dataset=args.dataset, fold=args.fold, user=args.user,
            metric_name=args.metric, cv_score=args.score,
            name=args.name, description=args.description, agent=args.agent,
        )
        commit = _get_git_commit_hash() or '--'
        print(f"Recorded: {args.name} | {args.metric}={args.score:.4f} | commit={commit}")
        print(f"Saved to: {fp}")
    elif args.command == 'evaluate':
        evaluate_on_test(
            dataset=args.dataset, fold=args.fold, user=args.user,
            agent=args.agent, workspace_dir=args.workspace, commit=args.commit,
        )
    elif args.command == 'export':
        output_file = export_benchmark_data(
            output_file=args.output,
        )
        print(f"\nBenchmark data successfully exported to: {output_file}")
    else:
        parser.print_help()

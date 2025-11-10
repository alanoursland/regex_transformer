#!/usr/bin/env python
"""
List and browse saved experiment results.

Usage:
    python -m src.experiments.list_results
    python -m src.experiments.list_results --pattern aplus
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def format_timestamp(timestamp_str):
    """Convert timestamp string to readable format."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def list_all_results(results_dir: Path = None):
    """List all experiment results."""

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print(f"No results directory found at: {results_dir}")
        return

    print(f"\n{'='*70}")
    print(f"EXPERIMENT RESULTS: {results_dir}")
    print(f"{'='*70}\n")

    # Group by pattern
    pattern_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    if not pattern_dirs:
        print("No experiments found.")
        return

    total_experiments = 0

    for pattern_dir in pattern_dirs:
        pattern_name = pattern_dir.name

        # Find all timestamp subdirectories
        timestamp_dirs = sorted([d for d in pattern_dir.iterdir() if d.is_dir()])

        if not timestamp_dirs:
            continue

        print(f"\nPattern: {pattern_name}")
        print(f"-" * 70)

        for timestamp_dir in timestamp_dirs:
            total_experiments += 1

            # Load metadata if available
            metadata_file = timestamp_dir / "metadata.json"
            metrics_file = timestamp_dir / "metrics.json"

            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                pattern = metadata.get("pattern", "unknown")
                model_cfg = metadata.get("model_config", {})
                training = metadata.get("training", {})

                # Load metrics if available
                metrics_str = ""
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        test_metrics = metrics.get("test", {})
                        tok_acc = test_metrics.get("token_acc", 0.0)
                        cls_acc = test_metrics.get("class_acc", 0.0)
                        metrics_str = f"Tok:{tok_acc:.3f} Cls:{cls_acc:.3f}"

                print(f"  {format_timestamp(timestamp_dir.name)}")
                print(f"    Pattern: {pattern}")
                print(f"    Model: d={model_cfg.get('d_model')}, heads={model_cfg.get('n_heads')}, "
                      f"states={model_cfg.get('num_states')}")
                print(f"    Training: epochs={training.get('epochs')}, "
                      f"samples={training.get('n_samples')}, batch={training.get('batch_size')}")
                if metrics_str:
                    print(f"    Test Acc: {metrics_str}")
                print(f"    Path: {timestamp_dir}")
                print()

            else:
                print(f"  {timestamp_dir.name}")
                print(f"    (No metadata found)")
                print(f"    Path: {timestamp_dir}")
                print()

    print(f"{'='*70}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}\n")


def list_pattern_results(pattern_name: str, results_dir: Path = None):
    """List results for a specific pattern."""

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"

    pattern_dir = results_dir / pattern_name

    if not pattern_dir.exists():
        print(f"No results found for pattern: {pattern_name}")
        print(f"Looking in: {pattern_dir}")
        return

    print(f"\n{'='*70}")
    print(f"RESULTS FOR PATTERN: {pattern_name}")
    print(f"{'='*70}\n")

    timestamp_dirs = sorted([d for d in pattern_dir.iterdir() if d.is_dir()])

    for idx, timestamp_dir in enumerate(timestamp_dirs, 1):
        print(f"\n[{idx}] {format_timestamp(timestamp_dir.name)}")
        print(f"    Path: {timestamp_dir}")

        # Show files
        files = sorted(timestamp_dir.glob("*.pt")) + sorted(timestamp_dir.glob("*.json"))
        print(f"    Files:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"      - {f.name:25s} ({size_kb:8.1f} KB)")

        # Show metrics if available
        metrics_file = timestamp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                print(f"\n    Test Metrics:")
                test = metrics.get("test", {})
                for key, value in test.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")

    print(f"\n{'='*70}")
    print(f"Total runs: {len(timestamp_dirs)}")
    print(f"{'='*70}\n")


def find_best_result(pattern_name: str = None, metric: str = "class_acc", results_dir: Path = None):
    """Find the best result based on a metric."""

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print(f"No results directory found at: {results_dir}")
        return

    best_result = None
    best_value = -float('inf') if metric.endswith('_acc') else float('inf')

    # Search patterns
    if pattern_name:
        pattern_dirs = [results_dir / pattern_name]
    else:
        pattern_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    for pattern_dir in pattern_dirs:
        if not pattern_dir.exists():
            continue

        for timestamp_dir in pattern_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            metrics_file = timestamp_dir / "metrics.json"
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                metrics = json.load(f)
                test_metrics = metrics.get("test", {})

                if metric in test_metrics:
                    value = test_metrics[metric]

                    # Higher is better for accuracy, lower for loss
                    if metric.endswith('_acc'):
                        if value > best_value:
                            best_value = value
                            best_result = timestamp_dir
                    else:
                        if value < best_value:
                            best_value = value
                            best_result = timestamp_dir

    if best_result:
        print(f"\nBest result by {metric}: {best_value:.4f}")
        print(f"Path: {best_result}\n")
    else:
        print(f"\nNo results found with metric: {metric}\n")


def main():
    parser = argparse.ArgumentParser(
        description="List and browse experiment results"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Filter by pattern name"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show best result"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="class_acc",
        help="Metric to use for --best (default: class_acc)"
    )

    args = parser.parse_args()

    if args.best:
        find_best_result(args.pattern, args.metric)
    elif args.pattern:
        list_pattern_results(args.pattern)
    else:
        list_all_results()


if __name__ == "__main__":
    main()

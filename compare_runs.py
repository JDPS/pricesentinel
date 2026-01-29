# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI helper to compare trained runs for a country/model.

This script scans ``models/{country}/{model_name}``, reads ``metrics.json`` for
each run, and prints a compact comparison table including the training data
window (when available) and basic metrics.

Examples:
    uv run python compare_runs.py --country PT --model-name baseline
    uv run python compare_runs.py --country PT --model-name baseline_fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Compare trained runs for a given country and model.",
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country code (e.g. PT, XX)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline",
        help="Model name (default: baseline)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="Root directory for models (default: models/)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=[
            "modified_time",
            "run_id",
            "train_start",
            "train_end",
            "train_mae",
            "train_rmse",
            "val_mae",
            "val_rmse",
        ],
        default="modified_time",
        help=(
            "Field to sort by (default: modified_time, which uses filesystem "
            "modification time as recorded when the run directory was created)."
        ),
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending).",
    )
    parser.add_argument(
        "--start-on-or-after",
        type=str,
        help=(
            "Filter: keep only runs where train_start_date is on or after this date (YYYY-MM-DD)."
        ),
    )
    parser.add_argument(
        "--end-on-or-before",
        type=str,
        help=(
            "Filter: keep only runs where train_end_date is on or before this date (YYYY-MM-DD)."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["train_mae", "train_rmse", "val_mae", "val_rmse"],
        help=(
            "Metric name to use for filtering and/or sorting. When used together "
            "with --max-metric, only runs with metric <= value are kept. When "
            "combined with --sort-by, you can sort by this metric explicitly."
        ),
    )
    parser.add_argument(
        "--max-metric",
        type=float,
        help=(
            "Filter: when used with --metric, keep only runs where the selected "
            "metric is <= this value."
        ),
    )
    return parser.parse_args()


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    """
    Load metrics from a JSON file, returning an empty dict on failure.
    """
    try:
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if isinstance(loaded, dict):
        return cast(dict[str, Any], loaded)

    return {}


def collect_runs(country: str, model_name: str, models_root: Path) -> list[dict[str, Any]]:
    """
    Collects information about model runs from the specified directory.

    This function gathers details about available runs of a given model from a directory
    structure based on country and model name. Each run is represented by a folder containing
    a "metrics.json" file with relevant training and validation metrics. Runs are sorted
    by modification time in descending order. If the model directory does not exist, an empty
    list is returned.
    """
    country = country.upper()
    model_dir = models_root / country / model_name

    if not model_dir.exists():
        return []

    runs: list[dict[str, Any]] = []

    run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = _load_metrics(metrics_path)

        run_info: dict[str, Any] = {
            "run_id": run_dir.name,
            "metrics_path": str(metrics_path) if metrics_path.exists() else "<missing>",
            "train_start_date": metrics.get("train_start_date"),
            "train_end_date": metrics.get("train_end_date"),
            "train_mae": metrics.get("train_mae"),
            "train_rmse": metrics.get("train_rmse"),
            "val_mae": metrics.get("val_mae"),
            "val_rmse": metrics.get("val_rmse"),
        }
        runs.append(run_info)

    return runs


def _date_ge(date_str: str | None, threshold: str) -> bool:
    """
    Compare ISO date strings (YYYY-MM-DD) using lexicographical ordering.

    Returns True if date_str >= threshold. If date_str is None, returns False
    so that runs without date information are filtered out when a threshold
    is requested.
    """
    if date_str is None:
        return False
    return str(date_str) >= threshold


def _date_le(date_str: str | None, threshold: str) -> bool:
    """
    Compare ISO date strings (YYYY-MM-DD) using lexicographical ordering.

    Returns True if date_str <= threshold. If date_str is None, returns False
    so that runs without date information are filtered out when a threshold
    is requested.
    """
    if date_str is None:
        return False
    return str(date_str) <= threshold


def filter_runs(
    runs: list[dict[str, Any]],
    start_on_or_after: str | None,
    end_on_or_before: str | None,
    metric_name: str | None,
    max_metric: float | None,
) -> list[dict[str, Any]]:
    """
    Apply simple filters on training window and an optional metric.
    """
    filtered: list[dict[str, Any]] = []

    for run in runs:
        # Filter by training window start
        if start_on_or_after is not None:
            if not _date_ge(run.get("train_start_date"), start_on_or_after):
                continue

        # Filter by training window end
        if end_on_or_before is not None:
            if not _date_le(run.get("train_end_date"), end_on_or_before):
                continue

        # Filter by metric threshold
        if metric_name is not None and max_metric is not None:
            value = run.get(metric_name)
            if not isinstance(value, int | float) or value > max_metric:
                continue

        filtered.append(run)

    return filtered


def _sort_key(run: dict[str, Any], sort_by: str) -> Any:
    """
    Compute a sort key for a run based on the requested field.
    """
    if sort_by == "train_start":
        value = run.get("train_start_date")
    elif sort_by == "train_end":
        value = run.get("train_end_date")
    else:
        value = run.get(sort_by)

    if isinstance(value, int | float):
        return value

    if value is None:
        # Put runs with missing values at the end of the list.
        return ""

    return str(value)


def sort_runs(runs: list[dict[str, Any]], sort_by: str, ascending: bool) -> list[dict[str, Any]]:
    """
    Sort runs according to the requested field.

    When sort_by is 'modified_time', the original ordering from collect_runs
    (newest first) is preserved, optionally reversed when ascending=True.
    """
    if not runs:
        return runs

    if sort_by == "modified_time":
        # collect_runs already returns newest-first; just reverse when ascending.
        return list(reversed(runs)) if ascending else runs

    sorted_runs = sorted(runs, key=lambda r: _sort_key(r, sort_by), reverse=not ascending)
    return sorted_runs


def print_runs(country: str, model_name: str, runs: list[dict[str, Any]]) -> None:
    # Use simple text output for easy terminal inspection.
    print(  # noqa: T201
        f"\n=== Run Comparison: country={country.upper()}, model={model_name} "
        f"({len(runs)} runs) ===\n"
    )

    if not runs:
        print("No runs found.")  # noqa: T201
        print("\n=== End Comparison ===\n")  # noqa: T201
        return

    header = [
        "run_id",
        "train_start",
        "train_end",
        "train_mae",
        "train_rmse",
        "val_mae",
        "val_rmse",
    ]
    print(" | ".join(header))  # noqa: T201
    print("-" * 80)  # noqa: T201

    for run in runs:
        row = [
            run.get("run_id", ""),
            str(run.get("train_start_date") or "-"),
            str(run.get("train_end_date") or "-"),
            f"{run.get('train_mae'):.3f}" if isinstance(run.get("train_mae"), int | float) else "-",
            f"{run.get('train_rmse'):.3f}"
            if isinstance(run.get("train_rmse"), int | float)
            else "-",
            f"{run.get('val_mae'):.3f}" if isinstance(run.get("val_mae"), int | float) else "-",
            f"{run.get('val_rmse'):.3f}" if isinstance(run.get("val_rmse"), int | float) else "-",
        ]
        print(" | ".join(row))  # noqa: T201

    print("\n=== End Comparison ===\n")  # noqa: T201


def main() -> None:
    args = parse_args()
    runs = collect_runs(
        country=args.country,
        model_name=args.model_name,
        models_root=args.models_root,
    )
    runs = filter_runs(
        runs,
        start_on_or_after=args.start_on_or_after,
        end_on_or_before=args.end_on_or_before,
        metric_name=args.metric,
        max_metric=args.max_metric,
    )
    runs = sort_runs(runs, sort_by=args.sort_by, ascending=args.ascending)
    print_runs(args.country, args.model_name, runs)


if __name__ == "__main__":
    main()

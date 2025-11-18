#!/usr/bin/env python
"""CLI helper to inspect trained runs and forecasts.

Examples:
    uv run python inspect_run.py --country XX
    uv run python inspect_run.py --country PT --model-name baseline --run-id 20240101_120000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.inspection import RunInspection, inspect_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect metrics and forecasts for a trained run.",
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
        "--run-id",
        type=str,
        default=None,
        help="Run ID to inspect (default: latest run for the model)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="Root directory for models (default: models/)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for data (default: data/)",
    )
    return parser.parse_args()


def print_inspection(inspection: RunInspection) -> None:
    country = inspection.country_code
    model_name = inspection.model_name
    run_id = inspection.run_id or "<none>"

    # Use stdout printing for this small CLI helper (intentional side effects).
    print(  # noqa: T201
        f"\n=== Run Inspection: country={country}, model={model_name}, run_id={run_id} ===\n"
    )

    # Metrics
    if inspection.metrics_path and inspection.metrics is not None:
        print(f"Metrics file: {inspection.metrics_path}")  # noqa: T201
        for key, value in inspection.metrics.items():
            print(f"  {key}: {value}")  # noqa: T201
    else:
        print("Metrics: <none found>")  # noqa: T201

    # Forecast summaries
    if not inspection.forecast_summaries:
        print("\nForecasts: <none found>")  # noqa: T201
    else:
        print("\nForecasts:")  # noqa: T201
        for summary in inspection.forecast_summaries:
            print(f"  File: {summary.path}")  # noqa: T201
            print(f"    rows: {summary.rows}")  # noqa: T201
            if summary.rows > 0:
                print(  # noqa: T201
                    f"    time: {summary.min_timestamp} -> {summary.max_timestamp}"
                )
                print(  # noqa: T201
                    f"    price: min={summary.min_price:.2f}, "
                    f"max={summary.max_price:.2f}, mean={summary.mean_price:.2f}"
                )

    print("\n=== End Inspection ===\n")  # noqa: T201


def main() -> None:
    args = parse_args()
    inspection = inspect_run(
        country_code=args.country.upper(),
        model_name=args.model_name,
        run_id=args.run_id,
        models_root=args.models_root,
        data_root=args.data_root,
    )
    print_inspection(inspection)


if __name__ == "__main__":
    main()

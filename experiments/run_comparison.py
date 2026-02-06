# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI script to run model comparisons for PriceSentinel.

Compares multiple model trainers on the same dataset using walk-forward or
time-series split cross-validation and produces a ranked Markdown report.

Example usage:
    # Using a pre-built features CSV:
    uv run python experiments/run_comparison.py \
        --country PT \
        --models baseline,xgboost,lightgbm \
        --features-file outputs/PT/PT_electricity_features_20230101_20231231.csv

    # Auto-discover features via PipelineBuilder:
    uv run python experiments/run_comparison.py \
        --country PT \
        --models baseline,xgboost \
        --start 2023-01-01 \
        --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Add project root to path when running as a script
sys.path.insert(0, os.getcwd())

from core.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("run_comparison")


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run PriceSentinel model comparison",
    )

    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="ISO 3166-1 alpha-2 country code (e.g. PT)",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names to compare (e.g. baseline,xgboost,lightgbm)",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Path to a pre-built features CSV. If omitted, features are "
        "built via PipelineBuilder (requires --start and --end).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) for feature engineering (ignored if --features-file given)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for feature engineering (ignored if --features-file given)",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "time_series_split"],
        help="Cross-validation method (default: walk_forward)",
    )
    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=720,
        help="Initial training window size in samples (default: 720)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=24,
        help="Step size for walk-forward validation (default: 24)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for time_series_split (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Directory to save the Markdown report (default: outputs/reports)",
    )

    return parser.parse_args()


def _load_features_from_file(
    features_file: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features DataFrame from a CSV file.

    Args:
        features_file: Path to the features CSV.

    Returns:
        Tuple of (x, y) where x is the feature matrix and y the target series.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    import pandas as pd  # noqa: E402

    path = Path(features_file)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    logger.info("Loading features from %s", features_file)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = "target_price"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {features_file}")

    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
    feature_cols = [c for c in df_clean.columns if c not in ("timestamp", target_col)]
    x = df_clean[feature_cols].select_dtypes(include="number")
    y = df_clean[target_col]

    logger.info("Loaded %d samples with %d features", len(x), x.shape[1])
    return x, y


def _load_features_via_pipeline(
    country_code: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features using PipelineBuilder and return (x, y).

    Args:
        country_code: ISO country code.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        Tuple of (x, y).
    """
    import pandas as pd  # noqa: E402

    from core.pipeline_builder import PipelineBuilder  # noqa: E402
    from data_fetchers import auto_register_countries  # noqa: E402

    auto_register_countries()
    pipeline = PipelineBuilder.create_pipeline(country_code)
    pipeline.engineer_features(start_date, end_date)

    # Discover the latest feature file
    repo = pipeline.repository
    pattern = f"{country_code}_electricity_features_*.csv"
    feature_files = repo.list_processed_data(pattern)

    if not feature_files:
        raise FileNotFoundError(f"No feature files found for {country_code} after engineering")

    features_path = feature_files[0]
    logger.info("Using feature file: %s", features_path)

    df = pd.read_csv(features_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = "target_price"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from features")

    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
    feature_cols = [c for c in df_clean.columns if c not in ("timestamp", target_col)]
    x = df_clean[feature_cols].select_dtypes(include="number")
    y = df_clean[target_col]

    logger.info("Engineered %d samples with %d features", len(x), x.shape[1])
    return x, y


def main() -> None:
    """Entry point for the model comparison CLI."""
    args = _parse_args()
    setup_logging(level="INFO")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        logger.error("No model names provided via --models")
        sys.exit(1)

    logger.info(
        "Model comparison: country=%s, models=%s, cv=%s",
        args.country,
        model_names,
        args.cv,
    )

    # ----- Load features -----
    try:
        if args.features_file:
            x, y = _load_features_from_file(args.features_file)
        else:
            if not args.start or not args.end:
                logger.error("Either --features-file or both --start and --end are required")
                sys.exit(1)
            x, y = _load_features_via_pipeline(args.country, args.start, args.end)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load features: %s", exc)
        sys.exit(1)

    # ----- Run comparison -----
    from experiments.model_comparison import ModelComparison  # noqa: E402

    try:
        comparison = ModelComparison(
            country_code=args.country,
            model_names=model_names,
        )
        results = comparison.run(
            x=x,
            y=y,
            cv_method=args.cv,
            initial_train_size=args.initial_train_size,
            step_size=args.step_size,
            n_splits=args.n_splits,
        )
    except ValueError as exc:
        logger.error("Comparison failed: %s", exc)
        sys.exit(1)

    # ----- Print results -----
    print("\n" + "=" * 70)  # noqa: T201
    print(f"  MODEL COMPARISON RESULTS -- {args.country}")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print(results.to_markdown(index=False, floatfmt=".4f"))  # noqa: T201
    print("-" * 70)  # noqa: T201

    best = results.iloc[0]
    print(  # noqa: T201
        f"\nBest model: {best['model']}  " f"(MAE {best['mean_mae']:.4f} +/- {best['std_mae']:.4f})"
    )
    print("=" * 70 + "\n")  # noqa: T201

    # ----- Save Markdown report -----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"comparison_{args.country}_{'-'.join(model_names)}.md"
    report_text = comparison.generate_report(results)

    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    print(f"Report saved to {report_path}")  # noqa: T201


if __name__ == "__main__":
    main()

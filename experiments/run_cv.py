# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI script to run Cross-Validation for PriceSentinel.

Example usage:
    uv run python experiments/run_cv.py --country PT --start 2023-01-01 --end 2023-12-31 --splits 3
"""

import argparse
import logging
import os
import sys

# Add project root to path if running closely
sys.path.insert(0, os.getcwd())

from core.logging_config import setup_logging
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

logger = logging.getLogger("run_cv")


def main() -> None:
    """Run cross-validation CLI."""
    parser = argparse.ArgumentParser(description="Run PriceSentinel Cross-Validation")
    parser.add_argument(
        "--country", type=str, required=True, help="ISO 3166-1 alpha-2 country code (e.g. PT)"
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD) for CV period"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date (YYYY-MM-DD) for CV period"
    )
    parser.add_argument(
        "--splits", type=int, default=5, help="Number of TimeSeriesSplit folds (default: 5)"
    )

    args = parser.parse_args()

    # Setup environment
    setup_logging(level="INFO")

    logger.info(f"Setting up Cross-Validation for {args.country}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Splits: {args.splits}")

    try:
        # Register countries
        auto_register_countries()

        # Create pipeline
        pipeline = PipelineBuilder.create_pipeline(args.country)

        # Run CV
        results = pipeline.run_cross_validation(
            start_date=args.start, end_date=args.end, n_splits=args.splits
        )

        # Output results
        print("\n" + "=" * 60)  # noqa: T201
        print(f"CROSS-VALIDATION RESULTS ({args.splits} Folds)")  # noqa: T201
        print("=" * 60)  # noqa: T201
        print(results.to_markdown(index=False, floatfmt=".4f"))  # noqa: T201
        print("-" * 60)  # noqa: T201

        mean_mae = results["mae"].mean()
        std_mae = results["mae"].std()
        mean_imp = results["improvement_pct"].mean()

        print(f"\nAverage MAE: {mean_mae:.4f} ± {std_mae:.4f}")  # noqa: T201
        print(f"Average Improvement vs Naive: {mean_imp:.1f}%")  # noqa: T201
        print("=" * 60 + "\n")  # noqa: T201

        # Save report
        os.makedirs("outputs/reports", exist_ok=True)
        report_path = f"outputs/reports/cv_{args.country}_{args.end.replace('-', '')}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Cross-Validation Report: {args.country}\n\n")
            f.write(f"- **Period**: {args.start} to {args.end}\n")
            f.write(f"- **Folds**: {args.splits}\n")
            f.write(f"- **Average MAE**: {mean_mae:.4f}\n")
            f.write(f"- **Std MAE**: {std_mae:.4f}\n")
            f.write(f"- **Improvement**: {mean_imp:.1f}%\n\n")
            f.write("## Fold Details\n\n")
            f.write(results.to_markdown(index=False, floatfmt=".4f"))

        logger.info(f"Report saved to {report_path}")

    except Exception as e:
        logger.error(f"Cross-Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

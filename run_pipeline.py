# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
PriceSentinel: Event-Aware Energy Price Forecasting

Main command-line interface for running the forecasting pipeline.

Usage:
    # Run full pipeline
    python run_pipeline.py --country PT --all --start-date 2023-01-01 --end-date 2024-12-31

    # Run individual stages
    python run_pipeline.py --country PT --fetch --start-date 2024-01-01 --end-date 2024-01-31
    python run_pipeline.py --country PT --clean
    python run_pipeline.py --country PT --train
    python run_pipeline.py --country PT --forecast

    # Get pipeline info
    python run_pipeline.py --country PT --info
"""  # noqa: E501

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from config.country_registry import CountryRegistry
from core.logging_config import setup_logging
from core.pipeline import Pipeline
from data_fetchers import auto_register_countries


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PriceSentinel: Event-Aware Energy Price Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for Portugal
  python run_pipeline.py --country PT --all --start-date 2023-01-01 --end-date 2024-12-31

  # Fetch data only
  python run_pipeline.py --country PT --fetch --start-date 2024-01-01 --end-date 2024-01-31

  # Run forecast only
  python run_pipeline.py --country PT --forecast --forecast-date 2025-01-07

  # Get information about available data
  python run_pipeline.py --country PT --info
        """,  # noqa: E501
    )

    # Required arguments
    parser.add_argument(
        "--country", type=str, required=True, help="Country code (e.g., PT, ES, DE, XX for mock)"
    )

    # Pipeline stages
    parser.add_argument("--all", action="store_true", help="Run all pipeline stages")

    parser.add_argument("--fetch", action="store_true", help="Fetch raw data from APIs")

    parser.add_argument("--clean", action="store_true", help="Clean and verify data")

    parser.add_argument("--features", action="store_true", help="Engineer features")

    parser.add_argument("--train", action="store_true", help="Train forecasting model")

    parser.add_argument("--forecast", action="store_true", help="Generate forecasts")

    parser.add_argument("--info", action="store_true", help="Show pipeline and data information")

    # Date arguments
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")

    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    parser.add_argument(
        "--forecast-date", type=str, help="Forecast date (YYYY-MM-DD), defaults to today"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def validate_arguments(args):
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments

    Returns:
        True if valid, False otherwise
    """
    # Check if at least one action is specified
    actions = [
        args.all,
        args.fetch,
        args.clean,
        args.features,
        args.train,
        args.forecast,
        args.info,
    ]

    if not any(actions):
        print("Error: No action specified. Use --all or specify individual stages.")
        print("Use --help for usage information.")
        return False

    # Check date requirements
    if args.all or args.fetch:
        if not args.start_date or not args.end_date:
            print("Error: --start-date and --end-date are required for fetching data.")
            return False

        # Validate date format
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("Error: Dates must be in YYYY-MM-DD format.")
            return False

    return True


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Setup logging
    setup_logging(level=args.log_level)

    # Import logger after setup
    import logging

    logger = logging.getLogger(__name__)

    # Print header
    print("\n" + "=" * 70)
    print("PriceSentinel: Event-Aware Energy Price Forecasting")
    print("=" * 70 + "\n")

    # Auto-register all countries
    logger.info("Registering countries...")
    auto_register_countries()

    # List available countries
    available_countries = CountryRegistry.list_countries()
    logger.info(f"Available countries: {', '.join(available_countries)}")

    # Validate country code
    if args.country.upper() not in available_countries:
        logger.error(
            f"Country '{args.country}' not registered.\n"
            f"Available countries: {available_countries}"
        )
        sys.exit(1)

    try:
        # Initialize pipeline
        logger.info(f"Initializing pipeline for {args.country}...")
        pipeline = Pipeline(country_code=args.country)

        # Show info if requested
        if args.info:
            info = pipeline.get_info()
            print("\n" + "=" * 70)
            print(f"PIPELINE INFORMATION: {info['country_code']}")
            print("=" * 70)
            print(f"Country: {info['country_name']}")
            print(f"Timezone: {info['timezone']}")
            print(f"Run ID: {info['run_id']}")
            print(f"Data directory: {info['data_directory']}")
            print("\nData info:")
            for key, value in info["data_info"].items():
                if key != "sources":
                    print(f"  {key}: {value}")
            if "sources" in info["data_info"]:
                print("  Sources:")
                for source, count in info["data_info"]["sources"].items():
                    print(f"    - {source}: {count} files")
            print("=" * 70 + "\n")
            sys.exit(0)

        # Run pipeline stages
        if args.all:
            # Run full pipeline
            pipeline.run_full_pipeline(args.start_date, args.end_date, args.forecast_date)

        else:
            # Run individual stages
            if args.fetch:
                pipeline.fetch_data(args.start_date, args.end_date)

            if args.clean:
                pipeline.clean_and_verify()

            if args.features:
                pipeline.engineer_features()

            if args.train:
                pipeline.train_model()

            if args.forecast:
                pipeline.generate_forecast(args.forecast_date)

        print("\n" + "=" * 70)
        print(f"[OK] Pipeline completed successfully for {args.country}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
        print("\n" + "=" * 70)
        print(f"[FAIL] Pipeline failed for {args.country}")
        print(f"Error: {e}")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

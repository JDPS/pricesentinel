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

from __future__ import annotations

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

    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline",
        help="Model name to use for training (default: baseline)",
    )

    # Date arguments
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")

    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    parser.add_argument(
        "--forecast-date", type=str, help="Forecast date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--forecast-start-date",
        type=str,
        help="Start date for forecast range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forecast-end-date",
        type=str,
        help="End date for forecast range (YYYY-MM-DD)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--fast-train",
        action="store_true",
        help="Use fast training mode (e.g., smaller model or shorter run)",
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

    # Check forecast date / range requirements for forecast-only runs
    if args.forecast and not args.all:
        has_single = bool(args.forecast_date)
        has_range = bool(args.forecast_start_date and args.forecast_end_date)

        if not (has_single or has_range):
            print(
                "Error: Provide either --forecast-date or "
                "--forecast-start-date and --forecast-end-date for forecasting."
            )
            return False

        if has_range:
            try:
                start_dt = datetime.strptime(args.forecast_start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(args.forecast_end_date, "%Y-%m-%d")
            except ValueError:
                print("Error: Forecast dates must be in YYYY-MM-DD format.")
                return False

            if start_dt > end_dt:
                print("Error: --forecast-start-date must be <= --forecast-end-date.")
                return False

    return True


class PipelineCLI:
    """CLI runner for the PriceSentinel pipeline."""

    def __init__(self, args):
        import logging

        self.args = args
        # Initialise logger immediately to avoid Optional type issues
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Pipeline is initialised later in init_pipeline
        self.pipeline: Pipeline | None = None

    def setup_logging(self):
        setup_logging(level=self.args.log_level)

    @staticmethod
    def print_header():
        print("\n" + "=" * 70)
        print("PriceSentinel: Event-Aware Energy Price Forecasting")
        print("=" * 70 + "\n")

    def register_countries(self) -> list[str]:
        self.logger.info("Registering countries...")
        auto_register_countries()
        available_countries = CountryRegistry.list_countries()
        self.logger.info(f"Available countries: {', '.join(available_countries)}")
        return available_countries

    def validate_country(self, available_countries: list[str]) -> None:
        if self.args.country.upper() not in available_countries:
            self.logger.error(
                f"Country '{self.args.country}' not registered.\n"
                f"Available countries: {available_countries}"
            )
            sys.exit(1)

    def init_pipeline(self) -> None:
        self.logger.info(f"Initializing pipeline for {self.args.country}...")
        self.pipeline = Pipeline(country_code=self.args.country)

    def show_info_and_exit(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        info = self.pipeline.get_info()
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

    def run_stages(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        # Determine effective model name, incorporating fast-train flag if set
        model_name = self.args.model_name
        if getattr(self.args, "fast_train", False) and not model_name.endswith("_fast"):
            model_name = f"{model_name}_fast"

        if self.args.all:
            self.pipeline.run_full_pipeline(
                self.args.start_date,
                self.args.end_date,
                self.args.forecast_date,
                model_name=model_name,
            )
            return

        if self.args.fetch:
            self.pipeline.fetch_data(self.args.start_date, self.args.end_date)

        if self.args.clean:
            self.pipeline.clean_and_verify()

        if self.args.features:
            self.pipeline.engineer_features()

        if self.args.train:
            self.pipeline.train_model(model_name=model_name)

        if self.args.forecast:
            # Use range if provided, otherwise single forecast date
            if self.args.forecast_start_date and self.args.forecast_end_date:
                self.pipeline.generate_forecast_range(
                    self.args.forecast_start_date,
                    self.args.forecast_end_date,
                    model_name=model_name,
                )
            else:
                self.pipeline.generate_forecast(self.args.forecast_date, model_name=model_name)

    def run(self):
        self.setup_logging()
        self.print_header()
        available_countries = self.register_countries()
        self.validate_country(available_countries)

        try:
            self.init_pipeline()

            if self.args.info:
                self.show_info_and_exit()

            self.run_stages()

            print("\n" + "=" * 70)
            print(f"[OK] Pipeline completed successfully for {self.args.country}")
            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            self.logger.warning("\n\nPipeline interrupted by user")
            sys.exit(1)

        except Exception as e:
            self.logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
            print("\n" + "=" * 70)
            print(f"[FAIL] Pipeline failed for {self.args.country}")
            print(f"Error: {e}")
            print("=" * 70 + "\n")
            sys.exit(1)


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    cli = PipelineCLI(args)
    cli.run()


if __name__ == "__main__":
    main()

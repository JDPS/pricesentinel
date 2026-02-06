#!/usr/bin/env python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI script to run the forecasting pipeline for a specific date.

Usage:
    python models/run_forecast.py --country PT --date 2025-02-05
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta, timezone

from config.country_registry import ConfigLoader
from core.exceptions import PriceSentinelError
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the forecast pipeline."""
    parser = argparse.ArgumentParser(description="Run PriceSentinel Forecast")
    parser.add_argument(
        "--country",
        "-c",
        type=str,
        default="PT",
        help="Country ISO code (default: PT)",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),  # noqa: UP017
        help="Forecast date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--days-history",
        type=int,
        default=30,
        help="Days of history to fetch/use for context (default: 30)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="baseline",
        help="Model name to use (default: baseline)",
    )

    args = parser.parse_args()

    country_code = args.country.upper()
    forecast_date_str = args.date
    days_history = args.days_history
    model_name = args.model

    try:
        # Calculate context window
        forecast_dt = datetime.strptime(forecast_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        start_date = (forecast_dt - timedelta(days=days_history)).strftime("%Y-%m-%d")
        # For forecasting T, we technically need history up to T-1,
        # but fetching up to T ensures we have everything.
        end_date = forecast_date_str

        # Register countries
        auto_register_countries()

        logger.info(f"--- Starting Forecast Run for {country_code} ---")
        logger.info(f"Target Date: {forecast_date_str}")
        logger.info(f"Context: {start_date} to {end_date}")

        # Load config
        config_loader = ConfigLoader()
        _config = config_loader.load_country_config(country_code)  # noqa: F841

        # Build pipeline
        pipeline = PipelineBuilder.create_pipeline(country_code)

        # Run pipeline
        # 1. Fetch data (ensure we have recent context)
        logger.info("Step 1: Fetching data...")
        await pipeline.fetch_data(start_date, end_date)

        # 2. Clean & Verify
        logger.info("Step 2: Cleaning data...")
        pipeline.clean_and_verify(start_date, end_date)

        # 3. Engineer Features
        logger.info("Step 3: Engineering features...")
        pipeline.engineer_features(start_date, end_date)

        # 4. Generate Forecast
        logger.info(f"Step 4: Generating forecast using model '{model_name}'...")
        pipeline.generate_forecast(forecast_date=forecast_date_str, model_name=model_name)

        logger.info("--- Forecast Run Complete ---")

    except PriceSentinelError as e:
        logger.error(f"Forecast run failed: {e}", exc_info=True)
        sys.exit(1)
    except (ValueError, FileNotFoundError, OSError) as e:
        logger.error(f"Forecast run failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Forecast run interrupted by user.")
        sys.exit(130)

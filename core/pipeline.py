# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
4
Country-agnostic pipeline orchestration.

This module provides the main Pipeline class that orchestrates all stages
of the forecasting workflow while remaining completely country-agnostic.
"""

import logging
from datetime import datetime

from config.country_registry import ConfigLoader, FetcherFactory
from core.data_manager import CountryDataManager

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Country-agnostic pipeline orchestration.

    This class coordinates all stages of the forecasting pipeline:
    - Data fetching
    - Data cleaning and verification
    - Feature engineering
    - Model training
    - Forecast generation

    All country-specific logic is handled by adapters registered in the
    country registry, keeping the pipeline code generic.
    """

    def __init__(self, country_code: str):
        """
        Initialise the pipeline for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-2 code (e.g. 'PT')

        Raises:
            ValueError: If the country is not registered,
            FileNotFoundError: If country configuration doesn't exist
        """
        self.country_code = country_code.upper()
        self.config = ConfigLoader.load_country_config(self.country_code)
        self.fetchers = FetcherFactory.create_fetchers(self.country_code)
        self.data_manager = CountryDataManager(self.country_code)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure directories exist
        self.data_manager.create_directories()

        logger.info(f"Initialized pipeline for {self.country_code} " f"(run_id: {self.run_id})")

    def fetch_data(self, start_date: str, end_date: str):
        """
        Fetch all required data sources.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info(f"=== Stage 1: Fetching data for {self.country_code} ===")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Fetch electricity prices
        logger.info("Fetching electricity prices...")
        try:
            fetched_prices = self.fetchers["electricity"].fetch_prices(start_date, end_date)

            if len(fetched_prices) > 0:
                filename = self.data_manager.generate_filename(
                    "electricity_prices", start_date, end_date
                )
                output_path = self.data_manager.get_raw_path("electricity") / filename
                fetched_prices.to_csv(output_path, index=False)
                logger.info(f"Saved {len(fetched_prices)} electricity price records")
            else:
                logger.warning("No electricity price data fetched")

        except Exception as e:
            logger.error(f"Failed to fetch electricity prices: {e}")

        # Fetch electricity load
        logger.info("Fetching electricity load...")
        try:
            load_df = self.fetchers["electricity"].fetch_load(start_date, end_date)

            if len(load_df) > 0:
                filename = self.data_manager.generate_filename(
                    "electricity_load", start_date, end_date
                )
                output_path = self.data_manager.get_raw_path("electricity") / filename
                load_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(load_df)} electricity load records")
            else:
                logger.warning("No electricity load data fetched")

        except Exception as e:
            logger.error(f"Failed to fetch electricity load: {e}")

        # Fetch weather data
        logger.info("Fetching weather data...")
        try:
            weather_df = self.fetchers["weather"].fetch_weather(start_date, end_date)

            if len(weather_df) > 0:
                filename = self.data_manager.generate_filename("weather", start_date, end_date)
                output_path = self.data_manager.get_raw_path("weather") / filename
                weather_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(weather_df)} weather records")
            else:
                logger.warning("No weather data fetched")

        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")

        # Fetch gas prices
        logger.info("Fetching gas prices...")
        try:
            gas_df = self.fetchers["gas"].fetch_prices(start_date, end_date)

            if len(gas_df) > 0:
                filename = self.data_manager.generate_filename("gas_prices", start_date, end_date)
                output_path = self.data_manager.get_raw_path("gas") / filename
                gas_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(gas_df)} gas price records")
            else:
                logger.warning("No gas price data fetched")

        except Exception as e:
            logger.error(f"Failed to fetch gas prices: {e}")

        # Fetch events
        logger.info("Fetching holidays and events...")
        try:
            holidays_df = self.fetchers["events"].get_holidays(start_date, end_date)

            if len(holidays_df) > 0:
                holidays_path = self.data_manager.get_events_path() / "holidays.csv"
                holidays_df.to_csv(holidays_path, index=False)
                logger.info(f"Saved {len(holidays_df)} holidays")
            else:
                logger.info("No holidays in date range")

            # Manual events
            manual_events_df = self.fetchers["events"].get_manual_events()
            if len(manual_events_df) > 0:
                logger.info(f"Loaded {len(manual_events_df)} manual events")

        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")

        logger.info("=== Stage 1 complete: Data fetching ===\n")

    @staticmethod
    def clean_and_verify():
        """
        Clean and verify data quality.

        To be implemented in Phase 3.
        """
        logger.info("=== Stage 2: Cleaning and verifying data ===")
        logger.warning("Data cleaning not yet implemented (Phase 3)")
        logger.info("=== Stage 2 skipped ===\n")

    @staticmethod
    def engineer_features():
        """
        Generate features for modelling.

        To be implemented in Phase 4.
        """
        logger.info("=== Stage 3: Engineering features ===")
        logger.warning("Feature engineering not yet implemented (Phase 4)")
        logger.info("=== Stage 3 skipped ===\n")

    @staticmethod
    def train_model():
        """
        Train forecasting model.

        To be implemented in Phase 6.
        """
        logger.info("=== Stage 4: Training model ===")
        logger.warning("Model training not yet implemented (Phase 6)")
        logger.info("=== Stage 4 skipped ===\n")

    @staticmethod
    def generate_forecast(forecast_date: str | None = None):
        """
        Generate price forecasts.

        To be implemented in Phase 7.

        Args:
            forecast_date: Date to forecast (YYYY-MM-DD), defaults today
        """
        forecast_date = forecast_date or datetime.now().strftime("%Y-%m-%d")

        logger.info("=== Stage 5: Generating forecast ===")
        logger.info(f"Forecast date: {forecast_date}")
        logger.warning("Forecast generation not yet implemented (Phase 7)")
        logger.info("=== Stage 5 skipped ===\n")

    def run_full_pipeline(self, start_date: str, end_date: str, forecast_date: str | None = None):
        """
        Run the complete pipeline from data fetching to forecasting.

        Args:
            start_date: Historical data start date (YYYY-MM-DD)
            end_date: Historical data end date (YYYY-MM-DD)
            forecast_date: Date to forecast (YYYY-MM-DD), defaults today
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING FULL PIPELINE FOR {self.country_code}")
        logger.info(f"{'='*70}")
        logger.info(f"Historical period: {start_date} to {end_date}")
        logger.info(f"Run ID: {self.run_id}\n")

        try:
            # Stage 1: Fetch data
            self.fetch_data(start_date, end_date)

            # Stage 2: Clean and verify
            self.clean_and_verify()

            # Stage 3: Engineer features
            self.engineer_features()

            # Stage 4: Train model
            self.train_model()

            # Stage 5: Generate forecast
            self.generate_forecast(forecast_date)

            logger.info(f"{'='*70}")
            logger.info(f"PIPELINE COMPLETE FOR {self.country_code}")
            logger.info(f"{'='*70}\n")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def get_info(self) -> dict:
        """
        Get information about the pipeline and data.

        Returns:
            Dictionary with pipeline information
        """
        return {
            "country_code": self.country_code,
            "country_name": self.config.country_name,
            "timezone": self.config.timezone,
            "run_id": self.run_id,
            "data_directory": str(self.data_manager.base_path),
            "data_info": self.data_manager.get_directory_info(),
        }

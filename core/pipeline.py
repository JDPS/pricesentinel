# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Country-agnostic pipelines orchestration.

This module provides the main Pipeline class that orchestrates all stages
of the forecasting workflow while remaining completely country-agnostic.
"""

import logging
from datetime import date, datetime

from config.country_registry import ConfigLoader, FetcherFactory
from core.cleaning import DataCleaner
from core.data_manager import CountryDataManager
from core.features import FeatureEngineer
from models import DEFAULT_MODEL_NAME, get_trainer

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
        self.cleaner = DataCleaner(self.data_manager, self.country_code)
        self.feature_engineer = FeatureEngineer(self.country_code)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._last_start_date: str | None = None
        self._last_end_date: str | None = None

        # Ensure directories exist
        self.data_manager.create_directories()

        logger.info(f"Initialized pipeline for {self.country_code} (run_id: {self.run_id})")

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """
        Validate date strings and logical ordering.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Raises:
            ValueError: If format is invalid or start_date > end_date.
        """
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
        except ValueError as exc:
            raise ValueError(
                "Invalid date format. Expected YYYY-MM-DD for start_date and end_date"
            ) from exc

        if start_dt > end_dt:
            raise ValueError("start_date must be before or equal to end_date")

    def fetch_data(self, start_date: str, end_date: str) -> None:
        """
        Fetch all required data sources.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self._validate_dates(start_date, end_date)
        self._last_start_date = start_date
        self._last_end_date = end_date

        logger.info(f"=== Stage 1: Fetching data for {self.country_code} ===")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Fetch electricity prices
        logger.info("Fetching electricity prices...")
        self._fetch_and_store(
            fetch_fn=self.fetchers["electricity"].fetch_prices,
            dataset_key="electricity",
            filename_prefix="electricity_prices",
            start_date=start_date,
            end_date=end_date,
            empty_msg="No electricity price data fetched",
            success_msg="Saved {count} electricity price records",
        )

        # Fetch electricity load
        logger.info("Fetching electricity load...")
        self._fetch_and_store(
            fetch_fn=self.fetchers["electricity"].fetch_load,
            dataset_key="electricity",
            filename_prefix="electricity_load",
            start_date=start_date,
            end_date=end_date,
            empty_msg="No electricity load data fetched",
            success_msg="Saved {count} electricity load records",
        )

        # Fetch weather data
        logger.info("Fetching weather data...")
        self._fetch_and_store(
            fetch_fn=self.fetchers["weather"].fetch_weather,
            dataset_key="weather",
            filename_prefix="weather",
            start_date=start_date,
            end_date=end_date,
            empty_msg="No weather data fetched",
            success_msg="Saved {count} weather records",
        )

        # Fetch gas prices
        logger.info("Fetching gas prices...")
        self._fetch_and_store(
            fetch_fn=self.fetchers["gas"].fetch_prices,
            dataset_key="gas",
            filename_prefix="gas_prices",
            start_date=start_date,
            end_date=end_date,
            empty_msg="No gas price data fetched",
            success_msg="Saved {count} gas price records",
        )

        # Fetch events
        logger.info("Fetching holidays and events...")
        try:
            holidays_df = self.fetchers["events"].get_holidays(start_date, end_date)

            if len(holidays_df) > 0:
                holidays_path = self.data_manager.get_events_path() / "holidays.csv"
                holidays_df.to_csv(holidays_path, index=False)
                logger.info("Saved %d holidays", len(holidays_df))
            else:
                logger.info("No holidays in date range")

            manual_events_df = self.fetchers["events"].get_manual_events()
            if len(manual_events_df) > 0:
                manual_path = self.data_manager.get_events_path() / "manual_events.csv"
                manual_events_df.to_csv(manual_path, index=False)
                logger.info("Loaded %d manual events", len(manual_events_df))

        except (FileNotFoundError, ValueError) as exc:
            # Expected errors: missing files or invalid data formats.
            logger.error("Failed to fetch events for %s: %s", self.country_code, exc)
        except Exception:
            # Unexpected errors should surface for debugging rather than be swallowed.
            logger.exception("Unexpected error while fetching events for %s", self.country_code)
            raise

    def _fetch_and_store(
        self,
        fetch_fn,
        dataset_key: str,
        filename_prefix: str,
        start_date: str,
        end_date: str,
        empty_msg: str,
        success_msg: str,
    ) -> None:
        """
        Fetches data using the specified fetch function and stores it in a file.

        This method uses the provided fetch function to retrieve data between the given
        start and end dates. If data is retrieved successfully and is not empty, it
        saves the data to a CSV file at the specified location. Informational or error
        messages are logged based on the outcome.
        """
        try:
            df = fetch_fn(start_date, end_date)
            if len(df) > 0:
                filename = self.data_manager.generate_filename(
                    filename_prefix, start_date, end_date
                )
                output_path = self.data_manager.get_raw_path(dataset_key) / filename
                df.to_csv(output_path, index=False)
                logger.info(success_msg.format(count=len(df)))
            else:
                logger.warning(empty_msg)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to fetch %s for %s: %s", filename_prefix, self.country_code, exc)
        except Exception:
            logger.exception(
                "Unexpected error while fetching %s for %s", filename_prefix, self.country_code
            )
            raise

    def _ensure_dates_set(self) -> tuple[str, str]:
        """
        Ensure that start and end dates are known for downstream stages.

        Returns:
            Tuple with (start_date, end_date).

        Raises:
            ValueError: If dates have not been set by fetch_data.
        """
        if not self._last_start_date or not self._last_end_date:
            raise ValueError(
                "Date range not set. Call fetch_data(start_date, end_date) first "
                "or provide dates explicitly to each stage."
            )
        return self._last_start_date, self._last_end_date

    def clean_and_verify(self, start_date: str | None = None, end_date: str | None = None) -> None:
        """
        Clean and verify data quality.

        Currently focuses on electricity, weather, gas, and events for the
        specified date range.
        """
        logger.info("=== Stage 2: Cleaning and verifying data ===")

        if start_date is None or end_date is None:
            start_date, end_date = self._ensure_dates_set()

        self._validate_dates(start_date, end_date)

        self.cleaner.clean_electricity(start_date, end_date)
        self.cleaner.clean_weather(start_date, end_date)
        self.cleaner.clean_gas(start_date, end_date)
        self.cleaner.clean_events(start_date, end_date)

        logger.info("=== Stage 2 complete ===\n")

    def engineer_features(self, start_date: str | None = None, end_date: str | None = None) -> None:
        """
        Generate features for modelling.

        Builds feature matrices from cleaned data and stores them under the
        processed data directory.
        """
        logger.info("=== Stage 3: Engineering features ===")

        if start_date is None or end_date is None:
            start_date, end_date = self._ensure_dates_set()

        self._validate_dates(start_date, end_date)

        self.feature_engineer.build_electricity_features(self.data_manager, start_date, end_date)

        logger.info("=== Stage 3 complete ===\n")

    def train_model(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        Train forecasting model.

        Uses engineered features to fit a baseline model and saves the
        trained artefact and basic metrics.
        """
        logger.info("=== Stage 4: Training model ===")

        if start_date is None or end_date is None:
            start_date, end_date = self._ensure_dates_set()

        self._validate_dates(start_date, end_date)

        trainer = get_trainer(self.country_code, model_name=model_name)
        self.feature_engineer.train_with_trainer(
            trainer=trainer,
            data_manager=self.data_manager,
            country_code=self.country_code,
            run_id=self.run_id,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info("=== Stage 4 complete ===\n")

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
            self.clean_and_verify(start_date, end_date)

            # Stage 3: Engineer features
            self.engineer_features(start_date, end_date)

            # Stage 4: Train model
            self.train_model(start_date, end_date)

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

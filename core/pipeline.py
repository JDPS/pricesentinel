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
from typing import Any

import pandas as pd

from config.country_registry import CountryConfig
from core.cleaning import DataCleaner
from core.data_manager import CountryDataManager
from core.exceptions import DateRangeError, PriceSentinelError
from core.features import FeatureEngineer
from core.repository import DataRepository
from core.stages.fetch_stage import DataFetchStage
from core.types import PipelineInfo
from core.verification import DataVerifier
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

    def __init__(
        self,
        country_code: str,
        config: CountryConfig,
        data_manager: CountryDataManager,
        cleaner: DataCleaner,
        feature_engineer: FeatureEngineer,
        fetch_stage: DataFetchStage,
        verifier: DataVerifier,
        repository: DataRepository,
        model_registry: Any = None,  # Avoid circular import, initialized in builder
        runtime_guard: Any = None,  # Avoid circular
    ):
        """
        Initialise the pipeline with dependencies.

        Args:
            country_code: ISO 3166-1 alpha-2 code
            config: Loaded country configuration
            data_manager: Initialized data manager
            cleaner: Initialized data cleaner
            feature_engineer: Initialized feature engineer
            fetch_stage: Initialized data fetch stage
            verifier: Initialized data verifier
            repository: Initialized data repository
            model_registry: Initialized model registry
        """
        self.country_code = country_code.upper()
        self.config = config
        self.data_manager = data_manager
        self.cleaner = cleaner
        self.feature_engineer = feature_engineer
        self.fetch_stage = fetch_stage
        self.verifier = verifier
        self.repository = repository

        # Lazy import to avoid circular dependency if strictly typed,
        # or rely on builder to pass it.
        # Since we just use it in methods, storing Any is safe for runtime.
        from models.model_registry import ModelRegistry

        self.model_registry = model_registry or ModelRegistry()

        from core.guards import RuntimeGuard

        self.runtime_guard = runtime_guard or RuntimeGuard()

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._last_start_date: str | None = None
        self._last_end_date: str | None = None

        logger.info(f"Initialized pipeline for {self.country_code} (run_id: {self.run_id})")

    @staticmethod
    def _validate_dates(start_date: str, end_date: str) -> None:
        """
        Validate date strings and logical ordering.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Raises:
            ValueError: If a format is invalid or start_date > end_date.
        """
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
        except ValueError as exc:
            raise DateRangeError(
                "Invalid date format. Expected YYYY-MM-DD for start_date and end_date"
            ) from exc

        if start_dt > end_dt:
            raise DateRangeError("start_date must be before or equal to end_date")

    async def fetch_data(self, start_date: str, end_date: str) -> None:
        """
        Fetch all required data sources concurrently.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self._validate_dates(start_date, end_date)
        self._last_start_date = start_date
        self._last_end_date = end_date

        await self.fetch_stage.run(start_date, end_date)

    def _ensure_dates_set(self) -> tuple[str, str]:
        """
        Ensure that start and end dates are known for downstream stages.

        Returns:
            Tuple with (start_date, end_date).

        Raises:
            ValueError: If dates have not been set by fetch_data.
        """
        if not self._last_start_date or not self._last_end_date:
            raise DateRangeError(
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

        # Verification
        try:
            # Use repository to load data
            prices_df = self.repository.load_data(
                "electricity_prices_clean", start_date, end_date, source="processed"
            )
            load_df = self.repository.load_data(
                "electricity_load_clean", start_date, end_date, source="processed"
            )

            self.verifier.verify_electricity(prices_df, load_df)
        except (ValueError, KeyError) as e:
            logger.warning(f"Verification failed: {e}")

        logger.info("=== Stage 2 complete ===\n")

    def engineer_features(self, start_date: str | None = None, end_date: str | None = None) -> None:
        """
        Generate features for modelling.

        Builds feature matrices from cleaned data and store them under the
        processed data directory.
        """
        logger.info("=== Stage 3: Engineering features ===")

        if start_date is None or end_date is None:
            start_date, end_date = self._ensure_dates_set()

        self._validate_dates(start_date, end_date)

        self.feature_engineer.build_electricity_features(start_date, end_date)

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
            run_id=self.run_id,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info("=== Stage 4 complete ===\n")

    def generate_forecast(
        self, forecast_date: str | None = None, model_name: str = DEFAULT_MODEL_NAME
    ) -> None:
        """
        Generates electricity price forecasts for a specified date using a pre-trained model
        and engineered features.

        Forecast generation involves locating the latest set of processed feature files,
         applying a trained model to produce
        predictions, and saving the results into a structured output file. If no suitable
         features or model are found, the process will be skipped.

        Parameters:
            forecast_date: str | None
                The specific date for which forecasts need to be generated.
                 Defaults to the current date if not provided.
            model_name: str
                Name of the model to be used for predictions.
                 Defaults to the global DEFAULT_MODEL_NAME constant.

        Raises:
            ValueError
                If required inputs or dependencies are incorrectly configured or missing.
        """
        forecast_date = forecast_date or datetime.now().strftime("%Y-%m-%d")
        resolved_model_name = self.model_registry.resolve_model_name(self.country_code, model_name)

        logger.info("=== Stage 5: Generating forecast ===")
        logger.info("Forecast date: %s", forecast_date)
        logger.info("Requested model: %s | Resolved model: %s", model_name, resolved_model_name)

        # Find latest engineered feature file for this country
        pattern = f"{self.country_code}_electricity_features_*.csv"
        feature_files = self.repository.list_processed_data(pattern)

        if not feature_files:
            logger.warning(
                "No electricity_features files found for %s; skipping forecast",
                self.country_code,
            )
            logger.info("=== Stage 5 skipped ===\n")
            return

        features_path = feature_files[0]
        logger.info("Using features file for forecast: %s", features_path)

        df = pd.read_csv(features_path, parse_dates=["timestamp"])
        if df.empty:
            logger.warning("Features file %s is empty; skipping forecast", features_path)
            logger.info("=== Stage 5 skipped ===\n")
            return

        # Apply Runtime Guards
        df = self.runtime_guard.validate_and_clamp(df)

        # Build prediction matrix (numeric features only, excluding target)
        target_col = "target_price"
        feature_cols = [c for c in df.columns if c not in ("timestamp", target_col)]
        x = df[feature_cols].select_dtypes(include="number")

        if x.empty or x.shape[1] == 0:
            logger.warning("No numeric feature columns available for forecast; skipping")
            logger.info("=== Stage 5 skipped ===\n")
            return

        if x.empty or x.shape[1] == 0:
            logger.warning("No numeric feature columns available for forecast; skipping")
            logger.info("=== Stage 5 skipped ===\n")
            return

        # Load model using Registry
        try:
            model, _ = self.model_registry.load_model(
                self.country_code, resolved_model_name, run_id=self.run_id
            )
            logger.info(f"Loaded model {resolved_model_name} for run_id {self.run_id}")
        except FileNotFoundError:
            logger.warning(
                "Model for run_id %s not found; attempting to use latest model",
                self.run_id,
            )
            try:
                model, meta = self.model_registry.load_model(self.country_code, resolved_model_name)
                logger.info(f"Using latest available model (run_id={meta.get('run_id')})")
            except FileNotFoundError:
                logger.warning(
                    "No trained models found for %s/%s; cannot generate forecast",
                    self.country_code,
                    resolved_model_name,
                )
                logger.info("=== Stage 5 skipped ===\n")
                return

        # Predict next-hour prices; align forecast timestamp as t+1h
        preds = model.predict(x)
        forecast_timestamp = df["timestamp"] + pd.Timedelta(hours=1)

        forecast_df = pd.DataFrame(
            {
                "forecast_timestamp": forecast_timestamp,
                "forecast_price_eur_mwh": preds,
                "source_features_file": features_path.name,
                "model_name": resolved_model_name,
                "run_id": self.run_id,
            }
        )

        # Filter by requested forecast date
        # Interpret forecast_date as a simple calendar date (no timezone semantics needed)
        forecast_date_dt = date.fromisoformat(forecast_date)
        mask = forecast_df["forecast_timestamp"].dt.date == forecast_date_dt
        filtered = forecast_df.loc[mask].copy()

        if filtered.empty:
            logger.warning(
                "No forecast rows for date %s in features file %s; writing empty forecast file",
                forecast_date,
                features_path.name,
            )

        out_name = (
            f"{self.country_code}_forecast_"
            f"{forecast_date_dt.strftime('%Y%m%d')}_{resolved_model_name}.csv"
        )
        out_path = self.repository.save_forecast(filtered, out_name)

        logger.info(
            "Saved %d forecast rows for %s to %s",
            len(filtered),
            self.country_code,
            out_path,
        )
        logger.info("=== Stage 5 complete ===\n")

    def generate_forecast_range(
        self,
        start_date: str,
        end_date: str,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        Generate forecasts for a range of calendar dates.

        This is a thin wrapper that iterates over dates and calls
        generate_forecast() for each one.
        """
        self._validate_dates(start_date, end_date)

        current = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)

        while current <= end_dt:
            self.generate_forecast(current.isoformat(), model_name=model_name)
            current = current.fromordinal(current.toordinal() + 1)

    async def run_full_pipeline(
        self,
        start_date: str,
        end_date: str,
        forecast_date: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        Run the complete pipeline from data fetching to forecasting.

        Args:
            start_date: Historical data start date (YYYY-MM-DD)
            end_date: Historical data end date (YYYY-MM-DD)
            forecast_date: Date to forecast (YYYY-MM-DD), defaults today
            model_name: Name of the model to be used for predictions,
             defaults to DEFAULT_MODEL_NAME
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"STARTING FULL PIPELINE FOR {self.country_code}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Historical period: {start_date} to {end_date}")
        logger.info(f"Run ID: {self.run_id}\n")

        try:
            # Stage 1: Fetch data
            await self.fetch_data(start_date, end_date)

            # Stage 2: Clean and verify
            self.clean_and_verify(start_date, end_date)

            # Stage 3: Engineer features
            self.engineer_features(start_date, end_date)

            # Stage 4: Train model
            self.train_model(start_date, end_date, model_name=model_name)

            # Stage 5: Generate forecast
            self.generate_forecast(forecast_date, model_name=model_name)

            logger.info(f"{'=' * 70}")
            logger.info(f"PIPELINE COMPLETE FOR {self.country_code}")
            logger.info(f"{'=' * 70}\n")

        except PriceSentinelError:
            raise
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
            raise PriceSentinelError(str(e)) from e

    def run_cross_validation(
        self, start_date: str, end_date: str, n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Run Time Series Cross-Validation.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_splits: Number of splits

        Returns:
            DataFrame with CV results
        """
        logger.info(f"=== Running Cross-Validation ({n_splits} splits) ===")
        self._validate_dates(start_date, end_date)

        # Local import to avoid circular dependency
        from core.cross_validation import CrossValidator

        cv = CrossValidator(self, n_splits=n_splits)
        results = cv.run(start_date, end_date)

        logger.info("=== Cross-Validation complete ===\n")
        return results

    def get_info(self) -> PipelineInfo:
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

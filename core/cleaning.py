# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Data cleaning and verification utilities.

This module contains a class for performing basic data cleaning and
verification on sector-specific datasets. It provides functionality to
prepare raw datasets for use in further processing and analysis. Operations
focus on deterministic cleaning such as file filtering, timestamp adjustment,
and duplicate removal.
"""

import logging

import pandas as pd

from core.data_manager import CountryDataManager
from core.repository import DataRepository

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaning and verification utilities.

    This class focuses on simple, deterministic cleaning steps that prepare
    raw datasets for feature engineering and training. It operates on files
    managed by CountryDataManager.
    """

    def __init__(
        self, data_manager: CountryDataManager, repository: DataRepository, country_code: str
    ):
        self.data_manager = data_manager
        self.repository = repository
        self.country_code = country_code

    def _load_latest_raw(
        self,
        source: str,
        filename_prefix: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        Load and merge all raw files for a source/prefix over a date range.

        This method scans all raw CSV files whose name matches the given
        prefix (e.g. ``*_{filename_prefix}_*.csv``), concatenates them,
        normalises timestamps to UTC, filters by the requested window, and
        removes duplicate timestamps.

        It is intentionally simple and deterministic rather than optimised
        for very large numbers of files.
        """
        # Use repository to load raw frames
        frames = self.repository.load_matching_raw(source, filename_prefix)

        if not frames:
            logger.info(
                "All raw files for source=%s, prefix=%s are unlikely to contain data after loading",
                source,
                filename_prefix,
            )
            return None

        df = pd.concat(frames, ignore_index=True)

        start = pd.to_datetime(start_date, utc=True)
        end = pd.to_datetime(end_date, utc=True)

        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df = df.loc[mask].copy()

        if df.empty:
            return None

        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df

    def clean_electricity(self, start_date: str, end_date: str) -> None:
        """
        Clean electricity prices and load data.
        """
        # Prices
        prices_df = self._load_latest_raw(
            source="electricity",
            filename_prefix="electricity_prices",
            start_date=start_date,
            end_date=end_date,
        )

        if prices_df is not None:
            path = self.repository.save_data(
                prices_df, "electricity_prices_clean", start_date, end_date
            )
            logger.info("Saved cleaned electricity prices to %s", path)
        else:
            logger.warning("No electricity price data to clean for %s", self.country_code)

        # Load
        load_df = self._load_latest_raw(
            source="electricity",
            filename_prefix="electricity_load",
            start_date=start_date,
            end_date=end_date,
        )

        if load_df is not None:
            path = self.repository.save_data(
                load_df, "electricity_load_clean", start_date, end_date
            )
            logger.info("Saved cleaned electricity load to %s", path)
        else:
            logger.warning("No electricity load data to clean for %s", self.country_code)

    def clean_weather(self, start_date: str, end_date: str) -> None:
        """
        Clean weather data.
        """
        weather_df = self._load_latest_raw(
            source="weather",
            filename_prefix="weather",
            start_date=start_date,
            end_date=end_date,
        )

        if weather_df is None:
            logger.warning("No weather data to clean for %s", self.country_code)
            return

        path = self.repository.save_data(weather_df, "weather_clean", start_date, end_date)
        logger.info("Saved cleaned weather data to %s", path)

    def clean_gas(self, start_date: str, end_date: str) -> None:
        """
        Clean gas price data.
        """
        gas_df = self._load_latest_raw(
            source="gas",
            filename_prefix="gas_prices",
            start_date=start_date,
            end_date=end_date,
        )

        if gas_df is None:
            logger.warning("No gas price data to clean for %s", self.country_code)
            return

        path = self.repository.save_data(gas_df, "gas_prices_clean", start_date, end_date)
        logger.info("Saved cleaned gas prices to %s", path)

    def clean_events(self, start_date: str, end_date: str) -> None:
        """
        Clean holidays and manual events.
        """
        events_path = self.data_manager.get_events_path()

        # Holidays
        holidays_path = events_path / "holidays.csv"
        if holidays_path.exists():
            self.holidays_events(holidays_path, start_date, end_date)
        else:
            logger.info("No holidays file found at %s", holidays_path)

        # Manual events
        manual_path = events_path / "manual_events.csv"
        if manual_path.exists():
            self.manual_events(manual_path, start_date, end_date)
        else:
            logger.info("No manual events file found at %s", manual_path)

    def holidays_events(self, holidays_path, start_date, end_date):
        holidays_df = pd.read_csv(holidays_path, parse_dates=["timestamp"])

        if not holidays_df.empty:
            if holidays_df["timestamp"].dt.tz is None:
                holidays_df["timestamp"] = holidays_df["timestamp"].dt.tz_localize("UTC")
            else:
                holidays_df["timestamp"] = holidays_df["timestamp"].dt.tz_convert("UTC")

            start = pd.to_datetime(start_date, utc=True)
            end = pd.to_datetime(end_date, utc=True)

            mask = (holidays_df["timestamp"] >= start) & (holidays_df["timestamp"] <= end)
            holidays_df = holidays_df.loc[mask].copy()

            if not holidays_df.empty:
                path = self.repository.save_data(
                    holidays_df.sort_values("timestamp"),
                    "holidays_clean",
                    start_date,
                    end_date,
                )
                logger.info("Saved cleaned holidays to %s", path)

    def manual_events(self, manual_path, start_date, end_date):
        events_df = pd.read_csv(manual_path, parse_dates=["date_start", "date_end"])

        if not events_df.empty:
            for col in ["date_start", "date_end"]:
                if events_df[col].dt.tz is None:
                    events_df[col] = events_df[col].dt.tz_localize("UTC")
                else:
                    events_df[col] = events_df[col].dt.tz_convert("UTC")

            start = pd.to_datetime(start_date, utc=True)
            end = pd.to_datetime(end_date, utc=True)

            mask = (events_df["date_end"] >= start) & (events_df["date_start"] <= end)
            events_df = events_df.loc[mask].copy()

            if not events_df.empty:
                path = self.repository.save_data(
                    events_df, "manual_events_clean", start_date, end_date
                )
                logger.info("Saved cleaned manual events to %s", path)

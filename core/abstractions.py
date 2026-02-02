# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Core abstract base classes for PriceSentinel.

This module defines the interfaces that all
country-specific data fetchers must implement.
These abstractions ensure country-agnostic pipeline code.
"""

from abc import ABC, abstractmethod

import pandas as pd


class ElectricityDataFetcher(ABC):
    """
    Abstract base class for electricity market data fetching.

    All country-specific electricity fetchers must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    async def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
                - timestamp: pd.DatetimeIndex (UTC timezone-aware)
                - price_eur_mwh: float (price in EUR/MWh)
                - market: str (e.g. 'day_ahead')
                - quality_flag: int (0=good, 1=interpolated, 2=missing)
        """
        pass

    @abstractmethod
    async def fetch_load(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch actual load/demand data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
                - timestamp: pd.DatetimeIndex (UTC timezone-aware)
                - load_mw: float (load in MW)
                - quality_flag: int (0=good, 1=interpolated, 2=missing)
        """
        pass

    async def fetch_generation(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """
        Fetch generation data by source (optional).

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with generation by source, or None if not available
        """
        return None


class WeatherDataFetcher(ABC):
    """
    Abstract base class for weather data fetching.

    All weather data fetchers must implement this interface.
    """

    @abstractmethod
    async def fetch_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data for configured locations.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
                - timestamp: pd.DatetimeIndex (UTC timezone-aware)
                - location_name: str (name of the location)
                - temperature_c: float (temperature in Celsius)
                - wind_speed_ms: float (wind speed in m/s)
                - solar_radiation_wm2: float (solar irradiance in W/m²)
                - precipitation_mm: float (precipitation in mm)
                - quality_flag: int (0=good, 1=interpolated, 2=missing)
        """
        pass


class GasDataFetcher(ABC):
    """
    Abstract base class for gas price data fetching.

    All gas price fetchers must implement this interface.
    """

    @abstractmethod
    async def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch gas hub prices.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
                - timestamp: pd.DatetimeIndex (UTC timezone-aware, daily frequency)
                - price_eur_mwh: float (gas price in EUR/MWh)
                - hub_name: str (name of the gas hub)
                - quality_flag: int (0=good, 1=interpolated, 2=missing)
        """
        pass


class EventDataProvider(ABC):
    """
    Abstract base class for event data providers.

    Provides holidays, DST transitions, and manually curated events.
    """

    @abstractmethod
    def get_holidays(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get national holidays for the country.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns:
                - timestamp: pd.DatetimeIndex (date of holiday)
                - event_type: str (always 'holiday')
                - description: str (name of the holiday)
        """
        pass

    @abstractmethod
    def get_manual_events(self) -> pd.DataFrame:
        """
        Get manually curated events (policy changes, outages, etc.).

        Returns:
            DataFrame with columns:
                - date_start: pd.DatetimeIndex (event start date)
                - date_end: pd.DatetimeIndex (event end date)
                - event_type: str (type of event)
                - description: str (event description)
                - source: str (source of information)
        """
        pass


# Standard output schema documentation
time_stamp = "pd.DatetimeIndex (UTC)"
ELECTRICITY_PRICE_SCHEMA = {
    "timestamp": time_stamp,
    "price_eur_mwh": "float",
    "market": "str",
    "quality_flag": "int (0=good, 1=interpolated, 2=missing)",
}

ELECTRICITY_LOAD_SCHEMA = {"timestamp": time_stamp, "load_mw": "float", "quality_flag": "int"}

WEATHER_SCHEMA = {
    "timestamp": time_stamp,
    "location_name": "str",
    "temperature_c": "float",
    "wind_speed_ms": "float",
    "solar_radiation_wm2": "float",
    "precipitation_mm": "float",
    "quality_flag": "int",
}

GAS_PRICE_SCHEMA = {
    "timestamp": time_stamp,
    "price_eur_mwh": "float",
    "hub_name": "str",
    "quality_flag": "int",
}

EVENT_SCHEMA = {"timestamp": "pd.DatetimeIndex", "event_type": "str", "description": "str"}

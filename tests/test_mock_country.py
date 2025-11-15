# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for mock country implementation.

This module tests that the mock country generates synthetic data
correctly and that the full pipeline works with mock data.
"""

import pandas as pd
import pytest

from config.country_registry import CountryRegistry, FetcherFactory
from data_fetchers.mock import register_mock_country


@pytest.fixture
def mock_country_setup():
    """Register a mock country for testing."""
    CountryRegistry.clear()
    register_mock_country()
    yield
    CountryRegistry.clear()


def test_mock_country_registered(mock_country_setup):
    """Test that mock country is registered."""
    assert CountryRegistry.is_registered("XX")
    assert "XX" in CountryRegistry.list_countries()


def test_mock_fetchers_creation(mock_country_setup):
    """Test creating fetchers for mock country."""
    fetchers = FetcherFactory.create_fetchers("XX")

    assert "electricity" in fetchers
    assert "weather" in fetchers
    assert "gas" in fetchers
    assert "events" in fetchers


def test_mock_electricity_prices(mock_country_setup):
    """Test mock electricity price generation."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["electricity"].fetch_prices("2024-01-01", "2024-01-07")

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "price_eur_mwh" in df.columns
    assert "market" in df.columns
    assert "quality_flag" in df.columns

    # Check data properties
    assert len(df) == 7 * 24  # 7 days * 24 hours
    assert df["price_eur_mwh"].min() > 0  # Prices are positive
    assert df["timestamp"].is_monotonic_increasing  # Timestamps are sorted
    assert all(df["market"] == "day_ahead")
    assert all(df["quality_flag"] == 0)


def test_mock_electricity_load(mock_country_setup):
    """Test mock electricity load generation."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["electricity"].fetch_load("2024-01-01", "2024-01-07")

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "load_mw" in df.columns
    assert len(df) == 7 * 24
    assert df["load_mw"].min() > 0


def test_mock_weather(mock_country_setup):
    """Test mock weather data generation."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["weather"].fetch_weather("2024-01-01", "2024-01-07")

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "location_name" in df.columns
    assert "temperature_c" in df.columns
    assert "wind_speed_ms" in df.columns
    assert "solar_radiation_wm2" in df.columns
    assert "precipitation_mm" in df.columns

    # Check physical constraints
    assert df["wind_speed_ms"].min() >= 0
    assert df["solar_radiation_wm2"].min() >= 0
    assert df["precipitation_mm"].min() >= 0


def test_mock_gas_prices(mock_country_setup):
    """Test mock gas price generation."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["gas"].fetch_prices("2024-01-01", "2024-01-07")

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "price_eur_mwh" in df.columns
    assert "hub_name" in df.columns
    assert len(df) == 7  # Daily data
    assert df["price_eur_mwh"].min() > 0


def test_mock_holidays(mock_country_setup):
    """Test mock holiday generation."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["events"].get_holidays("2024-01-01", "2024-12-31")

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "event_type" in df.columns
    assert "description" in df.columns
    assert len(df) > 0  # Should have some holidays
    assert all(df["event_type"] == "holiday")


def test_mock_manual_events(mock_country_setup):
    """Test mock manual events."""
    fetchers = FetcherFactory.create_fetchers("XX")

    df = fetchers["events"].get_manual_events()

    assert isinstance(df, pd.DataFrame)
    assert "date_start" in df.columns
    assert "date_end" in df.columns
    assert "event_type" in df.columns
    assert "description" in df.columns
    assert "source" in df.columns


def test_mock_data_reproducibility(mock_country_setup):
    """Test that mock data is reproducible (same seed)."""
    fetchers1 = FetcherFactory.create_fetchers("XX")
    fetchers2 = FetcherFactory.create_fetchers("XX")

    df1 = fetchers1["electricity"].fetch_prices("2024-01-01", "2024-01-07")
    df2 = fetchers2["electricity"].fetch_prices("2024-01-01", "2024-01-07")

    # Should be identical
    pd.testing.assert_frame_equal(df1, df2)

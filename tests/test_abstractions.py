# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for core abstractions.

This module tests that abstract base classes enforce proper implementation
and that subclasses must implement all required methods.
"""

import pytest

from core.abstractions import (
    ElectricityDataFetcher,
    EventDataProvider,
    GasDataFetcher,
    WeatherDataFetcher,
)


def test_abstract_electricity_fetcher_cannot_be_instantiated():
    """Verify ElectricityDataFetcher cannot be instantiated directly."""

    with pytest.raises(TypeError):
        ElectricityDataFetcher()


def test_abstract_weather_fetcher_cannot_be_instantiated():
    """Verify WeatherDataFetcher cannot be instantiated directly."""

    with pytest.raises(TypeError):
        WeatherDataFetcher()


def test_abstract_gas_fetcher_cannot_be_instantiated():
    """Verify GasDataFetcher cannot be instantiated directly."""

    with pytest.raises(TypeError):
        GasDataFetcher()


def test_abstract_event_provider_cannot_be_instantiated():
    """Verify EventDataProvider cannot be instantiated directly."""

    with pytest.raises(TypeError):
        EventDataProvider()


def test_incomplete_electricity_fetcher_fails():
    """Verify incomplete implementations fail."""

    class IncompleteElectricityFetcher(ElectricityDataFetcher):
        """Missing required methods"""

        pass

    with pytest.raises(TypeError):
        IncompleteElectricityFetcher()


def test_complete_electricity_fetcher_succeeds():
    """Verify complete implementation succeeds."""

    class CompleteElectricityFetcher(ElectricityDataFetcher):
        """Complete implementation"""

        def fetch_prices(self, start_date, end_date):
            return None

        def fetch_load(self, start_date, end_date):
            return None

    # Should not raise
    fetcher = CompleteElectricityFetcher()
    assert fetcher.fetch_prices(None, None) is None


def test_complete_weather_fetcher_succeeds():
    """Verify the complete weather fetcher implementation."""

    class CompleteWeatherFetcher(WeatherDataFetcher):
        """Complete implementation"""

        def fetch_weather(self, start_date, end_date):
            return None

    fetcher = CompleteWeatherFetcher()
    assert fetcher.fetch_weather(None, None) is None


def test_complete_gas_fetcher_succeeds():
    """Verify the complete gas fetcher implementation."""

    class CompleteGasFetcher(GasDataFetcher):
        """Complete implementation"""

        def fetch_prices(self, start_date, end_date):
            return None

    fetcher = CompleteGasFetcher()
    assert fetcher.fetch_prices(None, None) is None


def test_complete_event_provider_succeeds():
    """Verify complete event provider implementation."""

    class CompleteEventProvider(EventDataProvider):
        """Complete implementation"""

        def get_holidays(self, start_date, end_date):
            return None

        def get_manual_events(self):
            return None

    provider = CompleteEventProvider()
    assert provider.get_holidays(None, None) is None

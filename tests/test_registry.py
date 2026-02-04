# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for a country registry and factory pattern.

This module tests the CountryRegistry and FetcherFactory to ensure
proper registration and retrieval of country-specific adapters.
"""

import pytest

from config.country_registry import CountryRegistry
from core.exceptions import ConfigurationError, CountryNotRegisteredError
from data_fetchers.mock import (
    MockElectricityFetcher,
    MockEventProvider,
    MockGasFetcher,
    MockWeatherFetcher,
)


@pytest.fixture
def clean_registry():
    """Clear registry before and after each test."""
    CountryRegistry.clear()
    yield
    CountryRegistry.clear()


def test_register_and_retrieve_country(clean_registry):
    """Test registering and retrieving country adapters."""
    CountryRegistry.register(
        "TEST",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    adapters = CountryRegistry.get_adapters("TEST")

    assert "electricity" in adapters
    assert "weather" in adapters
    assert "gas" in adapters
    assert "events" in adapters
    assert adapters["electricity"] == MockElectricityFetcher


def test_list_countries(clean_registry):
    """Test listing registered countries."""
    CountryRegistry.register(
        "PT",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    CountryRegistry.register(
        "ES",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    countries = CountryRegistry.list_countries()

    assert "PT" in countries
    assert "ES" in countries
    assert len(countries) == 2


def test_missing_country_raises_error(clean_registry):
    """Test that requesting an unknown country raises error."""
    with pytest.raises(CountryNotRegisteredError):
        CountryRegistry.get_adapters("ZZ")


def test_is_registered(clean_registry):
    """Test checking if the country is registered."""
    CountryRegistry.register(
        "PT",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    assert CountryRegistry.is_registered("PT")
    assert not CountryRegistry.is_registered("ZZ")


def test_register_incomplete_adapters_fails(clean_registry):
    """Test that registering without all required adapters fails."""
    with pytest.raises(ConfigurationError):
        CountryRegistry.register(
            "BAD",
            {
                "electricity": MockElectricityFetcher,
                # Missing weather, gas, events
            },
        )


def test_clear_registry(clean_registry):
    """Test clearing the registry."""
    CountryRegistry.register(
        "PT",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    assert len(CountryRegistry.list_countries()) == 1

    CountryRegistry.clear()

    assert len(CountryRegistry.list_countries()) == 0

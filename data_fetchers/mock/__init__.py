# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock country implementation for testing abstraction layer.

This package provides synthetic data generators that implement the
standard fetcher interfaces, allowing the pipeline to be tested
without requiring real API access.
"""

import logging

from config.country_registry import CountryRegistry
from data_fetchers.mock.electricity import MockElectricityFetcher
from data_fetchers.mock.events import MockEventProvider
from data_fetchers.mock.gas import MockGasFetcher
from data_fetchers.mock.weather import MockWeatherFetcher

logger = logging.getLogger(__name__)


def register_mock_country():
    """
    Register a mock country (XX) in the country registry.

    This allows the mock country to be used for testing the entire
    pipeline without requiring real API credentials or data sources.

    Usage:
        >>> from data_fetchers.mock import register_mock_country
        >>> register_mock_country()
        >>> fetchers = FetcherFactory.create_fetchers('XX')
    """

    CountryRegistry.register(
        "XX",
        {
            "electricity": MockElectricityFetcher,
            "weather": MockWeatherFetcher,
            "gas": MockGasFetcher,
            "events": MockEventProvider,
        },
    )

    logger.debug("Registered mock country (XX)")


__all__ = [
    "MockElectricityFetcher",
    "MockWeatherFetcher",
    "MockGasFetcher",
    "MockEventProvider",
    "register_mock_country",
]

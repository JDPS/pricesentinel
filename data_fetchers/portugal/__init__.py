# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Portugal-specific data fetchers.

This package contains Portugal-specific implementations of data fetchers,
as well as the registration function for the country registry.
"""

import logging

from config.country_registry import CountryRegistry
from data_fetchers.portugal.electricity import PortugalElectricityFetcher
from data_fetchers.portugal.events import PortugalEventProvider
from data_fetchers.shared.open_meteo import OpenMeteoWeatherFetcher
from data_fetchers.shared.ttf_gas import TTFGasFetcher

logger = logging.getLogger(__name__)


def register_portugal():
    """
    Register Portugal's data fetchers in the country registry.

    This function maps Portugal (PT) to its specific fetcher implementations:
    - Electricity: ENTSO-E API (Portugal-specific)
    - Weather: Open-Meteo API (reusable across countries)
    - Gas: TTF gas prices (reusable for most EU countries)
    - Events: Portuguese holidays and manual events (Portugal-specific)

    Usage:
        >>> from data_fetchers.portugal import register_portugal
        >>> register_portugal()
        >>> fetchers = FetcherFactory.create_fetchers('PT')
    """
    CountryRegistry.register(
        "PT",
        {
            "electricity": PortugalElectricityFetcher,
            "weather": OpenMeteoWeatherFetcher,
            "gas": TTFGasFetcher,
            "events": PortugalEventProvider,
        },
    )

    logger.debug("Registered Portugal (PT) in country registry")


__all__ = ["PortugalElectricityFetcher", "PortugalEventProvider", "register_portugal"]

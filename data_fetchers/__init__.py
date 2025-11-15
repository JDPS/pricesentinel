# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Data fetchers for PriceSentinel.

This package contains country-specific data fetchers and the auto-registration
system for countries.
"""

import logging

from config.country_registry import CountryRegistry

logger = logging.getLogger(__name__)


def auto_register_countries():
    """
    Automatically register all implemented countries.

    This function should be called at application startup to ensure
    all countries are available in the registry.
    """
    logger.info("Auto-registering countries...")

    # Register a mock country for testing
    try:
        from data_fetchers.mock import register_mock_country

        register_mock_country()
        logger.info("Registered mock country (XX)")
    except ImportError as e:
        logger.warning(f"Could not register mock country: {e}")

    # Register Portugal
    try:
        from data_fetchers.portugal import register_portugal

        register_portugal()
        logger.info("Registered Portugal (PT)")
    except ImportError as e:
        logger.warning(f"Could not register Portugal: {e}")

    # Future countries will be registered here:
    # try:
    #     from data_fetchers.spain import register_spain
    #     register_spain()
    #     logger.info("Registered Spain (ES)")
    # except ImportError as e:
    #     logger.warning(f"Could not register Spain: {e}")

    registered = CountryRegistry.list_countries()
    logger.info(f"Registration complete. Available countries: {registered}")


__all__ = ["auto_register_countries"]

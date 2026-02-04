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


def auto_register_countries() -> None:
    """
    Automatically register all implemented countries.

    This function should be called at application startup to ensure
    all countries are available in the registry.
    """
    logger.info("Auto-registering countries...")

    # Register a mock country for testing
    try:
        from data_fetchers.mock import register_mock_country
    except ImportError as e:
        logger.warning(f"Could not register mock country: {e}")

        def register_mock_country():
            """
            Registers a mock country for testing purposes if the mock fetcher
            module is not available. If unavailable, this function performs
            no operation and logs a debugging message.

            Returns:
                None: This function does not return anything.
            """
            # Fallback no-op if mock fetcher is unavailable
            logger.debug("Mock country registration skipped; module unavailable.")
            return

        register_mock_country()  # type: ignore[no-untyped-call]
    else:
        register_mock_country()  # type: ignore[no-untyped-call]
        logger.info("Registered mock country (XX)")

    # Register Portugal
    try:
        from data_fetchers.portugal import register_portugal
    except ImportError as e:
        logger.warning(f"Could not register Portugal: {e}")

        def register_portugal():
            """
            Registers a mock country for testing purposes if the mock fetcher
            module is not available. If unavailable, this function performs
            no operation and logs a debugging message.

            Returns:
                None: This function does not return anything.
            """
            # Fallback no-op if mock fetcher is unavailable
            logger.debug("Portugal registration skipped; module unavailable.")
            return

        register_portugal()  # type: ignore[no-untyped-call]
    else:
        register_portugal()  # type: ignore[no-untyped-call]
        logger.info("Registered Portugal (PT)")

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

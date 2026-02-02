# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Core exception hierarchy for PriceSentinel.

This module defines the base exception class and specific error types
used throughout the application.
"""


class PriceSentinelError(Exception):
    """Base exception for all PriceSentinel errors."""

    pass


class ConfigurationError(PriceSentinelError):
    """Raised when configuration is invalid or missing."""

    pass


class CountryNotRegisteredError(ConfigurationError):
    """
    Raised when a requested country is not registered.

    Attributes:
        country_code: The requested country code
        available_countries: List of currently registered countries
    """

    def __init__(self, country_code: str, available_countries: list[str] | None = None):
        self.country_code = country_code
        self.available_countries = available_countries or []
        msg = f"Country '{country_code}' not registered."
        if self.available_countries:
            msg += f" Available: {self.available_countries}"
        super().__init__(msg)


class DataFetchError(PriceSentinelError):
    """Raised when data fetching fails."""

    pass


class DataValidationError(PriceSentinelError):
    """Raised when data fails validation checks."""

    pass

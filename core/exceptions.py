# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Custom exception hierarchy for PriceSentinel."""

from typing import Any


class PriceSentinelError(Exception):
    """Base exception for all PriceSentinel errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """
        Initialize PriceSentinel error.

        Args:
            message: Human-readable error message
            details: Optional dict with additional context
        """
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(PriceSentinelError):
    """Raised when configuration is invalid or missing."""

    pass


class DataFetchError(PriceSentinelError):
    """Raised when data fetching fails."""

    pass


class APIError(DataFetchError):
    """Raised when external API calls fail."""

    def __init__(
        self, message: str, status_code: int | None = None, response_body: str | None = None
    ):
        """
        Initialize API error.

        Args:
            message: Error description
            status_code: HTTP status code if available
            response_body: Raw response body for debugging
        """
        super().__init__(
            message, details={"status_code": status_code, "response_body": response_body}
        )
        self.status_code = status_code
        self.response_body = response_body


class DataValidationError(PriceSentinelError):
    """Raised when data validation fails."""

    pass


class DateRangeError(PriceSentinelError):
    """Raised when date range is invalid."""

    pass


class CountryNotRegisteredError(ConfigurationError):
    """Raised when country code is not registered."""

    def __init__(self, country_code: str, available_countries: list[str]):
        """
        Initialize country not registered error.

        Args:
            country_code: The unregistered country code
            available_countries: List of available country codes
        """
        message = (
            f"Country '{country_code}' is not registered.\n"
            f"Available countries: {', '.join(available_countries)}"
        )
        super().__init__(message, details={"country_code": country_code})
        self.country_code = country_code
        self.available_countries = available_countries


class DataQualityError(PriceSentinelError):
    """Raised when data quality checks fail."""

    def __init__(self, message: str, failed_checks: list[str] | None = None):
        """
        Initialize data quality error.

        Args:
            message: Error description
            failed_checks: List of failed quality check names
        """
        super().__init__(message, details={"failed_checks": failed_checks or []})
        self.failed_checks = failed_checks or []


class ModelError(PriceSentinelError):
    """Raised when model operations fail."""

    pass


class TrainingError(ModelError):
    """Raised when model training fails."""

    pass


class HyperparameterError(ModelError):
    """Raised when hyperparameter optimization fails."""

    pass


class EnsembleError(ModelError):
    """Raised when ensemble construction or prediction fails."""

    pass


class ForecastError(PriceSentinelError):
    """Raised when forecast generation fails."""

    pass

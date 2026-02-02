# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter registry for managing country-specific implementations."""

from typing import Any, Protocol, runtime_checkable

from core.abstractions import (
    ElectricityDataFetcher,
    EventDataProvider,
    GasDataFetcher,
    WeatherDataFetcher,
)
from core.exceptions import CountryNotRegisteredError


@runtime_checkable
class AdapterSet(Protocol):
    """Protocol defining required adapters for a country."""

    electricity: type[ElectricityDataFetcher]
    weather: type[WeatherDataFetcher]
    gas: type[GasDataFetcher]
    events: type[EventDataProvider]


class AdapterRegistry:
    """Instance-based registry for country adapters."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._adapters: dict[str, dict[str, type[Any]]] = {}

    def register(
        self,
        country_code: str,
        *,
        electricity: type[ElectricityDataFetcher],
        weather: type[WeatherDataFetcher],
        gas: type[GasDataFetcher],
        events: type[EventDataProvider],
    ) -> None:
        """
        Register adapters for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code
            electricity: Electricity data fetcher class
            weather: Weather data fetcher class
            gas: Gas data fetcher class
            events: Event provider class
        """
        self._adapters[country_code] = {
            "electricity": electricity,
            "weather": weather,
            "gas": gas,
            "events": events,
        }

    def get_adapters(self, country_code: str) -> dict[str, type[Any]]:
        """
        Get adapters for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code

        Returns:
            Dictionary of adapter classes

        Raises:
            CountryNotRegisteredError: If country not registered
        """
        if country_code not in self._adapters:
            # For now, pass empty list as available_countries to avoid circular imports.
            # Ideally we would fetch available countries from registry.
            raise CountryNotRegisteredError(country_code, list(self._adapters.keys()))
        return self._adapters[country_code]

    def is_registered(self, country_code: str) -> bool:
        """Check if country is registered."""
        return country_code in self._adapters

    def list_countries(self) -> list[str]:
        """Get list of registered country codes."""
        return list(self._adapters.keys())

    def unregister(self, country_code: str) -> None:
        """
        Unregister a country (primarily for testing).

        Args:
            country_code: ISO 3166-1 alpha-2 code
        """
        self._adapters.pop(country_code, None)

    def clear(self) -> None:
        """Clear all registrations (primarily for testing)."""
        self._adapters.clear()


# Global registry instance (for backward compatibility and default usage)
_default_registry = AdapterRegistry()


def get_default_registry() -> AdapterRegistry:
    """Get the default global registry instance."""
    return _default_registry


def create_registry() -> AdapterRegistry:
    """Create a new isolated registry instance (for testing)."""
    return AdapterRegistry()

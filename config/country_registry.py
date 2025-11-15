# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Country registry and factory pattern implementation.

This module provides the central registry for mapping country codes to
their specific data fetcher implementations, and a factory for creating
fetcher instances.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

configure_directory: str = "config/countries"


class CountryConfig:
    """
    Parsed country configuration.

    This class provides a convenient interface to access country-specific
    configuration parameters loaded from YAML files.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize CountryConfig from a dictionary.

        Args:
            config_dict: dictionary containing country configuration
        """
        self.country_code = config_dict["country_code"]
        self.country_name = config_dict.get("country_name", self.country_code)
        self.timezone = config_dict["timezone"]
        self.electricity_config = config_dict.get("electricity", {})
        self.weather_config = config_dict.get("weather", {})
        self.gas_config = config_dict.get("gas", {})
        self.events_config = config_dict.get("events", {})
        self.features_config = config_dict.get("features", {})

    @classmethod
    def from_yaml(
        cls, country_code: str, config_dir: str = "configure_directory"
    ) -> "CountryConfig":
        """
        Load country configuration from the YAML file.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g. 'PT')
            config_dir: Directory containing country configuration files

        Returns:
            CountryConfig instance

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        # Find project root (where the config directory is located)
        # This works whether running from project root or subdirectories
        current_file = Path(__file__).resolve()  # country_registry.py location
        project_root = current_file.parent.parent  # Go up two levels to the project root

        config_path = project_root / config_dir / f"{country_code}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a configuration file for country '{country_code}'"
            )

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["country_code", "timezone"]
        missing_fields = [field for field in required_fields if field not in config_dict]

        if missing_fields:
            raise ValueError(
                f"Configuration for {country_code} is missing required fields: {missing_fields}"
            )

        return cls(config_dict)

    def __repr__(self) -> str:
        return (
            f"CountryConfig(code={self.country_code}, name={self.country_name}, tz={self.timezone})"
        )


class CountryRegistry:
    """
    Registry mapping country codes to data fetcher implementations.

    This registry uses the adapter pattern to allow country-specific
    implementations while maintaining a consistent interface.
    """

    _registry: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, country_code: str, adapters: dict):
        """
        Register adapters for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code (e.g. 'PT')
            adapters: dictionary with keys:
                - 'electricity': ElectricityDataFetcher subclass
                - 'weather': WeatherDataFetcher subclass
                - 'gas': GasDataFetcher subclass
                - 'events': EventDataProvider subclass

        Example:
            >>> CountryRegistry.register('PT', {
            ...     'electricity': PortugalElectricityFetcher,
            ...     'weather': OpenMeteoWeatherFetcher,
            ...     'gas': TTFGasFetcher,
            ...     'events': PortugalEventProvider
            ... })
        """
        # Validate adapter types
        required_adapters = ["electricity", "weather", "gas", "events"]
        missing_adapters = [key for key in required_adapters if key not in adapters]

        if missing_adapters:
            raise ValueError(f"Missing required adapters for {country_code}: {missing_adapters}")

        cls._registry[country_code] = adapters
        logger.info(f"Registered country: {country_code}")

    @classmethod
    def get_adapters(cls, country_code: str) -> dict:
        """
        Retrieve registered adapters for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code

        Returns:
            dictionary of adapter classes

        Raises:
            ValueError: If the country is not registered
        """
        if country_code not in cls._registry:
            available = cls.list_countries()
            raise ValueError(
                f"Country '{country_code}' not registered.\n"
                f"Available countries: {available}\n"
                f"Please register the country first or check the country code."
            )

        return cls._registry[country_code]

    @classmethod
    def list_countries(cls) -> list:
        """
        List all registered countries.

        Returns:
            List of country codes
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, country_code: str) -> bool:
        """
        Check if a country is registered.

        Args:
            country_code: ISO 3166-1 alpha-2 code

        Returns:
            True if the country is registered, False otherwise
        """
        return country_code in cls._registry

    @classmethod
    def clear(cls):
        """
        Clear all registered countries (mainly for testing).
        """
        cls._registry.clear()
        logger.debug("Cleared country registry")


class FetcherFactory:
    """
    Factory for creating fetcher instances for a given country.

    This factory pattern allows the pipeline to remain country-agnostic
    while still using country-specific implementations.
    """

    @staticmethod
    def create_fetchers(country_code: str, config_dir: str = configure_directory) -> dict:
        """
        Create fetcher instances for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code
            config_dir: Directory containing country configuration files

        Returns:
            dictionary with initialised fetcher instances:
                - 'electricity': ElectricityDataFetcher instance
                - 'weather': WeatherDataFetcher instance
                - 'gas': GasDataFetcher instance
                - 'events': EventDataProvider instance

        Raises:
            ValueError: If the country is not registered,
            FileNotFoundError: If country configuration doesn't exist

        Example:
            >>> fetchers = FetcherFactory.create_fetchers('PT')
            >>> prices_df = fetchers['electricity'].fetch_prices('2024-01-01', '2024-01-31')
        """
        # Load country configuration
        config = CountryConfig.from_yaml(country_code, config_dir)

        # Get adapter classes
        adapters = CountryRegistry.get_adapters(country_code)

        # Instantiate fetchers with configuration
        try:
            fetchers = {
                "electricity": adapters["electricity"](config),
                "weather": adapters["weather"](config),
                "gas": adapters["gas"](config),
                "events": adapters["events"](config),
            }

            logger.info(f"Created fetchers for {country_code}")
            return fetchers

        except Exception as e:
            logger.error(f"Failed to create fetchers for {country_code}: {e}")
            raise


class ConfigLoader:
    """
    Utility class for loading country configurations.

    This provides a simple interface for loading configurations
    without needing to instantiate the full factory.
    """

    @staticmethod
    def load_country_config(
        country_code: str, config_dir: str = configure_directory
    ) -> CountryConfig:
        """
        Load and return country configuration.

        Args:
            country_code: ISO 3166-1 alpha-2 code
            config_dir: Directory containing country configuration files

        Returns:
            CountryConfig instance

        Raises:
            FileNotFoundError: If a configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        return CountryConfig.from_yaml(country_code, config_dir)

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration management for PriceSentinel.

This package handles country-specific configurations and the registry
of data fetcher adapters.
"""

from config.country_registry import (
    ConfigLoader,
    CountryConfig,
    CountryRegistry,
    FetcherFactory,
)

__all__ = ["CountryConfig", "CountryRegistry", "FetcherFactory", "ConfigLoader"]

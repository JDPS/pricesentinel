# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Core modules for PriceSentinel.

This package contains the core abstractions and utilities that are used
throughout the PriceSentinel system.
"""

from core.abstractions import (
    ElectricityDataFetcher,
    EventDataProvider,
    GasDataFetcher,
    WeatherDataFetcher,
)

__all__ = ["ElectricityDataFetcher", "WeatherDataFetcher", "GasDataFetcher", "EventDataProvider"]

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared data fetchers that are reusable across multiple countries.

These fetchers are country-agnostic and configured via country configuration files.
"""

from data_fetchers.shared.open_meteo import OpenMeteoWeatherFetcher

__all__ = ["OpenMeteoWeatherFetcher"]

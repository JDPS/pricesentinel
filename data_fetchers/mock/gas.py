# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock gas price data fetcher for testing.

This module generates synthetic gas hub prices with realistic patterns.
"""

import logging

import numpy as np
import pandas as pd

from core.abstractions import GasDataFetcher

logger = logging.getLogger(__name__)


class MockGasFetcher(GasDataFetcher):
    """
    Generates synthetic gas price data for testing.

    Creates realistic daily gas prices with:
    - Seasonal patterns (higher in winter)
    - Random variations
    - Occasional market shocks
    """

    def __init__(self, config):
        """
        Initialise mock gas fetcher.

        Args:
            config: CountryConfig instance
        """
        self.country_code = config.country_code
        self.hub_name = config.gas_config.get("hub_name", "MockHub")
        logger.debug(f"Initialized MockGasFetcher for {self.country_code}")

    def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate daily gas prices with realistic patterns.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic gas price data
        """
        # Create a daily timestamp range
        timestamps = pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC")

        # Use a dedicated RNG for reproducibility of gas prices
        rng = np.random.default_rng(46)

        # Base price (EUR/MWh)
        base_price = 30.0

        # A seasonal pattern: higher in winter (Dec-Feb), lower in summer
        day_of_year = timestamps.dayofyear
        # Peak in winter (day 1 and day 365), through in summer (day 180)
        seasonal_cycle = 15 * np.cos(2 * np.pi * (day_of_year - 1) / 365)

        # Random walk component (prices drift over time)
        random_walk = np.cumsum(rng.normal(0, 1, len(timestamps)))
        # Normalize random walk
        random_walk = (random_walk - random_walk.mean()) * 5

        # Daily noise
        daily_noise = rng.normal(0, 2, len(timestamps))

        # Occasional market shocks (2% of the time)
        shock_mask = rng.random(len(timestamps)) < 0.02
        shocks = np.where(shock_mask, rng.uniform(-10, 20, len(timestamps)), 0)

        # Combine all components
        prices = base_price + seasonal_cycle + random_walk + daily_noise + shocks

        # Ensure prices are positive and reasonable
        prices = np.maximum(prices, 5.0)
        prices = np.minimum(prices, 150.0)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price_eur_mwh": prices,
                "hub_name": self.hub_name,
                "quality_flag": 0,
            }
        )

        logger.info(
            f"Generated {len(df)} synthetic"
            f" gas price records from {start_date}"
            f" to {end_date}"
        )

        return df

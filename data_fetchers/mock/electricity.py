# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock electricity data fetcher for testing.

This module generates synthetic electricity price and the load data with
realistic patterns for testing the abstraction layer.
"""

import logging

import numpy as np
import pandas as pd

from core.abstractions import ElectricityDataFetcher

logger = logging.getLogger(__name__)


class MockElectricityFetcher(ElectricityDataFetcher):
    """
    Generates synthetic electricity price data for testing.

    This fetcher creates realistic-looking data with:
    - Daily cycles (higher prices during the day, lower at night)
    - Weekly patterns (lower prices on weekends)
    - Random variations
    - Occasional spikes
    """

    def __init__(self, config):
        """
        Initialise mock electricity fetcher.

        Args:
            config: CountryConfig instance (not used for mock,
             but required by an interface)
        """
        self.country_code = config.country_code
        logger.debug(f"Initialized MockElectricityFetcher for {self.country_code}")

    def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate hourly electricity prices with realistic patterns.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic price data
        """
        # Create an hourly timestamp range (inclusive of full end_date)
        # Add one day to end_date to include all 24 hours
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start_date, end=end_datetime, freq="1h", tz="UTC")

        np.random.seed(42)  # For reproducibility

        # Base price (EUR/MWh)
        base_price = 50.0

        # Daily cycle: higher during the day (8-20h), lower at night
        hour_of_day = timestamps.hour
        daily_cycle = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

        # Weekly cycle: lower on weekends
        day_of_week = timestamps.dayofweek
        weekend_discount = np.where(day_of_week >= 5, -10, 0)

        # Random noise
        noise = np.random.normal(0, 5, len(timestamps))

        # Occasional spikes (5% of the time)
        spike_mask = np.random.random(len(timestamps)) < 0.05
        spikes = np.where(spike_mask, np.random.uniform(20, 50, len(timestamps)), 0)

        # Combine all components
        prices = base_price + daily_cycle + weekend_discount + noise + spikes

        # Ensure prices are positive
        prices = np.maximum(prices, 5.0)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price_eur_mwh": prices,
                "market": "day_ahead",
                "quality_flag": 0,
            }
        )

        logger.info(
            f"Generated {len(df)} synthetic " f"price records from {start_date} to {end_date}"
        )

        return df

    def fetch_load(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate hourly load data with realistic patterns.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic load data
        """
        # Create an hourly timestamp range (inclusive of full end_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start_date, end=end_datetime, freq="1h", tz="UTC")

        np.random.seed(43)  # Different seed for the load

        # Base load (MW)
        base_load = 5000.0

        # Daily cycle: higher during the day
        hour_of_day = timestamps.hour
        daily_cycle = 2000 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

        # Weekly cycle: lower on weekends
        day_of_week = timestamps.dayofweek
        weekend_reduction = np.where(day_of_week >= 5, -500, 0)

        # Random noise
        noise = np.random.normal(0, 200, len(timestamps))

        # Combine all components
        load = base_load + daily_cycle + weekend_reduction + noise

        # Ensure load is reasonable
        load = np.maximum(load, 3000.0)
        load = np.minimum(load, 8000.0)

        df = pd.DataFrame({"timestamp": timestamps, "load_mw": load, "quality_flag": 0})

        logger.info(
            f"Generated {len(df)} synthetic load " f"records from {start_date} to {end_date}"
        )

        return df

    def fetch_generation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate synthetic generation data (optional feature).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic generation by source
        """
        # Create an hourly timestamp range (inclusive of full end_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start_date, end=end_datetime, freq="1h", tz="UTC")

        np.random.seed(44)

        # Simplified generation mix
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "solar_mw": np.maximum(0, 1000 * np.sin(2 * np.pi * timestamps.hour / 24)),
                "wind_mw": np.abs(np.random.normal(1500, 500, len(timestamps))),
                "hydro_mw": np.random.uniform(500, 1500, len(timestamps)),
                "gas_mw": np.random.uniform(1000, 3000, len(timestamps)),
                "quality_flag": 0,
            }
        )

        logger.info(f"Generated {len(df)} synthetic generation records")

        return df

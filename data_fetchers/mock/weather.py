# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock weather data fetcher for testing.

This module generates synthetic weather data with realistic patterns.
"""

import logging

import numpy as np
import pandas as pd

from core.abstractions import WeatherDataFetcher

logger = logging.getLogger(__name__)


class MockWeatherFetcher(WeatherDataFetcher):
    """
    Generates synthetic weather data for testing.

    Creates realistic patterns for:
    - Temperature (with daily and seasonal cycles)
    - Wind speed
    - Solar radiation (zero at night, peaks at noon)
    - Precipitation
    """

    def __init__(self, config):
        """
        Initialise mock weather fetcher.

        Args:
            config: CountryConfig instance
        """
        self.country_code = config.country_code
        # Use configured coordinates
        self.coordinates = config.weather_config.get(
            "coordinates", [{"name": "MockCity", "lat": 40.0, "lon": -8.0}]
        )
        logger.debug(f"Initialized MockWeatherFetcher for {self.country_code}")

    def fetch_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate hourly weather data with realistic patterns.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic weather data
        """
        # Create an hourly timestamp range (inclusive of full end_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start_date, end=end_datetime, freq="1h", tz="UTC")

        # Use a dedicated RNG for reproducibility of weather
        rng = np.random.default_rng(45)

        all_data = []

        for coord in self.coordinates:
            # Temperature (Celsius)
            base_temp = 15.0
            # Daily cycle: warmer during the day
            daily_temp_cycle = 8 * np.sin(2 * np.pi * (timestamps.hour - 6) / 24)
            # A seasonal cycle (simplified)
            day_of_year = timestamps.dayofyear
            seasonal_cycle = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            # Random variation
            temp_noise = rng.normal(0, 2, len(timestamps))
            temperature = base_temp + daily_temp_cycle + seasonal_cycle + temp_noise

            # Wind speed (m/s) - always positive, occasional gusts
            base_wind = 5.0
            wind_variation = np.abs(rng.normal(0, 3, len(timestamps)))
            # Occasional strong winds
            gust_mask = rng.random(len(timestamps)) < 0.05
            gusts = np.where(gust_mask, rng.uniform(10, 20, len(timestamps)), 0)
            wind_speed = base_wind + wind_variation + gusts
            wind_speed = np.maximum(wind_speed, 0)
            wind_speed = np.minimum(wind_speed, 25)  # Cap at 25 m/s

            # Solar radiation (W/m²) - zero at night, peaks at noon
            hour = timestamps.hour
            # Simple model: radiation only during daylight hours (6-18)
            solar_base = np.where(
                (hour >= 6) & (hour <= 18), 800 * np.sin(np.pi * (hour - 6) / 12), 0
            )
            # Cloud cover reduces radiation
            cloud_factor = rng.uniform(0.5, 1.0, len(timestamps))
            solar_radiation = solar_base * cloud_factor
            solar_radiation = np.maximum(solar_radiation, 0)
            solar_radiation = np.minimum(solar_radiation, 1200)

            # Precipitation (mm) - mostly zero, occasional rain
            precipitation = np.zeros(len(timestamps))
            rain_mask = rng.random(len(timestamps)) < 0.1  # 10% chance of rain
            precipitation[rain_mask] = rng.exponential(2, rain_mask.sum())
            precipitation = np.minimum(precipitation, 20)  # Cap at 20 mm

            location_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "location_name": coord["name"],
                    "temperature_c": temperature,
                    "wind_speed_ms": wind_speed,
                    "solar_radiation_wm2": solar_radiation,
                    "precipitation_mm": precipitation,
                    "quality_flag": 0,
                }
            )

            all_data.append(location_df)

        df = pd.concat(all_data, ignore_index=True)

        logger.info(
            f"Generated {len(df)} synthetic weather records from {start_date} to {end_date}"
        )

        return df

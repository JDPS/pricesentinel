# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic Open-Meteo weather data fetcher.

This fetcher is reusable across all countries that use Open-Meteo API.
It's country-agnostic and configured via country configuration files.
"""

import logging
from typing import Any

import httpx
import pandas as pd

from config.country_registry import CountryConfig
from core.abstractions import WeatherDataFetcher
from core.exceptions import APIError, ConfigurationError
from core.types import WeatherLocation

logger = logging.getLogger(__name__)


class OpenMeteoWeatherFetcher(WeatherDataFetcher):
    """
    Generic Open-Meteo weather fetcher (country-agnostic).

    Open-Meteo provides free weather data without requiring an API key.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config: CountryConfig):
        """
        Initialise Open-Meteo weather fetcher.

        Args:
            config: CountryConfig instance with weather coordinates

        Raises:
            ConfigurationError: If weather coordinates are missing
        """
        self.country_code = config.country_code
        self.coordinates = config.weather_config.get("coordinates", [])
        self.timezone = config.timezone

        if not self.coordinates:
            raise ConfigurationError(
                f"No weather coordinates configured for {self.country_code}",
                details={"country_code": self.country_code},
            )

        logger.debug(
            f"Initialized OpenMeteoWeatherFetcher for {self.country_code} "
            f"with {len(self.coordinates)} location(s)"
        )

    async def fetch_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data for configured locations.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weather data for all configured locations

        Raises:
            APIError: If API request fails
        """
        logger.info(
            f"Fetching weather data from {start_date} to {end_date} "
            f"for {len(self.coordinates)} location(s)"
        )

        all_data = []

        async with httpx.AsyncClient() as client:
            for coord in self.coordinates:
                try:
                    location_df = await self._fetch_location_weather(
                        client, coord, start_date, end_date
                    )
                    all_data.append(location_df)

                except APIError as e:
                    logger.error(f"Failed to fetch weather for {coord['name']}: {e}")
                    # Continue with other locations but log the specific API error

                except Exception as e:
                    logger.error(f"Unexpected error for {coord['name']}: {e}")
                    # Continue with other locations

        if not all_data:
            logger.warning("No weather data fetched for any location")
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "location_name",
                    "temperature_c",
                    "wind_speed_ms",
                    "solar_radiation_wm2",
                    "precipitation_mm",
                    "quality_flag",
                ]
            )

        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Fetched {len(df)} weather records")

        return df

    async def _fetch_location_weather(
        self, client: httpx.AsyncClient, coord: WeatherLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch weather data for a single location.

        Args:
            coord: WeatherLocation with 'name', 'lat', 'lon'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weather data for the location

        Raises:
            APIError: If HTTP request fails
        """
        params: dict[str, Any] = {
            "latitude": coord["lat"],
            "longitude": coord["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "windspeed_10m", "shortwave_radiation", "precipitation"],
            "timezone": "UTC",  # Always fetch in UTC
        }

        try:
            response = await client.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise APIError(
                f"Open-Meteo API error: {e.response.status_code}",
                status_code=e.response.status_code,
                response_body=e.response.text,
            ) from e
        except httpx.RequestError as e:
            raise APIError(f"Connection error to Open-Meteo: {str(e)}") from e

        data = response.json()

        if "hourly" not in data:
            raise APIError(
                "Invalid response format from Open-Meteo: missing 'hourly' key",
                response_body=str(data),
            )

        # Extract hourly data
        hourly = data["hourly"]

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"], utc=True),
                "location_name": coord["name"],
                "temperature_c": hourly["temperature_2m"],
                "wind_speed_ms": hourly["windspeed_10m"],
                "solar_radiation_wm2": hourly["shortwave_radiation"],
                "precipitation_mm": hourly["precipitation"],
                "quality_flag": 0,
            }
        )

        # Replace None values with NaN
        df = df.fillna(
            {
                "temperature_c": float("nan"),
                "wind_speed_ms": float("nan"),
                "solar_radiation_wm2": float("nan"),
                "precipitation_mm": 0.0,  # Missing precipitation = no rain
            }
        )

        logger.debug(f"Fetched {len(df)} weather records for {coord['name']}")

        return df

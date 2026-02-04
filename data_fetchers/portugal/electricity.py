# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Portugal electricity data fetcher using ENTSO-E Transparency Platform.

This module fetches day-ahead electricity prices and the load data for Portugal
from the ENTSO-E Transparency Platform API.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import httpx
import pandas as pd
import xmltodict

from core.abstractions import ElectricityDataFetcher
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class PortugalElectricityFetcher(ElectricityDataFetcher):
    """
    Fetches electricity data from ENTSO-E for Portugal.

    Requires ENTSOE_API_KEY environment variable to be set.
    """

    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    def __init__(self, config):
        """
        Initialise Portugal electricity fetcher.

        Args:
            config: CountryConfig instance

        Raises:
            ValueError: If ENTSOE_API_KEY is not set
        """
        self.country_code = config.country_code
        self.domain = config.electricity_config["entsoe_domain"]
        self.api_key = os.getenv("ENTSOE_API_KEY")

        if not self.api_key:
            raise ConfigurationError(
                "ENTSOE_API_KEY not found in environment variables.\n"
                "Please set it in your .env file or environment."
            )

        logger.debug(f"Initialized PortugalElectricityFetcher (domain: {self.domain})")

    async def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices for Portugal.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, price_eur_mwh, market, quality_flag

        Raises:
            httpx.RequestError: If API request fails
        """
        logger.info(f"Fetching electricity prices from {start_date} to {end_date}")

        # Convert dates to ENTSO-E format (YYYYMMDDhhmm)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        params = {
            "securityToken": self.api_key,
            "documentType": "A44",  # Day-ahead prices
            "in_Domain": self.domain,
            "out_Domain": self.domain,
            "periodStart": start_dt.strftime("%Y%m%d0000"),
            "periodEnd": end_dt.strftime("%Y%m%d0000"),
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

            # Parse XML response
            data = xmltodict.parse(response.content)

            # Extract time series
            df = self._parse_price_response(data)

            logger.info(f"Fetched {len(df)} price records")
            return df

        except httpx.RequestError as e:
            error_msg = str(e)
            logger.error(f"Failed to fetch electricity prices: {e}")

            # Provide helpful guidance for common errors
            if "400" in error_msg:
                logger.warning(
                    "ENTSO-E API returned 400 Bad Request. Common causes:\n"
                    "  1. Historical data too old (try more recent dates)\n"
                    "  2. Data not available for this domain/period\n"
                    "  3. API rate limiting\n"
                    f"  Requested period: {start_date} to {end_date}\n"
                    "  Suggestion: Try dates from the last 30 days"
                )
            elif "401" in error_msg or "403" in error_msg:
                logger.error(
                    "Authentication error. Please check_flag:\n"
                    "  1. ENTSOE_API_KEY is set correctly in .env\n"
                    "  2. API key is valid and not expired"
                )

            # Return empty DataFrame with the correct schema
            return pd.DataFrame(columns=["timestamp", "price_eur_mwh", "market", "quality_flag"])

    async def fetch_load(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch actual load data for Portugal.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, load_mw, quality_flag

        Raises:
            httpx.RequestError: If API request fails
        """
        logger.info(f"Fetching load data from {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        params = {
            "securityToken": self.api_key,
            "documentType": "A65",  # System total load
            "processType": "A16",  # Realised
            "outBiddingZone_Domain": self.domain,
            "periodStart": start_dt.strftime("%Y%m%d0000"),
            "periodEnd": end_dt.strftime("%Y%m%d0000"),
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

            data = xmltodict.parse(response.content)
            df = self._parse_load_response(data)

            logger.info(f"Fetched {len(df)} load records")
            return df

        except httpx.RequestError as e:
            error_msg = str(e)
            logger.error(f"Failed to fetch load data: {e}")

            # Provide helpful guidance for common errors
            if "400" in error_msg:
                logger.warning(
                    "ENTSO-E API returned 400 Bad Request for load data.\n"
                    f"  Requested period: {start_date} to {end_date}\n"
                    "  Suggestion: Try dates from the last 30 days"
                )

            return pd.DataFrame(columns=["timestamp", "load_mw", "quality_flag"])

    @staticmethod
    def _get_freq(f_resolution: str) -> str:
        if f_resolution == "PT15M":
            freq = "15min"
        else:
            freq = "1H"  # Default
        return freq

    def _parse_price_response(self, xml_data: dict[str, Any]) -> pd.DataFrame:
        """
        Parse ENTSO-E XML response for prices.

        Args:
            xml_data: Parsed XML dictionary

        Returns:
            DataFrame with price data
        """
        try:
            time_series = xml_data["Publication_MarketDocument"]["TimeSeries"]

            # Handle single time series vs multiple
            if not isinstance(time_series, list):
                time_series = [time_series]

            all_prices = []

            for ts in time_series:
                period = ts["Period"]

                # Handle multiple periods
                if not isinstance(period, list):
                    period = [period]

                for p in period:
                    start_time = pd.to_datetime(p["timeInterval"]["start"], utc=True)
                    resolution = p["resolution"]  # e.g. 'PT60M' for hourly

                    freq = self._get_freq(f_resolution=resolution)
                    logging.debug(f"freq: {freq}")

                    points = p["Point"]
                    if not isinstance(points, list):
                        points = [points]

                    for point in points:
                        position = int(point["position"]) - 1  # 0-indexed
                        price = float(point["price.amount"])

                        timestamp = start_time + pd.Timedelta(hours=position)

                        all_prices.append(
                            {
                                "timestamp": timestamp,
                                "price_eur_mwh": price,
                                "market": "day_ahead",
                                "quality_flag": 0,
                            }
                        )

            df = pd.DataFrame(all_prices)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Remove duplicates (keep first)
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

            return df

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse price response: {e}")
            return pd.DataFrame(columns=["timestamp", "price_eur_mwh", "market", "quality_flag"])

    def _parse_load_response(self, xml_data: dict[str, Any]) -> pd.DataFrame:
        """
        Parse ENTSO-E XML response for load data.

        Args:
            xml_data: Parsed XML dictionary

        Returns:
            DataFrame with load data
        """
        try:
            time_series = xml_data["GL_MarketDocument"]["TimeSeries"]

            if not isinstance(time_series, list):
                time_series = [time_series]

            all_loads = []

            for ts in time_series:
                period = ts["Period"]

                if not isinstance(period, list):
                    period = [period]

                for p in period:
                    start_time = pd.to_datetime(p["timeInterval"]["start"], utc=True)
                    resolution = p["resolution"]

                    freq = self._get_freq(f_resolution=resolution)
                    logging.debug(f"freq: {freq}")

                    points = p["Point"]
                    if not isinstance(points, list):
                        points = [points]

                    for point in points:
                        position = int(point["position"]) - 1
                        load = float(point["quantity"])

                        timestamp = start_time + pd.Timedelta(hours=position)

                        all_loads.append(
                            {"timestamp": timestamp, "load_mw": load, "quality_flag": 0}
                        )

            df = pd.DataFrame(all_loads)
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

            return df

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse load response: {e}")
            return pd.DataFrame(columns=["timestamp", "load_mw", "quality_flag"])

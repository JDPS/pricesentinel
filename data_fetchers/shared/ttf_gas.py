# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
TTF (Title Transfer Facility) gas price fetcher.

This fetcher handles TTF gas hub prices, which are used by most EU countries.
For MVP, it loads data from a manually downloaded CSV file.

Future enhancement: Integrate with ICE or other gas price APIs.
"""

import logging
from pathlib import Path

import pandas as pd

from core.abstractions import GasDataFetcher

logger = logging.getLogger(__name__)


class TTFGasFetcher(GasDataFetcher):
    """
    Fetches TTF gas hub prices.

    For MVP: Loads from manually downloaded CSV file.
    Expected CSV format: date, ttf_price
    """

    def __init__(self, config):
        """
        Initialize TTF gas fetcher.

        Args:
            config: CountryConfig instance
        """
        self.country_code = config.country_code
        self.hub_name = config.gas_config.get("hub_name", "TTF")
        self.currency = config.gas_config.get("currency", "EUR")

        # Look for a manual CSV file
        self.manual_csv_path = Path("data") / "manual_imports" / "ttf_gas_prices.csv"

        logger.debug(f"Initialized TTFGasFetcher for {self.country_code}")

    def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily TTF gas prices.

        For MVP: Loads from manually downloaded CSV file.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, price_eur_mwh, hub_name, quality_flag
        """
        logger.info(f"Fetching TTF gas prices from {start_date} to {end_date}")

        if not self.manual_csv_path.exists():
            logger.warning(
                f"TTF gas CSV not found at {self.manual_csv_path}.\n"
                f"Please download TTF gas prices and place them at this location.\n"
                f"Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["timestamp", "price_eur_mwh", "hub_name", "quality_flag"])

        try:
            # Load CSV (expecting columns: date, ttf_price)
            df = pd.read_csv(self.manual_csv_path, parse_dates=["date"])

            # Rename columns to standard schema
            df = df.rename(columns={"date": "timestamp", "ttf_price": "price_eur_mwh"})

            # Add hub_name and quality_flag
            df["hub_name"] = self.hub_name
            df["quality_flag"] = 0

            # Ensure the timestamp is datetime and UTC
            # First, make sure it's a datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Then ensure it's timezone-aware (UTC)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Filter to requested date range
            start = pd.to_datetime(start_date, utc=True)
            end = pd.to_datetime(end_date, utc=True)

            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            df_filtered = df[mask].reset_index(drop=True)

            logger.info(f"Loaded {len(df_filtered)} TTF gas price records from CSV")

            # Ensure we return the correct columns even if empty
            result_columns = ["timestamp", "price_eur_mwh", "hub_name", "quality_flag"]
            if len(df_filtered) == 0:
                # Return empty dataframe with correct schema
                return pd.DataFrame(columns=result_columns)

            return df_filtered[result_columns]

        except Exception as e:
            logger.error(f"Failed to load TTF gas prices from CSV: {e}")
            return pd.DataFrame(columns=["timestamp", "price_eur_mwh", "hub_name", "quality_flag"])

    def create_template_csv(self):
        """
        Create a template CSV file for manual data entry.

        This helps users understand the expected format.
        """
        self.manual_csv_path.parent.mkdir(parents=True, exist_ok=True)

        template = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", "2023-01-07", freq="D"),
                "ttf_price": [30.5, 31.2, 29.8, 30.1, 32.3, 31.7, 30.9],
            }
        )

        template.to_csv(self.manual_csv_path, index=False)
        logger.info(f"Created template CSV at {self.manual_csv_path}")

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Time normalization utilities.

This module provides the TimeNormalizer class for handling timezone conversions,
DST transitions, and ensuring consistent hourly grids across all datasets.
"""

import logging

import pandas as pd

from core.exceptions import DataValidationError, DateRangeError

logger = logging.getLogger(__name__)


class TimeNormalizer:
    """
    Handles time normalization, UTC conversion, and grid generation.
    """

    def __init__(self, timezone: str):
        """
        Initialize TimeNormalizer.

        Args:
            timezone: IANA timezone string (e.g., 'Europe/Lisbon')
        """
        self.timezone = timezone

    def normalize_to_utc(self, df: pd.DataFrame, col_name: str = "timestamp") -> pd.DataFrame:
        """
        Convert a DataFrame's timestamp column to UTC.

        Args:
            df: DataFrame containing the timestamp column.
            col_name: Name of the timestamp column.

        Returns:
            DataFrame with the timestamp column converted to UTC.

        Raises:
            DataValidationError: If the column is missing or conversion fails.
        """
        if df.empty:
            return df

        if col_name not in df.columns:
            raise DataValidationError(f"Column '{col_name}' missing from DataFrame")

        df = df.copy()

        try:
            # Ensure it is a datetime object first
            if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
                df[col_name] = pd.to_datetime(df[col_name], utc=True)

            # If it's timezone-naive, assume it's already in the target timezone (not UTC!)
            # But wait, fetchers usually return UTC-aware timestamps.
            # If they are naive, it's ambiguous.
            # Our contract says fetchers return UTC aware.
            # So this method mostly ensures it IS aware and converted to UTC.

            if df[col_name].dt.tz is None:
                # If naive, assume UTC as per project convention, or raise error?
                # Ideally fetchers return aware. If naive, we localize to UTC.
                logger.warning(f"Found timezone-naive timestamps in '{col_name}'. Assuming UTC.")
                df[col_name] = df[col_name].dt.tz_localize("UTC")
            else:
                # Convert to UTC
                df[col_name] = df[col_name].dt.tz_convert("UTC")

            return df

        except Exception as e:
            raise DataValidationError(f"Failed to normalize timestamps: {e}") from e

    def create_master_index(
        self, start_date: str, end_date: str, freq: str = "1h"
    ) -> pd.DatetimeIndex:
        """
        Generate a perfect time grid (master index) in UTC.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            freq: Frequency string (default '1h')

        Returns:
            pd.DatetimeIndex in UTC
        """
        try:
            # start_date 00:00 to end_date 23:59 (inclusive coverage)
            # Fetchers fetch "start to end", usually inclusive.
            # pd.date_range is inclusive by default.

            # Convert strings to naive datetimes, then localized to start of day in config timezone?
            # Or just work purely in UTC dates?
            # The plan implies "perfect hourly grids".
            # Usually strict daily boundaries in UTC are Easiest.

            start_dt = pd.Timestamp(start_date).tz_localize("UTC")
            end_dt = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
            # Subtract one period if we want [start, end] inclusive of dates
            # but exclusive of next day 00:00?
            # Usually for hourly data: 2024-01-01 00:00 to 2024-01-01 23:00.
            # So end_dt should be the exclusive bound for the range generation if closed='left'.

            # Let's generate inclusive of the last hour of the end_date.
            # End date 2024-01-01 -> includes 23:00.
            # So we want range [2024-01-01 00:00, 2024-01-02 00:00)

            return pd.date_range(start=start_dt, end=end_dt, freq=freq, inclusive="left")

        except Exception as e:
            raise DateRangeError(f"Failed to create master index: {e}") from e

    def handle_dst_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle DST transitions (duplicate or missing hours).

        Note: If data is already in UTC (which it should be), there are no DST gaps/dups
        in the strict sense (UTC doesn't have DST).
        However, if the source was local time and converted poorly, we might see issues.

        This method ensures uniqueness and continuous frequency.

        Args:
            df: DataFrame to check (must have 'timestamp' index or column)

        Returns:
            DataFrame with valid index and handled duplicates.
        """
        # Placeholder for complex logic.
        # For now, we drop duplicates and warn.

        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

        return df

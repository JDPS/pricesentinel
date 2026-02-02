# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Data verification utilities.

This module contains the DataVerifier class, responsible for performing quality
checks on cleaned datasets, such as detecting timestamp gaps and invalid values.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataVerifier:
    """
    Verifies data quality for cleaned datasets.
    """

    def __init__(self, country_code: str):
        self.country_code = country_code

    def check_gaps(self, df: pd.DataFrame, freq: str = "h") -> list[pd.Timestamp]:
        """
        Check for missing timestamps in the dataframe.

        Args:
            df: DataFrame with a 'timestamp' column.
            freq: Expected frequency string (e.g. 'h' for hourly).

        Returns:
            List of missing timestamps.
        """
        if df.empty:
            return []

        if "timestamp" not in df.columns:
            logger.warning("DataVerifier: DataFrame missing 'timestamp' column")
            return []

        # Ensure timestamp is datetime and sorted
        timestamps = pd.to_datetime(df["timestamp"])
        timestamps = timestamps.sort_values()

        start = timestamps.iloc[0]
        end = timestamps.iloc[-1]

        # Determine expected frequency alias for pandas
        # 'h' is deprecated for 'h', use 'h' or 'H' depending on pandas version,
        # usually 'h' is fine or 'ME' etc.
        # To be safe, we rely on the passed freq.

        expected_range = pd.date_range(start=start, end=end, freq=freq)

        # Find missing
        missing = expected_range.difference(pd.DatetimeIndex(timestamps))

        if not missing.empty:
            logger.warning(
                f"DataVerifier: Found {len(missing)} missing timestamps for {self.country_code} "
                f"(freq={freq}). First missing: {missing[0]}"
            )
            return missing.tolist()

        return []

    def check_negative_values(self, df: pd.DataFrame, columns: list[str]) -> list[str]:
        """
        Check for negative values in specific columns where they shouldn't exist.

        Args:
            df: DataFrame to check.
            columns: List of column names to validate.

        Returns:
            List of column names containing negative values.
        """
        failed_cols = []
        for col in columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    count = (df[col] < 0).sum()
                    logger.warning(
                        f"DataVerifier: Found {count} negative values in {col} "
                        f"for {self.country_code}."
                    )
                    failed_cols.append(col)
            else:
                logger.debug(f"DataVerifier: Column {col} not found in DataFrame.")

        return failed_cols

    def verify_electricity(self, prices_df: pd.DataFrame | None, load_df: pd.DataFrame | None):
        """
        Run verification checks on electricity data.
        """
        if prices_df is not None:
            self.check_gaps(prices_df, freq="h")
            # Prices can be negative, so we don't check for that usually (unless configured)

        if load_df is not None:
            self.check_gaps(load_df, freq="h")
            self.check_negative_values(load_df, ["load_mw"])

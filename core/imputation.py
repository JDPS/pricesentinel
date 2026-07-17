# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Time-series data imputation logic.

This module provides the TimeSeriesImputer for dealing with missing
timestamps and data gaps in energy time series (prices, load, weather).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeriesImputer:
    """
    Imputes missing values and reindexes time series to resolve gaps.
    """

    def __init__(self, country_code: str):
        self.country_code = country_code

    def impute_missing_timestamps(
        self, df: pd.DataFrame, time_col: str = "timestamp", freq: str = "h"
    ) -> pd.DataFrame:
        """
        Reindex DataFrame to ensure contiguous timestamps without gaps.
        """
        if df.empty or time_col not in df.columns:
            return df

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.sort_values(time_col).set_index(time_col)

        # Determine full range
        start = df.index.min()
        end = df.index.max()
        full_idx = pd.date_range(start=start, end=end, freq=freq)

        # Identify missing
        missing = full_idx.difference(pd.DatetimeIndex(df.index))
        if not missing.empty:
            logger.info(
                "TimeSeriesImputer: Reindexing %d missing timestamps for %s.",
                len(missing),
                self.country_code,
            )

        df = df.reindex(full_idx)
        df.index.name = time_col
        df = df.reset_index()

        return df

    def impute_column(self, df: pd.DataFrame, col: str, max_gap_hours: int = 3) -> pd.DataFrame:
        """
        Impute missing values in a column with linear interpolation (small gaps)
        or 7-day historical values (large gaps).
        Flags imputed rows with a boolean column.
        """
        if col not in df.columns:
            return df

        df = df.copy()
        missing_mask = df[col].isna()

        flag_col = f"is_imputed_{col}"
        df[flag_col] = missing_mask.astype("int8")

        if not missing_mask.any():
            return df

        logger.info(
            "TimeSeriesImputer: Imputing %d missing values in %s for %s.",
            missing_mask.sum(),
            col,
            self.country_code,
        )

        # 1. Linear interpolation for small gaps
        interpolated = df[col].interpolate(method="linear", limit=max_gap_hours)

        # 2. 7-day historical fallback for large gaps
        # Assumes df is regularly spaced! (freq="h" means 168 rows is 7 days)
        fallback = df[col].shift(168).fillna(df[col].shift(-168))

        # First fill with linear interpolation
        df[col] = interpolated

        # Fill remaining NaNs with fallback
        still_missing = df[col].isna()
        if still_missing.any():
            logger.info(
                "TimeSeriesImputer: Using 7-day fallback for %d values in %s.",
                still_missing.sum(),
                col,
            )
            df.loc[still_missing, col] = fallback[still_missing]

        # Absolute fallback: bfill/ffill for very edges
        final_missing = df[col].isna()
        if final_missing.any():
            logger.warning(
                "TimeSeriesImputer: %d values in %s still missing. Using fallback.",
                final_missing.sum(),
                col,
            )
            df[col] = df[col].bfill().ffill()

        return df

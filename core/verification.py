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

from core.types import ValidationLimits

logger = logging.getLogger(__name__)


class DataVerifier:
    """
    Verifies data quality for cleaned datasets.
    """

    def __init__(self, country_code: str, validation_config: ValidationLimits | None = None):
        self.country_code = country_code
        # Default limits if none provided
        self.validation_config: ValidationLimits = validation_config or {
            "price_min": -500.0,
            "price_max": 4000.0,
            "load_min": 0.0,
            "load_max": 100000.0,
        }

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

        expected_range = pd.date_range(start=start, end=end, freq=freq)

        # Find missing
        missing = expected_range.difference(pd.DatetimeIndex(timestamps))

        if not missing.empty:
            logger.warning(
                f"DataVerifier: Found {len(missing)} missing timestamps for {self.country_code} "
                f"(freq={freq}). First missing: {missing[0]}"
            )
            return list(missing)

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

    def check_limits(self, df: pd.DataFrame, col: str, min_val: float, max_val: float) -> bool:
        """
        Check if values in a column are within specified limits.

        Args:
            df: DataFrame to check.
            col: Column name.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            False if violations found, True otherwise.
        """
        if df.empty or col not in df.columns:
            return True

        violations = df[(df[col] < min_val) | (df[col] > max_val)]
        if not violations.empty:
            logger.warning(
                f"DataVerifier: Found {len(violations)} values outside limits "
                f"[{min_val}, {max_val}] in column '{col}' for {self.country_code}."
            )
            return False
        return True

    def clip_outliers(
        self, df: pd.DataFrame, col: str, min_val: float, max_val: float
    ) -> pd.DataFrame:
        """
        Clip values in a column to specified limits and flag clipped rows.
        """
        if df.empty or col not in df.columns:
            return df

        df = df.copy()

        # Identify violations
        lower_violations = df[col] < min_val
        upper_violations = df[col] > max_val
        violations_mask = lower_violations | upper_violations

        flag_col = f"is_clipped_{col}"
        df[flag_col] = violations_mask.astype("int8")

        if violations_mask.any():
            logger.warning(
                f"DataVerifier: Clipping {violations_mask.sum()} outliers in '{col}' "
                f"to [{min_val}, {max_val}] for {self.country_code}."
            )
            df[col] = df[col].clip(lower=min_val, upper=max_val)

        return df

    def verify_electricity(
        self, prices_df: pd.DataFrame | None, load_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Run verification checks on electricity data and clip outliers.
        Returns the potentially modified DataFrames.
        """
        if prices_df is not None:
            self.check_gaps(prices_df, freq="h")
            # Check price limits
            self.check_limits(
                prices_df,
                "price_eur_mwh",
                self.validation_config["price_min"],
                self.validation_config["price_max"],
            )
            prices_df = self.clip_outliers(
                prices_df,
                "price_eur_mwh",
                self.validation_config["price_min"],
                self.validation_config["price_max"],
            )

        if load_df is not None:
            self.check_gaps(load_df, freq="h")
            self.check_negative_values(load_df, ["load_mw"])
            # Check load limits
            self.check_limits(
                load_df,
                "load_mw",
                self.validation_config["load_min"],
                self.validation_config["load_max"],
            )
            load_df = self.clip_outliers(
                load_df,
                "load_mw",
                self.validation_config["load_min"],
                self.validation_config["load_max"],
            )

        return prices_df, load_df

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime guards for verifying and clamping input data during inference.
"""

import logging
from typing import TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


class RuntimeLimits(TypedDict, total=False):
    """Configuration for runtime limits."""

    price_max: float
    price_min: float
    load_max: float
    load_min: float


class RuntimeGuard:
    """
    Enforces runtime limits on feature data to prevent model extrapolation
    on wild inputs.
    """

    def __init__(self, limits: RuntimeLimits | None = None):
        self.limits = limits or {}

    def validate_and_clamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data against limits and clamp outliers.
        Returns a new DataFrame with clamped values.
        """
        if df.empty:
            return df

        df_guarded = df.copy()

        # Check Price Limits (if price is a feature, usually lags)
        self._clamp_column(df_guarded, "price_eur_mwh", "price_min", "price_max")

        # Check Load Limits
        self._clamp_column(df_guarded, "load_mw", "load_min", "load_max")

        # Check Lag Columns (dynamically)
        for col in df_guarded.columns:
            if col.startswith("price_lag_") or col.startswith("price_rolling_"):
                self._clamp_column(df_guarded, col, "price_min", "price_max")

        return df_guarded

    def _clamp_column(self, df: pd.DataFrame, col: str, min_key: str, max_key: str) -> None:
        """Helper to clamp a single column if it exists."""
        if col not in df.columns:
            return

        min_val = self.limits.get(min_key)
        max_val = self.limits.get(max_key)

        if min_val is not None:
            mask_min = df[col] < min_val
            if mask_min.any():
                logger.warning(
                    f"Guard: Clamping {mask_min.sum()} values in '{col}' to min {min_val}"
                )
                df.loc[mask_min, col] = min_val  # type: ignore

        if max_val is not None:
            mask_max = df[col] > max_val
            if mask_max.any():
                logger.warning(
                    f"Guard: Clamping {mask_max.sum()} values in '{col}' to max {max_val}"
                )
                df.loc[mask_max, col] = max_val  # type: ignore

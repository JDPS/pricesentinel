# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Feature engineering module for preparing data for training models.

This module includes functionality for loading cleaned datasets,
engineering features for electricity price forecasting, and training
models using a provided trainer. Designed specifically for country-based
datasets and supports additional exogenous features like an electricity load.

Classes:
    FeatureEngineer: Handles the feature engineering pipeline.
"""

import logging
import os
from collections.abc import Sequence

import pandas as pd

from core.repository import DataRepository
from models.base import BaseTrainer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for training models.

    Currently focuses on electricity price forecasting with an optional load
    as an exogenous variable.
    """

    def __init__(
        self,
        country_code: str,
        repository: DataRepository,
        features_config: dict | None = None,
    ):
        self.country_code = country_code
        self.repository = repository
        # Raw dict from CountryConfig.features_config; use simple boolean flags.
        self._features_config = features_config or {}

    def _is_enabled(self, flag: str, default: bool = True) -> bool:
        """
        Check if a feature flag is enabled in the configuration.

        Args:
            flag: Name of the flag in the features config dict
            default: Default value when the flag is not explicitly set
        """
        value = self._features_config.get(flag, default)
        return bool(value)

    def _load_cleaned(
        self,
        name: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        df = self.repository.load_data(name, start_date, end_date, source="processed")

        if df is None or df.empty:
            return None

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def build_electricity_features(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Build electricity price features for the given date range.
        """
        prices_df = self._load_cleaned("electricity_prices_clean", start_date, end_date)

        if prices_df is None:
            logger.warning(
                "Skipping feature engineering for %s: no cleaned electricity prices",
                self.country_code,
            )
            return

        load_df = self._load_cleaned("electricity_load_clean", start_date, end_date)

        df = prices_df.copy()

        # Target: next-hour price
        df["target_price"] = df["price_eur_mwh"].shift(-1)

        # Price lags
        lags: Sequence[int] = (1, 2, 24)
        for lag in lags:
            df[f"price_lag_{lag}"] = df["price_eur_mwh"].shift(lag)

        # Calendar features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Optional load feature
        if load_df is not None and "load_mw" in load_df.columns:
            load_small = load_df[["timestamp", "load_mw"]]
            df = df.merge(load_small, on="timestamp", how="left")

        # Drop rows with missing target or critical lags
        required_cols = ["target_price"] + [f"price_lag_{lag}" for lag in lags]
        df = df.dropna(subset=required_cols)

        if df.empty:
            logger.warning(
                "Feature DataFrame empty after dropping NaNs for %s; skipping save",
                self.country_code,
            )
            return

        # Optional weather features (aggregated by timestamp)
        if self._is_enabled("use_weather_features", True):
            weather_df = self._load_cleaned("weather_clean", start_date, end_date)
            if weather_df is not None:
                weather_agg = (
                    weather_df.groupby("timestamp")
                    .agg(
                        temperature_c=("temperature_c", "mean"),
                        wind_speed_ms=("wind_speed_ms", "mean"),
                        solar_radiation_wm2=("solar_radiation_wm2", "mean"),
                        precipitation_mm=("precipitation_mm", "sum"),
                    )
                    .reset_index()
                )
                df = df.merge(weather_agg, on="timestamp", how="left")

        # Optional gas price feature (daily, joined by date)
        if self._is_enabled("use_gas_features", True):
            gas_df = self._load_cleaned("gas_prices_clean", start_date, end_date)
            if gas_df is not None and "price_eur_mwh" in gas_df.columns:
                gas_df = gas_df.copy()
                gas_df["date"] = gas_df["timestamp"].dt.normalize()
                gas_daily = (
                    gas_df.sort_values("timestamp")
                    .drop_duplicates(subset=["date"], keep="last")[["date", "price_eur_mwh"]]
                    .rename(columns={"price_eur_mwh": "gas_price_eur_mwh"})
                )
                df["date"] = df["timestamp"].dt.normalize()
                df = df.merge(gas_daily, on="date", how="left")

        # Holiday and manual event flags (always present; data usage is configurable)
        df["is_holiday"] = 0
        df["is_event"] = 0

        if self._is_enabled("use_event_features", True):
            holidays_df = self._load_cleaned("holidays_clean", start_date, end_date)
            if holidays_df is not None:
                holidays_df = holidays_df.copy()
                holidays_df["date"] = holidays_df["timestamp"].dt.normalize()
                holiday_dates = set(holidays_df["date"])
                df_dates = df["timestamp"].dt.normalize()
                df.loc[df_dates.isin(holiday_dates), "is_holiday"] = 1

            # Manual events
            # Note: Using repository to load manual events as RAW processed data for now,
            # or rely on `load_cleaned` if we saved them as "manual_events_clean"
            events_df = self._load_cleaned("manual_events_clean", start_date, end_date)

            if events_df is not None and not events_df.empty:
                # Ensure date columns are parsed if repository didn't do it automatically enough
                # (Repository load_data returns DF, usually we expect basic parsing, but for
                # ranges we might need to check).
                # Re-parse if needed or assume cleaned data is good.
                # Since we saved it as Cleaned, let's assume it has proper dtypes or we cast.
                # Actually, CSV lost timezone info if not careful, but cleaning standardized it.
                if "date_start" in events_df.columns:
                    events_df["date_start"] = pd.to_datetime(events_df["date_start"], utc=True)
                    events_df["date_end"] = pd.to_datetime(events_df["date_end"], utc=True)

                    for _, row in events_df.iterrows():
                        mask = (df["timestamp"] >= row["date_start"]) & (
                            df["timestamp"] <= row["date_end"]
                        )
                        df.loc[mask, "is_event"] = 1

        # Ensure boolean flags are numeric for model training
        df["is_holiday"] = df["is_holiday"].astype("int8")
        df["is_event"] = df["is_event"].astype("int8")

        # Drop helper column if created
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        path = self.repository.save_data(df, "electricity_features", start_date, end_date)

        logger.info(
            "Saved electricity features for %s to %s (%d rows)",
            self.country_code,
            path,
            len(df),
        )

    def train_with_trainer(
        self,
        trainer: BaseTrainer,
        run_id: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Train the provided trainer using engineered electricity features.
        """
        df = self.repository.load_data(
            "electricity_features", start_date, end_date, source="processed"
        )

        if df is None:
            raise ValueError(
                f"Feature file not found for training (start={start_date}, end={end_date}). "
                f"Run engineer_features() first."
            )

        if df.empty:
            raise ValueError("Feature file is empty; cannot train model")

        if "target_price" not in df.columns:
            raise ValueError("Feature set missing 'target_price' column")

        # Build feature matrix and target vector
        target_col = "target_price"
        feature_cols = [c for c in df.columns if c not in ("timestamp", target_col)]

        # Use only numeric feature columns (exclude strings like 'market')
        x = df[feature_cols].select_dtypes(include="number")
        y = df[target_col]

        if x.empty or x.shape[1] == 0:
            raise ValueError("No numeric feature columns available for training")

        n_samples = len(df)
        if n_samples < 10:
            split_idx = max(1, n_samples // 2)
        else:
            split_idx = int(n_samples * 0.8)

        x_train = x.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        x_val = x.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        logger.info(
            "Training %s: %d samples, %d features (train=%d, val=%d)",
            self.country_code,
            n_samples,
            x.shape[1],
            len(x_train),
            len(x_val),
        )

        metrics = trainer.train(x_train, y_train, x_val, y_val)
        # Record the training data window for easier benchmarking/comparison.
        metrics["train_start_date"] = start_date
        metrics["train_end_date"] = end_date

        # A simple guardrail for obviously bad metrics
        train_mae = metrics.get("train_mae")
        train_rmse = metrics.get("train_rmse")
        bad_metrics = False

        if isinstance(train_mae, int | float) and train_mae > 1000:
            logger.warning(
                "High train MAE detected for %s (%.3f) – check data and feature config",
                self.country_code,
                train_mae,
            )
            bad_metrics = True

        if isinstance(train_rmse, int | float) and train_rmse > 1000:
            logger.warning(
                "High train RMSE detected for %s (%.3f) – check data and feature config",
                self.country_code,
                train_rmse,
            )
            bad_metrics = True

        skip_save_env = os.getenv("PRICESENTINEL_SKIP_SAVE_ON_BAD_METRICS", "0")
        skip_save = bad_metrics and skip_save_env == "1"

        if skip_save:
            logger.warning(
                "Skipping model save for %s (run_id=%s) due to poor metrics "
                "and PRICESENTINEL_SKIP_SAVE_ON_BAD_METRICS=1",
                self.country_code,
                run_id,
            )
        else:
            trainer.save(self.country_code, run_id, metrics=metrics)

        logger.info(
            "Training complete for %s (%d samples). Metrics: %s",
            self.country_code,
            n_samples,
            metrics,
        )

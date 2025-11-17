# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0


import logging
from collections.abc import Sequence

import pandas as pd

from core.data_manager import CountryDataManager
from models.base import BaseTrainer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for training models.

    Currently focuses on electricity price forecasting with optional load
    as an exogenous variable.
    """

    def __init__(self, country_code: str):
        self.country_code = country_code

    def _load_cleaned(
        self,
        data_manager: CountryDataManager,
        name: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        Load a cleaned dataset by logical name for a date range.
        """
        path = data_manager.get_processed_file_path(name, start_date, end_date)

        if not path.exists():
            logger.warning("Cleaned data file not found: %s", path)
            return None

        df = pd.read_csv(path, parse_dates=["timestamp"])

        if df.empty:
            logger.warning("Cleaned data file %s is empty", path)
            return None

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def build_electricity_features(
        self,
        data_manager: CountryDataManager,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Build electricity price features for the given date range.
        """
        prices_df = self._load_cleaned(
            data_manager, "electricity_prices_clean", start_date, end_date
        )

        if prices_df is None:
            logger.warning(
                "Skipping feature engineering for %s: no cleaned electricity prices",
                self.country_code,
            )
            return

        load_df = self._load_cleaned(data_manager, "electricity_load_clean", start_date, end_date)

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

        features_path = data_manager.get_processed_file_path(
            "electricity_features", start_date, end_date
        )
        features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(features_path, index=False)

        logger.info(
            "Saved electricity features for %s to %s (%d rows)",
            self.country_code,
            features_path,
            len(df),
        )

    def train_with_trainer(
        self,
        trainer: BaseTrainer,
        data_manager: CountryDataManager,
        country_code: str,
        run_id: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Train the provided trainer using engineered electricity features.
        """
        features_path = data_manager.get_processed_file_path(
            "electricity_features", start_date, end_date
        )

        if not features_path.exists():
            raise ValueError(
                f"Feature file not found for training: {features_path}. "
                f"Run engineer_features() first."
            )

        df = pd.read_csv(features_path)

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

        n_samples = len(df)
        if n_samples < 10:
            split_idx = max(1, n_samples // 2)
        else:
            split_idx = int(n_samples * 0.8)

        x_train = x.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        x_val = x.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        metrics = trainer.train(x_train, y_train, x_val, y_val)
        trainer.save(country_code, run_id, metrics=metrics)

        logger.info(
            "Training complete for %s (%d samples). Metrics: %s",
            country_code,
            n_samples,
            metrics,
        )

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Time Series Cross-Validation logic.

This module provides the CrossValidator class to perform robust model evaluation
using time-series aware splitting strategies.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# pylint: disable=cyclic-import
from core.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class FoldMetric:
    """Metrics for a single CV fold."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    mae: float
    rmse: float
    mae_naive: float


class CrossValidator:
    """
    Performs Time Series Cross-Validation on a Pipeline.
    """

    def __init__(self, pipeline: Pipeline, n_splits: int = 5):
        """
        Initialize CrossValidator.

        Args:
            pipeline: Configured Pipeline instance
            n_splits: Number of splits for TimeSeriesSplit
        """
        self.pipeline = pipeline
        self.n_splits = n_splits

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run Cross-Validation over the specified period.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame containing metrics for each fold
        """
        logger.info(
            "Starting %d-fold Time Series CV from %s to %s",
            self.n_splits,
            start_date,
            end_date,
        )

        # 1. Ensure features are engineered for the whole period
        repo = self.pipeline.repository

        self.pipeline.engineer_features(start_date, end_date)

        # Load the feature file
        pattern = f"{self.pipeline.country_code}_electricity_features_*.csv"
        feature_files = repo.list_processed_data(pattern)

        if not feature_files:
            raise FileNotFoundError(f"No feature files found for {self.pipeline.country_code}")

        # Use the latest file
        features_path = feature_files[0]
        df = pd.read_csv(features_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Apply Runtime Guards (mimic inference time safety)
        df = self.pipeline.runtime_guard.validate_and_clamp(df)

        # Filter strictly to the requested range
        mask = (df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) & (
            df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")
        )
        df = df[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError("No data available for the requested CV range")

        # Prepare X and y
        target_col = "target_price"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from features")

        # Drop rows where target is NaN (cannot train/test on them)
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

        feature_cols = [c for c in df_clean.columns if c not in ("timestamp", target_col)]
        # pylint: disable=invalid-name
        X = df_clean[feature_cols].select_dtypes(include="number")  # noqa: N806
        y = df_clean[target_col]
        timestamps = df_clean["timestamp"]

        logger.info("Data ready for CV: %d samples, %d features", len(df_clean), len(feature_cols))

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []

        # Get the trainer from the pipeline's country
        # We need to instantiate a fresh model for each fold
        # pylint: disable=import-outside-toplevel
        from models import get_trainer

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # noqa: N806
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            fold_train_start = timestamps.iloc[train_index].min()
            fold_train_end = timestamps.iloc[train_index].max()
            fold_test_start = timestamps.iloc[test_index].min()
            fold_test_end = timestamps.iloc[test_index].max()

            logger.info(
                "Fold %d/%d: Train [%s - %s], Test [%s - %s]",
                fold + 1,
                self.n_splits,
                fold_train_start,
                fold_train_end,
                fold_test_start,
                fold_test_end,
            )

            # Train
            trainer = get_trainer(self.pipeline.country_code)
            trainer.train(X_train, y_train)

            # Predict
            # BaseTrainer doesn't guarantee .model, but SklearnRegressorTrainer has it.
            # In a real app we might want a common Predictor interface.
            model = getattr(trainer, "model", None)
            if model is None or not hasattr(model, "predict"):
                raise AttributeError(
                    f"Trainer {type(trainer)} has no 'model' with 'predict' method"
                )

            y_pred = model.predict(X_test)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Naive baseline
            naive_col = "price_lag_24"
            if naive_col in X_test.columns:
                y_naive = X_test[naive_col]
                if y_naive.isnull().any():
                    # Fallback to mean of training as super-naive
                    y_naive = y_naive.fillna(y_train.mean())
                mae_naive = mean_absolute_error(y_test, y_naive)
            else:
                mae_naive = np.nan

            logger.info("Fold %d Result: MAE=%.4f, MAE_Naive=%.4f", fold + 1, mae, mae_naive)

            fold_results.append(
                {
                    "fold": fold + 1,
                    "train_start": fold_train_start,
                    "train_end": fold_train_end,
                    "test_start": fold_test_start,
                    "test_end": fold_test_end,
                    "mae": mae,
                    "rmse": rmse,
                    "mae_naive": mae_naive,
                    "improvement_pct": (
                        ((mae_naive - mae) / mae_naive * 100) if mae_naive > 0 else 0.0
                    ),
                }
            )

        results_df = pd.DataFrame(fold_results)
        return results_df

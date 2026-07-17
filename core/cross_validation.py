# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Time Series Cross-Validation logic.

This module provides the CrossValidator class to perform robust model evaluation
using time-series aware splitting strategies.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# pylint: disable=cyclic-import
from core.pipeline import Pipeline
from core.types import ModelConfig

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
        x_features, y, timestamps = self._prepare_data(start_date, end_date)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []

        # Get the trainer from the pipeline's country
        # We need to instantiate a fresh model for each fold
        # pylint: disable=import-outside-toplevel
        from models import get_trainer

        for fold, (train_index, test_index) in enumerate(tscv.split(x_features)):
            x_train, x_test = x_features.iloc[train_index], x_features.iloc[test_index]
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
            trainer.train(x_train, y_train)

            # Predict
            y_pred = trainer.predict(x_test)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Naive baseline
            naive_col = "price_lag_24"
            if naive_col in x_test.columns:
                y_naive = x_test[naive_col]
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

    def _prepare_data(
        self, start_date: str, end_date: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        repo = self.pipeline.repository
        self.pipeline.engineer_features(start_date, end_date)

        pattern = f"{self.pipeline.country_code}_electricity_features_*.csv"
        feature_files = repo.list_processed_data(pattern)

        if not feature_files:
            raise FileNotFoundError(f"No feature files found for {self.pipeline.country_code}")

        features_path = feature_files[0]
        df = pd.read_csv(features_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = self.pipeline.runtime_guard.validate_and_clamp(df)

        mask = (df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) & (
            df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")
        )
        df = df[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError("No data available for the requested CV range")

        target_col = "target_price"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from features")

        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

        feature_cols = [c for c in df_clean.columns if c not in ("timestamp", target_col)]
        x_features = df_clean[feature_cols].select_dtypes(include="number")
        y = df_clean[target_col]
        timestamps = df_clean["timestamp"]

        logger.info("Data ready: %d samples, %d features", len(df_clean), len(feature_cols))
        return x_features, y, timestamps

    def run_walk_forward(
        self,
        start_date: str,
        end_date: str,
        initial_train_size: int,
        step_size: int,
        model_name: str,
        model_config: ModelConfig | None = None,
        mode: str = "expanding",
    ) -> pd.DataFrame:
        """
        Run Walk-Forward Validation over the specified period.
        """
        logger.info("Starting Walk-Forward Validation from %s to %s", start_date, end_date)
        x_features, y, timestamps = self._prepare_data(start_date, end_date)

        # pylint: disable=import-outside-toplevel
        from models import get_trainer

        def trainer_factory() -> Any:
            return get_trainer(
                self.pipeline.country_code, model_name=model_name, config=model_config
            )

        validator = WalkForwardValidator(
            initial_train_size=initial_train_size, step_size=step_size, mode=mode
        )

        results_df = validator.run(x_features, y, trainer_factory)

        dates = []
        test_start = initial_train_size
        n_samples = len(x_features)
        while test_start < n_samples:
            dates.append(timestamps.iloc[test_start])
            test_start += step_size

        if len(dates) == len(results_df):
            results_df["test_date"] = dates

        return results_df


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.

    Simulates realistic model deployment by training on historical data and
    evaluating on the next unseen window, then stepping forward. Supports
    both expanding (growing training set) and sliding (fixed-size) window
    modes.

    Args:
        initial_train_size: Minimum number of samples in the first training
            window.
        step_size: Number of samples to advance on each step. Defaults to
            24 (one day of hourly data).
        mode: Window strategy -- ``"expanding"`` grows the training set each
            step; ``"sliding"`` keeps it fixed at *initial_train_size*.
    """

    def __init__(
        self,
        initial_train_size: int,
        step_size: int = 24,
        mode: str = "expanding",
    ) -> None:
        if initial_train_size < 1:
            raise ValueError("initial_train_size must be >= 1")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if mode not in ("expanding", "sliding"):
            raise ValueError(f"mode must be 'expanding' or 'sliding', got '{mode}'")

        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.mode = mode

    # pylint: disable=import-outside-toplevel
    def run(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        trainer_factory: Callable[[], Any],
    ) -> pd.DataFrame:
        """
        Execute walk-forward validation.

        For each step the validator:
        1. Creates a fresh trainer via *trainer_factory*.
        2. Trains on the current training window.
        3. Predicts on the next *step_size* samples.
        4. Records MAE and RMSE for the step.

        Args:
            x: Feature matrix (rows ordered chronologically).
            y: Target series aligned with *x*.
            trainer_factory: Zero-argument callable that returns a new
                :class:`~models.base.BaseTrainer` instance.

        Returns:
            DataFrame with columns ``step_idx``, ``train_size``,
            ``test_size``, ``mae``, and ``rmse``.

        Raises:
            ValueError: If the dataset is too small for even one step.
        """
        n_samples = len(x)

        if n_samples < self.initial_train_size + self.step_size:
            raise ValueError(
                f"Dataset too small ({n_samples} samples) for "
                f"initial_train_size={self.initial_train_size} + "
                f"step_size={self.step_size}"
            )

        step_results: list[dict[str, float | int]] = []
        step_idx = 0
        test_start = self.initial_train_size

        while test_start < n_samples:
            test_end = min(test_start + self.step_size, n_samples)

            if self.mode == "expanding":
                train_start = 0
            else:
                # sliding: keep training window fixed
                train_start = max(0, test_start - self.initial_train_size)

            x_train = x.iloc[train_start:test_start]
            y_train = y.iloc[train_start:test_start]
            x_test = x.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            logger.info(
                "WalkForward step %d: train [%d:%d] (%d), test [%d:%d] (%d), mode=%s",
                step_idx,
                train_start,
                test_start,
                len(x_train),
                test_start,
                test_end,
                len(x_test),
                self.mode,
            )

            trainer = trainer_factory()
            trainer.train(x_train, y_train)
            y_pred = trainer.predict(x_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            step_results.append(
                {
                    "step_idx": step_idx,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "mae": mae,
                    "rmse": rmse,
                }
            )

            logger.info("WalkForward step %d: MAE=%.4f, RMSE=%.4f", step_idx, mae, rmse)

            test_start += self.step_size
            step_idx += 1

        results_df = pd.DataFrame(step_results)
        logger.info(
            "WalkForward complete: %d steps, mean MAE=%.4f, mean RMSE=%.4f",
            len(results_df),
            results_df["mae"].mean(),
            results_df["rmse"].mean(),
        )
        return results_df

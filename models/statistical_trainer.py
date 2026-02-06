# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Statistical baseline trainers for price forecasting.

This module provides ARIMA and ETS (Exponential Smoothing) trainers that
wrap the corresponding statsmodels implementations.  Because these models
are univariate, ``train()`` uses only the target series *y_train* and
``predict()`` returns forecasts for ``len(x)`` steps ahead (the feature
matrix is used solely to determine the forecast horizon).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from core.exceptions import TrainingError

from .base import BaseTrainer
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from statsmodels.tsa.holtwinters import HoltWintersResultsWrapper
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

    from core.types import ModelConfig

logger = logging.getLogger(__name__)


class ARIMATrainer(BaseTrainer):
    """
    ARIMA / SARIMAX trainer for univariate price forecasting.

    Wraps :class:`statsmodels.tsa.statespace.sarimax.SARIMAX`.  The ARIMA
    order and optional seasonal order are read from
    ``config["hyperparameters"]``.

    Defaults:
        * ``order``: ``(1, 1, 1)``
        * ``seasonal_order``: ``(0, 0, 0, 0)``
    """

    def __init__(
        self,
        model_name: str = "arima",
        models_root: str | Path = "models",
        registry: ModelRegistry | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            models_root=models_root,
            registry=registry,
            config=config,
        )
        hp: dict[str, Any] = (config or {}).get("hyperparameters", {})

        order_raw = hp.get("order", (1, 1, 1))
        self.order: tuple[int, int, int] = (order_raw[0], order_raw[1], order_raw[2])
        seasonal_raw = hp.get("seasonal_order", (0, 0, 0, 0))
        self.seasonal_order: tuple[int, int, int, int] = (
            seasonal_raw[0],
            seasonal_raw[1],
            seasonal_raw[2],
            seasonal_raw[3],
        )
        self.model_fit: SARIMAXResultsWrapper | None = None
        self.metrics: dict[str, float | str] = {}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Fit the SARIMAX model on *y_train*.

        Args:
            x_train: Feature matrix (used only for length / index alignment).
            y_train: Target series to fit.
            x_val: Optional validation features (unused by ARIMA).
            y_val: Optional validation target for computing val metrics.

        Returns:
            Dictionary of training metrics including ``mae`` and
            ``model_type``.

        Raises:
            TrainingError: If statsmodels fitting fails.
        """
        _ = x_train  # ARIMA is univariate; x_train used only for interface compliance
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        logger.info(
            "Training ARIMA%s (seasonal=%s) on %d samples",
            self.order,
            self.seasonal_order,
            len(y_train),
        )

        try:
            model = SARIMAX(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = model.fit(disp=False)
        except Exception as exc:
            raise TrainingError(
                f"ARIMA fitting failed: {exc}",
                details={"order": self.order, "seasonal_order": self.seasonal_order},
            ) from exc

        # In-sample metrics
        y_pred_train = self.model_fit.fittedvalues
        train_mae = float(mean_absolute_error(y_train, y_pred_train))

        metrics: dict[str, float | str] = {
            "mae": train_mae,
            "model_type": "arima",
            "aic": float(self.model_fit.aic),
            "bic": float(self.model_fit.bic),
        }

        # Validation metrics (out-of-sample forecast)
        if x_val is not None and y_val is not None and len(y_val) > 0:
            forecast = self.model_fit.forecast(steps=len(y_val))
            metrics["val_mae"] = float(mean_absolute_error(y_val, forecast))

        self.metrics = metrics
        logger.info("ARIMA training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Forecast the next ``len(x)`` steps ahead.

        Args:
            x: Feature matrix whose length determines the forecast horizon.

        Returns:
            Array of forecasted values.

        Raises:
            TrainingError: If the model has not been fitted.
        """
        if self.model_fit is None:
            raise TrainingError("ARIMA model has not been fitted. Call train() first.")

        steps = len(x)
        forecast: np.ndarray = np.asarray(self.model_fit.forecast(steps=steps), dtype=np.float64)
        return forecast

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """Save the fitted ARIMA model and metrics via the registry."""
        if metrics is None:
            metrics = self.metrics

        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=self.model_fit,
            metrics=metrics,
        )


class ETSTrainer(BaseTrainer):
    """
    Exponential Smoothing (ETS) trainer for univariate price forecasting.

    Wraps :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`.
    Configuration is read from ``config["hyperparameters"]``.

    Defaults:
        * ``trend``: ``"add"``
        * ``seasonal``: ``None``
        * ``seasonal_periods``: ``None``
    """

    def __init__(
        self,
        model_name: str = "ets",
        models_root: str | Path = "models",
        registry: ModelRegistry | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            models_root=models_root,
            registry=registry,
            config=config,
        )
        hp: dict[str, Any] = (config or {}).get("hyperparameters", {})

        self.trend: str | None = hp.get("trend", "add")
        self.seasonal: str | None = hp.get("seasonal", None)
        self.seasonal_periods: int | None = hp.get("seasonal_periods", None)
        self.model_fit: HoltWintersResultsWrapper | None = None
        self.metrics: dict[str, float | str] = {}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Fit the Exponential Smoothing model on *y_train*.

        Args:
            x_train: Feature matrix (used only for length / index alignment).
            y_train: Target series to fit.
            x_val: Optional validation features (unused by ETS).
            y_val: Optional validation target for computing val metrics.

        Returns:
            Dictionary of training metrics including ``mae`` and
            ``model_type``.

        Raises:
            TrainingError: If statsmodels fitting fails.
        """
        _ = x_train  # ETS is univariate; x_train used only for interface compliance
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        logger.info(
            "Training ETS (trend=%s, seasonal=%s, periods=%s) on %d samples",
            self.trend,
            self.seasonal,
            self.seasonal_periods,
            len(y_train),
        )

        try:
            model = ExponentialSmoothing(
                y_train,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            )
            self.model_fit = model.fit(optimized=True)
        except Exception as exc:
            raise TrainingError(
                f"ETS fitting failed: {exc}",
                details={
                    "trend": self.trend,
                    "seasonal": self.seasonal,
                    "seasonal_periods": self.seasonal_periods,
                },
            ) from exc

        # In-sample metrics
        y_pred_train = self.model_fit.fittedvalues
        train_mae = float(mean_absolute_error(y_train, y_pred_train))

        metrics: dict[str, float | str] = {
            "mae": train_mae,
            "model_type": "ets",
            "aic": float(self.model_fit.aic),
        }

        # Validation metrics
        if x_val is not None and y_val is not None and len(y_val) > 0:
            forecast = self.model_fit.forecast(steps=len(y_val))
            metrics["val_mae"] = float(mean_absolute_error(y_val, forecast))

        self.metrics = metrics
        logger.info("ETS training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Forecast the next ``len(x)`` steps ahead.

        Args:
            x: Feature matrix whose length determines the forecast horizon.

        Returns:
            Array of forecasted values.

        Raises:
            TrainingError: If the model has not been fitted.
        """
        if self.model_fit is None:
            raise TrainingError("ETS model has not been fitted. Call train() first.")

        steps = len(x)
        forecast: np.ndarray = np.asarray(self.model_fit.forecast(steps=steps), dtype=np.float64)
        return forecast

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """Save the fitted ETS model and metrics via the registry."""
        if metrics is None:
            metrics = self.metrics

        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=self.model_fit,
            metrics=metrics,
        )

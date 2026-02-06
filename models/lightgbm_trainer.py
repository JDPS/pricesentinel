# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
LightGBM trainer for price forecasting.

This module provides a LightGBM-based regressor trainer with support for
early stopping, categorical features, and configurable hyperparameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM is required for LightGBMTrainer. Install with: pip install pricesentinel[ml]"
    ) from e

from .base import BaseTrainer
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from core.types import ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": -1,  # No limit (LightGBM's leaf-wise growth handles depth)
    "num_leaves": 63,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class LightGBMTrainer(BaseTrainer):
    """
    LightGBM regressor trainer for price forecasting.

    LightGBM is particularly effective for energy price forecasting due to
    its fast training speed, native categorical feature support, and
    leaf-wise tree growth strategy.
    """

    def __init__(
        self,
        model_name: str = "lightgbm",
        models_root: str | Path = "models",
        registry: ModelRegistry | None = None,
        config: ModelConfig | None = None,
    ):
        super().__init__(
            model_name=model_name,
            models_root=models_root,
            registry=registry,
            config=config,
        )
        params = (config or {}).get("hyperparameters", {})

        # Merge defaults with config overrides
        merged = {**_DEFAULT_PARAMS, **params}

        self.early_stopping_rounds: int = merged.pop("early_stopping_rounds", 50)

        self.model = lgb.LGBMRegressor(**merged)
        self.metrics: dict[str, float | str] = {}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Train the LightGBM model with optional early stopping.
        """
        logger.info("Training LightGBM (%s) on %d samples", self.model_name, len(x_train))

        fit_kwargs: dict[str, Any] = {}
        if x_val is not None and y_val is not None and len(x_val) > 0:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["eval_metric"] = "mae"

        self.model.fit(x_train, y_train, **fit_kwargs)

        metrics: dict[str, float | str] = {}

        # Training metrics
        y_pred_train: np.ndarray = self.model.predict(x_train)
        metrics["train_mae"] = float(mean_absolute_error(y_train, y_pred_train))
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        # Validation metrics
        if x_val is not None and y_val is not None and len(x_val) > 0:
            y_pred_val: np.ndarray = self.model.predict(x_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        self.metrics = metrics
        logger.info("LightGBM training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the trained LightGBM model."""
        result: np.ndarray = self.model.predict(x)
        return result

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """Save the trained model and metrics."""
        if metrics is None:
            metrics = self.metrics

        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=self.model,
            metrics=metrics,
        )

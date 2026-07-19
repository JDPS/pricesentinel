# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
XGBoost trainer for price forecasting.

This module provides an XGBoost-based regressor trainer with support for
early stopping on validation sets and configurable hyperparameters via
ModelConfig.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import optuna
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "XGBoost and Optuna required. Install with: pip install pricesentinel[ml]"
    ) from e

from .base import BaseTrainer
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from core.types import ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost regressor trainer for price forecasting.

    Supports early stopping when a validation set is provided.
    """

    def __init__(
        self,
        model_name: str = "xgboost",
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

        self.model = xgb.XGBRegressor(**merged)
        self.metrics: dict[str, float | str] = {}
        self._tuned_params: dict[str, Any] = {}

    def optimize_hyperparameters(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        Updates self.model with the best found parameters combined with the base config.
        """
        logger.info(f"Starting Optuna hyperparameter tuning for {n_trials} trials...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            param = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "n_estimators": 500,  # Fixed for tuning to allow early stopping
                "random_state": 42,
                "n_jobs": -1,
            }

            model = xgb.XGBRegressor(**param)

            fit_kwargs = {
                "eval_set": [(x_val, y_val)],
                "verbose": False,
            }

            model.fit(x_train, y_train, **fit_kwargs)

            y_pred_val = model.predict(x_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")

        # Update self.model with the new best params
        self._tuned_params = study.best_trial.params

        # Merge back with default/config to recreate model
        base_params = (self.config or {}).get("hyperparameters", {})
        merged = {**_DEFAULT_PARAMS, **base_params, **self._tuned_params}
        self.early_stopping_rounds = merged.pop("early_stopping_rounds", 50)

        self.model = xgb.XGBRegressor(**merged)
        return self._tuned_params

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Train the XGBoost model with optional early stopping.
        """
        logger.info("Training XGBoost (%s) on %d samples", self.model_name, len(x_train))

        fit_kwargs: dict[str, Any] = {}
        if x_val is not None and y_val is not None and len(x_val) > 0:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["verbose"] = False

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
        logger.info("XGBoost training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the trained XGBoost model."""
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

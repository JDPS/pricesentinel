# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
This module provides the SklearnRegressorTrainer class for building, training, and saving
a baseline price forecasting sklearn regressor using a RandomForestRegressor model.

The module integrates a structured approach to model training and evaluation by computing
basic metrics like mean absolute error (MAE) and root mean squared error (RMSE) for training
and validation datasets.

Classes:
    SklearnRegressorTrainer: A trainer class to handle regression tasks using sklearn's
    RandomForestRegressor with features like model saving and metric logging.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base import BaseTrainer

logger = logging.getLogger(__name__)


class SklearnRegressorTrainer(BaseTrainer):
    """
    Baseline sklearn regressor trainer for price forecasting.
    """

    def __init__(self, model_name: str = "baseline", models_root: str | Path = "models"):
        super().__init__(model_name=model_name, models_root=models_root)
        # Lightweight "fast" mode for quick demo runs
        fast_mode = model_name.endswith("_fast")
        n_estimators = 50 if fast_mode else 100
        max_depth = 5 if fast_mode else 10

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.metrics: dict[str, float] = {}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the underlying sklearn model and compute basic metrics.
        """
        logger.info("Training sklearn regressor (%s) on %d samples", self.model_name, len(x_train))
        self.model.fit(x_train, y_train)

        metrics: dict[str, float] = {}

        # Training metrics
        y_pred_train = self.model.predict(x_train)
        metrics["train_mae"] = float(mean_absolute_error(y_train, y_pred_train))
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        # Validation metrics (if provided)
        if x_val is not None and y_val is not None and len(x_val) > 0:
            y_pred_val = self.model.predict(x_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        self.metrics = metrics
        logger.info("Training metrics: %s", metrics)
        return metrics

    def _model_dir(self, country_code: str, run_id: str) -> Path:
        return self.models_root / country_code / self.model_name / run_id

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Save the trained model and metrics under the models directory.
        """
        if metrics is None:
            metrics = self.metrics

        model_dir = self._model_dir(country_code, run_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Saved model to %s", model_path)
        logger.info("Saved metrics to %s", metrics_path)

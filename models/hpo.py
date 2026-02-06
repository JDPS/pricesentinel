# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Hyperparameter optimization using Optuna.

This module provides Bayesian hyperparameter optimization for model trainers
using Optuna with time-series aware cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise ImportError(
        "Optuna is required for hyperparameter optimization. "
        "Install with: pip install pricesentinel[ml]"
    ) from e

from core.exceptions import HyperparameterError

logger = logging.getLogger(__name__)

# Default search spaces per algorithm
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "xgboost": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("int", 1, 10),
        "reg_alpha": ("float_log", 1e-8, 10.0),
        "reg_lambda": ("float_log", 1e-8, 10.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 1000),
        "num_leaves": ("int", 15, 127),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_samples": ("int", 5, 50),
        "reg_alpha": ("float_log", 1e-8, 10.0),
        "reg_lambda": ("float_log", 1e-8, 10.0),
    },
}


def _suggest_param(trial: optuna.Trial, name: str, spec: tuple[str, ...]) -> Any:
    """Suggest a hyperparameter value based on its type specification."""
    param_type = spec[0]
    if param_type == "int":
        return trial.suggest_int(name, int(spec[1]), int(spec[2]))
    elif param_type == "float":
        return trial.suggest_float(name, float(spec[1]), float(spec[2]))
    elif param_type == "float_log":
        return trial.suggest_float(name, float(spec[1]), float(spec[2]), log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, list(spec[1:]))
    else:
        msg = f"Unknown param type: {param_type}"
        raise HyperparameterError(msg)


class OptunaHPO:
    """
    Bayesian hyperparameter optimization using Optuna.

    Uses TimeSeriesSplit cross-validation internally to evaluate each
    trial, ensuring temporal ordering is respected.
    """

    def __init__(
        self,
        algorithm: str,
        n_trials: int = 50,
        metric: str = "mae",
        cv_splits: int = 3,
        search_space: dict[str, Any] | None = None,
    ):
        """
        Initialize the HPO runner.

        Args:
            algorithm: Algorithm name (e.g., "xgboost", "lightgbm", "random_forest").
            n_trials: Number of Optuna trials to run.
            metric: Optimization metric ("mae" or "rmse").
            cv_splits: Number of TimeSeriesSplit folds.
            search_space: Optional custom search space. If None, uses defaults.
        """
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.metric = metric
        self.cv_splits = cv_splits

        if search_space is not None:
            self.search_space = search_space
        elif algorithm in SEARCH_SPACES:
            self.search_space = SEARCH_SPACES[algorithm]
        else:
            raise HyperparameterError(
                f"No default search space for algorithm '{algorithm}'. "
                f"Available: {list(SEARCH_SPACES.keys())}. Provide a custom search_space."
            )

    def optimize(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization and return the best parameters.

        Args:
            x: Feature matrix.
            y: Target vector.

        Returns:
            Dictionary of best hyperparameters.
        """
        logger.info(
            "Starting Optuna HPO for '%s': %d trials, %d-fold CV, metric=%s",
            self.algorithm,
            self.n_trials,
            self.cv_splits,
            self.metric,
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(trial, x, y),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = study.best_value

        logger.info(
            "HPO complete. Best %s: %.4f. Best params: %s",
            self.metric,
            best_value,
            best_params,
        )

        return cast(dict[str, Any], best_params)

    def _objective(
        self,
        trial: optuna.Trial,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Objective function for a single Optuna trial."""
        params = {
            name: _suggest_param(trial, name, spec) for name, spec in self.search_space.items()
        }

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(x):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self._create_model(params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)

            if self.metric == "mae":
                score = mean_absolute_error(y_val, y_pred)
            elif self.metric == "rmse":
                score = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            else:
                score = mean_absolute_error(y_val, y_pred)

            scores.append(score)

        return float(np.mean(scores))

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create a model instance with the given hyperparameters."""
        if self.algorithm == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif self.algorithm == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **params)
        elif self.algorithm == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **params)
        else:
            raise HyperparameterError(f"Cannot create model for algorithm: {self.algorithm}")

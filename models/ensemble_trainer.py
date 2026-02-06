# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble methods for energy price forecasting.

This module provides two ensemble trainer implementations that combine
multiple base learners for improved prediction accuracy:

- **WeightedEnsembleTrainer**: computes a weighted average of sub-model
  predictions, where weights are derived from inverse-MAE on the
  validation set (or equal weights when no validation data is available).
- **StackingTrainer**: trains a Ridge meta-learner on out-of-fold
  predictions produced by the base learners via time-series cross-
  validation.

Both classes follow the :class:`~models.base.BaseTrainer` interface and
delegate sub-model construction to :func:`~models.get_trainer`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from core.exceptions import EnsembleError

from .base import BaseTrainer
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from core.types import ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_SUB_MODELS: list[str] = ["baseline"]


class WeightedEnsembleTrainer(BaseTrainer):
    """
    Ensemble trainer that combines sub-model predictions via weighted
    averaging.

    Weights are determined using inverse-MAE on the validation set.  When
    no validation data is supplied, all sub-models receive equal weight.

    The ``sub_models`` list is read from
    ``config["hyperparameters"]["sub_models"]``.
    """

    def __init__(
        self,
        model_name: str = "weighted_ensemble",
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
        params = (config or {}).get("hyperparameters", {})
        self.sub_model_names: list[str] = params.get("sub_models", list(_DEFAULT_SUB_MODELS))
        self.sub_trainers: list[BaseTrainer] = []
        self.weights: np.ndarray = np.array([])
        self.metrics: dict[str, float | str] = {}

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Train every sub-model and compute ensemble weights.

        Args:
            x_train: Training feature matrix.
            y_train: Training target series.
            x_val: Optional validation feature matrix used for weight
                optimisation.
            y_val: Optional validation target series.

        Returns:
            Dictionary of ensemble-level evaluation metrics.

        Raises:
            EnsembleError: If sub-model training or weight computation
                fails.
        """
        # Lazy import to break circular dependency (models/__init__.py
        # imports trainer modules at registration time).
        from models import get_trainer

        if not self.sub_model_names:
            raise EnsembleError(
                "No sub-models specified for the weighted ensemble.",
                details={"model_name": self.model_name},
            )

        logger.info(
            "Training weighted ensemble '%s' with sub-models: %s",
            self.model_name,
            self.sub_model_names,
        )

        self.sub_trainers = []
        for name in self.sub_model_names:
            try:
                trainer = get_trainer(
                    country_code="__ensemble__",
                    model_name=name,
                    models_root=str(self.models_root),
                )
                trainer.train(x_train, y_train, x_val, y_val)
                self.sub_trainers.append(trainer)
            except Exception as exc:
                raise EnsembleError(
                    f"Failed to train sub-model '{name}': {exc}",
                    details={"sub_model": name},
                ) from exc

        # --- Compute weights ------------------------------------------------
        self.weights = self._compute_weights(x_val, y_val)
        weight_map = dict(zip(self.sub_model_names, self.weights.tolist(), strict=True))
        logger.info("Ensemble weights: %s", weight_map)

        # --- Ensemble-level metrics -----------------------------------------
        metrics: dict[str, float | str] = {"model_type": "weighted_ensemble"}
        for idx, name in enumerate(self.sub_model_names):
            metrics[f"weight_{name}"] = float(self.weights[idx])

        y_pred_train = self.predict(x_train)
        metrics["train_mae"] = float(mean_absolute_error(y_train, y_pred_train))
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        if x_val is not None and y_val is not None and len(x_val) > 0:
            y_pred_val = self.predict(x_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        self.metrics = metrics
        logger.info("Ensemble training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Return the weighted average of sub-model predictions.

        Args:
            x: Feature matrix.

        Returns:
            1-D array of ensemble predictions.

        Raises:
            EnsembleError: If no sub-models have been trained yet.
        """
        if not self.sub_trainers:
            raise EnsembleError(
                "Cannot predict before training the ensemble.",
                details={"model_name": self.model_name},
            )

        predictions = np.column_stack([t.predict(x) for t in self.sub_trainers])
        result: np.ndarray = predictions @ self.weights
        return result

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """
        Persist every sub-model and the ensemble weights via the registry.

        Args:
            country_code: ISO country code.
            run_id: Unique run identifier.
            metrics: Optional metrics override.
        """
        if metrics is None:
            metrics = self.metrics

        # Save individual sub-models
        for trainer in self.sub_trainers:
            trainer.save(country_code=country_code, run_id=run_id)

        # Save ensemble metadata (weights) through the registry
        ensemble_artifact = {
            "sub_model_names": self.sub_model_names,
            "weights": self.weights.tolist(),
        }
        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=ensemble_artifact,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        x_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> np.ndarray:
        """
        Derive sub-model weights from validation performance.

        Uses inverse-MAE weighting when a validation set is available;
        falls back to equal weights otherwise.
        """
        n = len(self.sub_trainers)

        if x_val is None or y_val is None or len(x_val) == 0:
            logger.info("No validation set provided; using equal weights.")
            return np.ones(n) / n

        mae_scores: list[float] = []
        for trainer in self.sub_trainers:
            y_pred = trainer.predict(x_val)
            mae = float(mean_absolute_error(y_val, y_pred))
            # Guard against a perfect (zero-error) model causing division
            # by zero.
            mae_scores.append(max(mae, 1e-9))

        inverse_mae = np.array([1.0 / m for m in mae_scores])
        weights: np.ndarray = inverse_mae / inverse_mae.sum()
        return weights


class StackingTrainer(BaseTrainer):
    """
    Stacking ensemble that trains a Ridge meta-learner on out-of-fold
    predictions from base sub-models.

    Configuration keys (under ``config["hyperparameters"]``):

    * ``sub_models`` -- list of base-learner model names.
    * ``meta_model`` -- name of the meta-learner algorithm
      (currently only ``"ridge"`` is supported).
    * ``meta_alpha`` -- Ridge regularisation parameter (default ``1.0``).
    * ``n_splits`` -- number of ``TimeSeriesSplit`` folds (default ``3``).
    """

    def __init__(
        self,
        model_name: str = "stacking",
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
        params = (config or {}).get("hyperparameters", {})
        self.sub_model_names: list[str] = params.get("sub_models", list(_DEFAULT_SUB_MODELS))
        self.meta_model_name: str = params.get("meta_model", "ridge")
        self.meta_alpha: float = float(params.get("meta_alpha", 1.0))
        self.n_splits: int = int(params.get("n_splits", 3))

        self.sub_trainers: list[BaseTrainer] = []
        self.meta_model: Ridge | None = None
        self.metrics: dict[str, float | str] = {}

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Train base learners and a Ridge meta-learner via stacking.

        Steps:
            1. Generate out-of-fold predictions for each base learner
               using :class:`~sklearn.model_selection.TimeSeriesSplit`.
            2. Fit a Ridge regression meta-learner on the stacked OOF
               prediction matrix.
            3. Re-train every base learner on the full training set so
               that ``predict()`` uses models fitted on all available
               data.

        Args:
            x_train: Training feature matrix.
            y_train: Training target series.
            x_val: Optional validation feature matrix.
            y_val: Optional validation target series.

        Returns:
            Dictionary of stacking-level evaluation metrics.

        Raises:
            EnsembleError: If training of any component fails.
        """
        from models import get_trainer

        if not self.sub_model_names:
            raise EnsembleError(
                "No sub-models specified for the stacking ensemble.",
                details={"model_name": self.model_name},
            )

        logger.info(
            "Training stacking ensemble '%s' with sub-models: %s (meta: %s)",
            self.model_name,
            self.sub_model_names,
            self.meta_model_name,
        )

        # ---- Step 1: out-of-fold predictions ----------------------------
        oof_matrix = self._build_oof_matrix(x_train, y_train)

        # ---- Step 2: fit meta-learner -----------------------------------
        try:
            self.meta_model = Ridge(alpha=self.meta_alpha)
            self.meta_model.fit(oof_matrix, y_train)
        except Exception as exc:
            raise EnsembleError(
                f"Meta-learner training failed: {exc}",
                details={"meta_model": self.meta_model_name},
            ) from exc

        # ---- Step 3: re-train base learners on full training data -------
        self.sub_trainers = []
        for name in self.sub_model_names:
            try:
                trainer = get_trainer(
                    country_code="__ensemble__",
                    model_name=name,
                    models_root=str(self.models_root),
                )
                trainer.train(x_train, y_train, x_val, y_val)
                self.sub_trainers.append(trainer)
            except Exception as exc:
                raise EnsembleError(
                    f"Failed to train sub-model '{name}' on full data: {exc}",
                    details={"sub_model": name},
                ) from exc

        # ---- Metrics ----------------------------------------------------
        metrics: dict[str, float | str] = {"model_type": "stacking"}
        y_pred_train = self.predict(x_train)
        metrics["train_mae"] = float(mean_absolute_error(y_train, y_pred_train))
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        if x_val is not None and y_val is not None and len(x_val) > 0:
            y_pred_val = self.predict(x_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        meta_coefs = self.meta_model.coef_.tolist()
        for idx, name in enumerate(self.sub_model_names):
            metrics[f"meta_coef_{name}"] = float(meta_coefs[idx])

        self.metrics = metrics
        logger.info("Stacking training metrics: %s", metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate stacked ensemble predictions.

        Each base learner produces its predictions, which are stacked
        column-wise and passed through the trained meta-learner.

        Args:
            x: Feature matrix.

        Returns:
            1-D array of ensemble predictions.

        Raises:
            EnsembleError: If the ensemble has not been trained yet.
        """
        if not self.sub_trainers or self.meta_model is None:
            raise EnsembleError(
                "Cannot predict before training the stacking ensemble.",
                details={"model_name": self.model_name},
            )

        base_preds = np.column_stack([t.predict(x) for t in self.sub_trainers])
        result: np.ndarray = self.meta_model.predict(base_preds)
        return result

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """
        Persist all sub-models and the meta-learner via the registry.

        Args:
            country_code: ISO country code.
            run_id: Unique run identifier.
            metrics: Optional metrics override.
        """
        if metrics is None:
            metrics = self.metrics

        # Save individual sub-models
        for trainer in self.sub_trainers:
            trainer.save(country_code=country_code, run_id=run_id)

        # Save meta-learner and ensemble metadata
        stacking_artifact = {
            "sub_model_names": self.sub_model_names,
            "meta_model": self.meta_model,
            "meta_model_name": self.meta_model_name,
            "meta_alpha": self.meta_alpha,
        }
        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=stacking_artifact,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_oof_matrix(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> np.ndarray:
        """
        Build the out-of-fold prediction matrix using TimeSeriesSplit.

        For each fold, every sub-model is trained on the fold's training
        portion and predicts the fold's validation portion.  The
        predictions are assembled into a matrix of shape
        ``(len(x_train), n_sub_models)``.

        Returns:
            2-D array where each column holds the OOF predictions of one
            base learner.

        Raises:
            EnsembleError: If OOF prediction generation fails.
        """
        from models import get_trainer

        n_samples = len(x_train)
        n_models = len(self.sub_model_names)
        oof_preds = np.zeros((n_samples, n_models))

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(x_train)):
            x_fold_train = x_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            x_fold_val = x_train.iloc[val_idx]

            for model_idx, name in enumerate(self.sub_model_names):
                try:
                    trainer = get_trainer(
                        country_code="__ensemble__",
                        model_name=name,
                        models_root=str(self.models_root),
                    )
                    trainer.train(x_fold_train, y_fold_train)
                    oof_preds[val_idx, model_idx] = trainer.predict(x_fold_val)
                except Exception as exc:
                    raise EnsembleError(
                        f"OOF prediction failed for '{name}' on fold {fold_idx}: {exc}",
                        details={"sub_model": name, "fold": fold_idx},
                    ) from exc

        return oof_preds

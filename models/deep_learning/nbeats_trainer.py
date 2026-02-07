# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
N-BEATS-inspired trainer for energy price forecasting.

Implements a simplified N-BEATS architecture using raw PyTorch with
fully connected blocks, early stopping, and integration with the
PriceSentinel model registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import TrainingError
from models.model_registry import ModelRegistry

from .base import DeepLearningTrainer

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from core.types import ModelConfig

logger = logging.getLogger(__name__)


class NBEATSTrainer(DeepLearningTrainer):
    """
    Trainer wrapping a simplified N-BEATS network.

    Each *block* is a pair of fully connected layers with ReLU activation.
    The blocks are stacked sequentially to produce a single scalar output
    (the predicted energy price).
    """

    def __init__(
        self,
        model_name: str = "nbeats",
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int) -> Any:
        """Construct the ``NBEATSModel`` and move it to the selected device."""
        import torch.nn as nn

        class NBEATSModel(nn.Module):
            """Simple N-BEATS-inspired fully connected network."""

            def __init__(self, n_features: int, hidden_size: int, n_layers: int) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                in_size = n_features
                for _ in range(n_layers):
                    layers.append(nn.Linear(in_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                    in_size = hidden_size
                layers.append(nn.Linear(hidden_size, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x: Any) -> Any:  # noqa: ANN401
                return self.network(x)

        device = self._select_device()
        model = NBEATSModel(n_features, self.hidden_size, self.n_layers).to(device)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float | str]:
        """
        Train the N-BEATS model.

        Args:
            x_train: Training features.
            y_train: Training targets.
            x_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Dictionary of training metrics including ``mae``,
            ``model_type``, and ``epochs_trained``.

        Raises:
            TrainingError: If training fails for any reason.
        """
        import torch
        import torch.nn as nn

        try:
            device = self._select_device()
            n_features = x_train.shape[1]
            self.model = self._build_model(n_features)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            train_loader = self._create_dataloader(x_train, y_train, self.batch_size, shuffle=True)

            val_loader = None
            if x_val is not None and y_val is not None and len(x_val) > 0:
                val_loader = self._create_dataloader(x_val, y_val, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            patience_counter = 0
            epochs_trained = 0

            for epoch in range(self.epochs):
                # -- Training phase --
                self.model.train()
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    preds = self.model(x_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()

                epochs_trained = epoch + 1

                # -- Early stopping on validation loss --
                if val_loader is not None:
                    val_loss = self._evaluate_loss(val_loader, criterion, device)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            logger.info(
                                "Early stopping at epoch %d (patience=%d)",
                                epochs_trained,
                                self.patience,
                            )
                            break

            # -- Compute final metrics --
            metrics = self._compute_metrics(x_train, y_train, x_val, y_val, device)
            metrics["model_type"] = "nbeats"
            metrics["epochs_trained"] = epochs_trained

            self.metrics = metrics
            logger.info("N-BEATS training complete: %s", metrics)
            return metrics

        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(
                f"N-BEATS training failed: {exc}",
                details={"model_name": self.model_name},
            ) from exc

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the trained N-BEATS model.

        Args:
            x: Feature DataFrame.

        Returns:
            Numpy array of predictions.

        Raises:
            TrainingError: If the model has not been trained yet.
        """
        import torch

        if self.model is None:
            raise TrainingError(
                "Model has not been trained yet. Call train() first.",
                details={"model_name": self.model_name},
            )

        device = self._select_device()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x.values, dtype=torch.float32).to(device)
            preds = self.model(x_tensor).cpu().numpy().flatten()

        result: np.ndarray = preds
        return result

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float | str] | None = None,
    ) -> None:
        """
        Persist the trained model state dict via the model registry.

        Args:
            country_code: Country code for the model.
            run_id: Unique identifier for this training run.
            metrics: Optional metrics dictionary; falls back to stored metrics.
        """
        if metrics is None:
            metrics = self.metrics

        self.registry.save_model(
            country_code=country_code,
            model_name=self.model_name,
            run_id=run_id,
            model=self.model.state_dict() if self.model is not None else None,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_loss(
        self,
        loader: Any,
        criterion: Any,
        device: Any,
    ) -> float:
        """Compute mean loss over a dataloader in eval mode."""
        import torch

        self.model.eval()  # type: ignore[union-attr]
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = self.model(x_batch)
                total_loss += criterion(preds, y_batch).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _compute_metrics(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        device: Any,
    ) -> dict[str, float | str]:
        """Compute MAE on training (and optionally validation) data."""
        _ = device  # Used by caller; kept for API consistency
        import numpy as np

        metrics: dict[str, float | str] = {}

        train_preds = self.predict(x_train)
        metrics["mae"] = float(np.mean(np.abs(y_train.values - train_preds)))

        if x_val is not None and y_val is not None and len(x_val) > 0:
            val_preds = self.predict(x_val)
            metrics["val_mae"] = float(np.mean(np.abs(y_val.values - val_preds)))

        return metrics

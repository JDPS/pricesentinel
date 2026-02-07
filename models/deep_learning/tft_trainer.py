# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Simplified Temporal Fusion Transformer trainer for energy price forecasting.

Implements a TFT-inspired architecture with variable selection, a Gated
Residual Network, and multi-head self-attention using raw PyTorch.
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


class TFTTrainer(DeepLearningTrainer):
    """
    Trainer wrapping a simplified Temporal Fusion Transformer network.

    The architecture consists of:
    * Variable selection network (linear projection of raw features).
    * Gated Residual Network (GRN) for non-linear feature processing.
    * Multi-head self-attention for capturing temporal dependencies.
    * Linear output head producing a single scalar prediction.
    """

    def __init__(
        self,
        model_name: str = "tft",
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
        """Construct the ``SimpleTFTModel`` and move it to the selected device."""
        import torch
        import torch.nn as nn

        class SimpleTFTModel(nn.Module):
            """Simplified TFT-inspired architecture."""

            def __init__(self, n_features: int, hidden_size: int) -> None:
                super().__init__()
                # Variable selection network
                self.variable_selection = nn.Linear(n_features, hidden_size)

                # Gated Residual Network (GRN)
                self.grn_linear = nn.Linear(hidden_size, hidden_size)
                self.grn_gate_linear = nn.Linear(hidden_size, hidden_size)
                self.grn_layer_norm = nn.LayerNorm(hidden_size)

                # Self-attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size, num_heads=2, batch_first=True
                )

                # Output projection
                self.output_layer = nn.Linear(hidden_size, 1)

            def forward(self, x: Any) -> Any:  # noqa: ANN401
                # Variable selection
                selected = torch.relu(self.variable_selection(x))

                # GRN with gating
                grn_out = torch.relu(self.grn_linear(selected))
                gate = torch.sigmoid(self.grn_gate_linear(selected))
                gated = gate * grn_out + (1 - gate) * selected
                gated = self.grn_layer_norm(gated)

                # Self-attention (add sequence dim, then squeeze)
                attn_input = gated.unsqueeze(1)  # (batch, 1, hidden)
                attn_out, _ = self.attention(attn_input, attn_input, attn_input)
                attn_out = attn_out.squeeze(1)  # (batch, hidden)

                # Residual connection
                combined = attn_out + gated

                return self.output_layer(combined)

        device = self._select_device()
        model = SimpleTFTModel(n_features, self.hidden_size).to(device)
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
        Train the simplified TFT model.

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
            metrics["model_type"] = "tft"
            metrics["epochs_trained"] = epochs_trained

            self.metrics = metrics
            logger.info("TFT training complete: %s", metrics)
            return metrics

        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(
                f"TFT training failed: {exc}",
                details={"model_name": self.model_name},
            ) from exc

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the trained TFT model.

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
                preds = self.model(x_batch)  # type: ignore[misc]
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

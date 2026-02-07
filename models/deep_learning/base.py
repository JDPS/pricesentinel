# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for deep learning trainers.

Provides shared PyTorch boilerplate including device selection, dataloader
creation, and common hyperparameter extraction from config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from models.base import BaseTrainer
from models.model_registry import ModelRegistry

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from core.types import ModelConfig

logger = logging.getLogger(__name__)


class DeepLearningTrainer(BaseTrainer):
    """
    Abstract base for PyTorch-based model trainers.

    Extracts common hyperparameters from config and provides helpers for
    device selection and dataloader creation.  Subclasses must implement
    ``train``, ``predict``, and ``save``.
    """

    def __init__(
        self,
        model_name: str,
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
        params: dict[str, Any] = (config or {}).get("hyperparameters", {})

        self.hidden_size: int = params.get("hidden_size", 32)
        self.n_layers: int = params.get("n_layers", 2)
        self.epochs: int = params.get("epochs", 50)
        self.learning_rate: float = params.get("learning_rate", 0.001)
        self.batch_size: int = params.get("batch_size", 32)
        self.patience: int = params.get("patience", 5)

        self.model: Any | None = None
        self.metrics: dict[str, float | str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        """Return CUDA device when available, otherwise CPU."""
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _create_dataloader(
        x: pd.DataFrame,
        y: pd.Series,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """
        Convert pandas structures into a PyTorch DataLoader.

        Args:
            x: Feature DataFrame.
            y: Target Series.
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle samples each epoch.

        Returns:
            A ``DataLoader`` yielding ``(x_batch, y_batch)`` tensors.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

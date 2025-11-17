# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseTrainer(ABC):
    """
    Base interface for model trainers.
    """

    def __init__(self, model_name: str, models_root: str | Path = "models"):
        self.model_name = model_name
        self.models_root = Path(models_root)

    @abstractmethod
    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the model and return evaluation metrics.
        """

    @abstractmethod
    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Persist the trained model (and optionally metrics) to disk.
        """

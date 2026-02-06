# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Model trainer registry and factory.

This module provides a registry-based factory for retrieving model trainers.
Trainers register themselves at import time, and get_trainer() dispatches
based on model_name.

Usage:
    from models import get_trainer
    trainer = get_trainer("PT", model_name="xgboost")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import ModelError

from .base import BaseTrainer
from .sklearn_trainer import SklearnRegressorTrainer

if TYPE_CHECKING:
    from core.types import ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "baseline"

# Registry maps model names to trainer classes
_TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {}


def register_trainer(name: str, trainer_cls: type[BaseTrainer]) -> None:
    """
    Register a trainer class under a given name.

    Args:
        name: The model name key (e.g., "baseline", "xgboost").
        trainer_cls: The trainer class to register.
    """
    _TRAINER_REGISTRY[name] = trainer_cls
    logger.debug("Registered trainer '%s': %s", name, trainer_cls.__name__)


def get_trainer(
    country_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    models_root: str | Path = "models",
    config: ModelConfig | None = None,
) -> BaseTrainer:
    """
    Factory for trainers.

    Dispatches to the registered trainer class based on model_name.

    Args:
        country_code: Country code (reserved for country-specific dispatch).
        model_name: Name of the model/trainer to retrieve.
        models_root: Root directory for model artifacts.
        config: Optional model configuration with hyperparameters.

    Returns:
        An instance of the requested trainer.

    Raises:
        ModelError: If the model name is not registered.
    """
    _ = country_code  # Reserved for future country-specific dispatch

    # Normalize: names ending with _fast map to the base trainer
    base_name = model_name.removesuffix("_fast")

    trainer_cls = _TRAINER_REGISTRY.get(base_name)
    if trainer_cls is None:
        available = ", ".join(sorted(_TRAINER_REGISTRY.keys()))
        raise ModelError(
            f"Unknown model: '{model_name}'. Available: {available}",
            details={"model_name": model_name, "available": list(_TRAINER_REGISTRY.keys())},
        )

    return trainer_cls(model_name=model_name, models_root=models_root, config=config)


def list_registered_trainers() -> list[str]:
    """Return a sorted list of all registered trainer names."""
    return sorted(_TRAINER_REGISTRY.keys())


# Register built-in trainers
register_trainer("baseline", SklearnRegressorTrainer)

# Conditionally register optional trainers
try:
    from .xgboost_trainer import XGBoostTrainer

    register_trainer("xgboost", XGBoostTrainer)
except ImportError:
    pass

try:
    from .lightgbm_trainer import LightGBMTrainer

    register_trainer("lightgbm", LightGBMTrainer)
except ImportError:
    pass

try:
    from .statistical_trainer import ARIMATrainer, ETSTrainer

    register_trainer("arima", ARIMATrainer)
    register_trainer("ets", ETSTrainer)
except ImportError:
    pass

from .ensemble_trainer import StackingTrainer, WeightedEnsembleTrainer  # noqa: E402

register_trainer("ensemble_weighted", WeightedEnsembleTrainer)
register_trainer("ensemble_stacking", StackingTrainer)

try:
    from .deep_learning.nbeats_trainer import (
        NBEATSTrainer,  # noqa: E402 # type: ignore[import-untyped]
    )

    register_trainer("nbeats", NBEATSTrainer)
except ImportError:
    pass

try:
    from .deep_learning.tft_trainer import TFTTrainer  # noqa: E402 # type: ignore[import-untyped]

    register_trainer("tft", TFTTrainer)
except ImportError:
    pass


__all__ = [
    "BaseTrainer",
    "SklearnRegressorTrainer",
    "DEFAULT_MODEL_NAME",
    "get_trainer",
    "register_trainer",
    "list_registered_trainers",
]

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from .base import BaseTrainer
from .sklearn_trainer import SklearnRegressorTrainer

DEFAULT_MODEL_NAME = "baseline"


def get_trainer(
    country_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    models_root: str | Path = "models",
) -> BaseTrainer:
    """
    Factory for trainers.

    For now, returns a single baseline sklearn regressor trainer. In the
    future this can dispatch on country_code and model_name.
    """
    _ = country_code  # Reserved for future use
    return SklearnRegressorTrainer(model_name=model_name, models_root=models_root)


__all__ = ["BaseTrainer", "SklearnRegressorTrainer", "DEFAULT_MODEL_NAME", "get_trainer"]

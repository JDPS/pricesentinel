# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for machine learning trainers.

This module provides a factory method for retrieving trainers used in
machine learning tasks. Currently, it offers support for a baseline sklearn
regressor trainer. Future enhancements may include dispatching based on
country-specific or model-specific variations.

Constants:
DEFAULT_MODEL_NAME: str - Default model name used by the trainer.

Functions:
get_trainer: Factory method to retrieve an appropriate trainer based on
country code and model specification.
"""

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

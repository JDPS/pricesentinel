# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model trainer registry and factory."""

import pytest

from core.exceptions import ModelError
from models import (
    get_trainer,
    list_registered_trainers,
)
from models.sklearn_trainer import SklearnRegressorTrainer


def test_baseline_is_registered():
    """The baseline trainer should always be registered."""
    trainers = list_registered_trainers()
    assert "baseline" in trainers


def test_get_trainer_returns_baseline():
    """get_trainer should return SklearnRegressorTrainer for 'baseline'."""
    trainer = get_trainer("XX", model_name="baseline")
    assert isinstance(trainer, SklearnRegressorTrainer)


def test_get_trainer_fast_mode():
    """get_trainer should handle _fast suffix."""
    trainer = get_trainer("XX", model_name="baseline_fast")
    assert isinstance(trainer, SklearnRegressorTrainer)
    assert trainer.model_name == "baseline_fast"


def test_get_trainer_unknown_raises():
    """get_trainer should raise ModelError for unknown model names."""
    with pytest.raises(ModelError, match="Unknown model"):
        get_trainer("XX", model_name="nonexistent_model")


def test_get_trainer_with_config():
    """get_trainer should pass config to the trainer."""
    config = {"hyperparameters": {"n_estimators": 25, "max_depth": 3}}
    trainer = get_trainer("XX", model_name="baseline", config=config)
    assert isinstance(trainer, SklearnRegressorTrainer)
    assert trainer.model.n_estimators == 25
    assert trainer.model.max_depth == 3


def test_xgboost_is_registered():
    """XGBoost trainer should be registered if available."""
    trainers = list_registered_trainers()
    assert "xgboost" in trainers


def test_lightgbm_is_registered():
    """LightGBM trainer should be registered if available."""
    trainers = list_registered_trainers()
    assert "lightgbm" in trainers


def test_get_xgboost_trainer():
    """get_trainer should return XGBoostTrainer for 'xgboost'."""
    from models.xgboost_trainer import XGBoostTrainer

    trainer = get_trainer("XX", model_name="xgboost")
    assert isinstance(trainer, XGBoostTrainer)


def test_get_lightgbm_trainer():
    """get_trainer should return LightGBMTrainer for 'lightgbm'."""
    from models.lightgbm_trainer import LightGBMTrainer

    trainer = get_trainer("XX", model_name="lightgbm")
    assert isinstance(trainer, LightGBMTrainer)

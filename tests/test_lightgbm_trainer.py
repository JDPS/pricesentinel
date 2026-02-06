# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LightGBM trainer."""

import numpy as np
import pandas as pd
import pytest

from models.lightgbm_trainer import LightGBMTrainer


@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    x = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(x["feature_1"] * 2 + x["feature_3"] * 0.1 + np.random.randn(n) * 0.5)
    return x, y


def test_train_returns_metrics(synthetic_data):
    """Training should return a dict with train_mae and train_rmse."""
    x, y = synthetic_data
    trainer = LightGBMTrainer(model_name="lgbm_test")
    metrics = trainer.train(x, y)

    assert "train_mae" in metrics
    assert "train_rmse" in metrics
    assert metrics["train_mae"] >= 0
    assert metrics["train_rmse"] >= 0


def test_train_with_validation(synthetic_data):
    """Training with validation should also return val metrics."""
    x, y = synthetic_data
    split = int(len(x) * 0.8)

    trainer = LightGBMTrainer(model_name="lgbm_test")
    metrics = trainer.train(
        x.iloc[:split],
        y.iloc[:split],
        x.iloc[split:],
        y.iloc[split:],
    )

    assert "val_mae" in metrics
    assert "val_rmse" in metrics


def test_predict_shape(synthetic_data):
    """Predict should return array of correct length."""
    x, y = synthetic_data
    trainer = LightGBMTrainer(model_name="lgbm_test")
    trainer.train(x, y)

    preds = trainer.predict(x)
    assert len(preds) == len(x)
    assert isinstance(preds, np.ndarray)


def test_save_and_load(synthetic_data, tmp_path):
    """Trainer should save and be loadable via registry."""
    x, y = synthetic_data
    trainer = LightGBMTrainer(model_name="lgbm_test", models_root=tmp_path / "models")
    metrics = trainer.train(x, y)
    trainer.save("XX", "run_001", metrics)

    model_dir = tmp_path / "models" / "XX" / "lgbm_test" / "run_001"
    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "metrics.json").exists()


def test_custom_hyperparameters(synthetic_data):
    """Trainer should respect config hyperparameters."""
    x, y = synthetic_data
    config = {"hyperparameters": {"n_estimators": 10, "num_leaves": 15}}
    trainer = LightGBMTrainer(model_name="lgbm_test", config=config)

    assert trainer.model.n_estimators == 10
    assert trainer.model.num_leaves == 15

    metrics = trainer.train(x, y)
    assert metrics["train_mae"] >= 0

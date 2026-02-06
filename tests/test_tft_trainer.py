# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simplified TFT deep learning trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for deep learning tests")

from core.exceptions import TrainingError  # noqa: E402
from models.deep_learning.tft_trainer import TFTTrainer  # noqa: E402


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create small synthetic dataset for testing."""
    np.random.seed(42)
    n = 100
    x = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(
        x["feature_1"] * 2 + x["feature_3"] * 0.1 + np.random.randn(n) * 0.5,
        name="target",
    )
    return x, y


@pytest.fixture
def tiny_config() -> dict:
    """Minimal config for fast tests."""
    return {"hyperparameters": {"hidden_size": 4, "n_layers": 1, "epochs": 2}}


def test_train_returns_metrics(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tiny_config: dict,
) -> None:
    """Training should return a dict with expected metric keys."""
    x, y = synthetic_data
    trainer = TFTTrainer(model_name="tft_test", config=tiny_config)
    metrics = trainer.train(x, y)

    assert "mae" in metrics
    assert "model_type" in metrics
    assert "epochs_trained" in metrics
    assert metrics["model_type"] == "tft"
    assert metrics["mae"] >= 0
    assert metrics["epochs_trained"] == 2


def test_train_with_validation(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tiny_config: dict,
) -> None:
    """Training with validation data should also return val_mae."""
    x, y = synthetic_data
    split = int(len(x) * 0.8)
    trainer = TFTTrainer(model_name="tft_test", config=tiny_config)
    metrics = trainer.train(x.iloc[:split], y.iloc[:split], x.iloc[split:], y.iloc[split:])

    assert "val_mae" in metrics
    assert metrics["val_mae"] >= 0


def test_predict_shape(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tiny_config: dict,
) -> None:
    """Predict should return array of correct length."""
    x, y = synthetic_data
    trainer = TFTTrainer(model_name="tft_test", config=tiny_config)
    trainer.train(x, y)

    preds = trainer.predict(x)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(x)


def test_predict_before_train_raises() -> None:
    """Predicting without training should raise TrainingError."""
    trainer = TFTTrainer(model_name="tft_test")
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(TrainingError, match="not been trained"):
        trainer.predict(x)


def test_custom_hyperparameters() -> None:
    """Trainer should respect config hyperparameters."""
    config = {
        "hyperparameters": {
            "hidden_size": 16,
            "n_layers": 3,
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 64,
            "patience": 8,
        }
    }
    trainer = TFTTrainer(model_name="tft_test", config=config)

    assert trainer.hidden_size == 16
    assert trainer.n_layers == 3
    assert trainer.epochs == 10
    assert trainer.learning_rate == 0.01
    assert trainer.batch_size == 64
    assert trainer.patience == 8


def test_save(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tiny_config: dict,
    tmp_path: object,
) -> None:
    """Trainer should save the model state dict via the registry."""
    x, y = synthetic_data
    trainer = TFTTrainer(
        model_name="tft_test",
        models_root=tmp_path / "models",  # type: ignore[operator]
        config=tiny_config,
    )
    metrics = trainer.train(x, y)
    trainer.save("XX", "run_001", metrics)

    model_dir = tmp_path / "models" / "XX" / "tft_test" / "run_001"  # type: ignore[operator]
    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "metrics.json").exists()

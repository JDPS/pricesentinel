# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ARIMA and ETS statistical trainers."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import TrainingError
from models.statistical_trainer import ARIMATrainer, ETSTrainer


def _make_synthetic_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic time-series data for testing."""
    rng = np.random.default_rng(42)
    trend = np.arange(n, dtype=float) * 0.1
    noise = rng.standard_normal(n) * 2
    y = pd.Series(50.0 + trend + noise, name="price")
    x = pd.DataFrame({"feature_1": rng.standard_normal(n)})
    return x, y


class TestARIMATrainer:
    """Tests for the ARIMATrainer class."""

    def test_train_returns_metrics(self) -> None:
        x, y = _make_synthetic_data()
        trainer = ARIMATrainer()
        metrics = trainer.train(x, y)

        assert "mae" in metrics
        assert "model_type" in metrics
        assert metrics["model_type"] == "arima"
        assert "aic" in metrics
        assert "bic" in metrics

    def test_predict_returns_correct_shape(self) -> None:
        x, y = _make_synthetic_data()
        trainer = ARIMATrainer()
        trainer.train(x, y)

        x_test = pd.DataFrame({"feature_1": np.zeros(24)})
        predictions = trainer.predict(x_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 24

    def test_predict_before_train_raises(self) -> None:
        trainer = ARIMATrainer()
        x_test = pd.DataFrame({"feature_1": np.zeros(10)})
        with pytest.raises(TrainingError, match="not been fitted"):
            trainer.predict(x_test)

    def test_validation_metrics(self) -> None:
        x, y = _make_synthetic_data(200)
        x_val = pd.DataFrame({"feature_1": np.zeros(24)})
        y_val = pd.Series(np.full(24, 60.0))

        trainer = ARIMATrainer()
        metrics = trainer.train(x, y, x_val=x_val, y_val=y_val)

        assert "val_mae" in metrics

    def test_custom_order(self) -> None:
        x, y = _make_synthetic_data(100)
        config = {"hyperparameters": {"order": (2, 1, 0)}}
        trainer = ARIMATrainer(config=config)
        metrics = trainer.train(x, y)

        assert "mae" in metrics


class TestETSTrainer:
    """Tests for the ETSTrainer class."""

    def test_train_returns_metrics(self) -> None:
        x, y = _make_synthetic_data()
        trainer = ETSTrainer()
        metrics = trainer.train(x, y)

        assert "mae" in metrics
        assert "model_type" in metrics
        assert metrics["model_type"] == "ets"
        assert "aic" in metrics

    def test_predict_returns_correct_shape(self) -> None:
        x, y = _make_synthetic_data()
        trainer = ETSTrainer()
        trainer.train(x, y)

        x_test = pd.DataFrame({"feature_1": np.zeros(24)})
        predictions = trainer.predict(x_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 24

    def test_predict_before_train_raises(self) -> None:
        trainer = ETSTrainer()
        x_test = pd.DataFrame({"feature_1": np.zeros(10)})
        with pytest.raises(TrainingError, match="not been fitted"):
            trainer.predict(x_test)

    def test_validation_metrics(self) -> None:
        x, y = _make_synthetic_data(200)
        x_val = pd.DataFrame({"feature_1": np.zeros(24)})
        y_val = pd.Series(np.full(24, 60.0))

        trainer = ETSTrainer()
        metrics = trainer.train(x, y, x_val=x_val, y_val=y_val)

        assert "val_mae" in metrics

    def test_no_trend_config(self) -> None:
        x, y = _make_synthetic_data(100)
        config = {"hyperparameters": {"trend": None}}
        trainer = ETSTrainer(config=config)
        metrics = trainer.train(x, y)

        assert "mae" in metrics

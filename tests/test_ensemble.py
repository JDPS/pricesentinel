# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ensemble trainers."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import EnsembleError
from models.ensemble_trainer import StackingTrainer, WeightedEnsembleTrainer


def _make_synthetic_data(
    n: int = 200,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create synthetic data with train/val split."""
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "feature_1": rng.standard_normal(n),
            "feature_2": rng.standard_normal(n),
            "feature_3": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(x["feature_3"] * 0.5 + rng.standard_normal(n) * 2, name="target")

    split = int(n * 0.8)
    x_train, x_val = x.iloc[:split], x.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    return x_train, y_train, x_val, y_val


class TestWeightedEnsembleTrainer:
    """Tests for the WeightedEnsembleTrainer."""

    def test_single_model_ensemble(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": ["baseline"]}}
        trainer = WeightedEnsembleTrainer(config=config)
        metrics = trainer.train(x_train, y_train, x_val, y_val)

        assert "model_type" in metrics
        assert metrics["model_type"] == "weighted_ensemble"
        assert "train_mae" in metrics

    def test_predict_returns_correct_shape(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": ["baseline"]}}
        trainer = WeightedEnsembleTrainer(config=config)
        trainer.train(x_train, y_train)

        predictions = trainer.predict(x_val)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(x_val)

    def test_equal_weights_without_validation(self) -> None:
        x_train, y_train, _, _ = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": ["baseline"]}}
        trainer = WeightedEnsembleTrainer(config=config)
        trainer.train(x_train, y_train)

        # With single model and no validation, weight should be 1.0
        assert np.allclose(trainer.weights, [1.0])

    def test_weights_sum_to_one(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": ["baseline", "baseline"]}}
        trainer = WeightedEnsembleTrainer(config=config)
        trainer.train(x_train, y_train, x_val, y_val)

        assert np.isclose(trainer.weights.sum(), 1.0)

    def test_predict_before_train_raises(self) -> None:
        trainer = WeightedEnsembleTrainer()
        x = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(EnsembleError, match="before training"):
            trainer.predict(x)

    def test_no_sub_models_raises(self) -> None:
        x_train, y_train, _, _ = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": []}}
        trainer = WeightedEnsembleTrainer(config=config)
        with pytest.raises(EnsembleError, match="No sub-models"):
            trainer.train(x_train, y_train)


class TestStackingTrainer:
    """Tests for the StackingTrainer."""

    def test_stacking_trains_successfully(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {
            "hyperparameters": {
                "sub_models": ["baseline"],
                "n_splits": 2,
            }
        }
        trainer = StackingTrainer(config=config)
        metrics = trainer.train(x_train, y_train)

        assert "model_type" in metrics
        assert metrics["model_type"] == "stacking"
        assert "train_mae" in metrics

    def test_stacking_predict_shape(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {
            "hyperparameters": {
                "sub_models": ["baseline"],
                "n_splits": 2,
            }
        }
        trainer = StackingTrainer(config=config)
        trainer.train(x_train, y_train)

        predictions = trainer.predict(x_val)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(x_val)

    def test_stacking_predict_before_train_raises(self) -> None:
        trainer = StackingTrainer()
        x = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(EnsembleError, match="before training"):
            trainer.predict(x)

    def test_stacking_no_sub_models_raises(self) -> None:
        x_train, y_train, _, _ = _make_synthetic_data()
        config = {"hyperparameters": {"sub_models": []}}
        trainer = StackingTrainer(config=config)
        with pytest.raises(EnsembleError, match="No sub-models"):
            trainer.train(x_train, y_train)

    def test_stacking_with_validation(self) -> None:
        x_train, y_train, x_val, y_val = _make_synthetic_data()
        config = {
            "hyperparameters": {
                "sub_models": ["baseline"],
                "n_splits": 2,
            }
        }
        trainer = StackingTrainer(config=config)
        metrics = trainer.train(x_train, y_train, x_val, y_val)

        assert "val_mae" in metrics

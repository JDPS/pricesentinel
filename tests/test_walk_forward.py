# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for WalkForwardValidator."""

import numpy as np
import pandas as pd
import pytest

from core.cross_validation import WalkForwardValidator
from models.sklearn_trainer import SklearnRegressorTrainer


def _make_synthetic_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic time-series data for testing."""
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "feature_1": rng.standard_normal(n),
            "feature_2": rng.standard_normal(n),
            "feature_3": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(x["feature_3"] * 0.5 + rng.standard_normal(n) * 2, name="target")
    return x, y


def _trainer_factory() -> SklearnRegressorTrainer:
    return SklearnRegressorTrainer(model_name="baseline_fast")


class TestWalkForwardValidator:
    """Tests for the WalkForwardValidator class."""

    def test_expanding_mode_returns_steps(self) -> None:
        x, y = _make_synthetic_data(200)
        validator = WalkForwardValidator(initial_train_size=100, step_size=24, mode="expanding")
        results = validator.run(x, y, _trainer_factory)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert set(results.columns) >= {"step_idx", "train_size", "test_size", "mae", "rmse"}

    def test_sliding_mode_fixed_train_size(self) -> None:
        x, y = _make_synthetic_data(200)
        validator = WalkForwardValidator(initial_train_size=100, step_size=24, mode="sliding")
        results = validator.run(x, y, _trainer_factory)

        # In sliding mode, all training sizes should equal initial_train_size
        assert (results["train_size"] == 100).all()

    def test_expanding_mode_growing_train_size(self) -> None:
        x, y = _make_synthetic_data(200)
        validator = WalkForwardValidator(initial_train_size=50, step_size=24, mode="expanding")
        results = validator.run(x, y, _trainer_factory)

        # Train sizes should be non-decreasing in expanding mode
        train_sizes = results["train_size"].tolist()
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[-1] > train_sizes[0]

    def test_metrics_are_positive(self) -> None:
        x, y = _make_synthetic_data(200)
        validator = WalkForwardValidator(initial_train_size=100, step_size=24)
        results = validator.run(x, y, _trainer_factory)

        assert (results["mae"] > 0).all()
        assert (results["rmse"] > 0).all()

    def test_dataset_too_small_raises(self) -> None:
        x, y = _make_synthetic_data(10)
        validator = WalkForwardValidator(initial_train_size=50, step_size=24)
        with pytest.raises(ValueError, match="too small"):
            validator.run(x, y, _trainer_factory)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            WalkForwardValidator(initial_train_size=100, mode="invalid")

    def test_invalid_step_size_raises(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardValidator(initial_train_size=100, step_size=0)

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Optuna hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import HyperparameterError
from models.hpo import OptunaHPO


@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset for HPO testing."""
    np.random.seed(42)
    n = 300
    x = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(x["feature_1"] * 2 + x["feature_3"] * 0.1 + np.random.randn(n) * 0.5)
    return x, y


def test_hpo_random_forest(synthetic_data):
    """HPO should return valid hyperparameters for random forest."""
    x, y = synthetic_data
    hpo = OptunaHPO(algorithm="random_forest", n_trials=2, cv_splits=2)
    best_params = hpo.optimize(x, y)

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert best_params["n_estimators"] >= 50
    assert best_params["max_depth"] >= 3


def test_hpo_xgboost(synthetic_data):
    """HPO should return valid hyperparameters for XGBoost."""
    x, y = synthetic_data
    hpo = OptunaHPO(algorithm="xgboost", n_trials=2, cv_splits=2)
    best_params = hpo.optimize(x, y)

    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert best_params["learning_rate"] > 0


def test_hpo_lightgbm(synthetic_data):
    """HPO should return valid hyperparameters for LightGBM."""
    x, y = synthetic_data
    hpo = OptunaHPO(algorithm="lightgbm", n_trials=2, cv_splits=2)
    best_params = hpo.optimize(x, y)

    assert isinstance(best_params, dict)
    assert "num_leaves" in best_params


def test_hpo_unknown_algorithm():
    """HPO should raise for unknown algorithms without custom space."""
    with pytest.raises(HyperparameterError, match="No default search space"):
        OptunaHPO(algorithm="unknown_algo")


def test_hpo_custom_search_space(synthetic_data):
    """HPO should accept custom search spaces."""
    x, y = synthetic_data
    custom_space = {
        "n_estimators": ("int", 10, 50),
        "max_depth": ("int", 2, 5),
    }
    hpo = OptunaHPO(
        algorithm="random_forest",
        n_trials=2,
        cv_splits=2,
        search_space=custom_space,
    )
    best_params = hpo.optimize(x, y)

    assert 10 <= best_params["n_estimators"] <= 50
    assert 2 <= best_params["max_depth"] <= 5


def test_hpo_rmse_metric(synthetic_data):
    """HPO should work with RMSE metric."""
    x, y = synthetic_data
    hpo = OptunaHPO(algorithm="random_forest", n_trials=2, cv_splits=2, metric="rmse")
    best_params = hpo.optimize(x, y)

    assert isinstance(best_params, dict)

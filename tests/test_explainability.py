# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SHAP explainability module."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import ModelError
from models.explainability import SHAPExplainer
from models.sklearn_trainer import SklearnRegressorTrainer


def _make_trained_model() -> tuple:
    """Train a simple model and return (model, x_test)."""
    rng = np.random.default_rng(42)
    n = 100
    x = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.standard_normal(n),
            "feature_c": np.arange(n, dtype=float),
        }
    )
    y = pd.Series(x["feature_c"] * 0.5 + rng.standard_normal(n) * 2, name="target")

    trainer = SklearnRegressorTrainer(model_name="baseline_fast")
    trainer.train(x, y)
    return trainer.model, x


class TestSHAPExplainer:
    """Tests for the SHAPExplainer class."""

    def test_none_model_raises(self) -> None:
        with pytest.raises(ModelError, match="None model"):
            SHAPExplainer(model=None)

    def test_compute_shap_values_shape(self) -> None:
        model, x = _make_trained_model()
        explainer = SHAPExplainer(model)
        shap_values = explainer.compute_shap_values(x)

        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == x.shape

    def test_feature_importance_sorted(self) -> None:
        model, x = _make_trained_model()
        explainer = SHAPExplainer(model)
        importance = explainer.feature_importance(x)

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        # Should be sorted descending
        values = importance["importance"].tolist()
        assert values == sorted(values, reverse=True)

    def test_feature_importance_with_names(self) -> None:
        model, x = _make_trained_model()
        names = ["alpha", "beta", "gamma"]
        explainer = SHAPExplainer(model, feature_names=names)
        importance = explainer.feature_importance(x)

        assert set(importance["feature"].tolist()) == set(names)

    def test_save_summary(self, tmp_path) -> None:
        model, x = _make_trained_model()
        explainer = SHAPExplainer(model)
        output = tmp_path / "shap_summary.md"
        explainer.save_summary(x, output)

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "SHAP Feature Importance Summary" in content
        assert "Feature Rankings" in content

    def test_empty_dataframe_raises(self) -> None:
        model, _ = _make_trained_model()
        explainer = SHAPExplainer(model)
        with pytest.raises(ModelError, match="empty"):
            explainer.compute_shap_values(pd.DataFrame())

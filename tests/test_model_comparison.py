# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ModelComparison framework."""

import numpy as np
import pandas as pd
import pytest

from experiments.model_comparison import ModelComparison


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


class TestModelComparison:
    """Tests for the ModelComparison class."""

    def test_single_model_walk_forward(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        results = comparison.run(
            x, y, cv_method="walk_forward", initial_train_size=100, step_size=24
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results.iloc[0]["model"] == "baseline"
        assert results.iloc[0]["mean_mae"] > 0

    def test_multiple_models_walk_forward(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline", "xgboost"],
        )
        results = comparison.run(
            x, y, cv_method="walk_forward", initial_train_size=100, step_size=24
        )

        assert len(results) == 2
        # Results should be sorted by mean_mae ascending
        assert results.iloc[0]["mean_mae"] <= results.iloc[1]["mean_mae"]

    def test_time_series_split_method(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        results = comparison.run(x, y, cv_method="time_series_split", n_splits=3)

        assert len(results) == 1
        assert results.iloc[0]["n_steps"] == 3

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            ModelComparison(
                country_code="XX",
                model_names=["nonexistent_model"],
            )

    def test_generate_report(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        results = comparison.run(
            x, y, cv_method="walk_forward", initial_train_size=100, step_size=24
        )
        report = comparison.generate_report(results)

        assert "Model Comparison Report" in report
        assert "baseline" in report
        assert "Ranking" in report

    def test_generate_report_without_run_raises(self) -> None:
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        with pytest.raises(ValueError, match="No results"):
            comparison.generate_report()

    def test_results_stored_internally(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        results = comparison.run(
            x, y, cv_method="walk_forward", initial_train_size=100, step_size=24
        )

        assert comparison.results_ is not None
        assert comparison.results_.equals(results)

    def test_invalid_cv_method_raises(self) -> None:
        x, y = _make_synthetic_data()
        comparison = ModelComparison(
            country_code="XX",
            model_names=["baseline"],
        )
        with pytest.raises(ValueError, match="cv_method"):
            comparison.run(x, y, cv_method="invalid")

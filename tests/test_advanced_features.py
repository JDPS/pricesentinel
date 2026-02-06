# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for advanced feature engineering (Fourier, volatility, momentum, selection)."""

import numpy as np
import pandas as pd
import pytest

from core.data_manager import CountryDataManager
from core.feature_selection import FeatureSelector
from core.features import FeatureEngineer
from core.repository import CsvDataRepository


def _make_engineer(tmp_path):
    """Create a FeatureEngineer with advanced features enabled."""
    manager = CountryDataManager("XX", base_path=str(tmp_path / "data"))
    manager.create_directories()
    repo = CsvDataRepository(manager)

    config = {
        "use_cross_border_flows": False,
        "feature_windows": {"lags": [1, 2], "rolling_windows": [], "rolling_stats": ["mean"]},
        "use_weather_features": False,
        "use_gas_features": False,
        "use_event_features": False,
        "neighbors": [],
        "custom_feature_plugins": [],
        "use_fourier_features": True,
        "fourier_periods": [24, 168],
        "use_price_volatility": True,
        "use_price_momentum": True,
    }

    return FeatureEngineer("XX", repository=repo, features_config=config), manager


def _create_prices_df(n_hours=200):
    """Create a synthetic electricity prices DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h", tz="UTC")
    np.random.seed(42)
    prices = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.randn(n_hours) * 3
    return pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})


class TestFourierFeatures:
    def test_fourier_values_in_range(self, tmp_path, monkeypatch):
        """Fourier features should produce values in [-1, 1]."""
        monkeypatch.chdir(tmp_path)
        engineer, manager = _make_engineer(tmp_path)

        df = _create_prices_df()
        df["hour"] = df["timestamp"].dt.hour

        result = engineer._add_fourier_features(df.copy())

        for col in ["sin_24h", "cos_24h", "sin_168h", "cos_168h"]:
            assert col in result.columns
            assert result[col].min() >= -1.0 - 1e-10
            assert result[col].max() <= 1.0 + 1e-10

    def test_fourier_custom_periods(self, tmp_path, monkeypatch):
        """Fourier should accept custom periods."""
        monkeypatch.chdir(tmp_path)
        engineer, _ = _make_engineer(tmp_path)

        df = _create_prices_df()
        df["hour"] = df["timestamp"].dt.hour

        result = engineer._add_fourier_features(df.copy(), periods=[12])

        assert "sin_12h" in result.columns
        assert "cos_12h" in result.columns
        assert "sin_24h" not in result.columns


class TestVolatilityFeatures:
    def test_volatility_columns_created(self, tmp_path, monkeypatch):
        """Volatility features should create expected columns."""
        monkeypatch.chdir(tmp_path)
        engineer, _ = _make_engineer(tmp_path)

        df = _create_prices_df()
        result = engineer._add_volatility_features(df.copy())

        for window in [12, 24, 48]:
            assert f"price_volatility_{window}" in result.columns
            assert f"price_range_{window}" in result.columns

    def test_volatility_nan_only_warmup(self, tmp_path, monkeypatch):
        """Volatility should be NaN only for initial warmup rows."""
        monkeypatch.chdir(tmp_path)
        engineer, _ = _make_engineer(tmp_path)

        df = _create_prices_df(n_hours=100)
        result = engineer._add_volatility_features(df.copy())

        # After warmup (max window = 48), volatility should not be NaN
        assert result["price_volatility_48"].iloc[50:].notna().all()

    def test_volatility_non_negative(self, tmp_path, monkeypatch):
        """Standard deviation (volatility) should be non-negative."""
        monkeypatch.chdir(tmp_path)
        engineer, _ = _make_engineer(tmp_path)

        df = _create_prices_df()
        result = engineer._add_volatility_features(df.copy())

        for window in [12, 24, 48]:
            valid = result[f"price_volatility_{window}"].dropna()
            assert (valid >= 0).all()


class TestMomentumFeatures:
    def test_momentum_columns_created(self, tmp_path, monkeypatch):
        """Momentum features should create expected columns."""
        monkeypatch.chdir(tmp_path)
        engineer, _ = _make_engineer(tmp_path)

        df = _create_prices_df()
        result = engineer._add_momentum_features(df.copy())

        for window in [6, 12, 24]:
            assert f"price_roc_{window}" in result.columns


class TestFeatureSelector:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n = 300
        x = pd.DataFrame(
            {
                "important_1": np.random.randn(n),
                "important_2": np.random.randn(n),
                "noise_1": np.random.randn(n) * 0.01,
                "noise_2": np.random.randn(n) * 0.01,
            }
        )
        y = pd.Series(x["important_1"] * 3 + x["important_2"] * 2 + np.random.randn(n) * 0.1)
        return x, y

    def test_select_returns_non_empty(self, synthetic_data):
        """Feature selection should return at least one feature."""
        x, y = synthetic_data
        selector = FeatureSelector(threshold=0.001)
        selected = selector.select(x, y)

        assert len(selected) > 0
        assert all(col in x.columns for col in selected)

    def test_important_features_ranked_higher(self, synthetic_data):
        """Important features should be ranked above noise features."""
        x, y = synthetic_data
        selector = FeatureSelector()
        ranking = selector.get_importance_ranking(x, y)

        top_features = ranking["feature"].head(2).tolist()
        assert "important_1" in top_features or "important_2" in top_features

    def test_max_features_limit(self, synthetic_data):
        """max_features should limit the number of selected features."""
        x, y = synthetic_data
        selector = FeatureSelector(threshold=0.0, max_features=2)
        selected = selector.select(x, y)

        assert len(selected) <= 2

    def test_threshold_filters_noise(self, synthetic_data):
        """High threshold should filter out low-importance features."""
        x, y = synthetic_data
        selector = FeatureSelector(threshold=0.3)
        selected = selector.select(x, y)

        # With high threshold, likely only important features survive
        assert len(selected) <= 3

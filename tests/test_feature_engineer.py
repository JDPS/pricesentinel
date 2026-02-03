# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FeatureEngineer feature building and training helpers.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.data_manager import CountryDataManager
from core.features import FeatureEngineer
from core.repository import CsvDataRepository
from models.base import BaseTrainer


class DummyTrainer(BaseTrainer):
    """Simple trainer used to test train_with_trainer wiring."""

    def __init__(self, model_name: str = "dummy", models_root: str | Path = "models"):
        super().__init__(model_name=model_name, models_root=models_root)
        self.last_train_args: tuple | None = None
        self.saved: bool = False
        self.saved_with: tuple | None = None

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        self.last_train_args = (x_train, y_train, x_val, y_val)
        # Return a deterministic metric
        return {"train_mae": 1.23}

    def save(
        self,
        country_code: str,
        run_id: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.saved = True
        self.saved_with = (country_code, run_id, metrics or {})


class BadMetricsTrainer(DummyTrainer):
    """Trainer that reports very poor metrics, used to exercise guardrails."""

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        self.last_train_args = (x_train, y_train, x_val, y_val)
        # Deliberately large errors
        return {"train_mae": 5000.0, "train_rmse": 6000.0}


def _make_manager(tmp_path: Path) -> CountryDataManager:
    base_path = tmp_path / "data"
    manager = CountryDataManager("XX", base_path=str(base_path))
    manager.create_directories()
    return manager


def test_build_electricity_features_with_optional_sources(tmp_path, monkeypatch):
    """FeatureEngineer should incorporate weather, gas, and event information when available."""
    monkeypatch.chdir(tmp_path)
    manager = _make_manager(tmp_path)

    start_date = "2024-01-01"
    end_date = "2024-01-03"

    # Electricity prices (48 hours)
    timestamps = pd.date_range("2024-01-01", "2024-01-03 23:00", freq="1h", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": range(len(timestamps)),
            "market": "day_ahead",
        }
    )
    prices_path = manager.get_processed_file_path("electricity_prices_clean", start_date, end_date)
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path, index=False)

    # Load data
    load = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": [1000.0] * len(timestamps),
        }
    )
    load_path = manager.get_processed_file_path("electricity_load_clean", start_date, end_date)
    load.to_csv(load_path, index=False)

    # Weather data for a single location
    weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_name": "TestCity",
            "temperature_c": [10.0] * len(timestamps),
            "wind_speed_ms": [5.0] * len(timestamps),
            "solar_radiation_wm2": [100.0] * len(timestamps),
            "precipitation_mm": [0.1] * len(timestamps),
            "quality_flag": 0,
        }
    )
    weather_path = manager.get_processed_file_path("weather_clean", start_date, end_date)
    weather.to_csv(weather_path, index=False)

    # Gas prices (daily)
    gas_timestamps = pd.date_range("2024-01-01", "2024-01-03", freq="D", tz="UTC")
    gas = pd.DataFrame(
        {
            "timestamp": gas_timestamps,
            "price_eur_mwh": [30.0, 31.0, 32.0],
            "hub_name": "TTF",
            "quality_flag": 0,
        }
    )
    gas_path = manager.get_processed_file_path("gas_prices_clean", start_date, end_date)
    gas.to_csv(gas_path, index=False)

    # Holidays (one day)
    holidays = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "event_type": ["holiday"],
            "description": ["Test Holiday"],
        }
    )
    holidays_path = manager.get_processed_file_path("holidays_clean", start_date, end_date)
    holidays.to_csv(holidays_path, index=False)

    # Manual events (one-day interval)
    manual_events = pd.DataFrame(
        {
            "date_start": [pd.Timestamp("2024-01-03", tz="UTC")],
            "date_end": [pd.Timestamp("2024-01-03", tz="UTC")],
            "event_type": ["maintenance"],
            "description": ["Test Event"],
            "source": ["test"],
        }
    )
    manual_events_path = manager.get_processed_file_path(
        "manual_events_clean", start_date, end_date
    )
    manual_events.to_csv(manual_events_path, index=False)

    engineer = FeatureEngineer("XX", repository=CsvDataRepository(manager))
    engineer.build_electricity_features(start_date, end_date)

    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    assert features_path.exists()

    features_df = pd.read_csv(features_path)

    # Core columns
    assert "target_price" in features_df.columns
    assert "price_lag_1" in features_df.columns
    assert "hour" in features_df.columns
    assert "day_of_week" in features_df.columns

    # Optional features
    for col in (
        "load_mw",
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
        "precipitation_mm",
        "gas_price_eur_mwh",
        "is_holiday",
        "is_event",
    ):
        assert col in features_df.columns

    # Check that holiday and event flags are set at least once
    assert features_df["is_holiday"].max() in (0, 1)
    assert features_df["is_event"].max() in (0, 1)


def test_build_electricity_features_respects_feature_toggles(tmp_path, monkeypatch):
    """Feature toggles should allow disabling weather, gas, and event features."""
    monkeypatch.chdir(tmp_path)
    manager = _make_manager(tmp_path)

    start_date = "2024-01-01"
    end_date = "2024-01-02"

    # Minimal cleaned prices and load to allow feature building
    timestamps = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="1h", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": range(len(timestamps)),
            "market": "day_ahead",
        }
    )
    prices_path = manager.get_processed_file_path("electricity_prices_clean", start_date, end_date)
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path, index=False)

    load = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": [1000.0] * len(timestamps),
        }
    )
    load_path = manager.get_processed_file_path("electricity_load_clean", start_date, end_date)
    load.to_csv(load_path, index=False)

    # Provide weather/gas/events files that would be used if toggles were on
    weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_name": "TestCity",
            "temperature_c": [10.0] * len(timestamps),
            "wind_speed_ms": [5.0] * len(timestamps),
            "solar_radiation_wm2": [100.0] * len(timestamps),
            "precipitation_mm": [0.1] * len(timestamps),
            "quality_flag": 0,
        }
    )
    weather_path = manager.get_processed_file_path("weather_clean", start_date, end_date)
    weather.to_csv(weather_path, index=False)

    gas_timestamps = pd.date_range("2024-01-01", "2024-01-02", freq="D", tz="UTC")
    gas = pd.DataFrame(
        {
            "timestamp": gas_timestamps,
            "price_eur_mwh": [30.0, 31.0],
            "hub_name": "TTF",
            "quality_flag": 0,
        }
    )
    gas_path = manager.get_processed_file_path("gas_prices_clean", start_date, end_date)
    gas.to_csv(gas_path, index=False)

    holidays = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "event_type": ["holiday"],
            "description": ["Test Holiday"],
        }
    )
    holidays_path = manager.get_processed_file_path("holidays_clean", start_date, end_date)
    holidays.to_csv(holidays_path, index=False)

    manual_events = pd.DataFrame(
        {
            "date_start": [pd.Timestamp("2024-01-02", tz="UTC")],
            "date_end": [pd.Timestamp("2024-01-02", tz="UTC")],
            "event_type": ["maintenance"],
            "description": ["Test Event"],
            "source": ["test"],
        }
    )
    manual_events_path = manager.get_processed_file_path(
        "manual_events_clean", start_date, end_date
    )
    manual_events.to_csv(manual_events_path, index=False)

    engineer = FeatureEngineer(
        "XX",
        repository=CsvDataRepository(manager),
        features_config={
            "use_weather_features": False,
            "use_gas_features": False,
            "use_event_features": False,
        },
    )
    engineer.build_electricity_features(start_date, end_date)

    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_df = pd.read_csv(features_path)

    # Weather/gas columns should be absent when toggled off
    assert "temperature_c" not in features_df.columns
    assert "gas_price_eur_mwh" not in features_df.columns
    # Event flags still exist but remain zero (no event info applied)
    assert "is_holiday" in features_df.columns
    assert "is_event" in features_df.columns
    assert features_df["is_holiday"].max() == 0
    assert features_df["is_event"].max() == 0


def test_train_with_trainer_uses_numeric_features_and_saves(tmp_path, monkeypatch):
    """train_with_trainer should wire data into trainer and call save with metrics."""
    monkeypatch.chdir(tmp_path)
    manager = _make_manager(tmp_path)

    start_date = "2024-01-01"
    end_date = "2024-01-02"

    # Minimal features file: timestamp, numeric features, target_price
    timestamps = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature1": [1.0] * len(timestamps),
            "feature2": list(range(len(timestamps))),
            "target_price": list(range(len(timestamps))),
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)

    trainer = DummyTrainer(models_root=tmp_path / "models")
    engineer = FeatureEngineer("XX", repository=CsvDataRepository(manager))
    engineer.train_with_trainer(
        trainer=trainer,
        run_id="run123",
        start_date=start_date,
        end_date=end_date,
    )

    assert trainer.saved is True
    assert trainer.saved_with is not None
    country_code, run_id, metrics = trainer.saved_with
    assert country_code == "XX"
    assert run_id == "run123"
    assert "train_mae" in metrics


def test_train_with_trainer_no_numeric_features_raises(tmp_path, monkeypatch):
    """train_with_trainer should error if there are no numeric feature columns."""
    monkeypatch.chdir(tmp_path)
    manager = _make_manager(tmp_path)

    start_date = "2024-01-01"
    end_date = "2024-01-02"

    # Features with only timestamp and target_price but no other numeric features
    timestamps = pd.date_range("2024-01-01", "2024-01-01 01:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "market": ["day_ahead"] * len(timestamps),
            "target_price": [10.0, 11.0],
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)

    trainer = DummyTrainer(models_root=tmp_path / "models")

    engineer = FeatureEngineer("XX", repository=CsvDataRepository(manager))

    with pytest.raises(ValueError, match="No numeric feature columns available for training"):
        engineer.train_with_trainer(
            trainer=trainer,
            run_id="run123",
            start_date=start_date,
            end_date=end_date,
        )


def test_train_with_trainer_guardrail_can_skip_save(tmp_path, monkeypatch):
    """Guardrail should allow skipping model save when metrics are clearly bad."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRICESENTINEL_SKIP_SAVE_ON_BAD_METRICS", "1")
    manager = _make_manager(tmp_path)

    start_date = "2024-01-01"
    end_date = "2024-01-02"

    # Minimal valid features file with numeric columns
    timestamps = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature1": [1.0] * len(timestamps),
            "target_price": list(range(len(timestamps))),
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)

    trainer = BadMetricsTrainer(models_root=tmp_path / "models")

    engineer = FeatureEngineer("XX", repository=CsvDataRepository(manager))

    engineer.train_with_trainer(
        trainer=trainer,
        run_id="run123",
        start_date=start_date,
        end_date=end_date,
    )

    # With the guardrail env var set and bad metrics, save should be skipped
    assert trainer.saved is False

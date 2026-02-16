# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the forecast generation stage in Pipeline.
"""

import pandas as pd
import pytest

from config.country_registry import CountryRegistry
from core.data_manager import CountryDataManager
from core.pipeline_builder import PipelineBuilder
from data_fetchers.mock import register_mock_country
from models.sklearn_trainer import SklearnRegressorTrainer


@pytest.fixture
def mock_country_pipeline(tmp_path, monkeypatch):
    """Initialise a pipeline for the mock country with a temporary data/models root."""
    monkeypatch.chdir(tmp_path)
    CountryRegistry.clear()
    register_mock_country()
    pipeline = PipelineBuilder.create_pipeline("XX")
    yield pipeline
    CountryRegistry.clear()


def test_generate_forecast_uses_latest_features_and_model(tmp_path, mock_country_pipeline):
    """
    Forecast generation should load latest features and corresponding model and produce a CSV.
    """
    pipeline = mock_country_pipeline

    # Prepare synthetic features file
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    start_date = "2024-01-01"
    end_date = "2024-01-02"
    timestamps = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="1h", tz="UTC")

    features_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature1": range(len(timestamps)),
            "target_price": range(len(timestamps)),
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)

    # Train a small model and save it under the pipeline's run_id
    trainer = SklearnRegressorTrainer(models_root=tmp_path / "models")
    x = features_df[["feature1"]]
    y = features_df["target_price"]
    trainer.train(x_train=x, y_train=y)
    trainer.save(country_code="XX", run_id=pipeline.run_id, metrics={"train_mae": 0.0})

    # Run forecast generation for a specific date
    forecast_date = "2024-01-02"
    pipeline.generate_forecast(forecast_date, model_name=trainer.model_name)

    forecasts_dir = manager.get_processed_path() / "forecasts"
    forecast_files = list(forecasts_dir.glob("XX_forecast_20240102_*.csv"))
    assert forecast_files, "Expected a forecast CSV to be created"

    forecast_df = pd.read_csv(forecast_files[0], parse_dates=["forecast_timestamp"])
    assert "forecast_price_eur_mwh" in forecast_df.columns
    assert "forecast_p10_eur_mwh" in forecast_df.columns
    assert "forecast_p50_eur_mwh" in forecast_df.columns
    assert "forecast_p90_eur_mwh" in forecast_df.columns
    # All forecasts should correspond to the requested date
    assert all(forecast_df["forecast_timestamp"].dt.date.astype(str) == forecast_date)


def test_generate_forecast_resolves_champion_model(tmp_path, mock_country_pipeline):
    """`model_name=champion` should resolve to champion.json model_name."""
    pipeline = mock_country_pipeline

    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    start_date = "2024-01-01"
    end_date = "2024-01-02"
    timestamps = pd.date_range("2024-01-01", "2024-01-02 23:00", freq="1h", tz="UTC")

    features_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature1": range(len(timestamps)),
            "target_price": range(len(timestamps)),
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)

    trainer = SklearnRegressorTrainer(models_root=tmp_path / "models")
    x = features_df[["feature1"]]
    y = features_df["target_price"]
    trainer.train(x_train=x, y_train=y)
    trainer.save(country_code="XX", run_id=pipeline.run_id, metrics={"train_mae": 0.0})

    pipeline.model_registry.set_champion(
        country_code="XX",
        model_name=trainer.model_name,
        run_id=pipeline.run_id,
        trained_window={"start": start_date, "end": end_date},
    )

    pipeline.generate_forecast("2024-01-02", model_name="champion")

    forecasts_dir = manager.get_processed_path() / "forecasts"
    forecast_files = list(forecasts_dir.glob("XX_forecast_20240102_baseline.csv"))
    assert forecast_files, "Expected champion-resolved forecast file to be created"


def test_generate_forecast_applies_scorecard_interval_calibration(tmp_path, mock_country_pipeline):
    """Recent low interval coverage should widen forecast intervals via calibration."""
    pipeline = mock_country_pipeline

    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    start_date = "2024-01-31"
    end_date = "2024-02-01"
    forecast_date = "2024-02-01"
    timestamps = pd.date_range("2024-01-31", "2024-02-01 23:00", freq="1h", tz="UTC")

    features_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature1": range(len(timestamps)),
            "target_price": range(len(timestamps)),
        }
    )
    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)

    trainer = SklearnRegressorTrainer(models_root=tmp_path / "models")
    x = features_df[["feature1"]]
    y = features_df["target_price"]
    trainer.train(x_train=x, y_train=y)
    trainer.save(country_code="XX", run_id=pipeline.run_id, metrics={"train_mae": 1.0})

    scorecard_dir = manager.get_processed_path() / "scorecards"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    score_df = pd.DataFrame(
        {
            "country_code": ["XX"] * 12,
            "target_date": [f"2024-01-{d:02d}" for d in range(20, 32)],
            "status": ["ok"] * 12,
            "quantile_coverage_10_90": [0.50] * 12,
        }
    )
    score_df.to_csv(scorecard_dir / "daily_scorecard.csv", index=False)

    pipeline.generate_forecast(forecast_date, model_name=trainer.model_name)

    forecasts_dir = manager.get_processed_path() / "forecasts"
    forecast_files = list(forecasts_dir.glob("XX_forecast_20240201_*.csv"))
    assert forecast_files, "Expected a calibrated forecast CSV to be created"

    forecast_df = pd.read_csv(forecast_files[0])
    assert (forecast_df["uncertainty_source"] == "scorecard_coverage_calibrated").all()
    assert float(forecast_df["interval_calibration_factor"].iloc[0]) > 1.0
    assert int(forecast_df["interval_calibration_samples"].iloc[0]) >= 7

# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for daily ops forecast evaluation and scorecard upserts."""

import argparse
from pathlib import Path

import pandas as pd
import pytest

from core.data_manager import CountryDataManager
from experiments.daily_ops import (
    DailyScoreRecord,
    _evaluate_daily,
    _run_forecast_mode,
    _upsert_score_record,
)


def test_upsert_score_record_keeps_single_canonical_row(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    first = DailyScoreRecord(
        country_code="XX",
        target_date="2024-01-01",
        status="deferred",
        reason="actuals_incomplete",
        forecast_file="f1.csv",
        model_name="baseline",
        rows_merged=0,
        mae=None,
        rmse=None,
        mape=None,
        directional_accuracy=None,
        peak_hour_abs_error=None,
        generated_at_utc="2026-01-01T00:00:00+00:00",
    )
    _upsert_score_record(manager, first)

    second = DailyScoreRecord(
        country_code="XX",
        target_date="2024-01-01",
        status="ok",
        reason="evaluation_complete",
        forecast_file="f1.csv",
        model_name="baseline",
        rows_merged=24,
        mae=1.0,
        rmse=2.0,
        mape=3.0,
        directional_accuracy=0.5,
        peak_hour_abs_error=4.0,
        generated_at_utc="2026-01-01T01:00:00+00:00",
    )
    csv_path, _ = _upsert_score_record(manager, second)

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.loc[0, "status"] == "ok"
    assert int(df.loc[0, "rows_merged"]) == 24


def test_evaluate_daily_defers_when_forecast_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    record = _evaluate_daily(manager, "XX", "2024-01-01")

    assert record.status == "deferred"
    assert record.reason == "forecast_missing"


def test_evaluate_daily_computes_metrics_when_data_complete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    target_date = "2024-01-01"
    timestamps = pd.date_range(target_date, periods=24, freq="h", tz="UTC")

    forecast_df = pd.DataFrame(
        {
            "forecast_timestamp": timestamps,
            "forecast_price_eur_mwh": [50.0] * 24,
            "forecast_p10_eur_mwh": [48.0] * 24,
            "forecast_p50_eur_mwh": [50.0] * 24,
            "forecast_p90_eur_mwh": [52.0] * 24,
            "model_name": ["baseline"] * 24,
            "run_id": ["run_x"] * 24,
        }
    )
    forecasts_dir = manager.get_processed_path() / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecasts_dir / "XX_forecast_20240101_baseline.csv", index=False)

    actual_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": [52.0] * 24,
        }
    )
    actual_path = manager.get_processed_file_path(
        "electricity_prices_clean", target_date, target_date
    )
    actual_df.to_csv(actual_path, index=False)

    record = _evaluate_daily(manager, "XX", target_date)

    assert record.status == "ok"
    assert record.rows_merged == 24
    assert record.mae == 2.0
    assert record.rmse == 2.0
    assert record.model_name == "baseline"
    assert record.quantile_coverage_10_90 == 1.0
    assert record.pinball_loss_avg is not None
    assert record.interval_width_avg == 4.0


@pytest.mark.asyncio
async def test_run_forecast_mode_uses_only_historical_context(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    class _DummyModelRegistry:
        @staticmethod
        def resolve_model_name(country_code: str, requested_model_name: str) -> str:
            _ = country_code
            _ = requested_model_name
            return "baseline"

    class _DummyPipeline:
        def __init__(self) -> None:
            self.data_manager = CountryDataManager("XX", base_path="data")
            self.data_manager.create_directories()
            self.model_registry = _DummyModelRegistry()
            self.fetch_window: tuple[str, str] | None = None
            self.cleaned_window: tuple[str, str] | None = None
            self.feature_window: tuple[str, str] | None = None
            self.forecast_args: tuple[str, str] | None = None

        async def fetch_data(self, start_date: str, end_date: str) -> None:
            self.fetch_window = (start_date, end_date)

        def clean_and_verify(self, start_date: str, end_date: str) -> None:
            self.cleaned_window = (start_date, end_date)

        def engineer_features(self, start_date: str, end_date: str) -> None:
            self.feature_window = (start_date, end_date)

        def generate_forecast(self, target_date: str, model_name: str) -> None:
            self.forecast_args = (target_date, model_name)
            forecasts_dir = self.data_manager.get_processed_path() / "forecasts"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"XX_forecast_{target_date.replace('-', '')}_baseline.csv"
            (forecasts_dir / file_name).write_text(
                "forecast_timestamp,forecast_price_eur_mwh\n", encoding="utf-8"
            )

    pipeline = _DummyPipeline()

    monkeypatch.setattr("experiments.daily_ops.auto_register_countries", lambda: None)
    monkeypatch.setattr(
        "experiments.daily_ops.PipelineBuilder.create_pipeline",
        lambda _country: pipeline,
    )
    monkeypatch.setattr(
        "experiments.daily_ops.generate_health_summary",
        lambda _country, as_of_date=None: {
            "json_path": "health.json",
            "markdown_path": "health.md",
            "summary": {"alerts": {"overall_status": "ok"}},
            "as_of_date": as_of_date,
        },
    )

    args = argparse.Namespace(
        country="XX",
        target_date="2024-02-10",
        model_name="champion",
        history_days=45,
        skip_fetch=False,
    )

    result = await _run_forecast_mode(args)

    assert pipeline.fetch_window is not None
    assert pipeline.fetch_window[0] == "2023-12-27"
    assert pipeline.fetch_window[1] == "2024-02-09"
    assert pipeline.cleaned_window == pipeline.fetch_window
    assert pipeline.feature_window == pipeline.fetch_window
    assert pipeline.forecast_args == ("2024-02-10", "champion")
    assert result["exists"] is True
    assert Path(result["forecast_path"]).exists()

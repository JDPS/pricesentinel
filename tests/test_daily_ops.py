# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for daily ops forecast evaluation and scorecard upserts."""

import pandas as pd

from core.data_manager import CountryDataManager
from experiments.daily_ops import DailyScoreRecord, _evaluate_daily, _upsert_score_record


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

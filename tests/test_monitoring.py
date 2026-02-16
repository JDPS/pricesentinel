# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for monitoring health summary generation."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from core.data_manager import CountryDataManager
from core.monitoring import generate_health_summary


def _touch_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("col\n1\n", encoding="utf-8")


def test_generate_health_summary_warns_when_no_data(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    out = generate_health_summary("XX", as_of_date="2024-01-05")

    summary = out["summary"]
    assert summary["alerts"]["overall_status"] in {"warn", "critical"}
    assert Path(out["json_path"]).exists()
    assert Path(out["markdown_path"]).exists()


def test_generate_health_summary_ok_with_recent_data_and_good_scores(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("XX", base_path="data")
    manager.create_directories()

    # Create recent raw files for freshness checks
    for source in ("electricity", "weather", "gas"):
        raw_file = manager.get_raw_path(source) / f"XX_{source}_20240101_20240105.csv"
        _touch_csv(raw_file)
        ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
        os.utime(raw_file, (ts, ts))

    # Create scorecard with good recent metrics
    score_dir = manager.get_processed_path() / "scorecards"
    score_dir.mkdir(parents=True, exist_ok=True)
    score = pd.DataFrame(
        {
            "country_code": ["XX"] * 7,
            "target_date": [f"2024-01-0{i}" for i in range(1, 8)],
            "status": ["ok"] * 7,
            "reason": ["evaluation_complete"] * 7,
            "forecast_file": ["f.csv"] * 7,
            "model_name": ["baseline"] * 7,
            "rows_merged": [24] * 7,
            "mae": [10.0] * 7,
            "rmse": [12.0] * 7,
            "mape": [15.0] * 7,
            "directional_accuracy": [0.7] * 7,
            "peak_hour_abs_error": [8.0] * 7,
            "generated_at_utc": [datetime.now(UTC).isoformat()] * 7,
        }
    )
    score.to_csv(score_dir / "daily_scorecard.csv", index=False)

    # Expected next-day forecast file
    forecast_dir = manager.get_processed_path() / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    _touch_csv(forecast_dir / "XX_forecast_20240106_baseline.csv")

    out = generate_health_summary("XX", as_of_date="2024-01-05")

    summary = out["summary"]
    assert summary["alerts"]["overall_status"] == "ok"
    assert summary["rolling"]["7d"]["records"] >= 5

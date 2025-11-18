"""Tests for inspection helpers that summarise runs and forecasts."""

import pandas as pd

from core.inspection import inspect_run


def test_inspect_run_with_metrics_and_forecast(tmp_path, monkeypatch):
    """inspect_run should load metrics and summarise forecast CSVs."""
    monkeypatch.chdir(tmp_path)

    # Create models/XX/baseline/run123/metrics.json
    models_root = tmp_path / "models" / "XX" / "baseline" / "run123"
    models_root.mkdir(parents=True, exist_ok=True)
    metrics_path = models_root / "metrics.json"
    metrics_path.write_text('{"train_mae": 1.23, "train_rmse": 2.34}', encoding="utf-8")

    # Create data/XX/processed/forecasts/XX_forecast_20240101_baseline.csv
    forecasts_dir = tmp_path / "data" / "XX" / "processed" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "forecast_timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]),
            "forecast_price_eur_mwh": [10.0, 20.0],
            "source_features_file": ["XX_electricity_features_20240101_20240102.csv"] * 2,
            "model_name": ["baseline"] * 2,
            "run_id": ["run123"] * 2,
        }
    )
    forecast_path = forecasts_dir / "XX_forecast_20240101_baseline.csv"
    df.to_csv(forecast_path, index=False)

    inspection = inspect_run("XX", model_name="baseline")

    assert inspection.country_code == "XX"
    assert inspection.model_name == "baseline"
    assert inspection.run_id == "run123"
    assert inspection.metrics is not None
    assert inspection.metrics.get("train_mae") == 1.23

    assert inspection.forecast_summaries
    summary = inspection.forecast_summaries[0]
    # summary.path is relative to the working directory; compare resolved paths
    assert summary.path.resolve() == forecast_path.resolve()
    assert summary.rows == 2
    assert summary.min_price == 10.0
    assert summary.max_price == 20.0

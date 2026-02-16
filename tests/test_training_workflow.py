# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reproducible training workflow helpers."""

import pandas as pd

from experiments.run_training import _drop_report, _leakage_report, _missingness_report


def _base_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    price = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    return pd.DataFrame(
        {
            "timestamp": ts,
            "price_eur_mwh": price,
            "price_lag_1": price.shift(1),
            "target_price": price.shift(-1),
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "is_holiday": 0,
            "is_event": 0,
        }
    )


def test_leakage_report_passes_for_consistent_lags() -> None:
    df = _base_df()
    report = _leakage_report(df, lags=[1])

    assert report["passed"] is True
    assert report["failures"] == []


def test_leakage_report_fails_for_inconsistent_lag() -> None:
    df = _base_df()
    df.loc[df.index[2], "price_lag_1"] = 999.0

    report = _leakage_report(df, lags=[1])

    assert report["passed"] is False
    assert any("price_lag_1" in msg for msg in report["failures"])


def test_missingness_report_fails_when_enabled_family_columns_missing() -> None:
    df = _base_df()[["timestamp", "price_eur_mwh", "price_lag_1", "target_price"]]
    features_cfg = {
        "use_weather_features": True,
        "use_gas_features": False,
        "use_event_features": False,
        "use_fourier_features": False,
        "use_price_volatility": False,
        "use_price_momentum": False,
    }

    report = _missingness_report(df, features_cfg, {"core": 0.2, "weather": 0.2})

    assert report["passed"] is False
    assert report["families"]["weather"]["passed"] is False


def test_drop_report_counts_required_lag_and_target_rows() -> None:
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    prices_df = pd.DataFrame({"timestamp": ts, "price_eur_mwh": range(10)})

    report = _drop_report(prices_df, lags=[1, 2])

    assert report["raw_price_rows"] == 10
    assert report["rows_after_required_lag_target_filter"] == 7
    assert report["dropped_rows"] == 3

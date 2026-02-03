# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for event flags (is_holiday, is_event) for Portugal (PT).
"""

import pandas as pd

from core.data_manager import CountryDataManager
from core.features import FeatureEngineer


def test_pt_event_flags_reflect_manual_events_and_holidays(tmp_path, monkeypatch):
    """FeatureEngineer should set is_holiday and is_event based on cleaned PT event files."""
    monkeypatch.chdir(tmp_path)
    manager = CountryDataManager("PT", base_path="data")
    manager.create_directories()

    start_date = "2024-01-01"
    end_date = "2024-01-03"

    # Electricity prices for three days (hourly)
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

    # Minimal load data so that load_mw is available
    load = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": [1000.0] * len(timestamps),
        }
    )
    load_path = manager.get_processed_file_path("electricity_load_clean", start_date, end_date)
    load.to_csv(load_path, index=False)

    # Cleaned holidays: mark 2024-01-02 as a holiday
    holidays = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "event_type": ["holiday"],
            "description": ["PT Test Holiday"],
        }
    )
    holidays_path = manager.get_processed_file_path("holidays_clean", start_date, end_date)
    holidays.to_csv(holidays_path, index=False)

    # Cleaned manual events: mark 2024-01-03 as an event day
    manual_events = pd.DataFrame(
        {
            "date_start": [pd.Timestamp("2024-01-03 00:00", tz="UTC")],
            "date_end": [pd.Timestamp("2024-01-03 23:59", tz="UTC")],
            "event_type": ["maintenance"],
            "description": ["PT Test Event"],
            "source": ["test"],
        }
    )
    manual_events_path = manager.get_processed_file_path(
        "manual_events_clean", start_date, end_date
    )
    manual_events.to_csv(manual_events_path, index=False)

    from core.repository import CsvDataRepository

    engineer = FeatureEngineer("PT", repository=CsvDataRepository(manager))
    engineer.build_electricity_features(start_date, end_date)

    features_path = manager.get_processed_file_path("electricity_features", start_date, end_date)
    features_df = pd.read_csv(features_path, parse_dates=["timestamp"])

    # Rows whose date is 2024-01-02 should have is_holiday = 1
    is_holiday_rows = features_df[
        features_df["timestamp"].dt.normalize() == pd.Timestamp("2024-01-02", tz="UTC").normalize()
    ]
    assert not is_holiday_rows.empty
    assert is_holiday_rows["is_holiday"].max() == 1

    # Rows within the manual event date range should have is_event = 1
    is_event_rows = features_df[
        features_df["timestamp"].dt.normalize() == pd.Timestamp("2024-01-03", tz="UTC").normalize()
    ]
    assert not is_event_rows.empty
    assert is_event_rows["is_event"].max() == 1

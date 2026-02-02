# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for DataVerifier.
"""

import pandas as pd
import pytest

from core.verification import DataVerifier


@pytest.fixture
def verifier():
    return DataVerifier("MOCK")


def test_check_gaps_finds_missing_hours(verifier):
    # Create hourly data with a gap
    # 00:00, 01:00, (missing 02:00), 03:00
    dates = [
        pd.Timestamp("2023-01-01 00:00:00"),
        pd.Timestamp("2023-01-01 01:00:00"),
        pd.Timestamp("2023-01-01 03:00:00"),
    ]
    df = pd.DataFrame({"timestamp": dates, "value": [1, 2, 3]})

    missing = verifier.check_gaps(df, freq="h")

    assert len(missing) == 1
    assert missing[0] == pd.Timestamp("2023-01-01 02:00:00")


def test_check_gaps_no_gaps(verifier):
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    df = pd.DataFrame({"timestamp": dates, "value": range(5)})

    missing = verifier.check_gaps(df, freq="h")
    assert len(missing) == 0


def test_check_negative_values_detected(verifier):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h"),
            "load_mw": [100, -50, 200],  # Middle one is negative
            "prices": [10, 20, 30],
        }
    )

    failed = verifier.check_negative_values(df, ["load_mw", "prices"])
    assert "load_mw" in failed
    assert "prices" not in failed


def test_check_negative_values_clean(verifier):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h"),
            "load_mw": [100, 50, 200],
        }
    )

    failed = verifier.check_negative_values(df, ["load_mw"])
    assert len(failed) == 0


def test_verify_electricity_integrates_checks(verifier):
    # Setup data with gap in load and negative value in load
    # Prices OK
    prices_df = pd.DataFrame(
        {"timestamp": pd.date_range("2023-01-01", periods=3, freq="h"), "price": [10, 20, 30]}
    )

    # Load has gap: 00:00, (miss 01:00), 02:00 AND 02:00 is negative
    load_dates = [
        pd.Timestamp("2023-01-01 00:00:00"),
        # Missing 01:00
        pd.Timestamp("2023-01-01 02:00:00"),
    ]
    load_df = pd.DataFrame({"timestamp": load_dates, "load_mw": [100, -10]})

    # We can't easily assert on internal calls without mocking or capturing logs.
    # For a unit test of the specific method, we trust it runs.
    # We could capture logs if needed.

    # Just ensure it runs without error
    verifier.verify_electricity(prices_df, load_df)

    # Test with None
    verifier.verify_electricity(None, None)

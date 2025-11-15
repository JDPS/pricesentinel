# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Quick verification script to test that all fixes are working.

Run this script to verify the fixes before running the full test suite.
"""

from config.country_registry import FetcherFactory
from data_fetchers.mock import register_mock_country


def test_date_range_fix():
    """Test that mock data generates the correct number of records."""
    print("\n" + "=" * 60)
    print("Testing Date Range Fix")
    print("=" * 60)

    register_mock_country()
    fetchers = FetcherFactory.create_fetchers("XX")

    # Test electricity prices
    print("\n1. Testing electricity prices...")
    df = fetchers["electricity"].fetch_prices("2024-01-01", "2024-01-07")
    expected = 7 * 24  # 7 days * 24 hours
    actual = len(df)

    if actual == expected:
        print(f"   ✅ PASS: Got {actual} records (expected {expected})")
    else:
        print(f"   ❌ FAIL: Got {actual} records (expected {expected})")

    # Test electricity load
    print("\n2. Testing electricity load...")
    df = fetchers["electricity"].fetch_load("2024-01-01", "2024-01-07")
    actual = len(df)

    if actual == expected:
        print(f"   ✅ PASS: Got {actual} records (expected {expected})")
    else:
        print(f"   ❌ FAIL: Got {actual} records (expected {expected})")

    # Test weather
    print("\n3. Testing weather data...")
    df = fetchers["weather"].fetch_weather("2024-01-01", "2024-01-07")
    # Weather has one location in mock config
    actual = len(df)

    if actual == expected:
        print(f"   ✅ PASS: Got {actual} records (expected {expected})")
    else:
        print(f"   ❌ FAIL: Got {actual} records (expected {expected})")


def test_timezone_fix():
    """Test that timezone comparison works in holidays."""
    print("\n" + "=" * 60)
    print("Testing Timezone Comparison Fix")
    print("=" * 60)

    register_mock_country()
    fetchers = FetcherFactory.create_fetchers("XX")

    print("\n4. Testing holiday generation...")
    try:
        df = fetchers["events"].get_holidays("2024-01-01", "2024-12-31")
        print(f"   ✅ PASS: Generated {len(df)} holidays without timezone errors")
    except TypeError as e:
        if "tz-naive and tz-aware" in str(e):
            print(f"   ❌ FAIL: Timezone comparison error: {e}")
        else:
            raise


def test_no_deprecated_warnings():
    """Test that no FutureWarnings are generated."""
    print("\n" + "=" * 60)
    print("Testing No Deprecated Warnings")
    print("=" * 60)

    print("\n5. Checking for deprecated frequency usage...")
    print("   ℹ️  Run pytest to see if FutureWarnings are gone")
    print("   Expected: No warnings about 'H' being deprecated")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("VERIFICATION SCRIPT - Testing Fixes")
    print("=" * 60)

    try:
        test_date_range_fix()
        test_timezone_fix()
        test_no_deprecated_warnings()

        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print("\nNext step: Run full test suite with:")
        print("  pytest -v")
        print("\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Property-based tests for PriceSentinel core logic.
"""

from datetime import date

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from config.validation import validate_date_range


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    start=st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)),
    end=st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)),
)
def test_date_range_validation_properties(start, end):
    """
    Test invariants for date validation:
    1. If start > end, it MUST raise ValueError.
    2. If start <= end, it MUST return valid strings.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    if start > end:
        try:
            validate_date_range(start_str, end_str)
            pytest.fail("Should raise ValueError for inverted range")
        except ValueError:
            pass  # Expected
    else:
        s, e = validate_date_range(start_str, end_str)
        assert s == start_str
        assert e == end_str

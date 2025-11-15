# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock event data provider for testing.

This module generates synthetic holiday and event data.
"""

import logging

import pandas as pd

from core.abstractions import EventDataProvider

logger = logging.getLogger(__name__)


class MockEventProvider(EventDataProvider):
    """
    Generates synthetic event data for testing.

    Creates a simple set of holidays and manual events.
    """

    def __init__(self, config):
        """
        Initialise mock event provider.

        Args:
            config: CountryConfig instance
        """
        self.country_code = config.country_code
        logger.debug(f"Initialized MockEventProvider for {self.country_code}")

    def get_holidays(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate synthetic holidays.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with synthetic holiday data
        """
        # Create simple fixed holidays for testing
        start = pd.to_datetime(start_date, utc=True)
        end = pd.to_datetime(end_date, utc=True)

        # Define mock holidays (same each year)
        holiday_templates = [
            (1, 1, "New Year's Day"),
            (3, 15, "Mock National Day"),
            (5, 1, "Labour Day"),
            (8, 15, "Summer Holiday"),
            (12, 25, "Winter Holiday"),
            (12, 31, "New Year's Eve"),
        ]

        holidays = []

        # Generate holidays for each year in range
        for year in range(start.year, end.year + 1):
            for month, day, name in holiday_templates:
                date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                if start <= date <= end:
                    holidays.append(
                        {"timestamp": date, "event_type": "holiday", "description": name}
                    )

        df = pd.DataFrame(holidays)

        if len(df) > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Generated {len(df)} synthetic" f" holidays from {start_date} " f"to {end_date}"
        )

        return df

    def get_manual_events(self) -> pd.DataFrame:
        """
        Generate synthetic manual events.

        Returns:
            DataFrame with synthetic manual events
        """
        # Create a few example events for testing
        events = [
            {
                "date_start": pd.Timestamp("2024-03-01", tz="UTC"),
                "date_end": pd.Timestamp("2024-03-03", tz="UTC"),
                "event_type": "maintenance",
                "description": "Scheduled maintenance of transmission lines",
                "source": "mock_data",
            },
            {
                "date_start": pd.Timestamp("2024-07-15", tz="UTC"),
                "date_end": pd.Timestamp("2024-07-17", tz="UTC"),
                "event_type": "heatwave",
                "description": "Extreme temperatures affecting demand",
                "source": "mock_data",
            },
            {
                "date_start": pd.Timestamp("2024-11-10", tz="UTC"),
                "date_end": pd.Timestamp("2024-11-10", tz="UTC"),
                "event_type": "policy_change",
                "description": "New energy policy implementation",
                "source": "mock_data",
            },
        ]

        df = pd.DataFrame(events)

        logger.info(f"Generated {len(df)} synthetic manual events")

        return df

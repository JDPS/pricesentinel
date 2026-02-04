# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Portugal event data provider.

This module provides Portuguese holidays and manual events.
"""

import logging
from pathlib import Path

import pandas as pd

from core.abstractions import EventDataProvider

try:
    import holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PortugalEventProvider(EventDataProvider):
    """
    Provides Portuguese holidays and manually curated events.

    Uses the 'holidays' Python package for national holidays.
    """

    def __init__(self, config):
        """
        Initialise Portugal event provider.

        Args:
            config: CountryConfig instance
        """
        self.country_code = config.country_code
        self.manual_events_path = Path(config.events_config["manual_events_path"])

        if not HOLIDAYS_AVAILABLE:
            logger.warning(
                "The 'holidays' package is not installed. "
                "Holiday detection will not work. Install with: pip install holidays"
            )

        logger.debug("Initialized PortugalEventProvider")

    def get_holidays(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get Portuguese national holidays.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with holiday data
        """
        if not HOLIDAYS_AVAILABLE:
            logger.warning("Holidays package not available, returning empty DataFrame")
            return pd.DataFrame(columns=["timestamp", "event_type", "description"])

        logger.info(f"Fetching Portuguese holidays from {start_date} to {end_date}")

        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # Get years covered
            years = range(start.year, end.year + 1)

            # Get Portuguese holidays
            pt_holidays = holidays.Portugal(years=years)

            # Create a list of holidays in the date range
            holiday_list = []

            dates = pd.date_range(start_date, end_date, freq="D")

            for date in dates:
                date_obj = date.date()
                if date_obj in pt_holidays:
                    holiday_list.append(
                        {
                            "timestamp": pd.Timestamp(date_obj, tz="UTC"),
                            "event_type": "holiday",
                            "description": pt_holidays[date_obj],
                        }
                    )

            df = pd.DataFrame(holiday_list)

            logger.info(f"Found {len(df)} holidays")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch holidays: {e}")
            return pd.DataFrame(columns=["timestamp", "event_type", "description"])

    def get_manual_events(self) -> pd.DataFrame:
        """
        Load manually curated events from CSV.

        Returns:
            DataFrame with manual events
        """
        logger.info(f"Loading manual events from {self.manual_events_path}")

        if not self.manual_events_path.exists():
            logger.warning(
                f"Manual events file not found: {self.manual_events_path}\n"
                f"Creating template file..."
            )
            self._create_manual_events_template()
            return self._load_manual_events_file()

        return self._load_manual_events_file()

    def _load_manual_events_file(self) -> pd.DataFrame:
        """Load manual events CSV file."""
        try:
            df = pd.read_csv(self.manual_events_path, parse_dates=["date_start", "date_end"])

            # Ensure timestamps are UTC
            for col in ["date_start", "date_end"]:
                if df[col].dt.tz is None:
                    df[col] = pd.to_datetime(df[col], utc=True)
                else:
                    df[col] = df[col].dt.tz_convert("UTC")

            logger.info(f"Loaded {len(df)} manual events")

            return df

        except Exception as e:
            logger.error(f"Failed to load manual events: {e}")
            return pd.DataFrame(
                columns=["date_start", "date_end", "event_type", "description", "source"]
            )

    def _create_manual_events_template(self) -> None:
        """Create a template manual events CSV file."""
        self.manual_events_path.parent.mkdir(parents=True, exist_ok=True)

        template = pd.DataFrame(
            {
                "date_start": ["2024-03-01", "2024-07-15"],
                "date_end": ["2024-03-03", "2024-07-17"],
                "event_type": ["maintenance", "heatwave"],
                "description": [
                    "Scheduled maintenance of transmission infrastructure",
                    "Extreme heat affecting electricity demand",
                ],
                "source": ["operator_notice", "weather_service"],
            }
        )

        template.to_csv(self.manual_events_path, index=False)
        logger.info(f"Created manual events template at {self.manual_events_path}")

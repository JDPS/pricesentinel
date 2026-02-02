# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Data fetching stage implementation.

Handles concurrent fetching of data from multiple sources.
"""

import asyncio
import logging

from core.data_manager import CountryDataManager
from core.stages.base import BaseStage

logger = logging.getLogger(__name__)


class DataFetchStage(BaseStage):
    """
    Stage 1: Fetch data from configured sources.

    Orchestrates parallel fetching of electricity prices, load, weather,
    and gas prices, plus synchronous fetching of events.
    """

    def __init__(self, country_code: str, fetchers: dict, data_manager: CountryDataManager):
        super().__init__(country_code)
        self.fetchers = fetchers
        self.data_manager = data_manager

    async def run(self, start_date: str, end_date: str) -> None:
        """
        Execute the fetch stage.

        Args:
            start_date: Start (YYYY-MM-DD)
            end_date: End (YYYY-MM-DD)
        """
        logger.info(f"=== Stage 1: Fetching data for {self.country_code} ===")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Define fetch tasks
        tasks = [
            # Electricity Prices
            self._fetch_and_store(
                fetch_fn=self.fetchers["electricity"].fetch_prices,
                dataset_key="electricity",
                filename_prefix="electricity_prices",
                start_date=start_date,
                end_date=end_date,
                empty_msg="No electricity price data fetched",
                success_msg="Saved {count} electricity price records",
            ),
            # Electricity Load
            self._fetch_and_store(
                fetch_fn=self.fetchers["electricity"].fetch_load,
                dataset_key="electricity",
                filename_prefix="electricity_load",
                start_date=start_date,
                end_date=end_date,
                empty_msg="No electricity load data fetched",
                success_msg="Saved {count} electricity load records",
            ),
            # Weather
            self._fetch_and_store(
                fetch_fn=self.fetchers["weather"].fetch_weather,
                dataset_key="weather",
                filename_prefix="weather",
                start_date=start_date,
                end_date=end_date,
                empty_msg="No weather data fetched",
                success_msg="Saved {count} weather records",
            ),
            # Gas Prices
            self._fetch_and_store(
                fetch_fn=self.fetchers["gas"].fetch_prices,
                dataset_key="gas",
                filename_prefix="gas_prices",
                start_date=start_date,
                end_date=end_date,
                empty_msg="No gas price data fetched",
                success_msg="Saved {count} gas price records",
            ),
        ]

        # Execute parallel fetches
        await asyncio.gather(*tasks)

        # Fetch events (synchronous)
        self._fetch_events(start_date, end_date)

    async def _fetch_and_store(
        self,
        fetch_fn,
        dataset_key: str,
        filename_prefix: str,
        start_date: str,
        end_date: str,
        empty_msg: str,
        success_msg: str,
    ) -> None:
        """Helper to fetch and save data."""
        try:
            df = await fetch_fn(start_date, end_date)
            if len(df) > 0:
                filename = self.data_manager.generate_filename(
                    filename_prefix, start_date, end_date
                )
                output_path = self.data_manager.get_raw_path(dataset_key) / filename
                df.to_csv(output_path, index=False)
                logger.info(success_msg.format(count=len(df)))
            else:
                logger.warning(empty_msg)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to fetch %s for %s: %s", filename_prefix, self.country_code, exc)
        except Exception:
            logger.exception(
                "Unexpected error while fetching %s for %s", filename_prefix, self.country_code
            )
            # We don't re-raise here to allow other concurrent fetches to proceed?
            # Original code raised on unexpected error. Let's keep that behavior if critical,
            # but usually partial failure is better than total crash if independent.
            # However, original code raised. I'll re-raise.
            raise

    def _fetch_events(self, start_date: str, end_date: str) -> None:
        """Fetch holidays and manual events."""
        logger.info("Fetching holidays and events...")
        try:
            holidays_df = self.fetchers["events"].get_holidays(start_date, end_date)

            if len(holidays_df) > 0:
                holidays_path = self.data_manager.get_events_path() / "holidays.csv"
                holidays_df.to_csv(holidays_path, index=False)
                logger.info("Saved %d holidays", len(holidays_df))
            else:
                logger.info("No holidays in date range")

            manual_events_df = self.fetchers["events"].get_manual_events()
            if len(manual_events_df) > 0:
                manual_path = self.data_manager.get_events_path() / "manual_events.csv"
                manual_events_df.to_csv(manual_path, index=False)
                logger.info("Loaded %d manual events", len(manual_events_df))

        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to fetch events for %s: %s", self.country_code, exc)
        except Exception:
            logger.exception("Unexpected error while fetching events for %s", self.country_code)
            raise

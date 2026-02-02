# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Data Repository abstraction.

This module implements the Repository pattern for data access, decoupling
the application logic from file I/O operations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from core.data_manager import CountryDataManager

logger = logging.getLogger(__name__)


class DataRepository(ABC):
    """
    Abstract base class for data repositories.

    Defines the contract for saving and loading data, agnostic of the storage mechanism.
    """

    @abstractmethod
    def save_data(
        self, df: pd.DataFrame, name: str, start_date: str, end_date: str, source: str = "processed"
    ) -> Path:
        """
        Save a DataFrame to storage.

        Args:
            df: DataFrame to save.
            name: Logical name of the dataset (e.g., 'electricity_prices').
            start_date: Start date of the data.
            end_date: End date of the data.
            source: Source category ('processed', 'raw', etc.).

        Returns:
            Path or identifier of the saved resource.
        """
        pass

    @abstractmethod
    def load_data(
        self, name: str, start_date: str, end_date: str, source: str = "processed"
    ) -> pd.DataFrame | None:
        """
        Load a DataFrame from storage.

        Args:
            name: Logical name of the dataset.
            start_date: Start date of the data.
            end_date: End date of the data.
            source: Source category.

        Returns:
            Loaded DataFrame or None if not found.
        """

    @abstractmethod
    def load_matching_raw(self, source: str, filename_prefix: str) -> list[pd.DataFrame]:
        """
        Load all raw files matching a specific prefix pattern.

        Args:
            source: Data source category (e.g. 'electricity').
            filename_prefix: File prefix to match (e.g. 'electricity_prices').

        Returns:
            List of loaded DataFrames.
        """
        pass


class CsvDataRepository(DataRepository):
    """
    Implementation of DataRepository using CSV files via CountryDataManager.
    """

    def __init__(self, data_manager: CountryDataManager):
        self.data_manager = data_manager

    def save_data(
        self, df: pd.DataFrame, name: str, start_date: str, end_date: str, source: str = "processed"
    ) -> Path:
        if source == "processed":
            path = self.data_manager.get_processed_file_path(name, start_date, end_date)
        elif source == "raw":
            path = self.data_manager.get_file_path(name, start_date, end_date)
        else:
            # Fallback for other sources if simpler path logic needed
            path = self.data_manager.get_file_path(name, start_date, end_date)

        df.to_csv(path, index=False)
        logger.debug(f"Saved data to {path}")
        return path

    def load_data(
        self, name: str, start_date: str, end_date: str, source: str = "processed"
    ) -> pd.DataFrame | None:
        if source == "processed":
            path = self.data_manager.get_processed_file_path(name, start_date, end_date)
        elif source == "raw":
            path = self.data_manager.get_file_path(name, start_date, end_date)
        else:
            path = self.data_manager.get_file_path(name, start_date, end_date)

        if not path.exists():
            return None

        return pd.read_csv(path)

    def load_matching_raw(self, source: str, filename_prefix: str) -> list[pd.DataFrame]:
        pattern = f"*_{filename_prefix}_*.csv"
        files = self.data_manager.list_files(source, pattern=pattern)

        frames = []
        for path in files:
            logger.info(f"Loading raw file for {source}/{filename_prefix}: {path}")
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")

        return frames

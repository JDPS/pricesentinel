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
from core.metadata_manager import MetadataManager

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
    def list_processed_data(self, pattern: str) -> list[Path]:
        """
        List processed data files matching a pattern.

        Args:
            pattern: Glob pattern to match files.

        Returns:
            List of matching file paths.
        """
        pass

    @abstractmethod
    def save_forecast(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save forecast data to the forecasts directory.

        Args:
            df: DataFrame to save.
            filename: Filename for the forecast file.

        Returns:
            Path to the saved file.
        """
        pass

    @abstractmethod
    def save_raw_data(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        filename_prefix: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Save raw data to a specific source directory with standardized naming.

        Args:
            df: DataFrame to save.
            dataset_key: The raw source key (e.g., 'electricity', 'weather').
            filename_prefix: The prefix for the filename (e.g., 'electricity_prices').
            start_date: Start date string.
            end_date: End date string.

        Returns:
            Path to the saved file.
        """
        pass

    @abstractmethod
    def save_event_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save event data to a specific filename in the events directory.

        Args:
            df: DataFrame to save.
            filename: Exact filename to use (e.g. 'holidays.csv').

        Returns:
            Path to the saved file.
        """
        pass

    @abstractmethod
    def load_event_data(self, filename: str) -> pd.DataFrame | None:
        """
        Load event data from a specific filename in the events directory.

        Args:
            filename: Exact filename to use (e.g. 'holidays.csv').

        Returns:
            DataFrame if file exists, None otherwise.
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

    def __init__(
        self, data_manager: CountryDataManager, metadata_manager: MetadataManager | None = None
    ):
        self.data_manager = data_manager
        self.metadata_manager = metadata_manager

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

        if self.metadata_manager:
            self.metadata_manager.register_file(path, source, start_date, end_date, df)

        return path

    def save_raw_data(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        filename_prefix: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        # Resolve path using data manager's logic for raw source subdirectories
        filename = self.data_manager.generate_filename(filename_prefix, start_date, end_date)
        # dataset_key maps to the subdirectory in raw/ (e.g. raw/electricity)
        output_path = self.data_manager.get_raw_path(dataset_key) / filename

        # Ensure parent exists (Repository should ensure storage is ready)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.debug(f"Saved raw data to {output_path}")

        if self.metadata_manager:
            self.metadata_manager.register_file(output_path, dataset_key, start_date, end_date, df)

        return output_path

    def save_event_data(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.data_manager.get_events_path() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug(f"Saved event data to {path}")
        # Metadata for events? Maybe "events" as source
        if self.metadata_manager:
            # For manual events/holidays, start/end date might be the
            # whole file's range or irrelevant.
            # We'll just pass empty strings if not applicable or extract from df if possible.
            # But usually these are static files.
            pass
        return path

    def load_event_data(self, filename: str) -> pd.DataFrame | None:
        path = self.data_manager.get_events_path() / filename
        if not path.exists():
            return None

        # We don't know the exact date columns here generically,
        # so we rely on the caller or use standard pandas inference.
        # For simplicity in this project context where this is only used for holidays/manual events:
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to read event data from {path}: {e}")
            return None

    def list_processed_data(self, pattern: str) -> list[Path]:
        processed_dir = self.data_manager.get_processed_path()
        if not processed_dir.exists():
            return []

        # Sort by modification time, newest first
        files = sorted(processed_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def save_forecast(self, df: pd.DataFrame, filename: str) -> Path:
        forecasts_dir = self.data_manager.get_processed_path() / "forecasts"
        forecasts_dir.mkdir(parents=True, exist_ok=True)

        path = forecasts_dir / filename
        df.to_csv(path, index=False)
        logger.debug(f"Saved forecast to {path}")
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

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Country-aware data management and directory structure.

This module provides utilities for managing country-specific data directories,
file naming conventions, and data organisation.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CountryDataManager:
    """
    Manages country-specific data directories and file naming.

    This class ensures consistent directory structure and file naming
    across all countries, enabling proper data isolation.
    """

    def __init__(self, country_code: str, base_path: str = "data"):
        """
        Initialize CountryDataManager.

        Args:
            country_code: ISO 3166-1 alpha-2 code (e.g. 'PT')
            base_path: Base directory for all data (default: 'data')
        """
        self.country_code = country_code.upper()
        self.base_path = Path(base_path) / self.country_code

    def get_raw_path(self, source: str | None = None) -> Path:
        """
        Get the path for raw data, optionally for a specific source.

        Args:
            source: Optional data source name (e.g. 'electricity', 'weather')

        Returns:
            Path to raw data directory
        """
        if source:
            return self.base_path / "raw" / source
        return self.base_path / "raw"

    def get_processed_path(self) -> Path:
        """
        Get the path for processed/cleaned data.

        Returns:
            Path to processed data directory
        """
        return self.base_path / "processed"

    def get_events_path(self) -> Path:
        """
        Get the path for event data.

        Returns:
            Path to events directory
        """
        return self.base_path / "events"

    def get_metadata_path(self) -> Path:
        """
        Get the path for metadata files.

        Returns:
            Path to metadata directory
        """
        return self.base_path / "metadata"

    def create_directories(self):
        """
        Create all required directories for a country.

        This method is idempotent - it's safe to call multiple times.
        """
        dirs = [
            self.get_raw_path(),
            self.get_raw_path("electricity"),
            self.get_raw_path("weather"),
            self.get_raw_path("gas"),
            self.get_processed_path(),
            self.get_events_path(),
            self.get_metadata_path(),
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")

        # Create .gitkeep files to preserve directory structure in git
        for dir_path in dirs:
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()

        logger.info(f"Created directory structure for {self.country_code}")

    def generate_filename(
        self, source: str, start_date: str, end_date: str, extension: str = "csv"
    ) -> str:
        """
        Generate a standardised filename.

        Format: {country}_{source}_{start}_{end}.{ext}
        Example: PT_electricity_20240101_20240131.csv

        Args:
            source: Data source name (e.g. 'electricity', 'weather')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            extension: File extension (default: 'csv')

        Returns:
            Standardized filename
        """
        # Remove hyphens from dates
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        return f"{self.country_code}_{source}_{start}_{end}.{extension}"

    def get_file_path(
        self, source: str, start_date: str, end_date: str, extension: str = "csv"
    ) -> Path:
        """
        Get full file path for a data file.

        Args:
            source: Data source name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            extension: File extension

        Returns:
            Full path to the file
        """
        filename = self.generate_filename(source, start_date, end_date, extension)
        return self.get_raw_path(source) / filename

    def get_processed_file_path(
        self, name: str, start_date: str, end_date: str, extension: str = "csv"
    ) -> Path:
        """
        Get full file path for a processed data file.

        Args:
            name: Logical name for the processed dataset
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            extension: File extension

        Returns:
            Full path to the processed file
        """
        filename = self.generate_filename(name, start_date, end_date, extension)
        return self.get_processed_path() / filename

    def get_latest_file(self, source: str, pattern: str = "*") -> Path | None:
        """
        Find the most recent file for a given source.

        Args:
            source: Data source name
            pattern: Glob pattern to match files (default: '*')

        Returns:
            Path to the most recent file, or None if no files are found

        Raises:
            FileNotFoundError: If no files match the pattern
        """
        raw_path = self.get_raw_path(source)

        if not raw_path.exists():
            raise FileNotFoundError(f"Source directory not found: {raw_path}")

        files = sorted(raw_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if not files:
            raise FileNotFoundError(
                f"No files found for source '{source}' matching pattern '{pattern}'"
            )

        logger.debug(f"Found latest file for {source}: {files[0].name}")
        return files[0]

    def list_files(self, source: str, pattern: str = "*") -> list:
        """
        List all files for a given source.

        Args:
            source: Data source name
            pattern: Glob pattern to match files

        Returns:
            List of file paths, sorted by modification time (newest first)
        """
        raw_path = self.get_raw_path(source)

        if not raw_path.exists():
            return []

        files = sorted(raw_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        return files

    def get_directory_size(self) -> int:
        """
        Calculate total size of country data directory.

        Returns:
            Total size in bytes
        """
        total_size = 0

        for dirpath, _, filenames in os.walk(self.base_path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                total_size += filepath.stat().st_size

        return total_size

    def get_directory_info(self) -> dict:
        """
        Get information about the country data directory.

        Returns:
            Dictionary with directory statistics
        """
        info = {
            "country_code": self.country_code,
            "base_path": str(self.base_path),
            "exists": self.base_path.exists(),
            "total_size_bytes": 0,
            "file_count": 0,
            "sources": {},
        }

        if not self.base_path.exists():
            return info

        # Calculate total size
        total_size = self.get_directory_size()
        info["total_size_bytes"] = total_size
        info["total_size_mb"] = str(round(total_size / (1024 * 1024), 2))

        # Count files by source
        file_count = 0
        sources: dict[str, int] = {}
        for source in ["electricity", "weather", "gas"]:
            source_path = self.get_raw_path(source)
            if source_path.exists():
                files = list(source_path.glob("*"))
                # Exclude .gitkeep files
                files = [f for f in files if f.name != ".gitkeep"]
                sources[source] = len(files)
                file_count += len(files)

        info["sources"] = sources
        info["file_count"] = file_count

        return info

    def __repr__(self) -> str:
        return f"CountryDataManager(country={self.country_code}, path={self.base_path})"


def setup_country_directories(country_code: str, base_path: str = "data"):
    """
    Convenience function to set up directories for a new country.

    Args:
        country_code: ISO 3166-1 alpha-2 code
        base_path: Base directory for all data

    Returns:
        CountryDataManager instance
    """
    manager = CountryDataManager(country_code, base_path)
    manager.create_directories()
    logger.info(f"Set up directories for country: {country_code}")
    return manager

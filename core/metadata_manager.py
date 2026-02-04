# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Metadata management for data lineage and cataloging.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from core.types import DatasetMetadata

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages metadata collection and storage for datasets.

    Maintains a catalog.json file in the country's metadata directory,
    tracking every file created or updated by the pipeline.
    """

    def __init__(self, metadata_dir: Path):
        """
        Initialize MetadataManager.

        Args:
            metadata_dir: Path to the metadata directory (e.g. data/PT/metadata)
        """
        self.metadata_dir = metadata_dir
        self.catalog_path = metadata_dir / "catalog.json"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure metadata directory exists."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def load_catalog(self) -> dict[str, DatasetMetadata]:
        """Load the current catalog."""
        if not self.catalog_path.exists():
            return {}

        try:
            with open(self.catalog_path, encoding="utf-8") as f:
                # Cast the result because json.load returns Any
                from typing import cast

                return cast(dict[str, DatasetMetadata], json.load(f))
        except json.JSONDecodeError:
            logger.warning("Corrupt catalog.json found at %s. Starting fresh.", self.catalog_path)
            return {}
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return {}

    def save_catalog(self, catalog: dict[str, DatasetMetadata]) -> None:
        """Save the catalog."""
        try:
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)
            logger.debug(f"Saved catalog to {self.catalog_path}")
        except Exception as e:
            logger.error(f"Failed to save catalog: {e}")

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute checksum for {file_path}: {e}")
            return "error"

    def register_file(
        self,
        file_path: Path,
        source: str,
        start_date: str,
        end_date: str,
        df: pd.DataFrame | None = None,
    ) -> None:
        """
        Register a file in the catalog.

        Args:
            file_path: Path to the data file.
            source: Source of the data (e.g. 'electricity', 'weather').
            start_date: Start date of the data.
            end_date: End date of the data.
            df: Optional available DataFrame to avoid reloading for metrics.
        """
        if not file_path.exists():
            logger.warning(f"Attempted to register non-existent file: {file_path}")
            return

        catalog = self.load_catalog()
        filename = file_path.name

        # Calculate file stats
        checksum = self.compute_checksum(file_path)
        file_size = file_path.stat().st_size

        row_count = 0
        column_count = 0

        if df is not None:
            row_count = len(df)
            column_count = len(df.columns)
        else:
            # Try to peek if CSV
            try:
                if file_path.suffix == ".csv":
                    # lightweight peek
                    temp_df = pd.read_csv(file_path, nrows=1)
                    column_count = len(temp_df.columns)
                    # For row count we need to read strictly, but for performance maybe we skip?
                    # Or just rely on caller passing df.
                    # Let's count lines for now if small enough? No, avoid IO.
                    pass
            except Exception as e:
                logger.debug(f"Failed to peek CSV for stats: {e}")

        metadata: DatasetMetadata = {
            "filename": filename,
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
            "fetch_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checksum": checksum,
            "file_size_bytes": file_size,
            "row_count": row_count,
            "column_count": column_count,
        }

        catalog[filename] = metadata
        self.save_catalog(catalog)
        logger.info(f"Registered {filename} in metadata catalog")

    def generate_data_readme(self, country_code: str) -> None:
        """
        Generate a README.md file for the data directory from the catalog.

        Args:
            country_code: The ISO country code (e.g., 'PT') for the title.
        """
        catalog = self.load_catalog()
        readme_path = self.metadata_dir.parent / "README.md"

        lines = [
            f"# Data Catalog: {country_code}",
            "",
            f"**Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "",
            "## Available Datasets",
            "",
            "| Filename | Source | Start Date | End Date | Size (Bytes) | Rows | Cols |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        if not catalog:
            lines.append("\n*No datasets registered yet.*")
        else:
            # Sort by source then filename for readability
            for filename, meta in sorted(catalog.items(), key=lambda x: (x[1]["source"], x[0])):
                lines.append(
                    f"| `{filename}` "
                    f"| {meta.get('source', 'N/A')} "
                    f"| {meta.get('start_date', 'N/A')} "
                    f"| {meta.get('end_date', 'N/A')} "
                    f"| {meta.get('file_size_bytes', 0):,} "
                    f"| {meta.get('row_count', 0):,} "
                    f"| {meta.get('column_count', 0)} |"
                )

        lines.append("")
        lines.append("## Verification")
        lines.append("")
        lines.append(
            "All files are verified with SHA256 checksums stored in `metadata/catalog.json`."
        )

        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"Generated data README at {readme_path}")
        except Exception as e:
            logger.error(f"Failed to generate data README: {e}")

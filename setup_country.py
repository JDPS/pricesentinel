# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility script to set up a directory structure for a new country.

Usage:
    python setup_country.py PT
    python setup_country.py ES
"""

import logging
import sys

from core.data_manager import setup_country_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for a country setup script.
    """
    if len(sys.argv) < 2:
        print("Usage: python setup_country.py <COUNTRY_CODE>")
        print("Example: python setup_country.py PT")
        sys.exit(1)

    country_code = sys.argv[1].upper()

    if len(country_code) != 2:
        print("Error: Country code must be 2 letters (e.g., PT, ES, DE)")
        print(f"Got: {country_code}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Setting up directory structure for: {country_code}")
    print(f"{'=' * 60}\n")

    try:
        manager = setup_country_directories(country_code)

        print("\n[OK] Successfully created directories:")
        print(f"  - {manager.get_raw_path()}")
        print(f"  - {manager.get_processed_path()}")
        print(f"  - {manager.get_events_path()}")
        print(f"  - {manager.get_metadata_path()}")

        print(f"\n[OK] Directory structure ready for {country_code}")

        # Show directory info
        info = manager.get_directory_info()
        print("\nDirectory info:")
        print(f"  Location: {info['base_path']}")
        print(f"  Exists: {info['exists']}")

        print("\n{'='*60}")
        print("Next steps:")
        print(f"1. Create configuration: config/countries/{country_code}.yaml")
        print("2. Implement data fetchers (if needed)")
        print("3. Register country in data_fetchers/__init__.py")
        print(f"{'=' * 60}\n")

    except Exception as e:
        logger.error(f"Failed to set up directories: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for logging configuration and CountryDataManager utilities.
"""

import pytest

from core.data_manager import CountryDataManager, setup_country_directories
from core.logging_config import get_logger, setup_logging


@pytest.fixture(autouse=True)
def clean_logging():
    """Cleanup logging handlers before and after test."""
    import logging

    # Clean before
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    yield

    # Clean after
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)


def test_setup_logging_creates_files_in_tmpdir(tmp_path, monkeypatch, caplog, clean_logging):
    """setup_logging should create log files in the given directory."""
    log_dir = tmp_path / "logs"

    # Ensure the environment does not override the level
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    # Reset logging handlers to avoid pollution
    import logging

    logging.root.handlers = []

    setup_logging(log_dir=str(log_dir), level="DEBUG", log_to_file=True)
    logger = get_logger(__name__)

    with caplog.at_level("INFO"):
        logger.info("test message")

    # Check that at least the main log file exists
    files = list(log_dir.glob("pricesentinel_*.log"))
    assert files, "Expected at least one main log file to be created"


def test_country_data_manager_directory_and_file_helpers(tmp_path, monkeypatch):
    """CountryDataManager should manage paths and directory info correctly."""
    # Use a temporary base path to avoid touching real data/
    base_path = tmp_path / "data"
    monkeypatch.chdir(tmp_path)

    manager = CountryDataManager("pt", base_path=str(base_path))
    manager.create_directories()

    # Basic path helpers
    assert manager.get_raw_path().is_dir()
    assert manager.get_raw_path("electricity").is_dir()
    assert manager.get_processed_path().is_dir()
    assert manager.get_events_path().is_dir()
    assert manager.get_metadata_path().is_dir()

    # File naming
    filename = manager.generate_filename("electricity", "2024-01-01", "2024-01-02")
    assert filename.startswith("PT_electricity_20240101_20240102")
    assert filename.endswith(".csv")

    raw_file = manager.get_file_path("electricity", "2024-01-01", "2024-01-02")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("timestamp,price\n")

    # Listing and latest helpers
    files = manager.list_files("electricity")
    assert files, "Expected at least one file for 'electricity' source"
    assert raw_file in files
    latest = manager.get_latest_file("electricity")
    assert latest == raw_file

    # Directory info should reflect at least one file
    info = manager.get_directory_info()
    assert info["exists"] is True
    assert info["file_count"] >= 1
    assert info["sources"]["electricity"] >= 1


def test_setup_country_directories_uses_manager(tmp_path, monkeypatch):
    """setup_country_directories should delegate to CountryDataManager."""
    base_path = tmp_path / "countries"
    monkeypatch.chdir(tmp_path)

    manager = setup_country_directories("ES", base_path=str(base_path))

    assert isinstance(manager, CountryDataManager)
    assert manager.country_code == "ES"
    # Base path should exist and have expected subdirectories
    assert manager.base_path.exists()
    assert manager.get_raw_path().exists()
    assert manager.get_processed_path().exists()
    assert manager.get_events_path().exists()
    assert manager.get_metadata_path().exists()

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Logging configuration for PriceSentinel.

This module provides centralised logging configuration with support for
both file and console output.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO", log_to_file: bool = True):
    """
    Configure logging for the entire application.

    Args:
        log_dir: Directory for log files (default: 'logs')
        level: Logging level (default: 'INFO')
        log_to_file: Whether to log to file (default: True)

    Example:
        >>> from core.logging_config import setup_logging
        >>> setup_logging()
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Create the logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Get log level from environment or parameter
    log_level_str = os.getenv("LOG_LEVEL", level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    simple_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (simpler format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler (detailed format)
    if log_to_file:
        # Main log file
        log_filename = log_path / f"pricesentinel_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Error log file (only errors and critical)
        error_log_filename = (
            log_path / f"pricesentinel_error_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_file_handler = logging.FileHandler(error_log_filename, mode="a", encoding="utf-8")
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_file_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    root_logger.info(f"Logging configured: level={log_level_str}, log_dir={log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Message")
    """
    return logging.getLogger(name)

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end testing of the training pipeline for a mock country.

This module provides functionality to test the full training pipeline for
a mock country, including fetching data, cleaning and verifying, engineering
features, and training the model. It ensures that all intermediate and final
steps are executed successfully and produce the expected artefacts.

Functions:
- mock_pipeline: Pytest fixture that initialises and tears down a mock pipeline.
- test_full_training_flow_mock_country: Test function for the end-to-end training flow.

"""

import pytest

from config.country_registry import CountryRegistry
from core.pipeline import Pipeline
from data_fetchers.mock import register_mock_country


@pytest.fixture
def mock_pipeline():
    """
    Initialise a pipeline for the mock country with a clean registry.
    """
    CountryRegistry.clear()
    register_mock_country()
    pipeline = Pipeline(country_code="XX")
    yield pipeline
    CountryRegistry.clear()


def test_full_training_flow_mock_country(tmp_path, mock_pipeline, monkeypatch):
    """
    End-to-end test: fetch -> clean -> features -> train for mock country.
    """
    # Use a short date range for quick tests
    start_date = "2024-01-01"
    end_date = "2024-01-07"

    # Run stages explicitly to keep the test readable
    mock_pipeline.fetch_data(start_date, end_date)
    mock_pipeline.clean_and_verify(start_date, end_date)
    mock_pipeline.engineer_features(start_date, end_date)
    mock_pipeline.train_model(start_date, end_date)

    # Verify that a model artefact was created
    from pathlib import Path

    models_root = Path("models") / "XX" / "baseline" / mock_pipeline.run_id
    assert (models_root / "model.pkl").exists()
    assert (models_root / "metrics.json").exists()

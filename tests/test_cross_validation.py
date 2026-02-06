# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Cross-Validation logic.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.cross_validation import CrossValidator
from core.pipeline import Pipeline


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    pipeline = MagicMock(spec=Pipeline)
    pipeline.country_code = "PT"

    # Mock repository
    pipeline.repository = MagicMock()

    # Mock runtime guard
    pipeline.runtime_guard = MagicMock()
    pipeline.runtime_guard.validate_and_clamp.side_effect = lambda df: df

    return pipeline


def test_cv_run_basic_flow(mock_pipeline):
    """Test standard CV run with mocked data."""
    # Setup mock data
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price_eur_mwh": [50.0] * len(dates),
            "target_price": [55.0] * len(dates),
            "price_lag_24": [50.0] * len(dates),
            "feature_1": [1.0] * len(dates),
        }
    )

    # Mock file listing and reading
    mock_pipeline.repository.list_processed_data.return_value = ["dummy_features.csv"]

    with patch("core.cross_validation.pd.read_csv") as mock_read:
        mock_read.return_value = df

        # Mock trainer
        with patch("models.get_trainer") as mock_get_trainer:
            mock_trainer = MagicMock()

            # predict() returns array of same length as input
            mock_trainer.predict.side_effect = lambda x: [52.0] * len(x)
            # trainer.train returns metrics dict (ignored)
            mock_trainer.train.return_value = {}

            mock_get_trainer.return_value = mock_trainer

            # Run CV
            cv = CrossValidator(mock_pipeline, n_splits=2)
            results = cv.run("2023-01-01", "2023-01-31")

            # Verify feature engineering was triggered
            mock_pipeline.engineer_features.assert_called_with("2023-01-01", "2023-01-31")

            # Verify results structure
            assert not results.empty
            assert len(results) == 2
            assert "mae" in results.columns
            assert "mae_naive" in results.columns
            assert "fold" in results.columns


def test_cv_no_files_raises(mock_pipeline):
    """Test CV raises FileNotFoundError if no feature files exist."""
    mock_pipeline.repository.list_processed_data.return_value = []

    cv = CrossValidator(mock_pipeline)

    with pytest.raises(FileNotFoundError):
        cv.run("2023-01-01", "2023-01-31")


def test_cv_empty_data_raises(mock_pipeline):
    """Test CV raises ValueError if filtered data is empty."""
    # Mock available file but empty data for range
    mock_pipeline.repository.list_processed_data.return_value = ["dummy.csv"]

    # Ensure timestamp is datetime aware even if empty
    df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "target_price": []})

    with patch("core.cross_validation.pd.read_csv") as mock_read:
        mock_read.return_value = df

        cv = CrossValidator(mock_pipeline)

        with pytest.raises(ValueError, match="No data available"):
            cv.run("2023-01-01", "2023-01-31")

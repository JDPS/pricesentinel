# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the run_pipeline CLI wiring and setup_country utility script.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

# We mock setup_country_main so we don't need to patch its internal logging
# from setup_country import main as setup_country_main
import pytest

from config.country_registry import CountryRegistry
from core.pipeline import Pipeline
from data_fetchers.mock import register_mock_country
from run_pipeline import PipelineCLI, validate_arguments


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Disable file logging for all CLI tests."""
    monkeypatch.setattr("run_pipeline.setup_logging", MagicMock())
    monkeypatch.setattr("core.logging_config.setup_logging", MagicMock())


def _make_args(**overrides):
    """Helper to create an argparse-like namespace for PipelineCLI."""
    defaults = {
        "country": "XX",
        "all": False,
        "fetch": False,
        "clean": False,
        "features": False,
        "train": False,
        "forecast": False,
        "info": False,
        "start_date": "2024-01-01",
        "end_date": "2024-01-07",
        "forecast_date": "2024-01-08",
        "forecast_start_date": None,
        "forecast_end_date": None,
        "log_level": "INFO",
        "model_name": "baseline",
        "fast_train": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_validate_arguments_requires_action(capsys):
    """validate_arguments should fail when no action flags are provided."""
    args = _make_args(
        all=False, fetch=False, clean=False, features=False, train=False, forecast=False, info=False
    )

    valid = validate_arguments(args)

    captured = capsys.readouterr()
    assert valid is False
    assert "No action specified" in captured.out


def test_pipeline_cli_runs_all_stages_for_mock_country(tmp_path, monkeypatch):
    """
    Tests that the pipeline cli functionality triggers all stages for a mock country
    in a controlled test environment. The test isolates the pipeline execution
    to prevent any heavy computation or side effects by utilising a lightweight
    spy class, `DummyPipeline`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path provided by pytest for safely creating and
        removing files and directories during the test.
    monkeypatch : pytest.MonkeyPatch
        A pytest fixture for safely patching and modifying attributes or the
        behaviour of objects during the test.
    """
    # Prepare mock country and registry
    CountryRegistry.clear()
    register_mock_country()

    # Patch CountryDataManager base path to the temporary directory via environment
    monkeypatch.chdir(tmp_path)

    args = _make_args(all=True)
    cli = PipelineCLI(args)

    # Replace Pipeline with a lightweight spy to avoid heavy work
    calls = []

    class DummyPipeline(Pipeline):  # type: ignore[misc]
        """
        A Dummy Pipeline for testing.
        """

        def __init__(self, country_code: str):  # noqa: D401
            # Skip parent initialisation; just record initialisation
            calls.append(("init", country_code))

        def run_full_pipeline(self, start_date, end_date, forecast_date, model_name="baseline"):
            calls.append(("run_full", start_date, end_date, forecast_date, model_name))

    monkeypatch.setattr("core.pipeline.Pipeline", DummyPipeline)
    monkeypatch.setattr("run_pipeline.Pipeline", DummyPipeline)

    cli.run()

    assert ("init", "XX") in calls
    assert ("run_full", "2024-01-01", "2024-01-07", "2024-01-08", "baseline") in calls


def test_pipeline_cli_fast_train_adjusts_model_name(tmp_path, monkeypatch):
    """
    Tests the behaviour of the CLI for pipeline execution when using the fast training option,
    ensuring that the model name is adjusted correctly.

    Attributes:
        None

    Raises:
        AssertionError: Raised if the test case fails due to
        missing or misbehaving pipeline execution.

    Parameters:
        tmp_path (Path): Temporary path object used as test-specific directory.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to modify or mock runtime behaviour.

    """
    CountryRegistry.clear()
    register_mock_country()
    monkeypatch.chdir(tmp_path)

    args = _make_args(all=True, fast_train=True, model_name="baseline")
    cli = PipelineCLI(args)

    calls = []

    class DummyPipeline(Pipeline):  # type: ignore[misc]
        """
        Dummy Pipeline for testing.
        """

        def __init__(self, country_code: str):  # noqa: D401
            super().__init__(country_code)
            calls.append(("init", country_code))

        def run_full_pipeline(self, start_date, end_date, forecast_date, model_name="baseline"):
            calls.append(("run_full", start_date, end_date, forecast_date, model_name))

    monkeypatch.setattr("run_pipeline.Pipeline", DummyPipeline)

    cli.run()

    assert ("init", "XX") in calls
    assert ("run_full", "2024-01-01", "2024-01-07", "2024-01-08", "baseline_fast") in calls


def test_pipeline_cli_forecast_range_uses_generate_forecast_range(tmp_path, monkeypatch):
    """CLI forecast range flags should call generate_forecast_range on the pipeline."""
    CountryRegistry.clear()
    register_mock_country()
    monkeypatch.chdir(tmp_path)

    args = _make_args(
        all=False,
        fetch=False,
        clean=False,
        features=False,
        train=False,
        forecast=True,
        forecast_start_date="2024-01-01",
        forecast_end_date="2024-01-03",
    )
    cli = PipelineCLI(args)

    calls = []

    class DummyPipeline(Pipeline):  # type: ignore[misc]
        def __init__(self, country_code: str):  # noqa: D401
            calls.append(("init", country_code))

        def generate_forecast_range(self, start_date, end_date, model_name="baseline"):
            calls.append(("range", start_date, end_date, model_name))

    monkeypatch.setattr("run_pipeline.Pipeline", DummyPipeline)

    cli.run()

    assert ("init", "XX") in calls
    assert ("range", "2024-01-01", "2024-01-03", "baseline") in calls


def test_setup_country_main_smoke(monkeypatch, tmp_path, capsys):
    """setup_country.main should create directories and print a summary."""
    # Patch basicConfig to prevent side effects during import
    monkeypatch.setattr("logging.basicConfig", lambda **kwargs: None)

    # Deferred import
    from setup_country import main as setup_country_main

    # Redirect current working directory so script writes under tmp_path
    monkeypatch.chdir(tmp_path)

    # Simulate command-line arguments
    monkeypatch.setattr(
        "sys.argv",
        ["setup_country.py", "PT"],
        raising=False,
    )

    setup_country_main()

    captured = capsys.readouterr().out
    assert "Setting up directory structure for: PT" in captured
    assert "Directory structure ready for PT" in captured

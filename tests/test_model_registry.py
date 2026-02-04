# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ModelRegistry.
"""

import pytest

from models.model_registry import ModelRegistry


def test_model_registry_save_and_load(tmp_path):
    """Registry should save and load models correctly."""
    registry_root = tmp_path / "models"
    registry = ModelRegistry(registry_root)

    country = "XX"
    model_name = "test_model"
    run_id = "run_1"

    # Dummy model (just a dict for pickling)
    model = {"a": 1, "b": 2}
    metrics = {"mae": 0.5}

    # Save
    registry.save_model(country, model_name, run_id, model, metrics)

    # Check directory structure
    run_dir = registry_root / country / model_name / run_id
    assert run_dir.exists()
    assert (run_dir / "model.pkl").exists()
    assert (run_dir / "metrics.json").exists()

    # Load specific run
    loaded_model, metadata = registry.load_model(country, model_name, run_id)
    assert loaded_model == model
    assert metadata["metrics"] == metrics
    assert metadata["run_id"] == run_id

    # Load latest run (should be the same since only one exists)
    loaded_model_latest, _ = registry.load_model(country, model_name)
    assert loaded_model_latest == model


def test_model_registry_list_models(tmp_path):
    """Registry should list multiple runs correctly."""
    registry_root = tmp_path / "models"
    registry = ModelRegistry(registry_root)

    country = "XX"
    model_name = "test_model"

    # Save multiple runs
    # We sleep or mock time if dependent on timestamp sorting,
    # but get_latest_run_id depends on file mtime.
    # To ensure distinct mtimes, we might need a small delay, or we trust OS file system resolution.
    # Alternatively, list_models uses directory iteration.

    registry.save_model(country, model_name, "run_1", {"v": 1})

    # Force a slight mtime difference if filesystem is fast (rarely needed for basic listing check)
    import time

    time.sleep(0.01)

    registry.save_model(country, model_name, "run_2", {"v": 2})

    models = registry.list_models(country)
    assert model_name in models
    runs = models[model_name]
    assert "run_1" in runs
    assert "run_2" in runs
    assert len(runs) == 2

    # Check latest
    latest_id = registry.get_latest_run_id(country, model_name)
    assert latest_id == "run_2"

    model_latest, _ = registry.load_model(country, model_name)
    assert model_latest == {"v": 2}


def test_load_nonexistent_model_raises_error(tmp_path):
    """Loading a missing model should raise FileNotFoundError."""
    registry = ModelRegistry(tmp_path / "models")

    with pytest.raises(FileNotFoundError):
        registry.load_model("XX", "missing_model")

    with pytest.raises(FileNotFoundError):
        registry.load_model("XX", "missing_model", "run_x")

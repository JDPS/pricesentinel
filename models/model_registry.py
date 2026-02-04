# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Model Registry for managing machine learning model artifacts and metadata.

This module provides a centralized interface for saving, loading, and listing
models, ensuring consistent storage structures and metadata tracking.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages storage and retrieval of ML models and their metadata.
    """

    def __init__(self, models_root: str | Path = "models"):
        """
        Initialize the registry.

        Args:
            models_root: Root directory for storing models.
        """
        self.models_root = Path(models_root)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, country_code: str, model_name: str, run_id: str) -> Path:
        """Get the directory path for a specific model run."""
        return self.models_root / country_code / model_name / run_id

    def save_model(
        self,
        country_code: str,
        model_name: str,
        run_id: str,
        model: Any,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a trained model and its metrics.

        Args:
            country_code: Country code for the model.
            model_name: Name/type of the model.
            run_id: Unique identifier for this training run.
            model: The trained model object (must be pickleable).
            metrics: Dictionary of evaluation metrics.

        Returns:
            Path to the saved model directory.
        """
        run_dir = self._get_run_dir(country_code, model_name, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        model_path = run_dir / "model.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to pickle model for {country_code}/{model_name}: {e}")
            raise

        # Save metrics and metadata
        timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017
        metadata = {
            "country_code": country_code,
            "model_name": model_name,
            "run_id": run_id,
            "timestamp": timestamp,
            "metrics": metrics or {},
        }

        metrics_path = run_dir / "metrics.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics for {country_code}/{model_name}: {e}")

        logger.info(f"Saved model {model_name} for {country_code} (run_id={run_id})")
        return run_dir

    def load_model(
        self,
        country_code: str,
        model_name: str,
        run_id: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Load a model and its metadata.

        Args:
            country_code: Country code.
            model_name: Model name.
            run_id: Specific run ID. If None, loads the latest available model.

        Returns:
            Tuple of (model_object, metadata_dict).

        Raises:
            FileNotFoundError: If the model or run directory is not found.
        """
        if run_id:
            run_dir = self._get_run_dir(country_code, model_name, run_id)
            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Model run not found: {country_code}/{model_name}/{run_id}"
                )
        else:
            # Find latest run
            latest_run = self.get_latest_run_id(country_code, model_name)
            if not latest_run:
                raise FileNotFoundError(f"No models found for {country_code}/{model_name}")
            run_dir = self._get_run_dir(country_code, model_name, latest_run)

        # Load model
        model_path = run_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact missing at {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        # Load metadata
        metrics_path = run_dir / "metrics.json"
        metadata = {}
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return model, metadata

    def get_latest_run_id(self, country_code: str, model_name: str) -> str | None:
        """
        Find the ID of the most recently created model run.

        Args:
            country_code: Country code.
            model_name: Model name.

        Returns:
            Run ID of the latest model, or None if no models exist.
        """
        model_root = self.models_root / country_code / model_name
        if not model_root.exists():
            return None

        # Look for subdirectories containing 'model.pkl'
        candidates = []
        for run_dir in model_root.iterdir():
            if run_dir.is_dir() and (run_dir / "model.pkl").exists():
                candidates.append(run_dir)

        if not candidates:
            return None

        # Sort by modification time of the model artifact file
        candidates.sort(key=lambda p: (p / "model.pkl").stat().st_mtime, reverse=True)
        return candidates[0].name

    def list_models(self, country_code: str) -> dict[str, list[str]]:
        """
        List all available models and their runs for a country.

        Args:
            country_code: Country code.

        Returns:
            Dictionary where keys are model names and strings are lists of run IDs.
        """
        country_root = self.models_root / country_code
        if not country_root.exists():
            return {}

        result = {}
        for model_dir in country_root.iterdir():
            if model_dir.is_dir():
                runs = []
                for run_dir in model_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "model.pkl").exists():
                        runs.append(run_dir.name)
                if runs:
                    result[model_dir.name] = sorted(runs, reverse=True)
        return result

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""MLflow experiment tracking wrapper for PriceSentinel.

Provides a thin wrapper around MLflow that gracefully degrades to no-ops
when MLflow is not installed or the tracking server is unreachable.
This ensures the forecasting pipeline never crashes due to tracking issues.

Usage:
    tracker = ExperimentTracker(experiment_name="pricesentinel")
    run_id = tracker.log_run(
        model_name="xgboost",
        metrics={"rmse": 5.2, "mae": 3.1},
        params={"n_estimators": 100},
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracking wrapper with graceful degradation.

    If MLflow is not installed or ``MLFLOW_TRACKING_URI`` is not set, all
    tracking methods become silent no-ops so the pipeline can run without
    any tracking infrastructure.

    Args:
        experiment_name: The MLflow experiment name to log runs under.
    """

    def __init__(self, experiment_name: str = "pricesentinel") -> None:
        self._experiment_name = experiment_name
        self._mlflow: Any | None = None
        self._enabled = False

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            logger.info("MLFLOW_TRACKING_URI not set; experiment tracking is disabled.")
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._enabled = True
            logger.info(
                "MLflow tracking enabled (uri=%s, experiment=%s).",
                tracking_uri,
                experiment_name,
            )
        except ImportError:
            logger.warning(
                "mlflow is not installed; experiment tracking is disabled. "
                "Install with: uv sync --extra tracking"
            )
        except Exception:
            logger.warning(
                "Failed to initialise MLflow tracking; continuing without it.",
                exc_info=True,
            )

    @property
    def enabled(self) -> bool:
        """Whether MLflow tracking is active."""
        return self._enabled

    def log_run(
        self,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> str | None:
        """Log a training run to MLflow.

        Args:
            model_name: Name of the model (used as the MLflow run name).
            metrics: Dictionary of metric name to value.
            params: Optional dictionary of hyperparameters.
            artifacts: Optional list of file paths to log as artifacts.

        Returns:
            The MLflow run ID if logging succeeded, otherwise ``None``.
        """
        if not self._enabled or self._mlflow is None:
            logger.debug("Tracking disabled; skipping log_run for '%s'.", model_name)
            return None

        mlflow = self._mlflow

        try:
            with mlflow.start_run(run_name=model_name) as run:
                if params:
                    mlflow.log_params(params)

                mlflow.log_metrics(metrics)

                if artifacts:
                    for artifact_path in artifacts:
                        mlflow.log_artifact(artifact_path)

                run_id: str = run.info.run_id
                logger.info(
                    "Logged MLflow run '%s' (run_id=%s) with %d metrics.",
                    model_name,
                    run_id,
                    len(metrics),
                )
                return run_id

        except Exception:
            logger.warning(
                "Failed to log MLflow run for '%s'; continuing without tracking.",
                model_name,
                exc_info=True,
            )
            return None

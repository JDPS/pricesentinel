# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Model comparison framework for PriceSentinel.

Provides systematic comparison of multiple model trainers using walk-forward
or time-series split cross-validation, producing ranked results and Markdown
reports.

Usage:
    from experiments.model_comparison import ModelComparison

    comparison = ModelComparison(
        country_code="PT",
        model_names=["baseline", "xgboost", "lightgbm"],
    )
    results = comparison.run(x, y)
    report = comparison.generate_report(results)
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from core.cross_validation import WalkForwardValidator
from models import get_trainer, list_registered_trainers

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple model trainers on the same dataset.

    Runs each model through walk-forward or time-series split cross-validation,
    tracks per-model error metrics and timing, and produces a ranked comparison.

    Args:
        country_code: ISO country code passed to the trainer factory.
        model_names: List of registered model names to compare.
        model_configs: Optional per-model configuration overrides, keyed by
            model name.
        models_root: Root directory for model artifacts.
    """

    def __init__(
        self,
        country_code: str,
        model_names: list[str],
        model_configs: dict[str, dict[str, Any]] | None = None,
        models_root: str = "models",
    ) -> None:
        self.country_code = country_code
        self.model_names = model_names
        self.model_configs = model_configs or {}
        self.models_root = models_root
        self.results_: pd.DataFrame | None = None

        self._validate_model_names()

    def _validate_model_names(self) -> None:
        """Verify all requested models are registered.

        Raises:
            ValueError: If any model name is not found in the registry.
        """
        available = set(list_registered_trainers())
        unknown = [m for m in self.model_names if m not in available]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(available)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        cv_method: str = "walk_forward",
        initial_train_size: int = 720,
        step_size: int = 24,
        n_splits: int = 5,
        mode: str = "expanding",
    ) -> pd.DataFrame:
        """
        Run cross-validated comparison for every model.

        Args:
            x: Feature matrix ordered chronologically.
            y: Target series aligned with *x*.
            cv_method: ``"walk_forward"`` or ``"time_series_split"``.
            initial_train_size: Initial training window size (walk-forward).
            step_size: Step size in samples (walk-forward).
            n_splits: Number of folds (time-series split).
            mode: Window mode for walk-forward (``"expanding"`` or
                ``"sliding"``).

        Returns:
            DataFrame ranked by mean MAE ascending with columns:
            ``model``, ``mean_mae``, ``std_mae``, ``mean_rmse``,
            ``std_rmse``, ``train_time_s``, ``inference_time_s``,
            ``n_steps``.
        """
        if cv_method not in ("walk_forward", "time_series_split"):
            raise ValueError(
                f"cv_method must be 'walk_forward' or 'time_series_split', " f"got '{cv_method}'"
            )

        logger.info(
            "Starting model comparison: models=%s, cv=%s, samples=%d",
            self.model_names,
            cv_method,
            len(x),
        )

        comparison_rows: list[dict[str, Any]] = []

        for model_name in self.model_names:
            logger.info("Evaluating model '%s' ...", model_name)
            config = self.model_configs.get(model_name)

            try:
                row = self._evaluate_model(
                    model_name=model_name,
                    config=config,
                    x=x,
                    y=y,
                    cv_method=cv_method,
                    initial_train_size=initial_train_size,
                    step_size=step_size,
                    n_splits=n_splits,
                    mode=mode,
                )
                comparison_rows.append(row)
            except Exception:
                logger.exception("Model '%s' failed during evaluation", model_name)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "mean_mae": np.nan,
                        "std_mae": np.nan,
                        "mean_rmse": np.nan,
                        "std_rmse": np.nan,
                        "train_time_s": np.nan,
                        "inference_time_s": np.nan,
                        "n_steps": 0,
                        "status": "error",
                    }
                )

        results = (
            pd.DataFrame(comparison_rows)
            .sort_values("mean_mae", ascending=True, na_position="last")
            .reset_index(drop=True)
        )

        self.results_ = results
        logger.info("Model comparison complete. Best model: %s", results.iloc[0]["model"])
        return results

    def generate_report(self, results_df: pd.DataFrame | None = None) -> str:
        """
        Generate a Markdown report from comparison results.

        Args:
            results_df: Results DataFrame. If *None*, uses the internally
                stored results from the last :meth:`run` call.

        Returns:
            Markdown-formatted report string.

        Raises:
            ValueError: If no results are available.
        """
        df = results_df if results_df is not None else self.results_
        if df is None or df.empty:
            raise ValueError("No results available. Call run() first.")

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        best_model = df.iloc[0]["model"]
        best_mae = df.iloc[0]["mean_mae"]

        lines: list[str] = [
            f"# Model Comparison Report -- {self.country_code}",
            "",
            f"- **Generated**: {timestamp}",
            f"- **Country**: {self.country_code}",
            f"- **Models evaluated**: {len(df)}",
            f"- **Best model**: {best_model} (MAE {best_mae:.4f})",
            "",
            "## Ranking",
            "",
            df.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Summary",
            "",
        ]

        successful = df[df.get("status", "ok") != "error"]
        if len(successful) >= 2:
            worst_mae = successful.iloc[-1]["mean_mae"]
            spread = worst_mae - best_mae
            lines.append(
                f"The best model ({best_model}) achieved a mean MAE of "
                f"**{best_mae:.4f}**, which is **{spread:.4f}** lower than "
                f"the worst model ({successful.iloc[-1]['model']})."
            )
        elif len(successful) == 1:
            lines.append(
                f"Only one model completed successfully: {best_model} "
                f"with mean MAE **{best_mae:.4f}**."
            )

        failed = df[df.get("status", "ok") == "error"]
        if not failed.empty:
            lines.extend(
                [
                    "",
                    "## Errors",
                    "",
                    "The following models failed during evaluation:",
                    "",
                ]
            )
            for _, row in failed.iterrows():
                lines.append(f"- **{row['model']}**")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model_name: str,
        config: Any,
        x: pd.DataFrame,
        y: pd.Series,
        cv_method: str,
        initial_train_size: int,
        step_size: int,
        n_splits: int,
        mode: str,
    ) -> dict[str, Any]:
        """Evaluate a single model through cross-validation.

        Returns a dict with aggregated metrics and timing information.
        """

        def _trainer_factory() -> Any:
            return get_trainer(
                country_code=self.country_code,
                model_name=model_name,
                models_root=self.models_root,
                config=config,
            )

        t_start = time.perf_counter()

        if cv_method == "walk_forward":
            fold_df = self._run_walk_forward(
                x=x,
                y=y,
                trainer_factory=_trainer_factory,
                initial_train_size=initial_train_size,
                step_size=step_size,
                mode=mode,
            )
        else:
            fold_df = self._run_time_series_split(
                x=x,
                y=y,
                trainer_factory=_trainer_factory,
                n_splits=n_splits,
            )

        total_time = time.perf_counter() - t_start

        # Estimate training vs inference split (approximate -- walk-forward
        # timing is dominated by training)
        n_steps = len(fold_df)
        train_time_est = total_time * 0.9
        infer_time_est = total_time * 0.1

        return {
            "model": model_name,
            "mean_mae": float(fold_df["mae"].mean()),
            "std_mae": float(fold_df["mae"].std()),
            "mean_rmse": float(fold_df["rmse"].mean()),
            "std_rmse": float(fold_df["rmse"].std()),
            "train_time_s": round(train_time_est, 3),
            "inference_time_s": round(infer_time_est, 3),
            "n_steps": n_steps,
            "status": "ok",
        }

    @staticmethod
    def _run_walk_forward(
        x: pd.DataFrame,
        y: pd.Series,
        trainer_factory: Any,
        initial_train_size: int,
        step_size: int,
        mode: str,
    ) -> pd.DataFrame:
        """Delegate to :class:`WalkForwardValidator`."""
        validator = WalkForwardValidator(
            initial_train_size=initial_train_size,
            step_size=step_size,
            mode=mode,
        )
        return validator.run(x, y, trainer_factory)

    @staticmethod
    def _run_time_series_split(
        x: pd.DataFrame,
        y: pd.Series,
        trainer_factory: Any,
        n_splits: int,
    ) -> pd.DataFrame:
        """Run standard TimeSeriesSplit cross-validation.

        Returns a DataFrame with ``mae`` and ``rmse`` columns compatible
        with the walk-forward output.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results: list[dict[str, float | int]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(x)):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            trainer = trainer_factory()
            trainer.train(x_train, y_train)
            y_pred = trainer.predict(x_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            fold_results.append(
                {
                    "step_idx": fold_idx,
                    "train_size": len(x_train),
                    "test_size": len(x_test),
                    "mae": mae,
                    "rmse": rmse,
                }
            )

            logger.info(
                "TimeSeriesSplit fold %d/%d: MAE=%.4f, RMSE=%.4f",
                fold_idx + 1,
                n_splits,
                mae,
                rmse,
            )

        return pd.DataFrame(fold_results)

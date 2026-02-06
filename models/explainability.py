# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
SHAP-based model explainability for PriceSentinel forecasting models.

Provides feature importance analysis and interpretability summaries
using SHAP (SHapley Additive exPlanations) values for tree-based
models such as sklearn RandomForest, XGBoost, and LightGBM.

Usage:
    from models.explainability import SHAPExplainer

    explainer = SHAPExplainer(trainer.model, feature_names=feature_names)
    importance_df = explainer.feature_importance(x_test)
    explainer.save_summary(x_test, "shap_summary.md")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import ModelError

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP explainability wrapper for tree-based forecasting models.

    Computes SHAP values using TreeExplainer and provides utilities
    for feature importance ranking and text-based summary generation.

    Attributes:
        model: A fitted tree-based model object (sklearn, xgboost, or lightgbm).
        feature_names: Optional list of feature names corresponding to model inputs.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the SHAP explainer.

        Args:
            model: A trained tree-based model object (e.g., RandomForestRegressor,
                xgb.Booster, lgb.Booster, or their sklearn-API wrappers).
            feature_names: Optional list of feature names. If None, names are
                inferred from the DataFrame passed to subsequent method calls.

        Raises:
            ModelError: If the provided model is None.
        """
        if model is None:
            raise ModelError(
                "Cannot create SHAPExplainer with a None model.",
                details={"model": None},
            )
        self.model = model
        self.feature_names = feature_names
        self._explainer_cache: Any | None = None

    def _get_explainer(self) -> Any:
        """
        Lazily create and cache a SHAP TreeExplainer.

        Returns:
            A shap.TreeExplainer instance bound to self.model.

        Raises:
            ModelError: If shap is not installed or the model is incompatible
                with TreeExplainer.
        """
        if self._explainer_cache is not None:
            return self._explainer_cache

        try:
            import shap  # noqa: F811
        except ImportError as exc:
            raise ModelError(
                "The 'shap' package is required for explainability. "
                "Install it with: pip install shap",
                details={"missing_package": "shap"},
            ) from exc

        try:
            self._explainer_cache = shap.TreeExplainer(self.model)
        except Exception as exc:
            raise ModelError(
                f"Failed to create SHAP TreeExplainer for model type "
                f"'{type(self.model).__name__}'. Ensure the model is a fitted "
                f"tree-based estimator.",
                details={"model_type": type(self.model).__name__, "error": str(exc)},
            ) from exc

        logger.debug(
            "Created SHAP TreeExplainer for model type '%s'.",
            type(self.model).__name__,
        )
        return self._explainer_cache

    def compute_shap_values(self, x: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for the given input features.

        Args:
            x: A DataFrame of input features with shape (n_samples, n_features).

        Returns:
            A numpy array of SHAP values with shape (n_samples, n_features).

        Raises:
            ModelError: If SHAP value computation fails.
        """
        if x.empty:
            raise ModelError(
                "Cannot compute SHAP values on an empty DataFrame.",
                details={"shape": x.shape},
            )

        explainer = self._get_explainer()

        try:
            shap_values = explainer.shap_values(x)
        except Exception as exc:
            raise ModelError(
                f"SHAP value computation failed: {exc}",
                details={"input_shape": x.shape, "error": str(exc)},
            ) from exc

        # Some multi-output models return a list of arrays; take the first.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return np.asarray(shap_values)

    def feature_importance(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean absolute SHAP importance per feature.

        Args:
            x: A DataFrame of input features.

        Returns:
            A DataFrame with columns ``feature`` and ``importance``
            (mean |SHAP value|), sorted in descending order of importance.
        """
        shap_values = self.compute_shap_values(x)
        mean_abs_shap: np.ndarray = np.abs(shap_values).mean(axis=0)

        names = self.feature_names if self.feature_names is not None else list(x.columns)

        if len(names) != mean_abs_shap.shape[0]:
            raise ModelError(
                f"Feature name count ({len(names)}) does not match "
                f"SHAP value dimension ({mean_abs_shap.shape[0]}).",
                details={
                    "feature_name_count": len(names),
                    "shap_dimension": mean_abs_shap.shape[0],
                },
            )

        importance_df = pd.DataFrame({"feature": names, "importance": mean_abs_shap})
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(
            drop=True
        )

        logger.info(
            "Computed SHAP feature importance for %d features over %d samples.",
            len(names),
            x.shape[0],
        )
        return importance_df

    def save_summary(
        self,
        x: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """
        Save a text-based SHAP feature importance summary to a file.

        Generates a Markdown-formatted report listing all features ranked
        by their mean absolute SHAP values.

        Args:
            x: A DataFrame of input features used to compute SHAP values.
            output_path: Destination file path for the summary report.

        Raises:
            ModelError: If the summary cannot be written.
        """
        importance_df = self.feature_importance(x)
        output_path = Path(output_path)

        total_importance = importance_df["importance"].sum()
        lines: list[str] = [
            "# SHAP Feature Importance Summary",
            "",
            f"- **Model type**: `{type(self.model).__name__}`",
            f"- **Number of samples**: {x.shape[0]}",
            f"- **Number of features**: {x.shape[1]}",
            "",
            "## Feature Rankings",
            "",
            "| Rank | Feature | Mean |SHAP| | Cumulative % |",
            "|-----:|:--------|-------------:|-------------:|",
        ]

        cumulative = 0.0
        for rank, row in enumerate(importance_df.itertuples(), start=1):
            imp = float(row.importance)  # type: ignore[arg-type]
            feat = str(row.feature)
            cumulative += imp
            pct = (cumulative / total_importance * 100.0) if total_importance > 0 else 0.0
            lines.append(f"| {rank} | {feat} | {imp:.6f} | {pct:.1f}% |")

        lines.append("")
        lines.append("---")
        lines.append(
            f"*Generated by PriceSentinel SHAPExplainer "
            f"({x.shape[0]} samples, {x.shape[1]} features).*"
        )
        lines.append("")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            raise ModelError(
                f"Failed to write SHAP summary to '{output_path}': {exc}",
                details={"output_path": str(output_path), "error": str(exc)},
            ) from exc

        logger.info("SHAP summary saved to '%s'.", output_path)

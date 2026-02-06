# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Automated feature selection for reducing noise and improving model performance.

This module provides importance-based and recursive feature elimination methods
for selecting the most relevant features from engineered datasets.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Automated feature selection using importance-based methods.

    Supports tree-based importance ranking and threshold-based filtering.
    """

    def __init__(
        self,
        method: str = "importance",
        threshold: float = 0.01,
        max_features: int | None = None,
    ):
        """
        Initialize the feature selector.

        Args:
            method: Selection method ("importance" for tree-based importance).
            threshold: Minimum feature importance to keep (for importance method).
            max_features: Maximum number of features to keep. None means no limit.
        """
        self.method = method
        self.threshold = threshold
        self.max_features = max_features

    def select(self, x: pd.DataFrame, y: pd.Series) -> list[str]:
        """
        Select the most important features.

        Args:
            x: Feature matrix (numeric columns only).
            y: Target vector.

        Returns:
            List of selected feature column names.
        """
        if self.method == "importance":
            return self._importance_based(x, y)
        else:
            logger.warning("Unknown method '%s', falling back to importance", self.method)
            return self._importance_based(x, y)

    def _importance_based(self, x: pd.DataFrame, y: pd.Series) -> list[str]:
        """Select features based on tree-based feature importance."""
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x, y)

        importances = pd.Series(
            model.feature_importances_,
            index=x.columns,
        ).sort_values(ascending=False)

        # Apply threshold
        selected = importances[importances >= self.threshold]

        # Apply max_features limit
        if self.max_features is not None and len(selected) > self.max_features:
            selected = selected.head(self.max_features)

        selected_names = selected.index.tolist()

        logger.info(
            "Feature selection: %d/%d features selected (threshold=%.3f)",
            len(selected_names),
            len(x.columns),
            self.threshold,
        )

        return cast(list[str], selected_names)

    def get_importance_ranking(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Get a ranked DataFrame of feature importances.

        Args:
            x: Feature matrix.
            y: Target vector.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted descending.
        """
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x, y)

        importances = np.array(model.feature_importances_)

        ranking = (
            pd.DataFrame(
                {
                    "feature": x.columns.tolist(),
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return ranking

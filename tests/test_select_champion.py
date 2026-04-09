# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for champion selection helpers."""

from experiments.select_champion import _resolve_allowed_models


def test_resolve_allowed_models_filters_unavailable() -> None:
    requested = ["baseline", "xgboost", "lightgbm"]
    allowed, missing = _resolve_allowed_models(requested, available_models={"baseline"})

    assert allowed == ["baseline"]
    assert missing == ["xgboost", "lightgbm"]

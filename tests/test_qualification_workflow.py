# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for onboarding qualification helpers."""

from pathlib import Path

import pandas as pd

from experiments.qualify_country import _check_schema, _phase_status


def test_check_schema_passes_when_expected_columns_present(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {"timestamp": ["2024-01-01"], "price_eur_mwh": [10.0], "quality_flag": [0]}
    ).to_csv(path, index=False)

    result = _check_schema(path, {"timestamp", "price_eur_mwh", "quality_flag"})

    assert result["status"] == "pass"
    assert result["missing_columns"] == []


def test_phase_status_progression() -> None:
    schemas = {
        "a": {"status": "pass"},
        "b": {"status": "pass"},
    }
    features = {"status": "pass"}

    phase_a_only = _phase_status({"a": {"status": "fail"}}, features, smoke_ok=True)
    assert phase_a_only["ready_phase"] == "not_ready"

    phase_b = _phase_status(schemas, features, smoke_ok=False)
    assert phase_b["ready_phase"] == "B"

    phase_c = _phase_status(schemas, features, smoke_ok=True)
    assert phase_c["ready_phase"] == "C"

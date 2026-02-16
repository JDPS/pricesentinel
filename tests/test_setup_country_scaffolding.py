# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup_country scaffolding flags."""

from pathlib import Path


def test_setup_country_scaffold_flags_create_expected_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("logging.basicConfig", lambda **kwargs: None)
    from setup_country import main as setup_country_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "setup_country.py",
            "ES",
            "--scaffold-config",
            "--scaffold-fetchers",
            "--scaffold-tests",
        ],
        raising=False,
    )

    setup_country_main()

    expected = [
        Path("config/countries/ES.yaml"),
        Path("config/selection_policies/ES.yaml"),
        Path("config/monitoring/ES.yaml"),
        Path("data_fetchers/es/__init__.py"),
        Path("data_fetchers/es/electricity.py"),
        Path("data_fetchers/es/weather.py"),
        Path("data_fetchers/es/gas.py"),
        Path("data_fetchers/es/events.py"),
        Path("tests/qualification/test_es_qualification.py"),
    ]

    for path in expected:
        assert path.exists(), f"expected scaffold file missing: {path}"

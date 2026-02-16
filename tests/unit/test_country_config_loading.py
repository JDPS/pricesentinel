# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict country-config loading and schema enforcement."""

from pathlib import Path

import pytest
import yaml

from config.country_registry import CountryConfig
from config.validation import generate_config_template
from core.exceptions import ConfigurationError


def _write_config(tmp_path: Path, country_code: str, payload: dict) -> str:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{country_code}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(config_dir)


def test_country_config_from_yaml_validates_successfully(tmp_path: Path) -> None:
    payload = generate_config_template("pt", "Portugal")
    config_dir = _write_config(tmp_path, "PT", payload)

    config = CountryConfig.from_yaml("PT", config_dir=config_dir)

    assert config.country_code == "PT"
    assert config.features_config["use_weather_features"] is True


def test_country_config_from_yaml_rejects_misnested_features(tmp_path: Path) -> None:
    payload = generate_config_template("pt", "Portugal")
    payload["events"]["features"] = {"use_event_features": False}
    config_dir = _write_config(tmp_path, "PT", payload)

    with pytest.raises(ConfigurationError) as exc:
        CountryConfig.from_yaml("PT", config_dir=config_dir)

    assert "events.features" in str(exc.value)


def test_country_config_from_yaml_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = generate_config_template("pt", "Portugal")
    payload["unexpected"] = "value"
    config_dir = _write_config(tmp_path, "PT", payload)

    with pytest.raises(ConfigurationError) as exc:
        CountryConfig.from_yaml("PT", config_dir=config_dir)

    assert "unexpected" in str(exc.value)

# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Utility script to set up country directories and optional scaffolding."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from config.validation import generate_config_template
from core.data_manager import setup_country_directories

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _write_yaml(path: Path, payload: dict[str, Any], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return True


def _write_text(path: Path, text: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _scaffold_country_config(country_code: str, overwrite: bool) -> list[Path]:
    code = country_code.upper()
    created: list[Path] = []

    config_payload = generate_config_template(code, f"{code} Country")
    config_path = Path("config") / "countries" / f"{code}.yaml"
    if _write_yaml(config_path, config_payload, overwrite):
        created.append(config_path)

    selection_policy = {
        "allowed_models": ["baseline", "xgboost", "lightgbm"],
        "cv_method": "walk_forward",
        "initial_train_size": 720,
        "step_size": 24,
        "n_splits": 5,
        "metrics_priority": ["mae", "rmse", "stability", "model_name"],
        "hpo": {
            "enabled": False,
            "shortlist_size": 1,
            "n_trials": 10,
            "cv_splits": 3,
        },
    }
    policy_path = Path("config") / "selection_policies" / f"{code}.yaml"
    if _write_yaml(policy_path, selection_policy, overwrite):
        created.append(policy_path)

    monitoring_policy = {
        "status_thresholds": {
            "mae_warn": 20.0,
            "mae_critical": 30.0,
            "drift_warn_pct": 0.2,
            "drift_critical_pct": 0.35,
            "min_coverage_7d": 0.7,
            "min_coverage_30d": 0.8,
        },
        "freshness_hours": {
            "electricity": 48,
            "weather": 72,
            "gas": 96,
        },
    }
    monitor_path = Path("config") / "monitoring" / f"{code}.yaml"
    if _write_yaml(monitor_path, monitoring_policy, overwrite):
        created.append(monitor_path)

    return created


def _scaffold_fetchers(country_code: str, overwrite: bool) -> list[Path]:
    code = country_code.upper()
    pkg = country_code.lower()
    base = Path("data_fetchers") / pkg

    init_text = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""{code} data fetchers."""

from .electricity import {code}ElectricityFetcher
from .events import {code}EventProvider
from .gas import {code}GasFetcher
from .weather import {code}WeatherFetcher


def register_{pkg}() -> None:
    """Register {code} adapters in the country registry."""
    from config.country_registry import CountryRegistry

    CountryRegistry.register(
        "{code}",
        {{
            "electricity": {code}ElectricityFetcher,
            "weather": {code}WeatherFetcher,
            "gas": {code}GasFetcher,
            "events": {code}EventProvider,
        }},
    )
'''

    electricity_text = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""{code} electricity fetcher placeholder."""

import pandas as pd

from core.abstractions import ElectricityDataFetcher


class {code}ElectricityFetcher(ElectricityDataFetcher):
    """Placeholder electricity fetcher for {code}."""

    def __init__(self, config: object):
        self.config = config

    async def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Implement fetch_prices for {code}")

    async def fetch_load(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Implement fetch_load for {code}")
'''

    weather_text = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""{code} weather fetcher placeholder."""

import pandas as pd

from core.abstractions import WeatherDataFetcher


class {code}WeatherFetcher(WeatherDataFetcher):
    """Placeholder weather fetcher for {code}."""

    def __init__(self, config: object):
        self.config = config

    async def fetch_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Implement fetch_weather for {code}")
'''

    gas_text = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""{code} gas fetcher placeholder."""

import pandas as pd

from core.abstractions import GasDataFetcher


class {code}GasFetcher(GasDataFetcher):
    """Placeholder gas fetcher for {code}."""

    def __init__(self, config: object):
        self.config = config

    async def fetch_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Implement gas price fetch for {code}")
'''

    events_text = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""{code} event provider placeholder."""

import pandas as pd

from core.abstractions import EventDataProvider


class {code}EventProvider(EventDataProvider):
    """Placeholder event provider for {code}."""

    def __init__(self, config: object):
        self.config = config

    def get_holidays(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Implement holiday provider for {code}")

    def get_manual_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date_start", "date_end", "event_type", "description", "source"]
        )
'''

    created: list[Path] = []
    for filename, content in {
        "__init__.py": init_text,
        "electricity.py": electricity_text,
        "weather.py": weather_text,
        "gas.py": gas_text,
        "events.py": events_text,
    }.items():
        path = base / filename
        if _write_text(path, content, overwrite):
            created.append(path)

    return created


def _scaffold_qualification_test(country_code: str, overwrite: bool) -> list[Path]:
    code = country_code.upper()
    name = country_code.lower()
    path = Path("tests") / "qualification" / f"test_{name}_qualification.py"
    content = f'''# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Qualification skeleton for country {code}."""

import pytest


@pytest.mark.skip(reason="Implement qualification checks for {code}")
def test_{name}_qualification_smoke() -> None:
    """Add fetch schema, feature completeness, and smoke forecast checks."""
    assert True
'''

    return [path] if _write_text(path, content, overwrite) else []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set up country directories and scaffolding")
    parser.add_argument("country_code", help="ISO country code (2 letters)")
    parser.add_argument(
        "--scaffold-config",
        action="store_true",
        help="Create config/countries, selection policy, and monitoring YAML files",
    )
    parser.add_argument(
        "--scaffold-fetchers",
        action="store_true",
        help="Create placeholder data_fetchers/{country}/ package",
    )
    parser.add_argument(
        "--scaffold-tests",
        action="store_true",
        help="Create qualification test skeleton under tests/qualification/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite scaffold files if they already exist",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    country_code = args.country_code.upper()
    if len(country_code) != 2 or not country_code.isalpha():
        parser.error("Country code must be exactly 2 letters (e.g., PT, ES, DE)")

    print(f"\n{'=' * 60}")
    print(f"Setting up directory structure for: {country_code}")
    print(f"{'=' * 60}\n")

    try:
        manager = setup_country_directories(country_code)

        print("\n[OK] Successfully created directories:")
        print(f"  - {manager.get_raw_path()}")
        print(f"  - {manager.get_processed_path()}")
        print(f"  - {manager.get_events_path()}")
        print(f"  - {manager.get_metadata_path()}")

        created_files: list[Path] = []
        if args.scaffold_config:
            created_files.extend(_scaffold_country_config(country_code, overwrite=args.overwrite))
        if args.scaffold_fetchers:
            created_files.extend(_scaffold_fetchers(country_code, overwrite=args.overwrite))
        if args.scaffold_tests:
            created_files.extend(
                _scaffold_qualification_test(country_code, overwrite=args.overwrite)
            )

        print(f"\n[OK] Directory structure ready for {country_code}")

        if created_files:
            print("\nScaffold files created:")
            for path in created_files:
                print(f"  - {path}")

        info = manager.get_directory_info()
        print("\nDirectory info:")
        print(f"  Location: {info['base_path']}")
        print(f"  Exists: {info['exists']}")

        print("\n" + "=" * 60)
        print("Next steps:")
        print(f"1. Review configuration: config/countries/{country_code}.yaml")
        print("2. Register fetchers in data_fetchers/__init__.py")
        qualification_hint = (
            f"3. Run qualification: uv run python experiments/qualify_country.py "
            f"--country {country_code} --start YYYY-MM-DD --end YYYY-MM-DD"
        )
        print(qualification_hint)
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.error("Failed to set up country '%s': %s", country_code, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
